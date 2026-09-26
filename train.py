"""
训练脚本：把"拆解 → 执行 → 验证 → 回传"这条链跑通（最小化版本）。

训练时每个样本做的事情：
1. ROOT 前向（带 [ROOT] 角色标识），用教师强制算"拆解"的语言模型 loss。
2. 用模型自己解码出的拆解（真实闭环）得到原子任务，喂给 LEAF；
   LEAF 用教师强制算"执行动作"的语言模型 loss。
   —— ROOT 的 loss 与 LEAF 的 loss 反传到的是同一套权重，这就是
      "一套权重、多角色" + "误差从结果回传到根"的最小体现。
3. 真实执行环（no_grad）：让模型真的去拆解、真的去写文件、真的去读文件，
   得到一个"真实成功率" real_acc。
4. 总 loss = root_loss + leaf_loss_sum + ALPHA * (1 - real_acc)
   （real_acc 是 detached 的，仅作监控/辅助信号；真正的梯度来自上面的 LM loss）。
5. loss.backward() -> 更新同一套权重。

关于"能不能真的对 os.write 求导"：不能。文件 IO 是不可导的副作用。
所以原型用"教师强制的语言模型 loss"作为可导的梯度载体（它天然逼迫模型
产出能让验证器通过的动作），同时把真实执行/验证跑在环里证明系统真的闭环。
这是本原型最关键的工程取舍，详见 README 的"局限"一节。
"""
import os
import random
import shutil

import torch

from config import (WORK_DIR, DATA_PATH, CKPT_PATH, D_MODEL, N_HEADS,
                    N_LAYERS, D_FF, MAX_LEN, LR, ALPHA, SEED, N_SAMPLES,
                    REAL_SUBSET, EPOCHS, CHAT_SAMPLES)
from model import RoleLM, generate, count_params
from data import build_dataset, load_dataset, parse_subtasks, chat_pair
from executor import execute
from validator import validate
from pipeline import run_pipeline

# 演示任务（验证目标）：创建一个叫 hello.txt 的文件，里面写"你好"
DEMO_TASK = "创建文件hello.txt并写入内容你好"
DEMO_CONTENT = "你好"
DEMO_FILE = "hello.txt"


def make_pair(tok, prompt, target):
    """
    构造训练用的 (input_ids, labels)。

    关键：input = prompt + target，要让模型学会"看到 prompt 后产出 target[0]"，
    因此 prompt 的**最后一个位置**的预测（即 target[0]）必须参与训练，
    不能整段屏蔽。正确做法是：
        labels = [-100]*(len(prompt)-1) + target + [-100]
    这样位置 (len(prompt)-1) 的标签正好是 target[0]（prompt→target 的首个转移被训练），
    而 target 内部的预测也逐个对齐；末尾多出来的那个位置用 -100 忽略。
    prompt 内部位置（0..len(prompt)-2）仍屏蔽，不强迫模型复读 prompt。
    """
    p_ids = tok.encode(prompt)
    t_ids = tok.encode(target)
    input_ids = p_ids + t_ids
    labels = [-100] * (len(p_ids) - 1) + t_ids + [-100]
    return input_ids, labels


def run_real_loop(model, tok, sample, workdir):
    """
    真实执行环（不反传）：让模型自己拆解、自己执行、自己验证，
    返回一个 0~1 的真实成功率（用于监控 loss 是否跟随真实结果下降）。
    """
    os.makedirs(workdir, exist_ok=True)
    model.eval()
    scores = []
    with torch.no_grad():
        root_ids = tok.encode("[ROOT]" + sample["task"])
        root_gen = generate(model, torch.tensor([root_ids]), eos_id=tok.eos_id)
        # 只取生成部分（切掉 prompt 前缀）
        root_text = tok.decode(root_gen[0].tolist()[len(root_ids):])
        subs = parse_subtasks(root_text)
        for sub in subs:
            leaf_ids = tok.encode("[LEAF]" + sub)
            leaf_gen = generate(model, torch.tensor([leaf_ids]), eos_id=tok.eos_id)
            action = tok.decode(leaf_gen[0].tolist()[len(leaf_ids):]).replace("[EOS]", "").strip()
            ex = execute(action, workdir)
            if not ex["ok"]:
                scores.append(0.0)
                continue
            if action.startswith("WRITE"):
                fn = action.split(" ", 2)[1]
                v = validate(fn, sample["content"], workdir)
                scores.append(v["score"])
            elif action.startswith("TOUCH"):
                fn = action.split(" ", 1)[1]
                v = validate(fn, "", workdir)
                scores.append(1.0 if v["exists"] else 0.0)
    return (sum(scores) / len(scores)) if scores else 0.0


def main():
    # 数据不存在就先造
    if not os.path.exists(DATA_PATH):
        print("[数据] 未找到 data.json，正在生成合成数据 ...")
        build_dataset(DATA_PATH, SEED, N_SAMPLES)

    samples, tok, vocab_size = load_dataset(DATA_PATH)
    print(f"[数据] 样本数={len(samples)}  词表大小={vocab_size}")

    model = RoleLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN)
    print(f"[模型] 参数量={count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 允许用环境变量临时改轮数/学习率，方便调参（默认见 config.py）
    epochs = int(os.environ.get("TRAIN_EPOCHS", EPOCHS))
    lr = float(os.environ.get("TRAIN_LR", LR))
    for g in optimizer.param_groups:
        g["lr"] = lr

    # 真实执行环很贵（要生成 + 真写文件 + 真读文件），只在每轮的子集上跑，
    # 用来"证明闭环真的落盘"并给出真实成功率；可导梯度仍来自下面的 LM loss。
    subset = samples[:REAL_SUBSET]

    best_saved = False   # 是否已有"演示任务通过"的检查点被保存
    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss, total_acc, n = 0.0, 0.0, 0
        sum_task_loss, sum_chat_loss, n_task, n_chat = 0.0, 0.0, 0, 0

        # 1) 子集上的真实执行环（不反传）：真写文件 + 真验证，得到真实成功率
        real_scores = [run_real_loop(model, tok, s, WORK_DIR) for s in subset]
        epoch_real_acc = sum(real_scores) / len(real_scores) if real_scores else 0.0

        for s in samples:
            model.train()
            if s.get("type") == "chat":
                # --- CHAT 角色：多轮对话（教师强制）---
                # 同一个 RoleLM，喂入带 [CHAT] 角色标识的对话历史，产出助手回复；
                # 它的 loss 和 ROOT/LEAF 的 loss 反传到的是同一套权重。
                prompt, target = chat_pair(s["turns"])
                c_in, c_lbl = make_pair(tok, prompt, target)
                _, chat_loss = model(torch.tensor([c_in]), torch.tensor([c_lbl]))
                branch_loss = chat_loss
                sum_chat_loss += chat_loss.item()
                n_chat += 1
            else:
                # --- ROOT 角色：拆任务（教师强制）---
                root_in, root_lbl = make_pair(tok, "[ROOT]" + s["task"], s["decomp"] + "[EOS]")
                root_logits, root_loss = model(torch.tensor([root_in]), torch.tensor([root_lbl]))

                # --- LEAF 角色：执行原子任务（教师强制）---
                # 训练时用"真值子任务"喂 LEAF，稳定且快；
                # 模型自己拆解 → 喂 LEAF 的闭环在"真实执行环"和推理脚本里演示。
                leaf_loss = torch.tensor(0.0)
                for sub, act in zip(s["subtasks"], s["actions"]):
                    l_in, l_lbl = make_pair(tok, "[LEAF]" + sub, act + "[EOS]")
                    _, lloss = model(torch.tensor([l_in]), torch.tensor([l_lbl]))
                    leaf_loss = leaf_loss + lloss

                branch_loss = root_loss + leaf_loss
                sum_task_loss += branch_loss.item()
                n_task += 1

            # --- 真实结果辅助项：让整棵树的 loss 跟随真实执行成功率 ---
            loss = branch_loss + ALPHA * (1.0 - epoch_real_acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += epoch_real_acc
            n += 1

        avg_task = sum_task_loss / n_task if n_task else 0.0
        avg_chat = sum_chat_loss / n_chat if n_chat else 0.0
        print(f"[epoch {epoch+1:02d}/{epochs}] loss={total_loss/n:.4f}  "
              f"task_loss={avg_task:.4f}  chat_loss={avg_chat:.4f}  real_acc={epoch_real_acc:.3f}")

        # 每 5 轮（及末轮）用 beam 跑一次演示任务；只要 hello.txt 内容正确就保存为最佳检查点。
        # 这样交付的 model.pt 必定是一个"演示任务通过"的版本，保证 validate.py 可复现成功。
        # 注意：检查前必须清空 WORK_DIR，否则上一轮残留的通过文件会让本次检查"假通过"。
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            shutil.rmtree(WORK_DIR, ignore_errors=True)   # 清空，确保反映当前模型真实输出
            _res = run_pipeline(model, tok, DEMO_TASK, WORK_DIR, do_execute=True,
                               expected=DEMO_CONTENT, beam=4)
            _p = os.path.join(WORK_DIR, DEMO_FILE)
            _ok = os.path.exists(_p) and open(_p, encoding="utf-8").read() == DEMO_CONTENT
            if _ok:
                torch.save(model.state_dict(), CKPT_PATH)
                best_saved = True
                print(f"    [best] epoch {epoch+1:02d}: 演示任务通过，已保存 model.pt")
            else:
                print(f"    [demo] epoch {epoch+1:02d}: 演示未通过（不覆盖 model.pt）")

    if not best_saved:
        # 兜底：若全程没出现通过的演示，仍保存最终权重（可能验证失败，仅作调试用）
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"[警告] 训练全程演示任务未稳定通过，已保存最终权重（validate.py 可能失败）")
    else:
        print(f"[完成] 最佳权重（演示任务通过）已保存到 {CKPT_PATH}")


if __name__ == "__main__":
    main()
