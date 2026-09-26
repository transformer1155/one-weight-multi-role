"""
推理流水线（闭环演示用）：ROOT 拆解 -> 每个 LEAF 执行 -> 执行器写文件 -> 验证器检查。

这个函数在训练脚本（train.py）和推理脚本（infer.py / validate.py）里被复用，
保证"演示"和"训练"走的是同一条链。
"""
import os
import torch

from config import CHAT_SEP
from model import generate, generate_beam
from data import parse_subtasks
from executor import execute
from validator import validate


def run_pipeline(model, tok, task, workdir, do_execute=True, expected=None, beam=4):
    """
    跑完整条链，返回结构化结果，方便打印/断言。

    task:        用户任务文本，例如 "创建文件hello.txt并写入内容你好"
    do_execute:  是否真的调用执行器写文件（训练里的"演示环"会设为 True）
    expected:    验证器用的期望内容（推理时一般不知道，传 None 只查存在性）
    beam:        推理用束搜索宽度（>=1；1 即贪心）。字符级生成用 beam 更稳。
    """
    os.makedirs(workdir, exist_ok=True)
    model.eval()

    # 解码函数：按 beam 选择贪心或束搜索
    do_decode = (generate_beam if beam and beam > 1 else generate)

    res = {
        "task": task,
        "root_text": "",
        "subtasks": [],
        "leaves": [],   # 每个叶子：subtask / action / exec / valid
    }

    with torch.no_grad():
        # 1) ROOT：喂入带 [ROOT] 角色标识的任务，产出拆解
        root_ids = tok.encode("[ROOT]" + task)
        root_gen = do_decode(model, torch.tensor([root_ids]), eos_id=tok.eos_id, beam=beam)
        # 只取"生成部分"（切掉 prompt 前缀），否则会把 [ROOT]任务 也当成输出
        root_text = tok.decode(root_gen[0].tolist()[len(root_ids):]).replace("[EOS]", "").strip()
        res["root_text"] = root_text

        # 2) 解析成原子任务
        subs = parse_subtasks(root_text)
        res["subtasks"] = subs

        # 3) 每个 LEAF：喂入带 [LEAF] 角色标识的子任务，产出动作
        for sub in subs:
            leaf_ids = tok.encode("[LEAF]" + sub)
            leaf_gen = do_decode(model, torch.tensor([leaf_ids]), eos_id=tok.eos_id, beam=beam)
            # 同样只取生成部分，避免 [LEAF]子任务 前缀混入动作
            action = tok.decode(leaf_gen[0].tolist()[len(leaf_ids):]).replace("[EOS]", "").strip()

            leaf = {"subtask": sub, "action": action, "exec": None, "valid": None}
            if do_execute:
                ex = execute(action, workdir)
                leaf["exec"] = ex
                # 4) 验证器：真读文件检查
                if action.startswith("WRITE"):
                    fn = action.split(" ", 2)[1]
                    leaf["valid"] = validate(fn, expected if expected is not None else None, workdir)
                elif action.startswith("TOUCH"):
                    fn = action.split(" ", 1)[1]
                    leaf["valid"] = validate(fn, None, workdir)
            res["leaves"].append(leaf)

    return res


def run_chat_turn(model, tok, history, beam=4, max_new=96):
    """
    跑一轮对话，返回助手回复文本（纯回复，已去掉"助手："前缀）。

    history: list of (speaker, text)，例如
        [("用户","你好"), ("助手","你好！"), ("用户","你会做什么")]
    调用时 history 的最后一个元素应是"用户"的最新输入（尚无助手回复），
    这正好构成喂给模型的 prompt： "[CHAT]用户：你好|助手：你好！|用户：你会做什么"

    内部逻辑：把 history 拼成带 [CHAT] 角色标识的文本 -> beam 解码 ->
    切掉 prompt 前缀 -> 去掉可能重复的"助手："前缀 -> 返回纯回复。
    与 ROOT/LEAF 一样，用的是同一个 RoleLM，只是输入带 [CHAT] 角色 token。
    """
    model.eval()
    prompt = "[CHAT]" + CHAT_SEP.join(f"{sp}：{tx}" for sp, tx in history)
    ids = tok.encode(prompt)
    do_decode = (generate_beam if beam and beam > 1 else generate)
    if do_decode is generate_beam:
        gen = do_decode(model, torch.tensor([ids]), eos_id=tok.eos_id, beam=beam, max_new=max_new)
    else:
        gen = do_decode(model, torch.tensor([ids]), max_new=max_new, eos_id=tok.eos_id)
    out = tok.decode(gen[0].tolist()[len(ids):]).replace("[EOS]", "").strip()
    # 去掉可能重复的"助手："前缀（target 模板里带这个前缀，模型可能原样吐出）
    if out.startswith("助手："):
        out = out[len("助手："):].strip()
    return out
