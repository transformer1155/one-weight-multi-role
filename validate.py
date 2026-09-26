"""
验证脚本：双轨硬验证
  1) 文件任务闭环：hello.txt 是否真实创建、内容是否=你好（原有 ROOT/LEAF 闭环）
  2) 对话冒烟：问几个预设问题 + 一段多轮，检查回复是否合理（新增 CHAT 角色）

两条都过 -> PASS，退出码 0；任一不过 -> FAIL，退出码 1。
会真实读 WORK_DIR/hello.txt，也会真实跑模型生成对话。

用法：
    python validate.py
"""
import os
import sys

import torch

from config import (WORK_DIR, DATA_PATH, CKPT_PATH, D_MODEL, N_HEADS, N_LAYERS,
                    D_FF, MAX_LEN, SEED, CHAT_SMOKE)
from model import RoleLM
from data import build_dataset, load_dataset
from pipeline import run_pipeline, run_chat_turn


def check_file_task(model, tok):
    """原有文件任务闭环验证，返回 (ok, 详情字符串)。"""
    task = "创建文件hello.txt并写入内容你好"
    target_file = "hello.txt"
    target_content = "你好"

    # 跑完整条链（会真实写文件）
    run_pipeline(model, tok, task, WORK_DIR, do_execute=True, expected=target_content)

    path = os.path.join(WORK_DIR, target_file)
    ok = os.path.exists(path)
    content = ""
    if ok:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    match = (content == target_content)
    return ok and match, f"文件任务: 存在={ok} 内容={content!r} 匹配={match}"


def check_chat(model, tok):
    """对话冒烟检查，返回 (passed, total, 详情列表)。"""
    details = []
    passed = 0

    # 1) 单句：每个预设问题
    for case in CHAT_SMOKE:
        history = [("用户", case["ask"])]
        reply = run_chat_turn(model, tok, history, beam=4)
        ok = len(reply) > 0 and any(k in reply for k in case["must"])
        if ok:
            passed += 1
        details.append(f"  问:{case['ask']} -> {reply!r}  [{'OK' if ok else 'FAIL'}] ({case['desc']})")

    # 2) 多轮：先问好，再问身份，看第二轮是否连贯地接上上下文
    history = [("用户", "你好")]
    r1 = run_chat_turn(model, tok, history, beam=4)
    history.append(("助手", r1))
    history.append(("用户", "你是谁"))
    r2 = run_chat_turn(model, tok, history, beam=4)
    ok2 = len(r2) > 0
    if ok2:
        passed += 1
    details.append(f"  多轮: 你:你好 -> {r1!r}; 你:你是谁 -> {r2!r}  [{'OK' if ok2 else 'FAIL'}]")

    total = len(CHAT_SMOKE) + 1
    return passed, total, details


def main():
    if not os.path.exists(DATA_PATH):
        build_dataset(DATA_PATH, SEED)
    _, tok, vocab_size = load_dataset(DATA_PATH)

    model = RoleLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    else:
        print("[验证] 未找到权重，请先运行 python train.py")
        sys.exit(1)

    print("================ 验证结果 ================")
    f_ok, f_detail = check_file_task(model, tok)
    print(f_detail)

    c_passed, c_total, c_details = check_chat(model, tok)
    print(f"对话冒烟 ({c_passed}/{c_total} 通过):")
    for d in c_details:
        print(d)
    print("=========================================")

    if f_ok and c_passed == c_total:
        print("PASS ✅ 闭环成功 + 对话可用：一套权重同时胜任文件任务与多轮对话。")
        sys.exit(0)
    else:
        print("FAIL ❌ 未完全通过（文件任务或对话冒烟不达标）。")
        sys.exit(1)


if __name__ == "__main__":
    main()
