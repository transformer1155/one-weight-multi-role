"""
对话推理脚本：跑一个简单的多轮对话模型（CHAT 角色）。

用法：
    python chat.py                # 交互式多轮对话（输入 exit 退出）
    python chat.py "你会做什么"    # 单句问答，打印助手回复

模型仍是那套"一套权重、多角色"的 RoleLM：喂入带 [CHAT] 角色标识的对话历史，
它就会切换到对话模式，产出助手回复；喂 [ROOT]/[LEAF] 仍会去拆任务/执行。
"""
import os
import sys

import torch

from config import DATA_PATH, CKPT_PATH, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN, SEED
from model import RoleLM
from data import build_dataset, load_dataset
from pipeline import run_chat_turn


def load_model():
    if not os.path.exists(DATA_PATH):
        build_dataset(DATA_PATH, SEED)
    _, tok, vocab_size = load_dataset(DATA_PATH)
    model = RoleLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        print(f"[对话] 已加载权重 {CKPT_PATH}")
    else:
        print("[对话] 未找到权重，使用随机初始化（回复大概率无意义，请先训练）。")
    return model, tok


def main():
    model, tok = load_model()
    history = []

    # 单句模式
    single = sys.argv[1] if len(sys.argv) > 1 else None
    if single:
        history.append(("用户", single))
        reply = run_chat_turn(model, tok, history, beam=4)
        history.append(("助手", reply))
        print(reply)
        return

    # 交互式多轮模式
    print("==== 多轮对话（输入 exit / quit / 退出 结束）====")
    while True:
        try:
            u = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if u.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            break
        if not u:
            continue
        history.append(("用户", u))
        reply = run_chat_turn(model, tok, history, beam=4)
        history.append(("助手", reply))
        print("助手：" + reply)


if __name__ == "__main__":
    main()
