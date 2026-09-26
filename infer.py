"""
推理脚本：对一个用户任务跑完整条链，并打印每一步，证明闭环真的跑通。

默认任务就是验证任务："创建一个叫 hello.txt 的文件，里面写'你好'"。
用法：
    python infer.py
    python infer.py "创建文件foo.txt并写入内容世界"
"""
import os
import sys

import torch

from config import WORK_DIR, DATA_PATH, CKPT_PATH, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN, SEED
from model import RoleLM
from data import build_dataset, load_dataset
from pipeline import run_pipeline


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "创建文件hello.txt并写入内容你好"

    if not os.path.exists(DATA_PATH):
        build_dataset(DATA_PATH, SEED)
    _, tok, vocab_size = load_dataset(DATA_PATH)

    model = RoleLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        print(f"[推理] 已加载权重 {CKPT_PATH}")
    else:
        print("[推理] 未找到权重，使用随机初始化（结果大概率是错的，先训练再跑）。")

    res = run_pipeline(model, tok, task, WORK_DIR, do_execute=True)

    # ---- 打印结果 ----
    print("\n================ 推理结果 ================")
    print(f"任务     : {res['task']}")
    print(f"ROOT 拆解: {res['root_text']}")
    print(f"原子任务 : {res['subtasks']}")
    for i, leaf in enumerate(res["leaves"], 1):
        print(f"  -- LEAF {i} --")
        print(f"     子任务 : {leaf['subtask']}")
        print(f"     动作   : {leaf['action']}")
        if leaf["exec"]:
            print(f"     执行   : ok={leaf['exec']['ok']} path={leaf['exec'].get('path')} {leaf['exec'].get('error','')}")
        if leaf["valid"]:
            v = leaf["valid"]
            print(f"     验证   : exists={v['exists']} match={v['match']} content={v['content']!r}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
