"""
执行器：真正调用操作系统写文件（不是模拟）。

它接收 LEAF 输出的"动作字符串"，解析后真的在 WORK_DIR 里创建/写入文件：
- "TOUCH <文件名>"      -> 创建一个空文件
- "WRITE <文件名> <内容>" -> 把内容写进文件（内容可含中文，无空格）

这是闭环里"真实副作用"的那一环：模型说了不算，得真写出来才算数。
"""
import os


def execute(action, workdir):
    """
    执行一条动作。返回执行结果字典。
    action: 例如 "WRITE hello.txt 你好"
    """
    action = action.strip()
    try:
        if action.startswith("TOUCH"):
            fn = action.split(" ", 1)[1].strip()
            path = os.path.join(workdir, fn)
            open(path, "w", encoding="utf-8").close()   # 真实创建空文件
            return {"ok": True, "cmd": "TOUCH", "path": path}

        if action.startswith("WRITE"):
            parts = action.split(" ", 2)                # 最多切 3 段：WRITE / 文件名 / 内容
            fn = parts[1].strip()
            content = parts[2] if len(parts) > 2 else ""
            path = os.path.join(workdir, fn)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)                        # 真实写入内容
            return {"ok": True, "cmd": "WRITE", "path": path}

        return {"ok": False, "cmd": None, "path": None, "error": f"未知动作: {action!r}"}
    except Exception as e:  # 文件名非法、路径错误等
        return {"ok": False, "cmd": None, "path": None, "error": str(e)}
