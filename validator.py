"""
验证器：真实检查文件是否创建成功、内容是否正确（不是模拟）。

它真的去读 WORK_DIR 里的文件：
- exists: 文件是否存在
- content: 实际读到的内容
- match:  内容是否与期望一致（expected 为 None 时只检查存在性）
- score:  0.0~1.0 的可微友好标量（这里是硬指标：符合得 1，否则 0）
"""
import os


def validate(filename, expected, workdir):
    """
    filename: 文件名（不含目录）
    expected: 期望内容；传 None 表示只检查"文件存在"
    """
    path = os.path.join(workdir, filename)
    result = {"path": path, "exists": False, "content": "", "match": False, "score": 0.0}

    if not os.path.exists(path):
        return result

    result["exists"] = True
    try:
        with open(path, "r", encoding="utf-8") as f:
            actual = f.read()
    except Exception as e:
        result["error"] = str(e)
        return result

    result["content"] = actual
    if expected is None:
        result["match"] = True          # 没给期望内容，只要存在就认为通过
        result["score"] = 1.0
    else:
        result["match"] = (actual == expected)
        result["score"] = 1.0 if result["match"] else 0.0
    return result
