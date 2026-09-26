"""
数据层：自造训练数据 + 字符级分词器（角色 token 特殊实现）。

一套权重、多角色。支持两类训练样本：
  1. 任务样本（type="task"）：任务 -> 拆解 -> 执行结果（ROOT/LEAF 角色）
  2. 对话样本（type="chat"）：多轮对话（CHAT 角色）
两类样本共享同一个词表、同一个 RoleLM，训练时 loss 一起反传到同一套权重。

角色标识：输入里写 "[ROOT]" / "[LEAF]" / "[CHAT]"，分词器把它当成单独的特殊 token。
对话内部用普通文字 "用户："/"助手：" 表示说话人，轮次之间用分隔符 CHAT_SEP 隔开。
"""
import json
import os
import random

from config import CHAT_SEP

# 特殊 token（角色标识 + 控制符）。id 固定，方便其它模块引用。
SPECIAL = {
    "[PAD]": 0,
    "[ROOT]": 1,   # 根角色：拆解任务
    "[LEAF]": 2,   # 叶角色：执行原子任务
    "[EOS]": 3,    # 结束符
    "[UNK]": 4,    # 未登录字符
    "[CHAT]": 5,   # 对话角色：多轮聊天（新增）
}


class Tokenizer:
    """字符级分词器，但特殊 token 作为整体处理。"""
    def __init__(self, char2id, id2char, special):
        self.char2id = char2id
        self.id2char = id2char
        self.special = special
        self.id2special = {v: k for k, v in special.items()}
        self.unk_id = special["[UNK]"]
        self.eos_id = special["[EOS]"]
        # 特殊 token 按长度降序匹配，避免 "[ROOT]" 被拆成 "[" "ROOT" "]"
        self._special_keys = sorted(special.keys(), key=len, reverse=True)

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            matched = None
            for st in self._special_keys:
                if text.startswith(st, i):
                    matched = st
                    break
            if matched is not None:
                ids.append(self.special[matched])
                i += len(matched)
            else:
                ch = text[i]
                ids.append(self.char2id.get(ch, self.unk_id))
                i += 1
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i in self.id2special:
                out.append(self.id2special[i])
            elif i in self.id2char:
                out.append(self.id2char[i])
            # 其它 id（如 PAD）直接跳过
        return "".join(out)


# ===================== 对话数据（CHAT 角色） =====================
# 单轮问答池：同池配对（保证语义合理）。键是类别，值是 (用户问句列表, 助手答句列表)。
_CHAT_POOL = {
    "greet": (
        ["你好", "你好呀", "早上好", "嗨", "在吗", "你好啊"],
        ["你好！我是你的助手。", "你好呀，有什么可以帮你？", "早上好！",
         "嗨，我在，有什么事吗？", "你好，很高兴见到你。"],
    ),
    "self": (
        ["你是谁", "你叫什么名字", "你会做什么", "你会聊天吗", "你会创建文件吗",
         "你能帮我写东西吗", "你是什么模型"],
        ["我是一个能创建文件、写内容，也能陪你聊天的助手。",
         "我可以帮你创建文件、写入文字，也能和你聊天。",
         "我叫小助，是个简单的对话模型。",
         "我既能聊天，也能执行创建文件这类小任务。"],
    ),
    "qa": (
        ["一加一等于几", "中国的首都是哪里", "地球上有几个大洲", "水的化学式是什么", "一年有几个月"],
        ["一加一等于二。", "中国的首都是北京。", "地球上有七大洲。",
         "水的化学式是H2O。", "一年有十二个月。"],
    ),
    "chat": (
        ["谢谢", "你真棒", "讲个笑话", "今天天气不错", "再见", "你吃饭了吗"],
        ["不客气！", "谢谢夸奖，我会继续努力。",
         "好的：小猫问妈妈为什么自己长不高，妈妈说因为你还在喵喵喵。",
         "是啊，天气真好。", "再见，祝你一天愉快！", "我是程序，不用吃饭，不过谢谢关心。"],
    ),
}

# 多轮剧本（训练上下文记忆）：每条最后一轮一定是"助手"，符合 chat_pair 的要求。
_CHAT_SCRIPTS = [
    [("用户", "你好"), ("助手", "你好！有什么可以帮你？"),
     ("用户", "你会做什么"), ("助手", "我可以创建文件、写内容，也能陪你聊天。")],
    [("用户", "你是谁"), ("助手", "我是一个简单的对话模型，能聊天也能建文件。"),
     ("用户", "帮我建个文件"), ("助手", "好的，请告诉我文件名和内容，我会创建它。")],
    [("用户", "一加一等于几"), ("助手", "一加一等于二。"),
     ("用户", "谢谢"), ("助手", "不客气！")],
    [("用户", "早上好"), ("助手", "早上好！今天想做点什么？"),
     ("用户", "你会创建文件吗"), ("助手", "会呀，告诉我文件名和内容就行。"),
     ("用户", "那太好了"), ("助手", "随时找我。")],
]


def build_chat_samples(n, seed=42):
    """
    自造 n 条对话样本，单轮 + 多轮混合，返回 list of {"type":"chat","turns":[...]}。
    turns 形如 [["用户", u], ["助手", a], ...]，最后一轮一定是助手轮。
    """
    random.seed(seed)
    keys = list(_CHAT_POOL.keys())
    samples = []
    # 先放剧本多轮（少量但高质量，训练上下文连贯性）
    for sc in _CHAT_SCRIPTS:
        samples.append({"type": "chat", "turns": [list(t) for t in sc]})
    # 再填单轮（同池随机配对）
    while len(samples) < n:
        k = random.choice(keys)
        u = random.choice(_CHAT_POOL[k][0])
        a = random.choice(_CHAT_POOL[k][1])
        samples.append({"type": "chat", "turns": [["用户", u], ["助手", a]]})
    return samples[:n]


def format_chat(turns, sep=CHAT_SEP):
    """把多轮 turns 拼成 "用户：...<sep>助手：...<sep>用户：..." 形式。"""
    return sep.join(f"{sp}：{tx}" for sp, tx in turns)


def chat_pair(turns, sep=CHAT_SEP):
    """
    把一条多轮对话转成 (prompt, target) 训练对，供教师强制使用。
    prompt  = "[CHAT]" + 去掉最后一个助手轮的历史
    target  = "助手：" + 最后一个助手回复 + "[EOS]"
    """
    hist = turns[:-1]                       # 去掉最后一个助手轮
    last_reply = turns[-1][1]               # 最后一个助手回复
    prompt = "[CHAT]" + format_chat(hist, sep)
    target = "助手：" + last_reply + "[EOS]"
    return prompt, target


# ===================== 任务数据（ROOT/LEAF 角色，原有逻辑） =====================
def build_dataset(path, seed=42, n=300, chat_n=250):
    """
    生成合成训练数据并构建词表，存成 data.json。
    同时包含任务样本（ROOT/LEAF）和对话样本（CHAT），共用一个 RoleLM。

    实现要点：直接由"组合池"取唯一组合，避免随机抽样凑不够 n 条导致死循环。
    """
    random.seed(seed)

    # 文件名池：少量具名文件 + 一批 file_xxx.txt，保证组合池远大于 n
    filenames = [
        "hello.txt", "a.txt", "foo.txt", "bar.txt", "test.txt", "demo.txt",
        "out.txt", "x.txt", "data.txt", "note.txt", "log.txt", "result.txt",
    ] + [f"file{i:03d}.txt" for i in range(40)]

    # 内容池：少量中文/英文短串（控制词表规模）
    contents = [
        "你好", "世界", "你好世界", "hello", "test", "abc123",
        "数据", "文件内容", "ok", "内容A", "内容B", "测试",
    ]

    # 构造唯一组合池，并打散
    combos = [(fn, ct) for fn in filenames for ct in contents]
    random.shuffle(combos)

    samples = []

    def add(fn, ct, stype="task"):
        task = f"创建文件{fn}并写入内容{ct}"
        decomp = f"子任务1:创建文件{fn};子任务2:向{fn}写入内容{ct}"
        subtasks = [f"创建文件{fn}", f"向{fn}写入内容{ct}"]
        actions = [f"TOUCH {fn}", f"WRITE {fn} {ct}"]
        samples.append({
            "type": stype,
            "filename": fn,
            "content": ct,
            "task": task,
            "decomp": decomp,
            "subtasks": subtasks,
            "actions": actions,
        })

    # 保证演示任务一定在训练集里（并把它从池里去掉，避免重复）
    add("hello.txt", "你好")
    combos = [(fn, ct) for (fn, ct) in combos if not (fn == "hello.txt" and ct == "你好")]

    # 取前 n-1 个唯一组合填满任务样本
    for fn, ct in combos[: n - 1]:
        add(fn, ct)

    # 把演示任务（hello.txt / 你好）额外复制若干份，强化记忆，
    # 确保最终验证目标稳定可达（原型阶段用重复样本换可靠性，合理）。
    for _ in range(20):
        add("hello.txt", "你好")

    # 加入对话样本（CHAT 角色）
    for cs in build_chat_samples(chat_n, seed):
        samples.append(cs)

    # 构建词表：扫描所有用到的字符（任务 + 对话 + 分隔符/冒号）
    chars = set([CHAT_SEP, "："])   # 对话格式化时会用到分隔符和中文冒号
    for s in samples:
        if s["type"] == "task":
            chars.update(s["task"])
            chars.update(s["decomp"])
            for sub in s["subtasks"]:
                chars.update(sub)
            for a in s["actions"]:
                chars.update(a)
        else:  # chat
            for sp, tx in s["turns"]:
                chars.update(sp)
                chars.update(tx)
    chars = sorted(chars)

    char2id = {c: i + len(SPECIAL) for i, c in enumerate(chars)}
    id2char = {v: k for k, v in char2id.items()}

    data = {
        "samples": samples,
        "vocab": {
            "special": SPECIAL,
            "char2id": char2id,
            "id2char": {str(k): v for k, v in id2char.items()},  # json 键必须是字符串
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def load_dataset(path):
    """读取 data.json，返回 (samples, tokenizer, vocab_size)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    vocab = data["vocab"]
    special = {k: int(v) for k, v in vocab["special"].items()}
    char2id = {k: int(v) for k, v in vocab["char2id"].items()}
    id2char = {int(k): v for k, v in vocab["id2char"].items()}
    tok = Tokenizer(char2id, id2char, special)
    vocab_size = len(char2id) + len(SPECIAL)  # id 从 0 开始连续编号
    return data["samples"], tok, vocab_size


def parse_subtasks(root_text):
    """
    把 ROOT 产出的拆解文本解析成原子任务列表。
    格式示例："子任务1:创建文件hello.txt;子任务2:向hello.txt写入内容你好"
    """
    text = root_text.replace("[EOS]", "")
    subs = []
    for piece in text.split(";"):
        if ":" in piece:
            sub = piece.split(":", 1)[1].strip()
            if sub:
                subs.append(sub)
    return subs
