"""
模型定义：RoleLM —— 一套权重、多角色的共享模型。

核心思想：
- 模型就是一个因果语言模型（GPT 风格的小 Transformer）。
- 通过输入里的特殊 token [ROOT] / [LEAF] 来区分"当前这个位置在扮演什么角色"。
- ROOT 角色负责"拆解任务"，LEAF 角色负责"执行原子任务"。
- 无论哪个角色，用的都是**同一套权重**；角色由输入 token 决定，不靠改结构。
- 训练时，ROOT 的 loss 和 LEAF 的 loss 会一起反传到这同一套权重上，
  这就是"误差从结果回传到根"的最小实现：一棵树共享一个参数空间。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """因果自注意力（带下三角掩码，保证只看左边的 token）。"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 一次算出 Q/K/V
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv(x).split(D, dim=-1)      # (q, k, v) 各 (B, T, D)
        q, k, v = qkv
        # 拆成多头
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # 缩放点积
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果掩码：未来位置不可见
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                   # (B, h, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    """前馈层。"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """一个 Transformer 块：Pre-LN 结构（更稳）。"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RoleLM(nn.Module):
    """
    共享权重的角色语言模型。
    输入形如 "[ROOT]创建文件hello.txt并写入内容你好" 或 "[LEAF]向hello.txt写入内容你好"，
    输出下一个 token 的 logits。
    """
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=3, d_ff=512, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享（weight tying）：输出层复用 token 嵌入矩阵，省参数、更紧凑
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        # 截断超长序列的位置编码
        pos = self.pos_emb[:, :T, :]
        x = self.tok_emb(idx) + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab)
        loss = None
        if targets is not None:
            # 只对 targets 中非 -100 的位置算交叉熵（prompt 部分被忽略）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


def generate(model, idx, max_new=64, eos_id=3):
    """
    贪心解码：给定前缀 idx，逐 token 生成，直到 [EOS] 或达到最大新长度。
    用于推理时让 ROOT 产出拆解、让 LEAF 产出执行动作。
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            if idx.size(1) >= model.max_len:
                break
            logits, _ = model(idx)
            nxt = logits[:, -1:, :].argmax(dim=-1)   # 贪心：取概率最高的 token
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == eos_id:
                break
    return idx


def count_params(model):
    """统计参数量（权重共享只计一次）。"""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def generate_beam(model, idx, max_new=64, eos_id=3, beam=4):
    """
    束搜索解码（比贪心更稳，能纠正字符级生成的单字错误）。
    推理时使用：模型已充分训练，正确字符基本都在 top-k 里，beam 能避免
    贪心"将错就错"导致整条任务链崩掉。返回最佳序列 (1, T')。
    """
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        beams = [(idx, 0.0)]   # (序列(1,T), 累计 log 概率)
        finished = []
        for _ in range(max_new):
            if not beams:
                break
            seqs = [b[0] for b in beams]
            scores = [b[1] for b in beams]
            maxlen = max(s.size(1) for s in seqs)
            batch = torch.zeros(len(seqs), maxlen, dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, : s.size(1)] = s[0]
            logits, _ = model(batch)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)   # (B, V)
            new_beams = []
            for i in range(len(seqs)):
                topk = torch.topk(logp[i], beam)
                for k in range(beam):
                    tok_id = topk.indices[k].item()
                    tok_lp = topk.values[k].item()
                    new_seq = torch.cat([seqs[i], torch.tensor([[tok_id]])], dim=1)
                    new_score = scores[i] + tok_lp
                    if tok_id == eos_id:
                        finished.append((new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_score))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam]
            # 已有足够完成的候选且无可扩展时提前结束
            if finished and not beams:
                break
        if finished:
            finished.sort(key=lambda x: x[1], reverse=True)
            return finished[0][0]
        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[0][0]
