#!/usr/bin/env bash
# ============================================================================
# push_to_github.sh —— 一键把本项目推送到 GitHub（自动建仓库 + 推送）
#
# 在【能访问 github.com】的环境里执行（你的本机 / 有公网出口的云主机）：
#
#   1) 自动建仓库 + 推送（推荐，用你提供的 fine-grained PAT）：
#      GITHUB_TOKEN=github_pat_xxx ./push_to_github.sh https://github.com/<用户>/<仓库>.git
#
#   2) 只推送（仓库已存在时）：
#      ./push_to_github.sh https://github.com/<用户>/<仓库>.git
#
#   3) 交互输入地址：
#      GITHUB_TOKEN=github_pat_xxx ./push_to_github.sh
#
# 说明：
#   - GITHUB_TOKEN 仅通过环境变量临时传入，绝不写入任何文件。
#   - 若仓库已存在（GitHub 返回 422），脚本会自动跳过建仓库、直接推送。
#   - 若 PAT 没有“创建仓库”权限，建仓库会失败，但脚本会继续尝试推送
#     （此时请先在 GitHub 网页上手动建好同名空仓库）。
# ============================================================================
set -euo pipefail

REMOTE="${1:-}"

# ---- 用 token 改写远程地址（oauth2 方式，免输密码） ----
if [ -n "${GITHUB_TOKEN:-}" ] && [[ "${REMOTE:-}" == https://github.com/* ]]; then
  REMOTE="https://oauth2:${GITHUB_TOKEN}@${REMOTE#https://}"
  echo "[info] 已用 GITHUB_TOKEN 改写远程地址（oauth2 方式，无需输密码）"
fi

# ---- 交互询问地址 ----
if [ -z "${REMOTE:-}" ]; then
  read -r -p "请输入 GitHub 远程仓库地址 (https://github.com/<用户>/<仓库>.git): " REMOTE
fi
if [ -z "${REMOTE:-}" ]; then
  echo "[error] 未提供远程地址，退出。" >&2
  exit 1
fi

# ---- 从地址里解析 owner / repo（用于调用建仓库 API） ----
OWNER=""
REPO=""
if [[ "$REMOTE" =~ github\.com[/:]([^/]+)/(.+?)(\.git)?$ ]]; then
  OWNER="${BASH_REMATCH[1]}"
  REPO="${BASH_REMATCH[2]}"
  REPO="${REPO%.git}"
fi

# ---- 若提供了 token，先尝试在 GitHub 上创建仓库（幂等） ----
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "$REPO" ]; then
  echo "[info] 尝试在 GitHub 创建仓库 $OWNER/$REPO ..."
  CREATE_RESP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
    -X POST "https://api.github.com/user/repos" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -d "{\"name\":\"${REPO}\",\"private\":false,\"auto_init\":false}")
  case "$CREATE_RESP" in
    201) echo "[ok] 仓库已创建 ($OWNER/$REPO)" ;;
    422) echo "[info] 仓库已存在，跳过创建（422）" ;;
    *)   echo "[warn] 建仓库返回 HTTP $CREATE_RESP（可能无权限）；若推送失败，请先在 GitHub 网页手动建好同名空仓库" ;;
  esac
fi

# ---- git 初始化 / 提交 ----
if [ ! -d .git ]; then
  git init -q
  echo "[ok] git init 完成"
fi
git checkout -B main -q 2>/dev/null || git checkout -b main -q

cat > .gitignore <<'EOF'
sandbox/
__pycache__/
*.pyc
train.log
one-weight-multi-role.zip
EOF

git add .
echo "===== git status ====="
git status --short

if git diff --cached --quiet; then
  echo "[info] 没有需要提交的更改，跳过 commit。"
else
  git commit -q -m "feat: one-weight multi-role LM (ROOT/LEAF/CHAT) + task闭环 + 对话"
  echo "[ok] commit 完成"
fi

# ---- 推送（已带 token 的地址优先；否则用原始地址，由凭据助手/credential 提供） ----
PUSH_REMOTE="$REMOTE"
if [ -n "${GITHUB_TOKEN:-}" ] && [[ "$REMOTE" == https://github.com/* ]]; then
  PUSH_REMOTE="https://oauth2:${GITHUB_TOKEN}@${REMOTE#https://}"
fi
git remote remove origin 2>/dev/null || true
git remote add origin "$PUSH_REMOTE"
echo "[info] 推送 main -> origin ..."
git push -u origin main

echo "[done] 已推送到 $REMOTE"
