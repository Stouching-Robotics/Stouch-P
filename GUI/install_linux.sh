#!/bin/bash
# ============================================================
# SighTac-GUI Linux 启动器安装脚本
#
# 【这是什么】
#   把打包好的 SighTac-GUI 程序注册到 Linux 桌面系统，
#   让应用菜单 / Dock / 任务栏显示 logo 图标。
#
# 【怎么用】
#   1. 把下面 3 个文件放进同一个文件夹（可打包成 zip 发送）：
#        SighTac-GUI_XXXXXX          ← 可执行程序
#        install_linux.sh            ← 本脚本
#        logo.png                    ← 图标文件
#   2. 在该文件夹里打开终端，执行：
#        chmod +x install_linux.sh
#        ./install_linux.sh
#   3. 看到"✅ 启动器已创建"即安装成功。
#   4. 按 Win(Super) 键搜索 "SighTac" 即可从应用菜单带 logo 启动。
#      若菜单里没出现，按 Alt+F2 输入 r 回车刷新桌面后再试。
#
#   ⚠️ 重要：安装完成后请不要移动/重命名这个文件夹！
#      启动器记录的是程序的绝对路径，移动后需重跑一次本脚本。
#
# 【其他用法】
#   ./install_linux.sh                        # 自动检测（同目录 或 仓库 dist/）
#   ./install_linux.sh <可执行文件路径>        # 手动指定程序路径（最通用）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
ICON_DIR="$DATA_HOME/icons"
APPS_DIR="$DATA_HOME/applications"
ICON_NAME="SighTac-GUI.png"
DESKTOP_FILE="$APPS_DIR/SighTac-GUI.desktop"

# ---------- 1. 确定可执行程序路径 ----------
if [ -n "$1" ]; then
    # readlink -f 对不存在的路径会返回空并报错，这里用 || 兜底避免 set -e 提前退出
    BIN_PATH="$(readlink -f "$1" 2>/dev/null || echo "$1")"
else
    # 自动查找：优先脚本同目录（外发模式：程序+脚本放一起），其次仓库 dist/（开发模式）
    BIN_PATH="$(ls -t "$SCRIPT_DIR"/SighTac-GUI* 2>/dev/null | head -1)"
    [ -z "$BIN_PATH" ] && BIN_PATH="$(ls -t "$REPO_ROOT"/dist/SighTac-GUI* 2>/dev/null | head -1)"
fi

if [ -z "$BIN_PATH" ] || [ ! -f "$BIN_PATH" ]; then
    echo "❌ 找不到可执行程序。请指定路径：$0 <可执行文件路径>" >&2
    exit 1
fi
if [ ! -x "$BIN_PATH" ]; then
    echo "⚠️  ${BIN_PATH} 没有执行权限，先尝试添加..." >&2
    chmod +x "$BIN_PATH"
fi
echo "✅ 可执行程序: $BIN_PATH"

# ---------- 2. 确定 logo 图标来源 ----------
ICON_SRC=""
for cand in \
    "$SCRIPT_DIR/../../logo/logo.png" \
    "$SCRIPT_DIR/../logo/logo.png" \
    "$SCRIPT_DIR/logo.png" \
    "$SCRIPT_DIR/../logo/logo_hard.png" \
    "$REPO_ROOT/dist/SighTac-GUI.png"; do
    if [ -f "$cand" ]; then
        ICON_SRC="$cand"
        break
    fi
done

# ---------- 3. 安装图标 ----------
mkdir -p "$ICON_DIR" "$APPS_DIR"
if [ -n "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$ICON_DIR/$ICON_NAME"
    echo "✅ 图标已安装: $ICON_DIR/$ICON_NAME"
    ICON_LINE="Icon=$ICON_DIR/$ICON_NAME"
else
    echo "⚠️  未找到 logo.png（可从 ../logo/ 或脚本同目录提供），将使用系统默认图标" >&2
    ICON_LINE="Icon=applications-development"
fi

# ---------- 4. 生成 .desktop 启动器 ----------
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=SighTac-GUI
Comment=触觉力反馈传感器上位机软件
Exec=$BIN_PATH
${ICON_LINE}
Terminal=false
Type=Application
Categories=Development;
StartupNotify=true
EOF
chmod +x "$DESKTOP_FILE"
echo "✅ 启动器已创建: $DESKTOP_FILE"

# ---------- 5. 刷新桌面数据库 ----------
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi

echo ""
echo "🎉 安装完成！"
echo "   按 Super(Win) 键搜索 SighTac，即可从应用菜单启动（带 logo）。"
echo "   若菜单未出现，按 Alt+F2 输入 r 回车刷新桌面后再试。"
