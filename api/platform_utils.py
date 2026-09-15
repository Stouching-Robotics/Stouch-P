"""
平台检测工具模块

在模块导入时一次性完成 OS 和 CPU 架构检测，
对外暴露三个模块级常量:OS、ARCH、TAG。供 xu_camera 中方法构建平台特定的库文件搜索路径。
示例:
    from .platform_utils import OS, ARCH, TAG
    # Linux x86_64  → OS='linux',    ARCH='x86_64',  TAG='linux_x86_64'
    # Windows 64bit → OS='windows',  ARCH='x86_64',  TAG='windows_x86_64'
    # Linux aarch64 → OS='linux',    ARCH='aarch64', TAG='linux_aarch64'
"""

import platform
import sys
from struct import calcsize














def _get_os() -> str:
    os = platform.system().lower()

    os = ('cygwin' if os.startswith('cygwin') else
          'linux' if os.startswith('linux') else
          'freebsd' if os.startswith(
              ('freebsd', 'openbsd', 'isilon onefs')) else os)

    if os == 'linux':
        if hasattr(sys, 'getandroidapilevel'):
            os = 'android'
        else:
            cname, _ = platform.libc_ver()
            if cname == 'musl':
                os = 'alpine'
            elif cname == 'libc':
                os = 'android'
    return os





def _get_arch() -> str:
    arch = platform.machine().lower()
    arch_table = {
        'x86':      ('i386', 'i486', 'i586', 'i686', 'x86'),
        'x86_64':   ('x64', 'x86_64', 'amd64', 'intel'),
        'arm':      ('armv5', 'armv5l', 'armv5te', 'armv5tel', 'armv5tejl'),
        'armv6':    ('armv6l',),
        'armv7':    ('armv7l',),
        'aarch32':  ('aarch32',),
        'aarch64':  ('aarch64', 'arm64'),
        'ppc64le':  ('ppc64le',),
        'mips32el': ('mipsel', 'mips32el'),
        'mips64el': ('mips64el',),
        'riscv64':  ('riscv64',),
    }
    for alias, names in arch_table.items():
        if arch in names:
            arch = alias
            break

    
    if arch == 'x86_64' and platform.system() == 'Windows':
        if calcsize('P') * 8 == 32:
            arch = 'x86'

    return arch




OS   = _get_os()
ARCH = _get_arch()
TAG  = f'{OS}_{ARCH}'
