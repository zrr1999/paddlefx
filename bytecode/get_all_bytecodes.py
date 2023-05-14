from __future__ import annotations

import dis
import sys

assert (
    sys.version_info.major == 3 and sys.version_info.minor >= 8
), "Python 3.8+ required"
# print and save all bytecode in python
with open(f'./all_bytecodes/3.{sys.version_info.minor}.txt', 'w') as f:
    f.write('')
with open(f'./all_bytecodes/3.{sys.version_info.minor}.txt', 'a') as f:
    for i in range(256):
        opname = dis.opname[i]
        if not opname.startswith('<'):
            print(opname)
            f.write(opname + '\n')
