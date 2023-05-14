from __future__ import annotations

map_bytecode_version = {}
map_bytecode_num = {}

for version in ["3.8", "3.9", "3.10", "3.11"]:
    with open(f'./unspported_bytecodes/{version}.txt', 'r') as f:
        for opname in f.readlines():
            opname = opname.strip()
            print(opname)
            if map_bytecode_version.get(opname, None) is None:
                map_bytecode_version[opname] = [version]
                map_bytecode_num[opname] = 1
            else:
                map_bytecode_version[opname].append(version)
                map_bytecode_num[opname] += 1


with open(f'./classified_bytecode.txt', 'w') as f:
    f.write('')
with open(f'./classified_bytecode.txt', 'a') as f:
    f.write("字节码,出现次数,存在版本\n")
    for k, v in map_bytecode_version.items():
        v = "/".join(v)
        f.write(f"{k},{map_bytecode_num[k]},{v}\n")
