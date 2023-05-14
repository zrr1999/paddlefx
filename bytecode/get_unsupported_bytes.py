from __future__ import annotations

from paddlefx.translator import InstructionTranslator

for version in ["3.8", "3.9", "3.10", "3.11"]:
    with open(f'./all_bytecodes/{version}.txt', 'r') as f:
        all_bytecode = f.read().split("\n")
    with open(f'./unspported_bytecodes/{version}.txt', 'w') as f:
        f.write('')
    with open(f'./unspported_bytecodes/{version}.txt', 'a') as f:
        for opname in all_bytecode:
            if not hasattr(InstructionTranslator, opname):
                print(opname)
                f.write(opname + '\n')
