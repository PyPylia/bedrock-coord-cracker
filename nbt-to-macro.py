from python_nbt.nbt import read_from_nbt_file
from pyperclip import copy

file = read_from_nbt_file("bedrock.nbt")
palette = []

for block in file["palette"]:
    palette.append(block["Name"].value)

checks = [[] for i in range(8)]

for block in file["blocks"]:
    value = block["state"].value
    if value == palette.index("minecraft:air"): continue
    pos = block["pos"]
    if value == palette.index("minecraft:bedrock"):
        checks[pos[1].value * 2].append((" ", pos[0].value, pos[1].value + 1, pos[2].value))
    else:
        checks[(3 - pos[1].value) * 2 + 1].append(("!", pos[0].value, pos[1].value + 1, pos[2].value))

while [] in checks: checks.remove([])

def parse_check(check):
    x = f"x{f' + {check[1]}' if check[1] else ''},".ljust(7, " ")
    z = f"z{f' + {check[3]}' if check[3] else ''}".ljust(6, " ")
    return f"    {check[0]}is_bedrock!({x} {check[2]}, {z})"

code = " &&\n".join(map(lambda x: " &&\n".join(map(parse_check, x)), checks))
copy(code)
print(code)