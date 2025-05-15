import re
import periodictable as pt

def get_atomic_mass(element):
    if hasattr(pt, element):
        return getattr(pt, element).mass
    else:
        raise ValueError(f"Unknown element: {element}")

def multiply_group(group, multiplier):

    return {elem: count * multiplier for elem, count in group.items()}

def merge_groups(base, addition):

    for elem, count in addition.items():
        base[elem] = base.get(elem, 0) + count
    return base

def parse_formula(formula):

    tokens = re.findall(r'([A-Z][a-z]?|\(|\)|\d+)', formula)
    stack = [{}]
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token == '(':
            stack.append({})
            i += 1
        elif token == ')':
            group = stack.pop()
            i += 1
            multiplier = 1
            if i < len(tokens) and tokens[i].isdigit():
                multiplier = int(tokens[i])
                i += 1
            group = multiply_group(group, multiplier)
            stack[-1] = merge_groups(stack[-1], group)
        elif re.match(r'[A-Z][a-z]?', token):
            elem = token
            count = 1
            i += 1
            if i < len(tokens) and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1
            stack[-1][elem] = stack[-1].get(elem, 0) + count
        else:
            i += 1

    return stack[0]

def calculate_mass(elements, mole=1.0):

    mass = {}
    for elem, count in elements.items():
        mass[elem] = count * get_atomic_mass(elem) * mole
    return mass

def display_mass_ratio(mass_dict):

    total_mass = sum(mass_dict.values())
    print("元素质量比:")
    for elem, mass in mass_dict.items():
        ratio = mass / total_mass
        print(f"{elem}: {ratio:.5f}%")

def direct_tuning():
    print("\n--- 直接微调制备 ---")
    formula = input("输入分子式 (例如 Fe2O3): ")
    mole = float(input("输入摩尔数: "))
    elements = parse_formula(formula)
    mass = calculate_mass(elements, mole)
    display_mass_ratio(mass)

def mof_derived():
    print("\n--- MOF 衍生制备 ---")
    formula = input("输入MOF分子式 (例如 C8H12N4Co): ")
    mole = float(input("输入摩尔数: "))
    elements = parse_formula(formula)
    mass = calculate_mass(elements, mole)
    display_mass_ratio(mass)

def mixed_calcination():
    print("\n--- 金属/磁混合煅烧法 ---")
    metal_formula = input("输入金属氧化物分子式 (例如 CuO): ")
    metal_mole = float(input("输入金属氧化物摩尔数: "))
    additive_formula = input("输入添加物分子式 (可以是C，或任意化合物): ")
    additive_mass = float(input("输入添加物质量 (单位g): "))

    # 解析金属氧化物
    metal_elements = parse_formula(metal_formula)
    mass_metal = calculate_mass(metal_elements, metal_mole)

    # 解析添加物
    additive_elements = parse_formula(additive_formula)
    mass_additive = calculate_mass(additive_elements, 1.0)

    # 归一化后按质量比例加进去
    total_additive_mass = sum(mass_additive.values())
    factor = additive_mass / total_additive_mass

    for elem, mass in mass_additive.items():
        mass_metal[elem] = mass_metal.get(elem, 0) + mass * factor

    display_mass_ratio(mass_metal)


def main():
    print("金属/磁性催化剂 质量比计算器")
    print("选择模式:")
    print("1. 直接微调制备")
    print("2. MOF 衍生制备")
    print("3. 金属/磁混合煅烧法")
    choice = input("输入选择 (1/2/3): ")

    if choice == '1':
        direct_tuning()
    elif choice == '2':
        mof_derived()
    elif choice == '3':
        mixed_calcination()
    else:
        print("无效选择")

if __name__ == "__main__":
    main()