import re
import streamlit as st
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
    result = {}
    for elem, mass in mass_dict.items():
        ratio = mass / total_mass
        result[elem] = ratio
    return result

def main():
    st.title("金属/磁性催化剂 质量比计算器")
    mode = st.radio("选择模式:", ("直接微调制备", "MOF 衍生制备", "金属/磁混合煅烧法"))

    if mode == "直接微调制备":
        formula = st.text_input("输入分子式 (例如 Fe2O3)")
        mole = st.number_input("输入摩尔数", min_value=0.0, format="%.3f")
        if st.button("计算"):
            elements = parse_formula(formula)
            mass = calculate_mass(elements, mole)
            result = display_mass_ratio(mass)
            st.write("元素质量比:")
            st.json(result)

    elif mode == "MOF 衍生制备":
        formula = st.text_input("输入MOF分子式 (例如 C8H12N4Co)")
        mole = st.number_input("输入摩尔数", min_value=0.0, format="%.3f")
        if st.button("计算"):
            elements = parse_formula(formula)
            mass = calculate_mass(elements, mole)
            result = display_mass_ratio(mass)
            st.write("元素质量比:")
            st.json(result)

    elif mode == "金属/磁混合煅烧法":
        metal_formula = st.text_input("输入金属氧化物分子式 (例如 CuO)")
        metal_mole = st.number_input("输入金属氧化物摩尔数", min_value=0.0, format="%.3f")
        additive_formula = st.text_input("输入添加物分子式 (可以是C，或任意化合物)")
        additive_mass = st.number_input("输入添加物质量 (单位g)", min_value=0.0, format="%.3f")

        if st.button("计算"):
            metal_elements = parse_formula(metal_formula)
            mass_metal = calculate_mass(metal_elements, metal_mole)

            additive_elements = parse_formula(additive_formula)
            mass_additive = calculate_mass(additive_elements, 1.0)

            total_additive_mass = sum(mass_additive.values())
            factor = additive_mass / total_additive_mass

            for elem, mass in mass_additive.items():
                mass_metal[elem] = mass_metal.get(elem, 0) + mass * factor

            result = display_mass_ratio(mass_metal)
            st.write("元素质量比:")
            st.json(result)

if __name__ == "__main__":
    main()
