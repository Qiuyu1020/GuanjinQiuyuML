import re
import streamlit as st
import periodictable as pt

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
            for elem, count in group.items():
                stack[-1][elem] = stack[-1].get(elem, 0) + count * multiplier
        elif isinstance(token, str) and re.match(r'[A-Z][a-z]?', token):
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

def calculate_element_masses(formula, mass):
    elements = parse_formula(formula)

    molar_mass_total = 0
    for elem, count in elements.items():
        if not hasattr(pt, elem):
            raise ValueError(f"元素 {elem} 不在元素周期表中！")
        molar_mass_total += count * getattr(pt, elem).mass

    element_masses = {}
    for elem, count in elements.items():
        element_mass = mass * (count * getattr(pt, elem).mass) / molar_mass_total
        element_masses[elem] = element_mass

    return element_masses

def combine_masses(mass_dict1, mass_dict2):
    combined = mass_dict1.copy()
    for elem, mass in mass_dict2.items():
        combined[elem] = combined.get(elem, 0) + mass
    return combined

def display_element_masses(element_masses):
    total_mass = sum(element_masses.values())
    result = {}
    for elem, mass in element_masses.items():
        result[elem] = (mass, mass / total_mass)
    return result

def main():
    st.title("输入两个分子式计算元素质量比")

    formula1 = st.text_input("输入第一个分子式 (例如 CuO)")
    mass1 = st.number_input("输入第一个物质质量 (g)", min_value=0.0, format="%.3f")
    formula2 = st.text_input("输入第二个分子式 (例如 C或CH4)")
    mass2 = st.number_input("输入第二个物质质量 (g)", min_value=0.0, format="%.3f")

    if st.button("计算"):
        try:
            mass1_elements = calculate_element_masses(formula1, mass1)
            mass2_elements = calculate_element_masses(formula2, mass2)
            combined_masses = combine_masses(mass1_elements, mass2_elements)

            result = display_element_masses(combined_masses)
            total_mass = sum([v[0] for v in result.values()])

            st.write(f"总质量: {total_mass:.4f} g")
            st.write("各元素质量和质量百分比：")
            for elem, (mass, percent) in result.items():
                st.write(f"{elem}: {mass:.4f} g ({percent:.5f})")

        except ValueError as e:
            st.error(f"错误: {e}")

if __name__ == "__main__":
    main()
