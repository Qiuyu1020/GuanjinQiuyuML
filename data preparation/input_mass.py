import re
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
    print(f"总质量: {total_mass:.4f} g")
    print("各元素质量和质量百分比：")
    for elem, mass in element_masses.items():
        print(f"{elem}: {mass:.4f} g ({mass/total_mass:.5f})")

if __name__ == "__main__":
    while True:
        try:
            formula1 = input("输入第一个分子式 (例如 CuO): ")
            mass1 = float(input("输入第一个物质质量 (g): "))
            formula2 = input("输入第二个分子式 (例如 C或CH4): ")
            mass2 = float(input("输入第二个物质质量 (g): "))

            mass1_elements = calculate_element_masses(formula1, mass1)
            mass2_elements = calculate_element_masses(formula2, mass2)
            combined_masses = combine_masses(mass1_elements, mass2_elements)

            display_element_masses(combined_masses)
            break
        except ValueError as e:
            print(f"输入错误：{e}")
            print("请重新输入！\n")
