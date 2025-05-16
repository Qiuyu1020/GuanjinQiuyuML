import streamlit as st
import periodictable as pt

def main():
    st.title("元素原子数百分比 ➔ 质量百分比计算器")

    st.write("请输入元素名称和对应的原子数百分比。")
    st.write("格式示例: Fe, C, O")
    elements_input = st.text_input("输入元素名称 (用逗号分隔)")

    st.write("输入对应的原子数百分比 (顺序对应元素)。")
    st.write("格式示例: 30, 50, 20")
    numbers_input = st.text_input("输入原子数百分比 (用逗号分隔)")

    if st.button("开始计算"):
        try:
            element_list = [e.strip() for e in elements_input.split(',')]
            number_list = list(map(float, numbers_input.split(',')))

            if len(element_list) != len(number_list):
                st.error("元素数量与原子数百分比数量必须对应！")
                return


            total_number = sum(number_list)
            normalized_number = [n / total_number for n in number_list]


            mass_contributions = {}
            for elem, fraction in zip(element_list, normalized_number):
                if not hasattr(pt, elem):
                    st.error(f"元素 {elem} 不在周期表中！")
                    return
                atomic_mass = getattr(pt, elem).mass
                mass_contributions[elem] = fraction * atomic_mass


            total_mass = sum(mass_contributions.values())
            result = {elem: (mass / total_mass ) for elem, mass in mass_contributions.items()}

            st.write("各元素质量百分比：")
            for elem, mass_percent in result.items():
                st.write(f"{elem}: {mass_percent:.5f}")

        except Exception as e:
            st.error(f"输入错误: {e}")

if __name__ == "__main__":
    main()
