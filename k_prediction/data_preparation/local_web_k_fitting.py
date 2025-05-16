import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    st.title("k 拟合小工具 (两种模式)")

    mode = st.radio("选择拟合模式:", ("输入C0和Ct", "直接输入Ct/C0"))

    if mode == "输入C0和Ct":
        st.subheader("模式一：输入初始浓度C0和5组Ct")
        C0 = st.number_input("输入初始浓度 C0", min_value=0.0, format="%.3f")
        Ct_input = st.text_area("输入5个Ct值 (用逗号分隔，例如: 26,26,27,28,29)")
        t_input = st.text_area("输入5个t值 (用逗号分隔，例如: 30,31,32,33,34)")

        if st.button("开始拟合", key="fit1"):
            try:
                Ct_list = list(map(float, Ct_input.split(',')))
                t_list = list(map(float, t_input.split(',')))

                if len(Ct_list) != 5 or len(t_list) != 5:
                    st.error("Ct和t必须各有5个数！")
                    return

                x = np.array(t_list)
                y = -np.log(np.array(Ct_list) / C0)

                plot_and_fit(x, y)

            except Exception as e:
                st.error(f"输入错误: {e}")

    elif mode == "直接输入Ct/C0":
        st.subheader("模式二：直接输入5组Ct/C0比值")
        ratio_input = st.text_area("输入5个Ct/C0值 (用逗号分隔，例如: 0.95,0.90,0.85,0.80,0.75)")
        t_input = st.text_area("输入5个t值 (用逗号分隔，例如: 30,31,32,33,34)")

        if st.button("开始拟合", key="fit2"):
            try:
                ratio_list = list(map(float, ratio_input.split(',')))
                t_list = list(map(float, t_input.split(',')))

                if len(ratio_list) != 5 or len(t_list) != 5:
                    st.error("Ct/C0和t必须各有5个数！")
                    return

                x = np.array(t_list)
                y = -np.log(np.array(ratio_list))

                plot_and_fit(x, y)

            except Exception as e:
                st.error(f"输入错误: {e}")

def plot_and_fit(x, y):
    x_reshape = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_reshape, y)
    k = model.coef_[0]
    b = model.intercept_
    y_pred = model.predict(x_reshape)

    r2 = r2_score(y, y_pred)

    st.success(f"斜率 k = {k:.4f}")
    st.success(f"R² = {r2:.4f}")

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data Points')
    ax.plot(x, y_pred, color='red', label=f'Fit Line: y={k:.4f}x+{b:.4f}')
    ax.set_xlabel('t')
    ax.set_ylabel('-ln(Ct/C0)')
    ax.set_title('Linear Fit')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    main()