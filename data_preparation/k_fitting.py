# input 5 C0 5 Ct 5 t
# output figure line fitting k value R^2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

####################################################################################
#              change here only !!!!!!!!!!!#################################################
C0 = 27
Ct_list = [26, 26, 27, 28, 29]
t_list = [30, 31, 32, 33, 34]
############################################################################################'


x = np.array(t_list)
y = -np.log(np.array(Ct_list) / C0)


x_reshape = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x_reshape, y)
k = model.coef_[0]
b = model.intercept_
y_pred = model.predict(x_reshape)


r2 = r2_score(y, y_pred)


print(f"斜率 k = {k:.4f}")
print(f"R^2 = {r2:.4f}")


plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label=f'Fit Line: y={k:.4f}x+{b:.4f}')
plt.xlabel('t')
plt.ylabel('-ln(Ct/C0)')
plt.title('Linear Fit')
plt.legend()
plt.grid(True)
plt.show()
