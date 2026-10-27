import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as al
import pandas as pd
from regression import GP


np.random.seed(42)
X_train = np.array([-4, -3, -2, 0, 2, 3, 4])
y_train = np.sin(X_train) + 0.1 * np.random.randn(len(X_train))

X_test = np.linspace(-5, 5, 100)

Y,sigma_test = GP(X_train,X_test,y_train,1,1,1,0)


plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(X_test), "r--", label="True function f(x) = sin(x)")
plt.plot(X_test, Y, "b", label="Prediction f(x)")
plt.fill_between(X_test, Y - 2*sigma_test, Y + 2*sigma_test,
                 color="blue", alpha=0.2, label="95% confidence interval")
plt.scatter(X_train, y_train, color="black", marker="x", label="Observations")
plt.legend()
plt.title("1D Gaussian Process Regression, Pivoted Incomplete Cholesky")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()