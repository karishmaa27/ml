import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

diabetes=datasets.load_diabetes()
# print(diabetes.DESCR)

diabetes_X = np.array([[1],[2],[3]])
diabetes_X_train=diabetes_X
diabetes_X_test=diabetes_X

diabetes_y_train=np.array([3,2,4])
diabetes_y_test=np.array([3,2,4])

model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_y_train)

diabetes_y_predicted=model.predict(diabetes_X_test)

print("MEan squared error is :" , mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("Weights : ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(diabetes_X_test,diabetes_y_test)
plt.plot(diabetes_X_test,diabetes_y_predicted)
plt.show()
# MEan squared error is : 3035.060115291269
# Weights :  [941.43097333]
# Intercept:  153.39713623331644