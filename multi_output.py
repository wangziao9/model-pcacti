import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# generate mock data from mock regressor
X_train = np.array([[3., 4.], [2., 3.], [1., 5.]])
y_train = np.array([[2., 1.], [4., 3.], [6., 4.]])
regr = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
X_test = np.array([[2., 6.], [1., 3.], [2., 2.]])
y_true = np.array([[1., 2.], [3., 5.], [4., 6.]])
y_pred = regr.predict(X_test) 
print("ground truth of output 1 is:", y_true[:,0])
print("ground truth of output 2 is:", y_true[:,1])

# calculate y variance
avg_true= np.average(y_true, axis=0)
print("average value of ground truth:", avg_true)
y_variance = np.average((y_true - avg_true)**2, axis=0)
print("y_variance:", y_variance)

# calculate mse
mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
print("mse:", mse)
my_mse = np.average((y_true - y_pred)**2, axis=0)
assert(np.max(np.abs(my_mse - mse)) < 1e-4)

# calculate Coef. of Determination for each dimension
my_cods = (1-mse/y_variance)
print("my coef. of determination (for each output):", my_cods)
sklearn_cods = r2_score(y_true, y_pred, multioutput="raw_values")
print("sklearn's coef. of determination (for each output):", sklearn_cods)
assert(np.max(np.abs(my_cods - sklearn_cods))< 1e-4)

# calculate Coef. of Determination, averaged
cod = regr.score(X_test, y_true)
print("regr.score, always a float:", cod)
my_cod_averaged = np.average(my_cods)
print("my coef. of determination, averaged across outputs:", my_cod_averaged)
assert(np.abs(cod - my_cod_averaged) < 1e-4)