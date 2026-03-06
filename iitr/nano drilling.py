import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
# E = pulse energy 
# N = number of pulse count
# z = focal position of the laser
# n = number of samples
np.random.seed(42)
n = 300
E = np.random.uniform(100, 200, n)
N = np.random.uniform(50, 300, n)        
z = np.random.uniform(-1,1,n)
# r = beam radius
# delta = optical penetration depth
# F_th = Ablation Threshold 
r = 10e-6
delta = 1e-6
F_th = 0.5

F = E/(np.pi*r**2)    # LASER FLUENCE EQUATION
D = N*delta*np.log(F/F_th)  # Beer-Lambert law for ablation debth model
d = 2*r*(1+0.1*z)*1e6    # Approximation of Gaussian_Beam propagation 
D = D + np.random.normal(0, 1e-6, n)  # Adding noise to the data
data = pd.DataFrame({
    "E": E,
    "N":N,                           # Setting up the data_frame
    "z":z,
    "d":d,
    "D":D*1e6
})
print(data.head())        

X = data.drop("D", axis=1)   # Laser parameter
Y = data["D"]                # Hole Diameter
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

rf_pred = rf.predict(X_test)
rf_rmse = mean_squared_error(Y_test, rf_pred)

svr = SVR(kernel='rbf')          # Support vector regression
svr.fit(X_train, Y_train)

svr_pred = svr.predict(X_test)
svr_rmse = np.sqrt(mean_squared_error(Y_test, svr_pred))
print(f"Random Forest RMSE:", rf_rmse)
print(f"Support Vector Regression RMSE:", svr_rmse)

imp = rf.feature_importances_
plt.bar(X.columns, imp)
plt.title("Feature importance for Random Forest")
plt.show()