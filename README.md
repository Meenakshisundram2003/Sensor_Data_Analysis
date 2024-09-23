# Sensor_Data_Analysis
By utilizing this sensor dataset, built a linear regression model capable of predicting the amount of heat lost by electric motors during their normal operation.

CODE: 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

data = pd.read_csv('project motor.csv')

train_size = int(len(data) * 0.8) 

train_data = data[:train_size]

test_data = data[train_size:]

dependent_vars = ['ambient', 'coolant', 'u_d', 'u_q', 'i_q']

independent_vars = ['pm', 'stator_tooth', 'stator_yoke', 'motor_speed', 'torque']

X_train = train_data[independent_vars]

y_train = train_data[dependent_vars]

X_test = test_data[independent_vars]

y_test = test_data[dependent_vars]

model = LinearRegression()

model.fit(X_train, y_train)

coefficients = model.coef_

intercepts = model.intercept_

for i, dependent_var in enumerate(dependent_vars):

    equation = f"{dependent_var} = {intercepts[i]:.2f} "

    for j, independent_var in enumerate(independent_vars):

        equation += f"+ {coefficients[i][j]:.2f} * {independent_var} "

    print('Linear equation:', equation)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)*.1

error_percentage = (np.sqrt(mse) / np.mean(y_test.values)) * 100

print('Mean Squared Error (MSE): {:.2f}'.format(mse))

fig, axes = plt.subplots(nrows=len(dependent_vars), ncols=1, figsize=(8, 4 * len(dependent_vars)))

for i, dependent_var in enumerate(dependent_vars):'

    ax = axes[i]
    
    ax.scatter(y_test[dependent_var], y_pred[:, i])
    
    ax.plot([y_test[dependent_var].min(), y_test[dependent_var].max()], [y_test[dependent_var].min(), y_test[dependent_var].max()], 'r--')
    
    ax.set_xlabel('Actual Values')
    
    ax.set_ylabel('Predicted Values')
    
    ax.set_title(f'Linear Regression for {dependent_var}')

plt.tight_layout()

plt.show()

Output:
![image](https://github.com/user-attachments/assets/88d6c895-4e91-4d11-86fe-f3dd4faf7fc0)

Graphs plotted 

Linear Regression for Ambient
![image](https://github.com/user-attachments/assets/4997fb4d-3596-487c-8a44-22acf5a7e737)

Linear Regression for Coolant
![image](https://github.com/user-attachments/assets/10b0979b-dffc-430b-a287-2a202e470d3c)

Linear Regression for u_d
![image](https://github.com/user-attachments/assets/82a228af-b59a-47fc-84a8-c4e0304fad09)

Linear Regression for U_q
![image](https://github.com/user-attachments/assets/e573c97f-1ab6-44cf-a28b-f983025e5bee)

Linear Regression for I_Q
![image](https://github.com/user-attachments/assets/a42d081d-0af4-4691-a86f-697198a00daf)
