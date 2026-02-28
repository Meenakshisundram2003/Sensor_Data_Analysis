# Electric Motor Sensor Data Analysis

## Overview
This project analyzes sensor data to predict the amount of heat lost by electric motors during normal operation. The pipeline includes data cleaning (handling missing values and normalization), exploratory data analysis, and predictive modeling using linear regression.

## Technologies Used
* **Data Processing:** Python (NumPy, Pandas)
* **Machine Learning:** Linear Regression, Mean Squared Error (MSE) evaluation
* **Visualization:** Matplotlib, Power BI
* **Analysis:** Excel

## Getting Started
1. Clone this repository to your local machine.
2. Ensure you have the required Python libraries installed (`pandas`, `numpy`, `matplotlib`, `scikit-learn`).
3. Run the main analysis script or Jupyter Notebook to view the model training and output.

## Results and Visualizations
The model's predictive accuracy was evaluated using MSE. Below are key visualizations from the dataset and model comparisons.

### Feature Distributions

| Ambient | Coolant |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/4997fb4d-3596-487c-8a44-22acf5a7e737" width="400" height="300"> | <img src="https://github.com/user-attachments/assets/10b0979b-dffc-430b-a287-2a202e470d3c" width="400" height="300"> |

| Direct-axis Voltage (u_d) | Quadrature-axis Voltage (u_q) | Quadrature-axis Current (i_q) |
|-----|-----|-----|
| <img src="https://github.com/user-attachments/assets/82a228af-b59a-47fc-84a8-c4e0304fad09" width="400" height="600"> | <img src="https://github.com/user-attachments/assets/e573c97f-1ab6-44cf-a28b-f983025e5bee" width="400" height="600"> | <img src="https://github.com/user-attachments/assets/a42d081d-0af4-4691-a86f-697198a00daf" width="400" height="600"> |

### Excel Data Comparisons

**Ambient vs PM**
<img src="https://github.com/user-attachments/assets/81ba64c7-e1e6-4c78-8b4d-35c3213564b4" width="700" height="400">

**Coolant vs Motor Speed**
<img src="https://github.com/user-attachments/assets/141e6f61-8871-4b5c-b037-39bb9edbebeb" width="700" height="400">

**Current vs Motor Speed**
<img src="https://github.com/user-attachments/assets/e473289b-12c4-4928-882c-07f375cb04c1" width="700" height="400">
