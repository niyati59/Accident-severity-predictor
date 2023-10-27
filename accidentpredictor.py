import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and Preprocess the Dataset
data = pd.read_csv('accident_data.csv')  # Replace with the actual dataset
# Perform data preprocessing steps like handling missing values and encoding categorical variables here.

# Step 2: Define Dependent and Independent Variables
X = data[['Weather', 'Road_Type', 'Vehicle_Type', 'Time_of_Day']]  # Independent variables
y = data['Accident_Severity']  # Dependent variable

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 6: Save the Model
joblib.dump(model, 'accident_severity_model.pkl')

# Step 7: Predict Accident Severity for a Hypothetical Set of Variables
# Replace these values with your hypothetical scenario
hypothetical_data = pd.DataFrame({
    'Weather': ['Clear'],
    'Road_Type': ['Urban'],
    'Vehicle_Type': ['Car'],
    'Time_of_Day': ['Morning']
})

predicted_severity = model.predict(hypothetical_data)
print("Predicted Accident Severity:", predicted_severity)
