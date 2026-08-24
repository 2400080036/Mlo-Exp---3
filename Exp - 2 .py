import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read dataset tracked by DVC
data = pd.read_csv("student_marks.csv")
X = data[["Hours"]]
y = data["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)