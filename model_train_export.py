from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import joblib
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

df = px.data.gapminder()
df.head()
df.shape

X = df[["year", "pop", "gdpPercap"]]
y = df["lifeExp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

model_filename = "gapminder_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")