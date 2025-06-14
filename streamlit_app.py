import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/sales_data.csv")
df["Month_Num"] = pd.to_datetime(df["Month"]).dt.month
X = df[["Month_Num"]]
y = df["Sales"]
model = LinearRegression().fit(X, y)

st.title("?? Sales Forecasting Dashboard")
month = st.slider("Select month (1-12)", 1, 12)
prediction = model.predict([[month]])
st.write(f"?? Predicted sales for month {month}: ?{prediction[0]:.2f}")
st.line_chart(df.set_index("Month")["Sales"])
