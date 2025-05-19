# This is a test ground for Plotly Open Source Graphing Library for Python

# #######################
# Line plot
# #######################
# import plotly.express as px
# df = px.data.gapminder().query("country=='China'")
# fig = px.line(df, x="year", y="lifeExp", color='country')
# fig.show()

# #######################
# Connected Scatterplots
# #######################
# import plotly.express as px
# df = px.data.gapminder().query("country in ['Canada', 'Botswana']")
# fig = px.line(df, x="year", y="gdpPercap", color="country", text="year")
# fig.update_traces(textposition="bottom right")
# fig.show()

# #######################
# ML Regression
# #######################
# Orinary least square (OLS)
# import plotly.express as px
# df = px.data.tips()
# fig = px.scatter(
#     df, x='total_bill', y='tip', opacity=0.65,
#     trendline='ols', trendline_color_override='darkblue'
# )
# fig.show()

# Linear regression with scikit-learn
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.tip)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()
