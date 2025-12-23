import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    "category": ["A", "A", "B", "B", "C", "C"],
    "value": [10, 15, 7, 11, 13, 9]
})

fig = px.bar(df, x="category", y="value", title="Plotly Test Chart")
fig.show()
