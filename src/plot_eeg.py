import plotly.graph_objects as go
import plotly.offline as pyo


def plot_single_lead(d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=d))
    return fig