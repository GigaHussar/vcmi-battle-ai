
import pandas as pd
import plotly.graph_objs as go
from _paths_do_not_touch import EXPORT_DIR, MASTER_LOG, BASE_PATH
from pathlib import Path

# Load the CSV file
BASE_PATH = Path(__file__).resolve().parent.parent
EXPORT_DIR = BASE_PATH / "export"
MASTER_LOG = EXPORT_DIR / "master_log.csv"
df = pd.read_csv(MASTER_LOG)

# Calculate absolute difference
df['absolute_difference'] = (df['reward'] - df['predicted_performance']).abs()

# Create the interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(y=df['reward'], mode='lines+markers', name='Reward', line=dict(color='blue')))
fig.add_trace(go.Scatter(y=df['predicted_performance'], mode='lines+markers', name='Predicted Performance', line=dict(color='orange')))
fig.add_trace(go.Scatter(y=df['absolute_difference'], mode='lines', name='Absolute Difference', line=dict(color='green', dash='dash')))

# Layout settings
fig.update_layout(
    title='Reward vs Predicted Performance with Absolute Difference',
    xaxis_title='Index',
    yaxis_title='Value',
    height=600,
    width=1400,
    xaxis=dict(rangeslider=dict(visible=True))  # Enables scrolling
)

fig.show()
