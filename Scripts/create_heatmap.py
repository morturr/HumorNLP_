import plotly.graph_objects as go
import numpy as np
from numpy.distutils.conv_template import header

# Sample data (replace with your own data)
header = ["Amazon", "Dad Jokes", "Headlines", "One Liners", "Yelp Reviews"]
rows = [
    [84, 61, 61, 58, 60],
    # [90, 59, 74, 72, 64],
    [66, 91, 62, 60, 62],
    # [69, 92, 62, 70, 63],
    [63, 59, 94, 62, 52],
    # [66, 63, 94, 70, 54],
    [62, 61, 54, 84, 54],
    # [71, 52, 69, 84, 51],
    [54, 53, 55, 60, 61],
    # [57, 49, 58, 51, 69]
]

index = header

# Convert data to NumPy array for processing
data = np.array(rows)

# # Calculate colors for heatmap effect
# colors = [[
#     f'rgba({int(255 * (1 - val / np.max(data)))}, {int(255 * (val / np.max(data)))}, 0, 0.8)'
#     for val in row] for row in data]

# Improved color calculation: Using a power transformation for more contrast
def get_color(val):
    normalized = (val / 100) ** 1.5  # Adjust this power for more/less contrast
    red = int(255 * (1 - normalized))
    green = int(255 * normalized)
    return f'rgba({red}, {green}, 0, 0.8)'

rgba50 = get_color(50)  # Test the function
rgba53 = get_color(53)  # Test the function
rgba52 = get_color(55)  # Test the function
rgba60 = get_color(60)  # Test the function
rgba94 = get_color(94)  # Test the function

# Generate colors using the improved color mapping
colors = [[get_color(val) for val in row] for row in data]
# Transpose the colors list to match Plotly's column-wise format
colors_transposed = list(map(list, zip(*colors)))

# Create the heatmap table
fig = go.Figure(data=[go.Table(
columnwidth=[1] + [1] * len(header),
    header=dict(values=[
        "<b>Train Dataset</b>", *header
    ],
        align='center',
        line_color='black',
        fill_color=['white'] * (len(header) + 1),
        font=dict(color='black', size=12)),

    cells=dict(
        values=[index] + [list(col) for col in zip(*rows)],
        fill_color=[['white'] * len(index)] + colors_transposed,
        align='center',
        line_color='black'
    )
)])

# Update layout
fig.update_layout(title_text="Heatmap Table (Red to Green)")

# Show the heatmap
fig.show()

# Export to PDF
fig.write_image("structured_heatmap_table.pdf", format="pdf")
