st.header("Interactive PDF for European Roulette Colour Betting")

# Slider for the number of spins (t)
t = st.slider("Number of Spins (t)", min_value=100, max_value=10000, value=1000, step=100)

# Calculate distribution parameters based on t
sigma = 6 * np.sqrt(38) / 37       # per-bet sigma
center = -1/37 * t                 # drift (mean)
std_dev = sigma * np.sqrt(t)       # standard deviation

# Define the x-axis range as ±3 standard deviations around the center
x_min = center - 3 * std_dev
x_max = center + 3 * std_dev

# Set default values for the bound sliders (e.g., 0.5 std_dev from the center)
default_lower = center - 0.5 * std_dev
default_upper = center + 0.5 * std_dev

# Create sliders for the lower and upper bounds with dynamic ranges
lower_bound = st.slider(
    "Lower Threshold", 
    min_value=float(x_min), 
    max_value=float(center), 
    value=float(default_lower), 
    step=1.0,
    key="lower_bound"
)
upper_bound = st.slider(
    "Upper Threshold", 
    min_value=float(center), 
    max_value=float(x_max), 
    value=float(default_upper), 
    step=1.0,
    key="upper_bound"
)

# Compute the PDF using scipy.stats.norm
x_values = np.linspace(x_min, x_max, 1000)
y_values = norm.pdf(x_values, loc=center, scale=std_dev)

# Calculate the tail probabilities (i.e. probability of being more extreme than the bounds)
p_lower = norm.cdf(lower_bound, loc=center, scale=std_dev)
p_upper = 1 - norm.cdf(upper_bound, loc=center, scale=std_dev)
extreme_prob = p_lower + p_upper

formatted_lower = f"-£{abs(lower_bound):,.2f}" if lower_bound < 0 else f"£{abs(lower_bound):,.2f}"
formatted_upper = f"-£{abs(upper_bound):,.2f}" if upper_bound < 0 else f"£{abs(upper_bound):,.2f}"

st.write(
    f"**Probability of balance being either less than {formatted_lower} or greater than {formatted_upper}: {extreme_prob:.2%}**"
)

# Create the Plotly figure
fig = go.Figure()

# Add the PDF curve
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_values,
    mode='lines',
    name='Normal PDF',
    line=dict(color='blue')
))

# Add vertical dashed lines for the lower and upper bounds
fig.add_shape(
    type="line", x0=lower_bound, x1=lower_bound, y0=0, y1=max(y_values),
    line=dict(color="red", dash="dash")
)
fig.add_shape(
    type="line", x0=upper_bound, x1=upper_bound, y0=0, y1=max(y_values),
    line=dict(color="red", dash="dash")
)

# Shade the left tail (x < lower_bound)
mask_left = x_values < lower_bound
fig.add_trace(go.Scatter(
    x=x_values[mask_left],
    y=y_values[mask_left],
    mode='lines',
    fill='tozeroy',
    fillcolor='rgba(173,216,230,0.5)',  # light blue fill
    line=dict(color='lightblue'),
    showlegend=False
))

# Shade the right tail (x > upper_bound)
mask_right = x_values > upper_bound
fig.add_trace(go.Scatter(
    x=x_values[mask_right],
    y=y_values[mask_right],
    mode='lines',
    fill='tozeroy',
    fillcolor='rgba(173,216,230,0.5)',  # light blue fill
    line=dict(color='lightblue'),
    showlegend=False
))

# Update layout of the plot
fig.update_layout(
    title=f"PDF for European Roulette Colour Betting (t = {t})",
    xaxis_title="Balance",
    yaxis_title="Probability Density",
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)