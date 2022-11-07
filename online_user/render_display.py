from business import GraphBuilder, StatsBuilder
from dash import Input, Output, State, dcc, html
from jupyter_dash import JupyterDash


app = JupyterDash(__name__)

gb = GraphBuilder()

sb = StatsBuilder()


app.layout = html.Div(
    [
        html.H1("Applicants Demographics"),
        dcc.Dropdown(
            options=["Nationality", "Age", "Education"],
            value="Nationality",
            id="demo-plots-dropdown"
        ),
        html.Div(id="demo-plots-display"),
        html.H1("Experiment"),
        dcc.Slider(min=0.1, max=0.8, step=0.1, value=0.2, id="effect-size-slider"),
        html.Div(id="effect-size-display"),
        html.H1("Choose experiment days"),
        dcc.Slider(min=1, max=20, step=1, value=1, id="experiment-days-slider"),
        html.Div(id="experiment-days-display"),
        html.H1("Results"),
        html.Button("Begin Experiment", id="start-experiment-button", n_clicks=0),
        html.Div(id="results-display")
    ]
)


@app.callback(
    Output("demo-plots-display", "children"),
    Input("demo-plots-dropdown", "value")
)
def display_demo_graph(graph_name):
    """Serves applicant demograhic visualization.

    Parameters
    ----------
    graph_name : str
        User input given via 'demo-plots-dropdown'. Name of Graph to be returned.
        Options are 'Nationality', 'Age', 'Education'.

    Returns
    -------
    dcc.Graph
        Plot that will be displayed in 'demo-plots-display' Div.
    """
    if graph_name == "Nationality":
        fig= gb.build_nat_choropleth()
    elif graph_name == "Age":
        fig= gb.build_age_hist()
    elif graph_name == "Education":
        fig= gb.build_ed_bar()
    
    return dcc.Graph(figure=fig)


@app.callback(
    Output("effect-size-display", "children"),
    Input("effect-size-slider", "value")
)
def display_group_size(effect_size):
    """Serves information about required group size.

    Parameters
    ----------
    effect_size : float
        Size of effect that user wants to detect. Provided via 'effect-size-slider'.

    Returns
    -------
    html.Div
        Text with information about required group size. will be displayed in
        'effect-size-display'.
    """
    n_obs =sb.calculate_n_obs(effect_size)
    text = f"To detect an effect size {effect_size}, you need {n_obs} observation"
    return html.Div(text)


@app.callback(
    Output("experiment-days-display", "children"),
    Input("effect-size-slider", "value"),
    Input("experiment-days-slider", "value")
)
def display_cdf_pct(effect_size, days):
    """Serves probability of getting desired number of obervations.

    Parameters
    ----------
    effect_size : float
        The effect size that user wants to detect. Provided via 'effect-size-slider'.
    days : int
        Duration of the experiment. Provided via 'experiment-days-slider'.

    Returns
    -------
    html.Div
        Text with information about probabilty. Goes to 'experiment-days-display'.
    """
    # Calculate number of observations
    n_obs = sb.calculate_n_obs(effect_size)
    # Calculate percentage
    pct = sb.calculate_cdf_pct(n_obs, days)
    # Create text
    text = f"The probability of getting {n_obs} observation in {days} days is {pct}"
    # Return Div with text
    return html.Div()


@app.callback(
    Output("results-display", "children"),
    Input("start-experiment-button", "n_clicks"),
    State("experiment-days-slider", "value"),
)

def display_results(n_clicks, days):
    """Serves results from experiment.

    Parameters
    ----------
    n_clicks : int
        Number of times 'start-experiment-button' button has been pressed.
    days : int
        Duration of the experiment. Provided via 'experiment-days-display'.

    Returns
    -------
    html.Div
        Experiment results. Goes to 'results-display'.
    """
    if n_clicks == 0:
        return html.Div()
    else:
        #Run experiment
        sb.run_experiment(days)
        #create side by side bar chart
        fig = gb.build_contingency_bar()
        #Run chi-square
        result = sb.run_chi_square()
        #Return results
        return html.Div(
            [
                html.H2("Contingency Chart"),
                dcc.Graph(fig),
                html.H2("Chi-Square-Test for Independence"),
                html.H3(f"Degree of Freedom: {result.df}"),
                html.H3(f"P-value: {result.pvalue}"),
                html.H3(f"Statistic: {result.statistic}")
            ]
        )
