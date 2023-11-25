import altair as alt
import numpy as np
import pandas as pd
alt.data_transformers.disable_max_rows()


def plot_variables(data: pd.DataFrame, 
                   variables: list, 
                   var_type: str = 'categorical', 
                   target_var: str = 'y', 
                   ignore_vars: list = None) -> alt.Chart:
    """
    Create grouped bar plots for categorical variables, histograms for continuous variables, 
    and histograms for log-transformed continuous variables. The plots are grouped and colored 
    by the target variable.

    Parameters:
    - data (DataFrame): The input DataFrame containing the variables to be plotted.
    - variables (list): The list of variables to be plotted.
    - var_type (str, optional): The type of variables ('categorical', 'continuous', or 'log'). Default is 'categorical'.
    - target_var (str, optional): The target variable for grouping and coloring the plots. Default is 'y'.
    - ignore_vars (list, optional): A list of variables to ignore when creating plots. Default is None.

    Returns:
    - alt.Chart: The Altair chart containing the visualizations.
    """
    charts = []

    for i, var in enumerate(variables):
        if var not in data.columns or (ignore_vars and var in ignore_vars):
            continue  

        if var_type == 'categorical':
            num_rows = len(data[var].unique())

            chart = alt.Chart(data).mark_bar(stroke=None).encode(
                x=alt.X('count()', title='Count'),
                y=alt.Y(target_var + ':N', title=None),
                color=alt.Color(target_var + ':N', scale=alt.Scale(range=['#3C6682', '#45A778'])),
                row=alt.Row(f'{var}:N')
            ).properties(
                width=300,
                height=300 / num_rows,
                title=f'Grouped Bar Plot for {var}',
                spacing=0
            )

        elif var_type == 'continuous':
            hist_chart = alt.Chart(data).mark_bar(opacity=0.7, color='steelblue').encode(
                x=alt.X(f'{var}:Q', bin=alt.Bin(maxbins=50), title=var),
                y=alt.Y('count():Q', stack=None, title='Count'),
            )

            kde_chart = alt.Chart(data).transform_density(
                var,
                as_=[var, 'density'],
            ).mark_line(color='red').encode(
                x=alt.X(f'{var}:Q', title=var),
                y=alt.Y('density:Q', title='Density'),
            )

            chart = alt.layer(hist_chart, kde_chart).resolve_scale(y='independent').properties(
                width=200,
                height=150,
                title=f'{var}'
            )

        elif var_type == 'log':
            log_var = f'{var}_log'
            data[log_var] = np.log1p(data[var])

            hist_chart = alt.Chart(data).mark_bar(opacity=0.7, color='steelblue').encode(
                x=alt.X(f'{log_var}:Q', bin=alt.Bin(maxbins=50), title=f'{var} (log-transformed)'),
                y=alt.Y('count():Q', stack=None, title='Count'),
            )

            kde_chart = alt.Chart(data).transform_density(
                log_var,
                as_=[log_var, 'density'],
            ).mark_line(color='red').encode(
                x=alt.X(f'{log_var}:Q', title=f'{var} (log-transformed)'),
                y=alt.Y('density:Q', title='Density'),
            )

            chart = alt.layer(hist_chart, kde_chart).resolve_scale(y='independent').properties(
                width=300,
                height=225,
                title=f'{var} (log-transformed)'
            )

        charts.append(chart)

    final_chart = alt.concat(*charts, columns=3).configure_axis(grid=False)

    return final_chart
