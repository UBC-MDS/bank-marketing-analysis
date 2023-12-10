import pandas as pd
import numpy as np
import altair as alt
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_viz import plot_variables

# Test data
data = pd.DataFrame({
    'var1': np.random.choice(['A', 'B', 'C'], size=100),
    'var2': np.random.normal(0, 1, size=100),
    'var3': np.random.choice([True, False], size=100),
    'y': np.random.choice(['yes', 'no'], size=100)
})

# Type returned Test
def test_plot_variables_returns_chart():
    cat_chart = plot_variables(data, variables=['var1'], var_type='categorical')
    cont_chart = plot_variables(data, variables=['var2'], var_type='continuous')
    log_chart = plot_variables(data, variables=['var2'], var_type='log')

    assert isinstance(cat_chart, alt.ConcatChart)
    assert isinstance(cont_chart, alt.ConcatChart)
    assert isinstance(log_chart, alt.ConcatChart)

# Chart contains variables from df that was passed.
def test_plot_variables_contains_categorical_variable():
    chart = plot_variables(data, variables=['var1', 'var2', 'var3', 'y', 'var2_log'], var_type='categorical')
    
    expected_variables = ['var1', 'var2', 'var3', 'y', 'var2_log']
    actual_variables = list(chart.data.columns)
    
    assert set(actual_variables) == set(expected_variables), f"Unexpected variables found: {set(actual_variables) - set(expected_variables)}"

def test_plot_variables_ignore_variables():
    ignore_chart = plot_variables(data, variables=['var1', 'var2'], var_type='categorical', ignore_vars=['var1'])
    assert len(ignore_chart.to_dict()['concat']) == 1




def test_plot_variables_not_empty():
    chart = plot_variables(data, variables=['var1'], var_type='categorical')
    assert chart is not None, "Generated chart is empty."



def test_plot_variables_valid_var_type():
    valid_var_types = ['categorical', 'continuous', 'log']
    
    for var_type in valid_var_types:
        if var_type == 'log':
            chart = plot_variables(data, variables=['var2'], var_type=var_type)
        else:
            chart = plot_variables(data, variables=['var1'], var_type=var_type)
        
        assert isinstance(chart, alt.ConcatChart), f"Failed for var_type: {var_type}"
