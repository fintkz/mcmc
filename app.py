import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_error_metric(predictions, actual):
    """Calculate MAPE"""
    return np.mean(np.abs((np.array(actual) - np.array(predictions)) / np.array(actual))) * 100

def create_model_plot(actual, predictions, feature_dates, selected_features, model_name, model_data):
    """Create plot for a specific model"""
    fig = go.Figure()

    # Add actual values
    fig.add_trace(
        go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        )
    )

    # Add predictions
    if model_name == 'bayesian':
        # Add uncertainty bands
        uncertainty = model_data['uncertainty']
        mean_pred = model_data['yhat']
        fig.add_trace(
            go.Scatter(
                y=[p + 2*u for p, u in zip(mean_pred, uncertainty)],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                fillcolor='rgba(0, 100, 255, 0.2)',
                fill='tonexty'
            )
        )
        fig.add_trace(
            go.Scatter(
                y=[p - 2*u for p, u in zip(mean_pred, uncertainty)],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(0, 100, 255, 0.2)',
                fill='tonexty'
            )
        )
        fig.add_trace(
            go.Scatter(
                y=mean_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='blue', width=2)
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                y=model_data['yhat'],
                mode='lines',
                name='Predicted',
                line=dict(color='blue', width=2)
            )
        )

    # Add feature markers
    colors = {
        'promotions': 'red',
        'weather': 'blue',
        'sports': 'green',
        'school': 'purple',
        'holidays': 'orange'
    }

    for feature in selected_features:
        if feature in feature_dates:
            for date in feature_dates[feature]:
                fig.add_vline(
                    x=date,
                    line=dict(
                        color=colors.get(feature, 'gray'),
                        width=1,
                        dash='dash'
                    ),
                    annotation=dict(
                        text=feature.title(),
                        textangle=-90,
                        yref="paper",
                        y=1,
                        font=dict(size=8)
                    )
                )

    # Calculate error
    error = get_error_metric(model_data['yhat'], actual)

    # Update layout
    fig.update_layout(
        title=f"{model_name.upper()} Model (MAPE: {error:.2f}%)",
        xaxis_title="Time",
        yaxis_title="Demand",
        height=400,
        showlegend=True,
        margin=dict(t=50)
    )
    
    return fig

def run_analysis(selected_features=None):
    # Load results
    with open('results/model_results.json', 'r') as f:
        results = json.load(f)
    
    actual = results['actual']
    feature_dates = results['feature_dates']
    
    # Use all features if none selected
    if selected_features is None:
        selected_features = ['promotions', 'weather', 'sports', 'school', 'holidays']
    
    # Get predictions for selected feature combination
    feature_key = '_'.join(sorted([f"{f}_active" if f == 'promotions' else f"{f}_event" 
                                 if f in ['weather', 'sports'] else f"{f}_term" 
                                 if f == 'school' else f"{f}" 
                                 for f in selected_features])) or 'baseline'
    
    predictions = results['predictions'].get(feature_key, results['predictions']['baseline'])
    
    # Create plots for each model
    plots = {}
    for model in ['prophet', 'tft', 'bayesian']:
        plots[model] = create_model_plot(
            actual, predictions, feature_dates,
            selected_features, model, predictions[model]
        )
    
    return plots

def streamlit_app():
    import streamlit as st
    
    st.set_page_config(layout="wide")
    st.title("Demand Forecasting Model Comparison")
    
    # Sidebar feature selection
    st.sidebar.header("Feature Selection")
    all_features = ['promotions', 'weather', 'sports', 'school', 'holidays']
    selected_features = []
    for feature in all_features:
        if st.sidebar.checkbox(feature.title(), value=True):
            selected_features.append(feature)
    
    # Get plots
    plots = run_analysis(selected_features)
    
    # Display plots in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(plots['prophet'], use_container_width=True)
    
    with col2:
        st.plotly_chart(plots['tft'], use_container_width=True)
    
    with col3:
        st.plotly_chart(plots['bayesian'], use_container_width=True)
    
    # Model Explanations
    st.markdown("""
    ### Model Characteristics:
    
    1. **Prophet**
    - Handles holidays and seasonality explicitly
    - Decomposes trend, seasonality, and holiday effects
    - Best for data with strong seasonal patterns
    
    2. **Temporal Fusion Transformer (TFT)**
    - Captures complex feature interactions
    - Uses attention mechanism for interpretability
    - Excels at long-term dependencies
    
    3. **Bayesian Ensemble**
    - Provides uncertainty estimates
    - Robust to outliers and noise
    - Combines multiple models for better predictions
    """)

if __name__ == "__main__":
    import sys
    import os
    
    # Handle both direct Python and Streamlit execution
    if "STREAMLIT_SCRIPT_PATH" in os.environ:
        streamlit_app()
    else:
        # Regular Python execution
        plots = run_analysis()
        for model, fig in plots.items():
            fig.show()
