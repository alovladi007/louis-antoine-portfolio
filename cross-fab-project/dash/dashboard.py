#!/usr/bin/env python3
"""
Cross-Fab Project: Dashboard Module
Interactive dashboard for wafer maps, drift timelines, prediction intervals, and controller moves.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CrossFabDashboard:
    """Interactive dashboard for Cross-Fab project monitoring."""
    
    def __init__(self, config: Dict = None):
        """Initialize dashboard."""
        self.config = config or {
            'wafer_size': 300,  # mm
            'field_size': 26,   # mm
            'zone_size': 8.67,  # mm (field_size / 3)
            'cd_target': 45.0,  # nm
            'overlay_target': 5.0,  # nm
            'cd_tolerance': 2.0,  # nm
            'overlay_tolerance': 3.0,  # nm
            'update_interval': 30  # seconds
        }
        
        self.data = {}
        self.prediction_data = {}
        self.control_data = {}
        self.monitoring_data = {}
        
    def load_data(self, data_dir: str = "data") -> None:
        """Load data from CSV files."""
        print("Loading data for dashboard...")
        
        try:
            self.data['lithography'] = pd.read_csv(f"{data_dir}/lithography_data.csv")
            self.data['etch_deposition'] = pd.read_csv(f"{data_dir}/etch_deposition_data.csv")
            self.data['metrology'] = pd.read_csv(f"{data_dir}/metrology_data.csv")
            self.data['context'] = pd.read_csv(f"{data_dir}/context_data.csv")
            print("Data loaded successfully!")
        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            print("Generating sample data...")
            self._generate_sample_data()
    
    def _generate_sample_data(self) -> None:
        """Generate sample data for demonstration."""
        np.random.seed(42)
        n_wafers = 50
        n_fields = 25
        n_zones = 9
        
        # Generate lithography data
        litho_data = []
        for wafer_id in range(n_wafers):
            for field_id in range(n_fields):
                field_x = (field_id % 5) - 2
                field_y = (field_id // 5) - 2
                litho_data.append({
                    'wafer_id': wafer_id,
                    'field_id': field_id,
                    'field_x': field_x,
                    'field_y': field_y,
                    'dose': 20.0 + np.random.normal(0, 0.5),
                    'focus': np.random.normal(0, 0.1),
                    'stage_temp': 22.0 + np.random.normal(0, 0.5),
                    'align_x': np.random.normal(0, 0.05),
                    'align_y': np.random.normal(0, 0.05),
                    'nils_proxy': 0.8 + np.random.normal(0, 0.1),
                    'slit_scan': np.random.uniform(0.8, 1.0)
                })
        
        self.data['lithography'] = pd.DataFrame(litho_data)
        
        # Generate metrology data
        metrology_data = []
        for wafer_id in range(n_wafers):
            for zone_id in range(n_zones):
                zone_x = (zone_id % 3) - 1
                zone_y = (zone_id // 3) - 1
                
                # CD with spatial variation
                cd_base = 45.0
                cd_variation = np.random.normal(0, 1.5)
                cd_zone_effect = 0.5 * (abs(zone_x) + abs(zone_y))
                cd = cd_base + cd_variation + cd_zone_effect
                
                # Overlay
                overlay_x = np.random.normal(0, 2.0)
                overlay_y = np.random.normal(0, 2.0)
                overlay = np.sqrt(overlay_x**2 + overlay_y**2)
                
                metrology_data.append({
                    'wafer_id': wafer_id,
                    'zone_id': zone_id,
                    'zone_x': zone_x,
                    'zone_y': zone_y,
                    'cd': cd,
                    'overlay_x': overlay_x,
                    'overlay_y': overlay_y,
                    'overlay': overlay
                })
        
        self.data['metrology'] = pd.DataFrame(metrology_data)
        
        # Generate prediction data
        self.prediction_data = {
            'cd_predictions': np.random.normal(45.0, 1.0, n_wafers),
            'overlay_predictions': np.random.normal(5.0, 0.5, n_wafers),
            'cd_confidence': np.random.uniform(0.8, 0.95, n_wafers),
            'overlay_confidence': np.random.uniform(0.8, 0.95, n_wafers)
        }
        
        # Generate control data
        n_control_steps = 20
        self.control_data = {
            'steps': list(range(n_control_steps)),
            'dose_adjustments': np.random.normal(0, 0.5, n_control_steps),
            'focus_adjustments': np.random.normal(0, 0.1, n_control_steps),
            'cd_controls': np.random.normal(0, 1.0, n_control_steps),
            'overlay_controls': np.random.normal(0, 0.5, n_control_steps)
        }
        
        print("Sample data generated!")
    
    def create_wafer_map(self, wafer_id: int, metric: str = 'cd') -> go.Figure:
        """Create wafer map for specified metric."""
        if metric not in ['cd', 'overlay']:
            raise ValueError("Metric must be 'cd' or 'overlay'")
        
        # Get data for specific wafer
        wafer_data = self.data['metrology'][self.data['metrology']['wafer_id'] == wafer_id].copy()
        
        if wafer_data.empty:
            print(f"No data found for wafer {wafer_id}")
            return go.Figure()
        
        # Create grid for wafer map
        x_coords = wafer_data['zone_x'].values
        y_coords = wafer_data['zone_y'].values
        values = wafer_data[metric].values
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=x_coords,
            y=y_coords,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title=f"{metric.upper()} (nm)")
        ))
        
        fig.update_layout(
            title=f"Wafer {wafer_id} - {metric.upper()} Map",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            width=600,
            height=500
        )
        
        return fig
    
    def create_drift_timeline(self) -> go.Figure:
        """Create drift timeline plot."""
        # Generate sample timeline data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Simulate drift
        base_cd = 45.0
        base_overlay = 5.0
        cd_drift = np.cumsum(np.random.normal(0, 0.1, 30))
        overlay_drift = np.cumsum(np.random.normal(0, 0.05, 30))
        
        cd_values = base_cd + cd_drift
        overlay_values = base_overlay + overlay_drift
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CD Drift Timeline', 'Overlay Drift Timeline'),
            vertical_spacing=0.1
        )
        
        # CD drift
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cd_values,
                mode='lines+markers',
                name='CD',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # CD target line
        fig.add_hline(
            y=self.config['cd_target'],
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
            row=1, col=1
        )
        
        # CD tolerance bands
        fig.add_hline(
            y=self.config['cd_target'] + self.config['cd_tolerance'],
            line_dash="dot",
            line_color="orange",
            row=1, col=1
        )
        fig.add_hline(
            y=self.config['cd_target'] - self.config['cd_tolerance'],
            line_dash="dot",
            line_color="orange",
            row=1, col=1
        )
        
        # Overlay drift
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=overlay_values,
                mode='lines+markers',
                name='Overlay',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Overlay target line
        fig.add_hline(
            y=self.config['overlay_target'],
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
            row=2, col=1
        )
        
        # Overlay tolerance bands
        fig.add_hline(
            y=self.config['overlay_target'] + self.config['overlay_tolerance'],
            line_dash="dot",
            line_color="orange",
            row=2, col=1
        )
        fig.add_hline(
            y=self.config['overlay_target'] - self.config['overlay_tolerance'],
            line_dash="dot",
            line_color="orange",
            row=2, col=1
        )
        
        fig.update_layout(
            title="Process Drift Timeline",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_prediction_intervals(self) -> go.Figure:
        """Create prediction intervals plot."""
        # Generate sample prediction data
        n_samples = 50
        wafer_ids = list(range(n_samples))
        
        cd_predictions = self.prediction_data['cd_predictions']
        overlay_predictions = self.prediction_data['overlay_predictions']
        cd_confidence = self.prediction_data['cd_confidence']
        overlay_confidence = self.prediction_data['overlay_confidence']
        
        # Calculate prediction intervals
        cd_std = (1 - cd_confidence) * 2  # Simplified uncertainty
        overlay_std = (1 - overlay_confidence) * 1  # Simplified uncertainty
        
        cd_upper = cd_predictions + 1.96 * cd_std
        cd_lower = cd_predictions - 1.96 * cd_std
        overlay_upper = overlay_predictions + 1.96 * overlay_std
        overlay_lower = overlay_predictions - 1.96 * overlay_std
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CD Predictions with Confidence Intervals', 
                          'Overlay Predictions with Confidence Intervals'),
            vertical_spacing=0.1
        )
        
        # CD predictions
        fig.add_trace(
            go.Scatter(
                x=wafer_ids,
                y=cd_predictions,
                mode='markers',
                name='CD Predictions',
                marker=dict(color='blue', size=8)
            ),
            row=1, col=1
        )
        
        # CD confidence intervals
        fig.add_trace(
            go.Scatter(
                x=wafer_ids + wafer_ids[::-1],
                y=list(cd_upper) + list(cd_lower[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='CD 95% CI',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # CD target line
        fig.add_hline(
            y=self.config['cd_target'],
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
            row=1, col=1
        )
        
        # Overlay predictions
        fig.add_trace(
            go.Scatter(
                x=wafer_ids,
                y=overlay_predictions,
                mode='markers',
                name='Overlay Predictions',
                marker=dict(color='green', size=8)
            ),
            row=2, col=1
        )
        
        # Overlay confidence intervals
        fig.add_trace(
            go.Scatter(
                x=wafer_ids + wafer_ids[::-1],
                y=list(overlay_upper) + list(overlay_lower[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Overlay 95% CI',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Overlay target line
        fig.add_hline(
            y=self.config['overlay_target'],
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
            row=2, col=1
        )
        
        fig.update_layout(
            title="Virtual Metrology Predictions",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_controller_moves(self) -> go.Figure:
        """Create controller moves plot."""
        steps = self.control_data['steps']
        dose_adj = self.control_data['dose_adjustments']
        focus_adj = self.control_data['focus_adjustments']
        cd_control = self.control_data['cd_controls']
        overlay_control = self.control_data['overlay_controls']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dose Adjustments', 'Focus Adjustments',
                          'CD Control Signal', 'Overlay Control Signal'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Dose adjustments
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=dose_adj,
                mode='lines+markers',
                name='Dose Adj',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Focus adjustments
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=focus_adj,
                mode='lines+markers',
                name='Focus Adj',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # CD control signal
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=cd_control,
                mode='lines+markers',
                name='CD Control',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Overlay control signal
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=overlay_control,
                mode='lines+markers',
                name='Overlay Control',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="R2R Controller Moves",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_pareto_chart(self) -> go.Figure:
        """Create Pareto chart of contributors."""
        # Generate sample contributor data
        contributors = [
            'Dose Variation', 'Focus Drift', 'Chamber Matching',
            'Temperature Effects', 'Alignment Errors', 'Tool Wear',
            'Reticle Issues', 'Edge Effects', 'Process Noise'
        ]
        
        contributions = [25, 20, 15, 12, 10, 8, 5, 3, 2]
        cumulative = np.cumsum(contributions)
        
        # Create Pareto chart
        fig = go.Figure()
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=contributors,
            y=contributions,
            name='Contribution (%)',
            marker_color='lightblue'
        ))
        
        # Cumulative line
        fig.add_trace(go.Scatter(
            x=contributors,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative (%)',
            yaxis='y2',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title='Pareto Chart of Process Contributors',
            xaxis_title='Contributors',
            yaxis=dict(title='Contribution (%)', side='left'),
            yaxis2=dict(title='Cumulative (%)', side='right', overlaying='y'),
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_dashboard_html(self, output_dir: str = "dash") -> str:
        """Create complete dashboard HTML."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Create all plots
        wafer_map = self.create_wafer_map(0, 'cd')
        drift_timeline = self.create_drift_timeline()
        prediction_intervals = self.create_prediction_intervals()
        controller_moves = self.create_controller_moves()
        pareto_chart = self.create_pareto_chart()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Fab Project Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .plot-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
                @media (max-width: 768px) {{
                    .grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cross-Fab Project Dashboard</h1>
                <p>Virtual Metrology + Run-to-Run APC for CD & Overlay Uniformity</p>
                <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="plot-container">
                <h2>Wafer Map - CD Distribution</h2>
                <div id="wafer-map"></div>
            </div>
            
            <div class="plot-container">
                <h2>Process Drift Timeline</h2>
                <div id="drift-timeline"></div>
            </div>
            
            <div class="plot-container">
                <h2>Virtual Metrology Predictions</h2>
                <div id="prediction-intervals"></div>
            </div>
            
            <div class="plot-container">
                <h2>R2R Controller Moves</h2>
                <div id="controller-moves"></div>
            </div>
            
            <div class="plot-container">
                <h2>Pareto Chart of Contributors</h2>
                <div id="pareto-chart"></div>
            </div>
            
            <script>
                // Wafer Map
                {wafer_map.to_html(include_plotlyjs=False, div_id="wafer-map")}
                
                // Drift Timeline
                {drift_timeline.to_html(include_plotlyjs=False, div_id="drift-timeline")}
                
                // Prediction Intervals
                {prediction_intervals.to_html(include_plotlyjs=False, div_id="prediction-intervals")}
                
                // Controller Moves
                {controller_moves.to_html(include_plotlyjs=False, div_id="controller-moves")}
                
                // Pareto Chart
                {pareto_chart.to_html(include_plotlyjs=False, div_id="pareto-chart")}
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = os.path.join(output_dir, 'dashboard.html')
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to {html_file}")
        return html_file

def main():
    """Main function to create dashboard."""
    print("Cross-Fab Project: Dashboard")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = CrossFabDashboard()
    
    # Create dashboard HTML
    html_file = dashboard.create_dashboard_html()
    
    print(f"Dashboard created successfully!")
    print(f"Open {html_file} in your browser to view the dashboard.")

if __name__ == "__main__":
    main()