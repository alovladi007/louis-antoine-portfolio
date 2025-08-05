#!/usr/bin/env python3
"""
Interactive Thermal Simulation Demo for InGaN/GaN HEMTs
This creates an HTML file with interactive plots using Plotly
"""

import numpy as np
import json

# Create interactive plots data
def generate_interactive_data():
    # Temperature distribution data
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    
    # Generate temperature fields for different power levels
    power_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    temperature_data = {}
    
    for power in power_levels:
        gate_x, gate_y = 5, 50
        T_base = 27
        T_peak = 27 + power * 50  # 50 K/W thermal resistance
        sigma = 5
        
        Z = T_base + (T_peak - T_base) * np.exp(-((X - gate_x)**2 + (Y - gate_y)**2) / (2 * sigma**2))
        drain_effect = power * 10 * np.exp(-((X - 7)**2 + (Y - gate_y)**2) / (2 * (sigma*1.5)**2))
        Z += drain_effect
        
        temperature_data[f'power_{power}'] = {
            'x': x.tolist(),
            'y': y.tolist(),
            'z': Z.tolist(),
            'power': power,
            'max_temp': float(np.max(Z)),
            'thermal_resistance': float((np.max(Z) - T_base) / power)
        }
    
    # Transient response data
    transient_data = {}
    for tau in [5, 10, 15, 20, 25]:  # Different time constants
        t = np.linspace(0, 100, 1000)  # μs
        T_amb = 27
        P_pulse = 2.0
        R_th = 50
        
        # Pulse train
        t_on = 10
        t_off = 40
        T = np.zeros_like(t)
        T[0] = T_amb
        
        for i in range(1, len(t)):
            cycle_time = t[i] % (t_on + t_off)
            P = P_pulse if cycle_time < t_on else 0
            
            dT = (P * R_th - (T[i-1] - T_amb)) * 0.1 / tau
            T[i] = T[i-1] + dT
        
        transient_data[f'tau_{tau}'] = {
            'time': t.tolist(),
            'temperature': T.tolist(),
            'tau': tau,
            'steady_state': float(T_amb + P_pulse * R_th),
            'time_to_90': float(tau * 2.3)  # Time to reach 90% of steady state
        }
    
    return temperature_data, transient_data

# Generate HTML with interactive plots
def create_interactive_html():
    temperature_data, transient_data = generate_interactive_data()
    
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Interactive Thermal Simulation - InGaN/GaN HEMT</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #333;
        }
        .plot-container {
            margin: 20px 0;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }
        .control-group {
            margin: 10px 0;
        }
        label {
            display: inline-block;
            width: 200px;
            font-weight: bold;
        }
        input[type="range"] {
            width: 300px;
        }
        .value-display {
            display: inline-block;
            width: 100px;
            text-align: right;
            font-weight: bold;
            color: #667eea;
        }
        .info-box {
            background-color: #e8f0ff;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .results {
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Thermal Simulation - InGaN/GaN HEMT</h1>
        
        <div class="info-box">
            <p><strong>Instructions:</strong> Use the sliders below to adjust parameters and see real-time thermal response.</p>
        </div>
        
        <h2>1. Steady-State Temperature Distribution</h2>
        <div class="controls">
            <div class="control-group">
                <label>Power Dissipation:</label>
                <input type="range" id="powerSlider" min="0.5" max="3.0" step="0.1" value="2.0">
                <span class="value-display" id="powerValue">2.0 W</span>
            </div>
        </div>
        
        <div id="temperaturePlot" class="plot-container"></div>
        
        <div class="results" id="steadyStateResults">
            <strong>Results:</strong><br>
            Max Temperature: <span id="maxTemp">127.0</span>°C<br>
            Thermal Resistance: <span id="thermalRes">50.0</span> K/W<br>
            Temperature Rise: <span id="tempRise">100.0</span> K
        </div>
        
        <h2>2. Transient Thermal Response</h2>
        <div class="controls">
            <div class="control-group">
                <label>Thermal Time Constant:</label>
                <input type="range" id="tauSlider" min="5" max="25" step="5" value="15">
                <span class="value-display" id="tauValue">15 μs</span>
            </div>
        </div>
        
        <div id="transientPlot" class="plot-container"></div>
        
        <div class="results" id="transientResults">
            <strong>Results:</strong><br>
            Time Constant: <span id="timeConstant">15</span> μs<br>
            Time to 90% Steady-State: <span id="timeTo90">34.5</span> μs<br>
            Steady-State Temperature: <span id="steadyTemp">127.0</span>°C
        </div>
        
        <h2>3. Parameter Sensitivity Analysis</h2>
        <div id="sensitivityPlot" class="plot-container"></div>
    </div>
    
    <script>
        // Temperature data
        const tempData = ''' + json.dumps(temperature_data) + ''';
        
        // Transient data
        const transientData = ''' + json.dumps(transient_data) + ''';
        
        // Initial plots
        function updateTemperaturePlot(power) {
            const key = 'power_' + power;
            const data = tempData[key];
            
            const trace = {
                x: data.x,
                y: data.y,
                z: data.z,
                type: 'contour',
                colorscale: 'Hot',
                contours: {
                    coloring: 'heatmap'
                },
                colorbar: {
                    title: 'Temperature (°C)',
                    titleside: 'right'
                }
            };
            
            const layout = {
                title: 'Temperature Distribution at ' + power + ' W',
                xaxis: {title: 'X Position (μm)'},
                yaxis: {title: 'Y Position (μm)'},
                width: 700,
                height: 500
            };
            
            Plotly.newPlot('temperaturePlot', [trace], layout);
            
            // Update results
            document.getElementById('maxTemp').textContent = data.max_temp.toFixed(1);
            document.getElementById('thermalRes').textContent = data.thermal_resistance.toFixed(1);
            document.getElementById('tempRise').textContent = (data.max_temp - 27).toFixed(1);
        }
        
        function updateTransientPlot(tau) {
            const key = 'tau_' + tau;
            const data = transientData[key];
            
            const trace = {
                x: data.time,
                y: data.temperature,
                type: 'scatter',
                mode: 'lines',
                name: 'Temperature',
                line: {color: 'red', width: 2}
            };
            
            const layout = {
                title: 'Transient Response (τ = ' + tau + ' μs)',
                xaxis: {title: 'Time (μs)'},
                yaxis: {title: 'Temperature (°C)'},
                width: 700,
                height: 400,
                shapes: [
                    // Add pulse indicators
                    {type: 'rect', x0: 0, x1: 10, y0: 20, y1: 140, 
                     fillcolor: 'blue', opacity: 0.2, line: {width: 0}},
                    {type: 'rect', x0: 50, x1: 60, y0: 20, y1: 140, 
                     fillcolor: 'blue', opacity: 0.2, line: {width: 0}}
                ]
            };
            
            Plotly.newPlot('transientPlot', [trace], layout);
            
            // Update results
            document.getElementById('timeConstant').textContent = tau;
            document.getElementById('timeTo90').textContent = data.time_to_90.toFixed(1);
            document.getElementById('steadyTemp').textContent = data.steady_state.toFixed(1);
        }
        
        function createSensitivityPlot() {
            // Power vs Temperature for different materials
            const powers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
            
            const trace1 = {
                x: powers,
                y: powers.map(p => 27 + p * 50),  // SiC
                type: 'scatter',
                mode: 'lines+markers',
                name: 'SiC Substrate',
                line: {color: 'blue', width: 2}
            };
            
            const trace2 = {
                x: powers,
                y: powers.map(p => 27 + p * 80),  // Si
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Si Substrate',
                line: {color: 'red', width: 2}
            };
            
            const trace3 = {
                x: powers,
                y: powers.map(p => 27 + p * 120),  // Sapphire
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Sapphire Substrate',
                line: {color: 'green', width: 2}
            };
            
            const layout = {
                title: 'Temperature vs Power for Different Substrates',
                xaxis: {title: 'Power Dissipation (W)'},
                yaxis: {title: 'Junction Temperature (°C)'},
                width: 700,
                height: 400,
                shapes: [{
                    type: 'line',
                    x0: 0.5, x1: 3.0,
                    y0: 150, y1: 150,
                    line: {color: 'red', width: 2, dash: 'dash'}
                }],
                annotations: [{
                    x: 2.5,
                    y: 155,
                    text: 'Max Safe Temperature',
                    showarrow: false,
                    font: {color: 'red'}
                }]
            };
            
            Plotly.newPlot('sensitivityPlot', [trace1, trace2, trace3], layout);
        }
        
        // Event listeners
        document.getElementById('powerSlider').addEventListener('input', function(e) {
            const value = parseFloat(e.target.value);
            document.getElementById('powerValue').textContent = value.toFixed(1) + ' W';
            
            // Find closest available power level
            const availablePowers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
            const closest = availablePowers.reduce((prev, curr) => 
                Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
            );
            
            updateTemperaturePlot(closest);
        });
        
        document.getElementById('tauSlider').addEventListener('input', function(e) {
            const value = parseInt(e.target.value);
            document.getElementById('tauValue').textContent = value + ' μs';
            updateTransientPlot(value);
        });
        
        // Initialize plots
        updateTemperaturePlot(2.0);
        updateTransientPlot(15);
        createSensitivityPlot();
    </script>
</body>
</html>'''
    
    # Save HTML file
    output_path = '/workspace/louis-antoine-portfolio/thermal-modeling-files/interactive_demo.html'
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive demo created: {output_path}")
    return output_path

# Run the demo creation
if __name__ == "__main__":
    create_interactive_html()
    print("\nTo view the interactive demo, open the HTML file in a web browser.")