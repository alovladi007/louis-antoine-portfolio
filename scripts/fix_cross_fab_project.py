#!/usr/bin/env python3

import re

def fix_cross_fab_project():
    """Fix the Cross-Fab project page with proper charts and links"""
    with open('cross-fab-vm-r2r-project.html', 'r') as f:
        content = f.read()
    
    # Add JavaScript to populate charts
    chart_script = '''
    <script>
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Virtual Metrology Chart
            const vmData = [{
                x: Array.from({length: 30}, (_, i) => i + 1),
                y: Array.from({length: 30}, (_, i) => 50 + Math.sin(i * 0.2) * 5 + Math.random() * 2),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted CD',
                line: { color: '#667eea' }
            }, {
                x: Array.from({length: 30}, (_, i) => i + 1),
                y: Array.from({length: 30}, (_, i) => 50 + Math.sin(i * 0.2) * 5 + Math.random() * 2 - 0.5),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual CD',
                line: { color: '#ff6b6b' }
            }];
            
            const vmLayout = {
                title: 'Virtual Metrology: Predicted vs Actual CD',
                xaxis: { title: 'Sample Number' },
                yaxis: { title: 'CD (nm)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('vm-chart', vmData, vmLayout);
            
            // Run-to-Run Control Chart
            const r2rData = [{
                x: Array.from({length: 50}, (_, i) => i + 1),
                y: Array.from({length: 50}, (_, i) => 50 + Math.sin(i * 0.1) * 3 + Math.random() * 1),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'CD Measurement',
                line: { color: '#667eea' }
            }, {
                x: Array.from({length: 50}, (_, i) => i + 1),
                y: Array.from({length: 50}, (_, i) => 50),
                type: 'scatter',
                mode: 'lines',
                name: 'Target',
                line: { color: '#22c55e', dash: 'dash' }
            }, {
                x: Array.from({length: 50}, (_, i) => i + 1),
                y: Array.from({length: 50}, (_, i) => 50 + Math.sin(i * 0.1) * 3 + Math.random() * 1 - 0.5),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Controlled CD',
                line: { color: '#ff6b6b' }
            }];
            
            const r2rLayout = {
                title: 'Run-to-Run Control Performance',
                xaxis: { title: 'Run Number' },
                yaxis: { title: 'CD (nm)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('r2r-control-chart', r2rData, r2rLayout);
            
            // SPC Chart
            const spcData = [{
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 50 + Math.sin(i * 0.05) * 2 + Math.random() * 0.5),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Process Value',
                line: { color: '#667eea' }
            }, {
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 52),
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Control Limit',
                line: { color: '#ff6b6b', dash: 'dash' }
            }, {
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 48),
                type: 'scatter',
                mode: 'lines',
                name: 'Lower Control Limit',
                line: { color: '#ff6b6b', dash: 'dash' }
            }, {
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 50),
                type: 'scatter',
                mode: 'lines',
                name: 'Center Line',
                line: { color: '#22c55e', dash: 'dot' }
            }];
            
            const spcLayout = {
                title: 'Statistical Process Control Chart',
                xaxis: { title: 'Sample Number' },
                yaxis: { title: 'CD (nm)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('spc-chart', spcData, spcLayout);
            
            // Optimization Chart
            const optimizationData = [{
                x: Array.from({length: 20}, (_, i) => i * 0.05),
                y: Array.from({length: 20}, (_, i) => 1 - i * 0.05 + Math.random() * 0.1),
                type: 'scatter',
                mode: 'markers',
                name: 'Pareto Solutions',
                marker: { color: '#667eea', size: 8 }
            }];
            
            const optimizationLayout = {
                title: 'Multi-Objective Optimization: Pareto Front',
                xaxis: { title: 'CD Uniformity' },
                yaxis: { title: 'Overlay Control' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('optimization-chart', optimizationData, optimizationLayout);
        });
    </script>
    '''
    
    # Add the script before closing body tag
    content = content.replace('</body>', chart_script + '\n</body>')
    
    with open('cross-fab-vm-r2r-project.html', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("Fixing Cross-Fab project page...")
    fix_cross_fab_project()
    print("Done!")