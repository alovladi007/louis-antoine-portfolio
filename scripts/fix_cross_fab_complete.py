#!/usr/bin/env python3

import re

def fix_cross_fab_complete():
    """Completely fix the Cross-Fab project page with working charts"""
    with open('cross-fab-vm-r2r-project.html', 'r') as f:
        content = f.read()
    
    # Remove any existing script and add comprehensive chart initialization
    content = re.sub(r'<script>.*?</script>', '', content, flags=re.DOTALL)
    
    # Add comprehensive JavaScript for all charts
    chart_script = '''
    <script>
        // Initialize all charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing Cross-Fab charts...');
            
            // Virtual Metrology Chart
            const vmData = [{
                x: Array.from({length: 30}, (_, i) => i + 1),
                y: Array.from({length: 30}, (_, i) => 50 + Math.sin(i * 0.2) * 5 + Math.random() * 2),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted CD',
                line: { color: '#667eea', width: 2 },
                marker: { size: 6 }
            }, {
                x: Array.from({length: 30}, (_, i) => i + 1),
                y: Array.from({length: 30}, (_, i) => 50 + Math.sin(i * 0.2) * 5 + Math.random() * 2 - 0.5),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual CD',
                line: { color: '#ff6b6b', width: 2 },
                marker: { size: 6 }
            }];
            
            const vmLayout = {
                title: {
                    text: 'Virtual Metrology: Predicted vs Actual CD',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Sample Number',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'CD (nm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('vm-chart', vmData, vmLayout);
            
            // Run-to-Run Control Chart
            const r2rData = [{
                x: Array.from({length: 50}, (_, i) => i + 1),
                y: Array.from({length: 50}, (_, i) => 50 + Math.sin(i * 0.1) * 3 + Math.random() * 1),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'CD Measurement',
                line: { color: '#667eea', width: 2 },
                marker: { size: 4 }
            }, {
                x: Array.from({length: 50}, (_, i) => i + 1),
                y: Array.from({length: 50}, (_, i) => 50),
                type: 'scatter',
                mode: 'lines',
                name: 'Target',
                line: { color: '#22c55e', width: 2, dash: 'dash' }
            }, {
                x: Array.from({length: 50}, (_, i) => i + 1),
                y: Array.from({length: 50}, (_, i) => 50 + Math.sin(i * 0.1) * 3 + Math.random() * 1 - 0.5),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Controlled CD',
                line: { color: '#ff6b6b', width: 2 },
                marker: { size: 4 }
            }];
            
            const r2rLayout = {
                title: {
                    text: 'Run-to-Run Control Performance',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Run Number',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'CD (nm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('r2r-control-chart', r2rData, r2rLayout);
            
            // SPC Chart
            const spcData = [{
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 50 + Math.sin(i * 0.05) * 2 + Math.random() * 0.5),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Process Value',
                line: { color: '#667eea', width: 2 },
                marker: { size: 3 }
            }, {
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 52),
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Control Limit',
                line: { color: '#ff6b6b', width: 2, dash: 'dash' }
            }, {
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 48),
                type: 'scatter',
                mode: 'lines',
                name: 'Lower Control Limit',
                line: { color: '#ff6b6b', width: 2, dash: 'dash' }
            }, {
                x: Array.from({length: 100}, (_, i) => i + 1),
                y: Array.from({length: 100}, (_, i) => 50),
                type: 'scatter',
                mode: 'lines',
                name: 'Center Line',
                line: { color: '#22c55e', width: 2, dash: 'dot' }
            }];
            
            const spcLayout = {
                title: {
                    text: 'Statistical Process Control Chart',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Sample Number',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'CD (nm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('spc-chart', spcData, spcLayout);
            
            // Optimization Chart
            const optimizationData = [{
                x: Array.from({length: 20}, (_, i) => i * 0.05),
                y: Array.from({length: 20}, (_, i) => 1 - i * 0.05 + Math.random() * 0.1),
                type: 'scatter',
                mode: 'markers',
                name: 'Pareto Solutions',
                marker: { 
                    color: '#667eea', 
                    size: 8,
                    opacity: 0.8
                }
            }];
            
            const optimizationLayout = {
                title: {
                    text: 'Multi-Objective Optimization: Pareto Front',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'CD Uniformity',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'Overlay Control',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('optimization-chart', optimizationData, optimizationLayout);
            
            console.log('All Cross-Fab charts initialized successfully!');
        });
    </script>
    '''
    
    # Add the script before closing body tag
    content = content.replace('</body>', chart_script + '\n</body>')
    
    with open('cross-fab-vm-r2r-project.html', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("Completely fixing Cross-Fab project page...")
    fix_cross_fab_complete()
    print("Done!")