#!/usr/bin/env python3

import re

def fix_duv_complete():
    """Completely fix the DUV project page with proper links and working charts"""
    with open('duv-energy-deposition-project.html', 'r') as f:
        content = f.read()
    
    # Fix Flare Level link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-sun demo-icon"></i>\s*<h3 class="demo-title">Flare Level</h3>\s*<p class="demo-description">Adjustable flare level parameter for realistic scattered light effects\.</p>\s*</div>',
        '<a href="duv-flare-demo.html" class="demo-card">\n                <i class="fas fa-sun demo-icon"></i>\n                <h3 class="demo-title">Flare Level</h3>\n                <p class="demo-description">Adjustable flare level parameter for realistic scattered light effects.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Long-Tail Effects link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-chart-line demo-icon"></i>\s*<h3 class="demo-title">Long-Tail Effects</h3>\s*<p class="demo-description">Third Gaussian component for accurate long-range flare modeling\.</p>\s*</div>',
        '<a href="duv-flare-demo.html" class="demo-card">\n                <i class="fas fa-chart-line demo-icon"></i>\n                <h3 class="demo-title">Long-Tail Effects</h3>\n                <p class="demo-description">Third Gaussian component for accurate long-range flare modeling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Duty Cycle Analysis link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-wave-square demo-icon"></i>\s*<h3 class="demo-title">Duty Cycle Analysis</h3>\s*<p class="demo-description">Swing curves vs duty cycle with contrast and NILS calculations\.</p>\s*</div>',
        '<a href="duv-swing-curves-demo.html" class="demo-card">\n                <i class="fas fa-wave-square demo-icon"></i>\n                <h3 class="demo-title">Duty Cycle Analysis</h3>\n                <p class="demo-description">Swing curves vs duty cycle with contrast and NILS calculations.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Pitch Dependence link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-chart-area demo-icon"></i>\s*<h3 class="demo-title">Pitch Dependence</h3>\s*<p class="demo-description">Contrast and NILS vs pitch analysis for lithography optimization\.</p>\s*</div>',
        '<a href="duv-swing-curves-demo.html" class="demo-card">\n                <i class="fas fa-chart-area demo-icon"></i>\n                <h3 class="demo-title">Pitch Dependence</h3>\n                <p class="demo-description">Contrast and NILS vs pitch analysis for lithography optimization.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Remove any existing script and add comprehensive chart initialization
    content = re.sub(r'<script>.*?</script>', '', content, flags=re.DOTALL)
    
    # Add comprehensive JavaScript for all charts
    chart_script = '''
    <script>
        // Initialize all charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing DUV charts...');
            
            // Monte Carlo Chart - Particle Distribution
            const monteCarloData = [{
                x: Array.from({length: 100}, (_, i) => (i - 50) * 0.02),
                y: Array.from({length: 100}, (_, i) => Math.exp(-((i - 50) * 0.02)**2 / (2 * 0.05**2)) + Math.random() * 0.1),
                type: 'scatter',
                mode: 'markers',
                name: 'Particle Distribution',
                marker: { 
                    color: '#667eea', 
                    size: 3,
                    opacity: 0.7
                }
            }];
            
            const monteCarloLayout = {
                title: {
                    text: 'Monte Carlo Particle Distribution',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Position (μm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'Energy Density',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('monte-carlo-chart', monteCarloData, monteCarloLayout);
            
            // Double Gaussian Chart - PSF Components
            const x = Array.from({length: 100}, (_, i) => (i - 50) * 0.02);
            const gaussian1 = x.map(xi => Math.exp(-xi**2 / (2 * 0.03**2)));
            const gaussian2 = x.map(xi => 0.3 * Math.exp(-xi**2 / (2 * 0.1**2)));
            const total = gaussian1.map((g1, i) => g1 + gaussian2[i]);
            
            const doubleGaussianData = [
                { 
                    x: x, 
                    y: gaussian1, 
                    type: 'scatter', 
                    mode: 'lines', 
                    name: 'Narrow Gaussian (σ=30nm)', 
                    line: { color: '#667eea', width: 2 } 
                },
                { 
                    x: x, 
                    y: gaussian2, 
                    type: 'scatter', 
                    mode: 'lines', 
                    name: 'Wide Gaussian (σ=100nm)', 
                    line: { color: '#ff6b6b', width: 2 } 
                },
                { 
                    x: x, 
                    y: total, 
                    type: 'scatter', 
                    mode: 'lines', 
                    name: 'Total PSF', 
                    line: { color: '#22c55e', width: 3, dash: 'dash' } 
                }
            ];
            
            const doubleGaussianLayout = {
                title: {
                    text: 'Double Gaussian PSF Components',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Position (μm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'PSF Value',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('double-gaussian-chart', doubleGaussianData, doubleGaussianLayout);
            
            // Partial Coherence Chart - Resolution and DOF
            const naValues = Array.from({length: 20}, (_, i) => 0.5 + i * 0.025);
            const resolution = naValues.map(na => 0.61 * 193 / na);
            const dof = naValues.map(na => 193 / (2 * na**2));
            
            const coherenceData = [
                { 
                    x: naValues, 
                    y: resolution, 
                    type: 'scatter', 
                    mode: 'lines+markers', 
                    name: 'Resolution (nm)', 
                    line: { color: '#667eea', width: 2 },
                    marker: { size: 6 }
                },
                { 
                    x: naValues, 
                    y: dof, 
                    type: 'scatter', 
                    mode: 'lines+markers', 
                    name: 'Depth of Focus (nm)', 
                    line: { color: '#ff6b6b', width: 2 },
                    marker: { size: 6 }
                }
            ];
            
            const coherenceLayout = {
                title: {
                    text: 'Resolution and DOF vs Numerical Aperture',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Numerical Aperture',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'Value (nm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('partial-coherence-chart', coherenceData, coherenceLayout);
            
            // Flare Chart - Flare Effects
            const flareX = Array.from({length: 100}, (_, i) => (i - 50) * 0.05);
            const mainPSF = flareX.map(xi => Math.exp(-xi**2 / (2 * 0.05**2)));
            const flarePSF = flareX.map(xi => 0.02 * Math.exp(-xi**2 / (2 * 0.2**2)) * Math.exp(-Math.abs(xi) * 0.5));
            const totalFlare = mainPSF.map((m, i) => m + flarePSF[i]);
            
            const flareData = [
                { 
                    x: flareX, 
                    y: mainPSF, 
                    type: 'scatter', 
                    mode: 'lines', 
                    name: 'Main PSF', 
                    line: { color: '#667eea', width: 2 } 
                },
                { 
                    x: flareX, 
                    y: flarePSF, 
                    type: 'scatter', 
                    mode: 'lines', 
                    name: 'Flare PSF', 
                    line: { color: '#ff6b6b', width: 2 } 
                },
                { 
                    x: flareX, 
                    y: totalFlare, 
                    type: 'scatter', 
                    mode: 'lines', 
                    name: 'Total PSF with Flare', 
                    line: { color: '#22c55e', width: 3, dash: 'dash' } 
                }
            ];
            
            const flareLayout = {
                title: {
                    text: 'Flare Effects on PSF',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Position (μm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'PSF Value',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('flare-chart', flareData, flareLayout);
            
            // Swing Curves Chart - Contrast and NILS
            const pitch = Array.from({length: 50}, (_, i) => 50 + i * 5);
            const contrast = pitch.map(p => Math.sin(Math.PI * p / 200) / (Math.PI * p / 200));
            const nils = pitch.map(p => contrast[pitch.indexOf(p)] * (2 * Math.PI / p) * (1 / 0.85));
            
            const swingData = [
                { 
                    x: pitch, 
                    y: contrast, 
                    type: 'scatter', 
                    mode: 'lines+markers', 
                    name: 'Contrast', 
                    line: { color: '#667eea', width: 2 },
                    marker: { size: 6 }
                },
                { 
                    x: pitch, 
                    y: nils, 
                    type: 'scatter', 
                    mode: 'lines+markers', 
                    name: 'NILS', 
                    line: { color: '#ff6b6b', width: 2 },
                    marker: { size: 6 }
                }
            ];
            
            const swingLayout = {
                title: {
                    text: 'Swing Curves: Contrast and NILS vs Pitch',
                    font: { color: '#e0e0e0', size: 16 }
                },
                xaxis: { 
                    title: 'Pitch (nm)',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                yaxis: { 
                    title: 'Value',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(102, 126, 234, 0.2)'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' },
                legend: { font: { color: '#e0e0e0' } }
            };
            
            Plotly.newPlot('swing-curves-chart', swingData, swingLayout);
            
            console.log('All DUV charts initialized successfully!');
        });
    </script>
    '''
    
    # Add the script before closing body tag
    content = content.replace('</body>', chart_script + '\n</body>')
    
    with open('duv-energy-deposition-project.html', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("Completely fixing DUV project page...")
    fix_duv_complete()
    print("Done!")