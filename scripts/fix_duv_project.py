#!/usr/bin/env python3

import re

def fix_duv_project():
    """Fix the DUV project page with proper links and charts"""
    with open('duv-energy-deposition-project.html', 'r') as f:
        content = f.read()
    
    # Fix PSF Calculation link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-wave-square demo-icon"></i>\s*<h3 class="demo-title">PSF Calculation</h3>\s*<p class="demo-description">Double Gaussian point spread function with narrow and wide components for accurate modeling\.</p>\s*</div>',
        '<a href="duv-double-gaussian-demo.html" class="demo-card">\n                <i class="fas fa-wave-square demo-icon"></i>\n                <h3 class="demo-title">PSF Calculation</h3>\n                <p class="demo-description">Double Gaussian point spread function with narrow and wide components for accurate modeling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Coherence Length link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-eye demo-icon"></i>\s*<h3 class="demo-title">Coherence Length</h3>\s*<p class="demo-description">Calculate coherence length from NA and illumination parameters for accurate modeling\.</p>\s*</div>',
        '<a href="duv-partial-coherence-demo.html" class="demo-card">\n                <i class="fas fa-eye demo-icon"></i>\n                <h3 class="demo-title">Coherence Length</h3>\n                <p class="demo-description">Calculate coherence length from NA and illumination parameters for accurate modeling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Pupil Function link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-circle demo-icon"></i>\s*<h3 class="demo-title">Pupil Function</h3>\s*<p class="demo-description">Pupil cut-off effects and illumination sigma parameter optimization\.</p>\s*</div>',
        '<a href="duv-partial-coherence-demo.html" class="demo-card">\n                <i class="fas fa-circle demo-icon"></i>\n                <h3 class="demo-title">Pupil Function</h3>\n                <p class="demo-description">Pupil cut-off effects and illumination sigma parameter optimization.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Add JavaScript to populate charts
    chart_script = '''
    <script>
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Monte Carlo Chart
            const monteCarloData = [{
                x: Array.from({length: 50}, (_, i) => i * 0.1),
                y: Array.from({length: 50}, (_, i) => Math.exp(-(i * 0.1 - 2.5)**2 / 2) + Math.random() * 0.1),
                type: 'scatter',
                mode: 'markers',
                name: 'Particle Distribution',
                marker: { color: '#667eea', size: 4 }
            }];
            
            const monteCarloLayout = {
                title: 'Monte Carlo Particle Distribution',
                xaxis: { title: 'Position (μm)' },
                yaxis: { title: 'Energy Density' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('monte-carlo-chart', monteCarloData, monteCarloLayout);
            
            // Double Gaussian Chart
            const x = Array.from({length: 100}, (_, i) => (i - 50) * 0.02);
            const gaussian1 = x.map(xi => Math.exp(-xi**2 / (2 * 0.03**2)));
            const gaussian2 = x.map(xi => 0.3 * Math.exp(-xi**2 / (2 * 0.1**2)));
            const total = gaussian1.map((g1, i) => g1 + gaussian2[i]);
            
            const doubleGaussianData = [
                { x: x, y: gaussian1, type: 'scatter', mode: 'lines', name: 'Narrow Gaussian', line: { color: '#667eea' } },
                { x: x, y: gaussian2, type: 'scatter', mode: 'lines', name: 'Wide Gaussian', line: { color: '#ff6b6b' } },
                { x: x, y: total, type: 'scatter', mode: 'lines', name: 'Total PSF', line: { color: '#22c55e', dash: 'dash' } }
            ];
            
            const doubleGaussianLayout = {
                title: 'Double Gaussian PSF Components',
                xaxis: { title: 'Position (μm)' },
                yaxis: { title: 'PSF Value' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('double-gaussian-chart', doubleGaussianData, doubleGaussianLayout);
            
            // Partial Coherence Chart
            const naValues = Array.from({length: 20}, (_, i) => 0.5 + i * 0.025);
            const resolution = naValues.map(na => 0.61 * 193 / na);
            const dof = naValues.map(na => 193 / (2 * na**2));
            
            const coherenceData = [
                { x: naValues, y: resolution, type: 'scatter', mode: 'lines+markers', name: 'Resolution (nm)', line: { color: '#667eea' } },
                { x: naValues, y: dof, type: 'scatter', mode: 'lines+markers', name: 'Depth of Focus (nm)', line: { color: '#ff6b6b' } }
            ];
            
            const coherenceLayout = {
                title: 'Resolution and DOF vs Numerical Aperture',
                xaxis: { title: 'Numerical Aperture' },
                yaxis: { title: 'Value (nm)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('partial-coherence-chart', coherenceData, coherenceLayout);
            
            // Flare Chart
            const flareX = Array.from({length: 100}, (_, i) => (i - 50) * 0.05);
            const mainPSF = flareX.map(xi => Math.exp(-xi**2 / (2 * 0.05**2)));
            const flarePSF = flareX.map(xi => 0.02 * Math.exp(-xi**2 / (2 * 0.2**2)) * Math.exp(-Math.abs(xi) * 0.5));
            const totalFlare = mainPSF.map((m, i) => m + flarePSF[i]);
            
            const flareData = [
                { x: flareX, y: mainPSF, type: 'scatter', mode: 'lines', name: 'Main PSF', line: { color: '#667eea' } },
                { x: flareX, y: flarePSF, type: 'scatter', mode: 'lines', name: 'Flare PSF', line: { color: '#ff6b6b' } },
                { x: flareX, y: totalFlare, type: 'scatter', mode: 'lines', name: 'Total PSF', line: { color: '#22c55e', dash: 'dash' } }
            ];
            
            const flareLayout = {
                title: 'Flare Effects on PSF',
                xaxis: { title: 'Position (μm)' },
                yaxis: { title: 'PSF Value' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('flare-chart', flareData, flareLayout);
            
            // Swing Curves Chart
            const pitch = Array.from({length: 50}, (_, i) => 50 + i * 5);
            const contrast = pitch.map(p => Math.sin(Math.PI * p / 200) / (Math.PI * p / 200));
            const nils = pitch.map(p => contrast[pitch.indexOf(p)] * (2 * Math.PI / p) * (1 / 0.85));
            
            const swingData = [
                { x: pitch, y: contrast, type: 'scatter', mode: 'lines+markers', name: 'Contrast', line: { color: '#667eea' } },
                { x: pitch, y: nils, type: 'scatter', mode: 'lines+markers', name: 'NILS', line: { color: '#ff6b6b' } }
            ];
            
            const swingLayout = {
                title: 'Swing Curves: Contrast and NILS vs Pitch',
                xaxis: { title: 'Pitch (nm)' },
                yaxis: { title: 'Value' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e0e0e0' }
            };
            
            Plotly.newPlot('swing-curves-chart', swingData, swingLayout);
        });
    </script>
    '''
    
    # Add the script before closing body tag
    content = content.replace('</body>', chart_script + '\n</body>')
    
    with open('duv-energy-deposition-project.html', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("Fixing DUV project page...")
    fix_duv_project()
    print("Done!")