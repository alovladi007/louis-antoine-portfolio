#!/usr/bin/env python3

import re

def fix_cross_fab_links():
    """Fix broken links in Cross-Fab project page"""
    with open('cross-fab-vm-r2r-project.html', 'r') as f:
        content = f.read()
    
    # Fix Double-EWMA Control link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-wave-square demo-icon"></i>\s*<h3 class="demo-title">Double-EWMA Control</h3>\s*<p class="demo-description">Exponentially weighted moving average control with trend compensation for robust process control\.</p>\s*</div>',
        '<a href="cross-fab-r2r-demo.html" class="demo-card">\n                <i class="fas fa-wave-square demo-icon"></i>\n                <h3 class="demo-title">Double-EWMA Control</h3>\n                <p class="demo-description">Exponentially weighted moving average control with trend compensation for robust process control.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Kalman Filtering link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-filter demo-icon"></i>\s*<h3 class="demo-title">Kalman Filtering</h3>\s*<p class="demo-description">State estimation and prediction with Kalman filters for optimal control performance\.</p>\s*</div>',
        '<a href="cross-fab-r2r-demo.html" class="demo-card">\n                <i class="fas fa-filter demo-icon"></i>\n                <h3 class="demo-title">Kalman Filtering</h3>\n                <p class="demo-description">State estimation and prediction with Kalman filters for optimal control performance.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    with open('cross-fab-vm-r2r-project.html', 'w') as f:
        f.write(content)

def fix_duv_links():
    """Fix broken links in DUV project page"""
    with open('duv-energy-deposition-project.html', 'r') as f:
        content = f.read()
    
    # Fix Particle Generation link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-database demo-icon"></i>\s*<h3 class="demo-title">Particle Generation</h3>\s*<p class="demo-description">Stochastic particle generation with energy distribution, spatial positioning, and directional sampling\.</p>\s*</div>',
        '<a href="duv-monte-carlo-demo.html" class="demo-card">\n                <i class="fas fa-database demo-icon"></i>\n                <h3 class="demo-title">Particle Generation</h3>\n                <p class="demo-description">Stochastic particle generation with energy distribution, spatial positioning, and directional sampling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Statistical Analysis link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-chart-bar demo-icon"></i>\s*<h3 class="demo-title">Statistical Analysis</h3>\s*<p class="demo-description">Comprehensive statistical analysis with uncertainty quantification and confidence intervals\.</p>\s*</div>',
        '<a href="duv-monte-carlo-demo.html" class="demo-card">\n                <i class="fas fa-chart-bar demo-icon"></i>\n                <h3 class="demo-title">Statistical Analysis</h3>\n                <p class="demo-description">Comprehensive statistical analysis with uncertainty quantification and confidence intervals.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix PSF Calculation link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-calculator demo-icon"></i>\s*<h3 class="demo-title">PSF Calculation</h3>\s*<p class="demo-description">Double Gaussian point spread function with narrow and wide components for accurate modeling\.</p>\s*</div>',
        '<a href="duv-double-gaussian-demo.html" class="demo-card">\n                <i class="fas fa-calculator demo-icon"></i>\n                <h3 class="demo-title">PSF Calculation</h3>\n                <p class="demo-description">Double Gaussian point spread function with narrow and wide components for accurate modeling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix FFT Convolution link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-bolt demo-icon"></i>\s*<h3 class="demo-title">FFT Convolution</h3>\s*<p class="demo-description">High-performance FFT-based convolution for efficient aerial image calculation\.</p>\s*</div>',
        '<a href="duv-double-gaussian-demo.html" class="demo-card">\n                <i class="fas fa-bolt demo-icon"></i>\n                <h3 class="demo-title">FFT Convolution</h3>\n                <p class="demo-description">High-performance FFT-based convolution for efficient aerial image calculation.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Coherence Length link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-ruler demo-icon"></i>\s*<h3 class="demo-title">Coherence Length</h3>\s*<p class="demo-description">Calculate coherence length from NA and illumination parameters for accurate modeling\.</p>\s*</div>',
        '<a href="duv-partial-coherence-demo.html" class="demo-card">\n                <i class="fas fa-ruler demo-icon"></i>\n                <h3 class="demo-title">Coherence Length</h3>\n                <p class="demo-description">Calculate coherence length from NA and illumination parameters for accurate modeling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Pupil Function link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-eye demo-icon"></i>\s*<h3 class="demo-title">Pupil Function</h3>\s*<p class="demo-description">Pupil cut-off effects and illumination sigma parameter optimization\.</p>\s*</div>',
        '<a href="duv-partial-coherence-demo.html" class="demo-card">\n                <i class="fas fa-eye demo-icon"></i>\n                <h3 class="demo-title">Pupil Function</h3>\n                <p class="demo-description">Pupil cut-off effects and illumination sigma parameter optimization.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Flare Level link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-adjust demo-icon"></i>\s*<h3 class="demo-title">Flare Level</h3>\s*<p class="demo-description">Adjustable flare level parameter for realistic scattered light effects\.</p>\s*</div>',
        '<a href="duv-flare-demo.html" class="demo-card">\n                <i class="fas fa-adjust demo-icon"></i>\n                <h3 class="demo-title">Flare Level</h3>\n                <p class="demo-description">Adjustable flare level parameter for realistic scattered light effects.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Long-Tail Effects link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-wave-square demo-icon"></i>\s*<h3 class="demo-title">Long-Tail Effects</h3>\s*<p class="demo-description">Third Gaussian component for accurate long-range flare modeling\.</p>\s*</div>',
        '<a href="duv-flare-demo.html" class="demo-card">\n                <i class="fas fa-wave-square demo-icon"></i>\n                <h3 class="demo-title">Long-Tail Effects</h3>\n                <p class="demo-description">Third Gaussian component for accurate long-range flare modeling.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
        content,
        flags=re.DOTALL
    )
    
    # Fix Duty Cycle Analysis link
    content = re.sub(
        r'<div class="demo-card">\s*<i class="fas fa-chart-line demo-icon"></i>\s*<h3 class="demo-title">Duty Cycle Analysis</h3>\s*<p class="demo-description">Swing curves vs duty cycle with contrast and NILS calculations\.</p>\s*</div>',
        '<a href="duv-swing-curves-demo.html" class="demo-card">\n                <i class="fas fa-chart-line demo-icon"></i>\n                <h3 class="demo-title">Duty Cycle Analysis</h3>\n                <p class="demo-description">Swing curves vs duty cycle with contrast and NILS calculations.</p>\n                <span class="demo-badge">Interactive Demo</span>\n            </a>',
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
    
    with open('duv-energy-deposition-project.html', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("Fixing Cross-Fab links...")
    fix_cross_fab_links()
    print("Fixing DUV links...")
    fix_duv_links()
    print("Done!")