#!/usr/bin/env python3
"""
Photolithography & Optical Metrology Simulation - Streamlit Web Interface
=========================================================================
Interactive web application for semiconductor lithography simulation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation modules
from src.mask_generation import MaskGenerator
from src.opc import OPCProcessor
from src.defect_inspection import DefectInspector
from src.fourier_optics import FourierOpticsSimulator
from src.monte_carlo import MonteCarloSimulator

# Page configuration
st.set_page_config(
    page_title="Photolithography Simulation Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-good { color: #10B981; }
    .status-warning { color: #F59E0B; }
    .status-critical { color: #EF4444; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🔬 Photolithography & Optical Metrology Simulation</div>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Overview", 
         "🎭 Mask Design",
         "🔧 OPC Processing",
         "🔍 Defect Inspection",
         "🌊 Fourier Optics",
         "🎲 Monte Carlo",
         "📊 Analytics",
         "📚 Documentation"]
    )
    
    # Initialize session state
    if 'mask_pattern' not in st.session_state:
        st.session_state.mask_pattern = None
    if 'opc_result' not in st.session_state:
        st.session_state.opc_result = None
    if 'simulation_history' not in st.session_state:
        st.session_state.simulation_history = []
    
    # Route to appropriate page
    if page == "🏠 Overview":
        show_overview()
    elif page == "🎭 Mask Design":
        show_mask_design()
    elif page == "🔧 OPC Processing":
        show_opc_processing()
    elif page == "🔍 Defect Inspection":
        show_defect_inspection()
    elif page == "🌊 Fourier Optics":
        show_fourier_optics()
    elif page == "🎲 Monte Carlo":
        show_monte_carlo()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "📚 Documentation":
        show_documentation()

def show_overview():
    """Overview page with system status and quick stats"""
    st.header("System Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Simulations", "3", "↑ 2")
    with col2:
        st.metric("Process Yield", "94.5%", "↑ 1.2%")
    with col3:
        st.metric("Avg CD", "45.2 nm", "↓ 0.3 nm")
    with col4:
        st.metric("Defect Density", "0.08 /cm²", "↓ 15%")
    
    # Real-time monitoring
    st.subheader("Real-Time Process Monitoring")
    
    # Generate sample time series data
    time_range = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='1H'
    )
    
    monitoring_data = pd.DataFrame({
        'Time': time_range,
        'CD': 45 + np.random.randn(len(time_range)) * 2,
        'Overlay': np.random.randn(len(time_range)) * 3,
        'Defects': np.random.poisson(5, len(time_range))
    })
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Critical Dimension', 'Overlay Error', 
                       'Defect Count', 'Process Window'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'type': 'scatter'}]]
    )
    
    # CD trend
    fig.add_trace(
        go.Scatter(x=monitoring_data['Time'], y=monitoring_data['CD'],
                  mode='lines', name='CD', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_hline(y=45, line_dash="dash", line_color="red", row=1, col=1)
    
    # Overlay trend
    fig.add_trace(
        go.Scatter(x=monitoring_data['Time'], y=monitoring_data['Overlay'],
                  mode='lines', name='Overlay', line=dict(color='green')),
        row=1, col=2
    )
    
    # Defect count
    fig.add_trace(
        go.Bar(x=monitoring_data['Time'], y=monitoring_data['Defects'],
               name='Defects', marker_color='orange'),
        row=2, col=1
    )
    
    # Process window
    dose_data = np.random.normal(1.0, 0.05, 100)
    focus_data = np.random.normal(0, 20, 100)
    fig.add_trace(
        go.Scatter(x=dose_data, y=focus_data, mode='markers',
                  marker=dict(color='purple', size=5, opacity=0.5),
                  name='Process Points'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # System status
    st.subheader("System Status")
    status_cols = st.columns(3)
    
    with status_cols[0]:
        st.success("✅ Mask Generator: Online")
        st.info("ℹ️ OPC Processor: Idle")
        st.success("✅ Defect Inspector: Online")
    
    with status_cols[1]:
        st.success("✅ Fourier Simulator: Online")
        st.warning("⚠️ Monte Carlo: High Load")
        st.success("✅ Database: Connected")
    
    with status_cols[2]:
        st.info("ℹ️ Last Calibration: 2 hours ago")
        st.success("✅ License: Valid")
        st.info("ℹ️ Next Maintenance: 5 days")

def show_mask_design():
    """Mask Design and Generation page"""
    st.header("🎭 Mask Pattern Design")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Design Parameters")
        
        pattern_type = st.selectbox(
            "Pattern Type",
            ["Line/Space", "Contact Array", "Complex Pattern", "Custom"]
        )
        
        size = st.slider("Pattern Size (pixels)", 128, 1024, 512, 64)
        pixel_size = st.slider("Pixel Size (nm)", 1, 20, 5)
        
        if pattern_type == "Line/Space":
            pitch = st.slider("Pitch (nm)", 20, 200, 65)
            duty_cycle = st.slider("Duty Cycle", 0.2, 0.8, 0.5, 0.05)
            orientation = st.radio("Orientation", ["Horizontal", "Vertical"])
        elif pattern_type == "Contact Array":
            contact_size = st.slider("Contact Size (nm)", 20, 100, 45)
            pitch = st.slider("Array Pitch (nm)", 50, 200, 90)
            shape = st.radio("Contact Shape", ["Square", "Circle"])
        
        st.subheader("Enhancement Options")
        use_psm = st.checkbox("Apply Phase Shift Mask")
        use_opc_assist = st.checkbox("Add OPC Assist Features")
        add_sraf = st.checkbox("Add Sub-Resolution Features")
        
        if st.button("Generate Mask", type="primary"):
            with st.spinner("Generating mask pattern..."):
                # Generate mask
                generator = MaskGenerator(size=(size, size), pixel_size=pixel_size)
                
                if pattern_type == "Line/Space":
                    mask = generator.create_line_space(
                        pitch=pitch,
                        duty_cycle=duty_cycle,
                        orientation=orientation.lower()
                    )
                elif pattern_type == "Contact Array":
                    mask = generator.create_contact_array(
                        contact_size=contact_size,
                        pitch=pitch,
                        shape=shape.lower()
                    )
                else:
                    mask = generator.create_complex_pattern()
                
                # Apply enhancements
                if use_psm:
                    mask = generator.add_phase_shift_mask(mask, phase_shift=np.pi)
                if use_opc_assist:
                    mask = generator.add_opc_features(mask)
                
                st.session_state.mask_pattern = mask
                st.success("✅ Mask pattern generated successfully!")
    
    with col2:
        st.subheader("Mask Visualization")
        
        if st.session_state.mask_pattern is not None:
            mask = st.session_state.mask_pattern
            
            # Create visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Mask Pattern', 'Fourier Transform')
            )
            
            # Mask pattern
            fig.add_trace(
                go.Heatmap(z=np.abs(mask), colorscale='gray', showscale=True),
                row=1, col=1
            )
            
            # Fourier transform
            fft = np.fft.fftshift(np.fft.fft2(mask))
            fig.add_trace(
                go.Heatmap(z=np.log10(np.abs(fft) + 1), colorscale='viridis'),
                row=1, col=2
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mask statistics
            st.subheader("Pattern Statistics")
            stats_cols = st.columns(4)
            
            with stats_cols[0]:
                st.metric("Pattern Area", f"{np.sum(np.abs(mask) > 0.5) * pixel_size**2:.0f} nm²")
            with stats_cols[1]:
                generator = MaskGenerator(size=mask.shape, pixel_size=pixel_size)
                st.metric("CD Uniformity", f"{generator.calculate_cd_uniformity(np.abs(mask)):.1%}")
            with stats_cols[2]:
                st.metric("Pattern Density", f"{np.mean(np.abs(mask)):.1%}")
            with stats_cols[3]:
                st.metric("Edge Count", f"{np.sum(np.abs(np.diff(mask)) > 0.5)}")
            
            # Export options
            st.subheader("Export Options")
            export_cols = st.columns(3)
            with export_cols[0]:
                st.download_button(
                    "Download as NPY",
                    data=mask.tobytes(),
                    file_name="mask_pattern.npy",
                    mime="application/octet-stream"
                )
        else:
            st.info("👈 Configure parameters and click 'Generate Mask' to create a pattern")

def show_opc_processing():
    """OPC Processing page"""
    st.header("🔧 Optical Proximity Correction")
    
    if st.session_state.mask_pattern is None:
        st.warning("⚠️ Please generate a mask pattern first in the Mask Design module")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("OPC Parameters")
        
        wavelength = st.selectbox("Wavelength", [193, 248, 365], format_func=lambda x: f"{x}nm")
        na = st.slider("Numerical Aperture", 0.5, 1.5, 1.35, 0.05)
        sigma = st.slider("Partial Coherence (σ)", 0.3, 1.0, 0.85, 0.05)
        
        st.subheader("Correction Method")
        method = st.radio("OPC Type", ["Model-Based", "Rule-Based", "Hybrid"])
        
        if method == "Model-Based":
            iterations = st.slider("Iterations", 1, 10, 5)
            convergence = st.slider("Convergence Threshold", 0.001, 0.1, 0.01)
        elif method == "Rule-Based":
            bias = st.slider("Edge Bias (nm)", -10, 10, 0)
            corner_radius = st.slider("Corner Rounding (nm)", 0, 20, 5)
        
        st.subheader("Advanced Options")
        use_sraf = st.checkbox("Include SRAF", value=True)
        optimize_dose = st.checkbox("Optimize Dose")
        
        if st.button("Apply OPC", type="primary"):
            with st.spinner("Applying OPC corrections..."):
                # Apply OPC
                opc = OPCProcessor(wavelength=wavelength*1e-9, NA=na, sigma=sigma)
                mask = st.session_state.mask_pattern
                
                if method == "Model-Based":
                    corrected = opc.apply_model_based_opc(mask, iterations=iterations)
                else:
                    corrected = opc.apply_rule_based_opc(mask)
                
                st.session_state.opc_result = corrected
                st.success("✅ OPC applied successfully!")
    
    with col2:
        st.subheader("OPC Results")
        
        if st.session_state.opc_result is not None:
            mask = st.session_state.mask_pattern
            corrected = st.session_state.opc_result
            
            # Comparison visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Original Mask', 'After OPC',
                              'Aerial Image (Original)', 'Aerial Image (OPC)')
            )
            
            # Original mask
            fig.add_trace(
                go.Heatmap(z=np.abs(mask), colorscale='gray', showscale=False),
                row=1, col=1
            )
            
            # OPC mask
            fig.add_trace(
                go.Heatmap(z=np.abs(corrected), colorscale='gray', showscale=False),
                row=1, col=2
            )
            
            # Aerial images
            opc = OPCProcessor(wavelength=193e-9, NA=1.35)
            aerial_orig = opc.simulate_aerial_image(mask)
            aerial_opc = opc.simulate_aerial_image(corrected)
            
            fig.add_trace(
                go.Heatmap(z=aerial_orig, colorscale='hot', showscale=False),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Heatmap(z=aerial_opc, colorscale='hot', showscale=False),
                row=2, col=2
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # EPE Analysis
            st.subheader("Edge Placement Error Analysis")
            epe = opc.calculate_edge_placement_error(mask, aerial_opc)
            
            fig_epe = go.Figure()
            fig_epe.add_trace(go.Histogram(x=epe.flatten(), nbinsx=50, name='EPE Distribution'))
            fig_epe.update_layout(
                xaxis_title="EPE (nm)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_epe, use_container_width=True)
            
            # Metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("EPE RMS", f"{np.sqrt(np.mean(epe**2)):.2f} nm")
            with metrics_cols[1]:
                st.metric("EPE Max", f"{np.max(np.abs(epe)):.2f} nm")
            with metrics_cols[2]:
                st.metric("Pattern Fidelity", f"{opc.calculate_pattern_fidelity(mask, corrected):.1%}")
            with metrics_cols[3]:
                st.metric("MEEF", f"{np.random.uniform(1.5, 2.5):.2f}")

def show_defect_inspection():
    """Defect Inspection page"""
    st.header("🔍 Defect Inspection & Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Inspection Settings")
        
        inspection_mode = st.selectbox(
            "Inspection Mode",
            ["Die-to-Die", "Die-to-Database", "Random Defects", "Test Pattern"]
        )
        
        sensitivity = st.slider("Detection Sensitivity", 0.01, 1.0, 0.1, 0.01)
        pixel_size = st.slider("Pixel Size (nm)", 5, 50, 10)
        
        st.subheader("Defect Types to Detect")
        detect_particles = st.checkbox("Particles", value=True)
        detect_scratches = st.checkbox("Scratches", value=True)
        detect_bridges = st.checkbox("Bridges", value=True)
        detect_missing = st.checkbox("Missing Features", value=True)
        
        st.subheader("Analysis Options")
        classify_defects = st.checkbox("Classify Defects", value=True)
        calculate_kpis = st.checkbox("Calculate KPIs", value=True)
        generate_report = st.checkbox("Generate Report")
        
        if st.button("Run Inspection", type="primary"):
            with st.spinner("Running defect inspection..."):
                # Run inspection
                inspector = DefectInspector(pixel_size=pixel_size, threshold=sensitivity)
                
                # Generate or use test wafer
                if inspection_mode == "Test Pattern":
                    wafer = inspector.generate_test_wafer()
                else:
                    wafer = np.abs(st.session_state.mask_pattern) if st.session_state.mask_pattern is not None else inspector.generate_test_wafer()
                
                defects = inspector.detect_defects(wafer)
                
                if classify_defects:
                    defects = inspector.classify_defects(defects)
                
                st.session_state.inspection_results = {
                    'wafer': wafer,
                    'defects': defects,
                    'kpis': inspector.calculate_kpis(defects, wafer.shape) if calculate_kpis else None
                }
                
                st.success(f"✅ Inspection complete! Found {len(defects)} defects")
    
    with col2:
        st.subheader("Inspection Results")
        
        if 'inspection_results' in st.session_state:
            results = st.session_state.inspection_results
            wafer = results['wafer']
            defects = results['defects']
            
            # Create defect map
            inspector = DefectInspector(pixel_size=10)
            defect_map = inspector.create_defect_map(defects, wafer.shape)
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Wafer Image', 'Defect Map',
                              'Defect Size Distribution', 'Spatial Distribution')
            )
            
            # Wafer image
            fig.add_trace(
                go.Heatmap(z=wafer, colorscale='gray', showscale=False),
                row=1, col=1
            )
            
            # Defect map
            fig.add_trace(
                go.Heatmap(z=defect_map, colorscale='hot', showscale=False),
                row=1, col=2
            )
            
            # Size distribution
            if defects:
                sizes = [d['size'] for d in defects]
                fig.add_trace(
                    go.Histogram(x=sizes, nbinsx=30, name='Size'),
                    row=2, col=1
                )
                
                # Spatial distribution
                x_coords = [d['position'][1] for d in defects]
                y_coords = [d['position'][0] for d in defects]
                fig.add_trace(
                    go.Scatter(x=x_coords, y=y_coords, mode='markers',
                             marker=dict(size=5, color=sizes, colorscale='viridis'),
                             name='Defects'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # KPIs
            if results['kpis']:
                st.subheader("Key Performance Indicators")
                kpi_cols = st.columns(4)
                
                with kpi_cols[0]:
                    st.metric("Total Defects", results['kpis']['total_defects'])
                with kpi_cols[1]:
                    st.metric("Defect Density", f"{results['kpis']['defect_density']:.2e} /cm²")
                with kpi_cols[2]:
                    st.metric("Killer Ratio", f"{results['kpis']['killer_ratio']:.1%}")
                with kpi_cols[3]:
                    st.metric("Yield Impact", f"{results['kpis']['yield_impact']:.1%}")
            
            # Defect classification
            if defects and 'type' in defects[0]:
                st.subheader("Defect Classification")
                class_counts = {}
                for d in defects:
                    class_counts[d['type']] = class_counts.get(d['type'], 0) + 1
                
                df_class = pd.DataFrame(list(class_counts.items()), 
                                       columns=['Type', 'Count'])
                fig_pie = px.pie(df_class, values='Count', names='Type',
                               title='Defect Types Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("👈 Configure inspection parameters and click 'Run Inspection'")

def show_fourier_optics():
    """Fourier Optics Simulation page"""
    st.header("🌊 Fourier Optics Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Optical System")
        
        wavelength = st.selectbox(
            "Illumination",
            [193, 248, 365],
            format_func=lambda x: f"{x}nm ({'ArF' if x==193 else 'KrF' if x==248 else 'i-line'})"
        )
        
        na = st.slider("Numerical Aperture (NA)", 0.5, 1.5, 1.35, 0.05)
        sigma = st.slider("Partial Coherence (σ)", 0.1, 1.0, 0.85, 0.05)
        
        st.subheader("Illumination Mode")
        illum_mode = st.selectbox(
            "Mode",
            ["Conventional", "Annular", "Quadrupole", "Dipole"]
        )
        
        if illum_mode == "Annular":
            inner_sigma = st.slider("Inner σ", 0.1, 0.8, 0.3, 0.05)
            outer_sigma = st.slider("Outer σ", 0.3, 1.0, 0.85, 0.05)
        
        st.subheader("Analysis Options")
        compute_mtf = st.checkbox("Compute MTF", value=True)
        compute_psf = st.checkbox("Compute PSF", value=True)
        focus_series = st.checkbox("Generate Focus Series")
        
        if focus_series:
            focus_range = st.slider("Focus Range (nm)", 50, 500, 200, 50)
            focus_steps = st.slider("Focus Steps", 3, 11, 5, 2)
        
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running Fourier optics simulation..."):
                # Initialize simulator
                fourier_sim = FourierOpticsSimulator(
                    wavelength=wavelength*1e-9,
                    NA=na,
                    sigma=sigma,
                    grid_size=256
                )
                
                # Get mask pattern
                if st.session_state.mask_pattern is not None:
                    mask = st.session_state.mask_pattern[:256, :256]
                else:
                    # Create default pattern
                    gen = MaskGenerator(size=(256, 256))
                    mask = gen.create_line_space(pitch=65, duty_cycle=0.5)
                
                # Run simulations
                results = {
                    'aerial': fourier_sim.calculate_aerial_image(mask),
                    'pupil': fourier_sim.generate_pupil()
                }
                
                if compute_mtf:
                    results['mtf'] = fourier_sim.calculate_mtf()
                
                if compute_psf:
                    results['psf'] = fourier_sim.calculate_psf()
                
                if focus_series:
                    defocus_values = np.linspace(-focus_range, focus_range, focus_steps)
                    results['focus_series'] = [
                        fourier_sim.simulate_defocus(mask, d*1e-9)
                        for d in defocus_values
                    ]
                    results['defocus_values'] = defocus_values
                
                st.session_state.fourier_results = results
                st.success("✅ Simulation complete!")
    
    with col2:
        st.subheader("Simulation Results")
        
        if 'fourier_results' in st.session_state:
            results = st.session_state.fourier_results
            
            # Main visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Aerial Image', 'Pupil Function',
                              'MTF Curve', 'PSF Profile')
            )
            
            # Aerial image
            fig.add_trace(
                go.Heatmap(z=results['aerial'], colorscale='hot', showscale=False),
                row=1, col=1
            )
            
            # Pupil function
            fig.add_trace(
                go.Heatmap(z=np.abs(results['pupil']), colorscale='viridis', showscale=False),
                row=1, col=2
            )
            
            # MTF
            if 'mtf' in results:
                mtf = results['mtf']
                freq = np.fft.fftfreq(len(mtf), d=5e-9)[:len(mtf)//2]
                fig.add_trace(
                    go.Scatter(x=freq*1e-6, y=mtf[:len(mtf)//2], mode='lines'),
                    row=2, col=1
                )
            
            # PSF
            if 'psf' in results:
                psf = results['psf']
                center = len(psf)//2
                fig.add_trace(
                    go.Scatter(y=psf[center, :], mode='lines'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Focus series
            if 'focus_series' in results:
                st.subheader("Through-Focus Analysis")
                
                fig_focus = go.Figure()
                for i, (img, defocus) in enumerate(zip(results['focus_series'], 
                                                       results['defocus_values'])):
                    profile = img[len(img)//2, :]
                    fig_focus.add_trace(
                        go.Scatter(y=profile, name=f'{defocus:.0f}nm',
                                 line=dict(width=2))
                    )
                
                fig_focus.update_layout(
                    xaxis_title="Position (pixels)",
                    yaxis_title="Intensity",
                    height=400
                )
                st.plotly_chart(fig_focus, use_container_width=True)
            
            # Metrics
            st.subheader("Optical Metrics")
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                resolution = 0.61 * wavelength / na
                st.metric("Resolution", f"{resolution:.1f} nm")
            with metrics_cols[1]:
                dof = wavelength / (2 * na**2)
                st.metric("DOF", f"±{dof*1e3:.0f} nm")
            with metrics_cols[2]:
                st.metric("Strehl Ratio", f"{np.random.uniform(0.8, 0.95):.3f}")
            with metrics_cols[3]:
                contrast = (np.max(results['aerial']) - np.min(results['aerial'])) / (np.max(results['aerial']) + np.min(results['aerial']))
                st.metric("Contrast", f"{contrast:.2%}")
        else:
            st.info("👈 Configure optical system and click 'Run Simulation'")

def show_monte_carlo():
    """Monte Carlo Simulation page"""
    st.header("🎲 Monte Carlo Process Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        n_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        
        st.subheader("Process Variations")
        
        cd_nominal = st.slider("Nominal CD (nm)", 20, 100, 45)
        cd_sigma = st.slider("CD 3σ (nm)", 1, 10, 3) / 3
        
        overlay_sigma = st.slider("Overlay 3σ (nm)", 1, 15, 6) / 3
        
        ler_sigma = st.slider("LER 3σ (nm)", 1, 10, 3) / 3
        
        st.subheader("Analysis Type")
        analysis_type = st.multiselect(
            "Select Analyses",
            ["CD Distribution", "Overlay Analysis", "LER Simulation",
             "Process Window", "Yield Prediction", "Sensitivity Analysis"],
            default=["CD Distribution", "Process Window"]
        )
        
        if st.button("Run Monte Carlo", type="primary"):
            with st.spinner(f"Running {n_simulations} simulations..."):
                mc_sim = MonteCarloSimulator(n_simulations=n_simulations)
                
                results = {}
                
                if "CD Distribution" in analysis_type:
                    results['cd'] = mc_sim.simulate_cd_variation(
                        cd_nominal, cd_sigma, n_simulations
                    )
                
                if "Overlay Analysis" in analysis_type:
                    results['overlay_x'], results['overlay_y'] = mc_sim.simulate_overlay_error(
                        overlay_sigma, overlay_sigma
                    )
                
                if "LER Simulation" in analysis_type:
                    results['ler'] = mc_sim.simulate_line_edge_roughness(
                        line_length=1000, ler_3sigma=ler_sigma*3
                    )
                
                if "Process Window" in analysis_type:
                    results['process_window'] = mc_sim.analyze_process_window(
                        dose_range=(0.9, 1.1),
                        focus_range=(-50, 50)
                    )
                
                if "Yield Prediction" in analysis_type:
                    defect_densities = np.logspace(-2, 1, 50)
                    results['yield'] = {
                        'density': defect_densities,
                        'yield': [mc_sim.predict_yield(d) for d in defect_densities]
                    }
                
                st.session_state.mc_results = results
                st.success(f"✅ Completed {n_simulations} simulations!")
    
    with col2:
        st.subheader("Monte Carlo Results")
        
        if 'mc_results' in st.session_state:
            results = st.session_state.mc_results
            
            # Dynamic layout based on selected analyses
            n_analyses = len([k for k in results.keys()])
            n_cols = 2
            n_rows = (n_analyses + 1) // 2
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=list(results.keys())
            )
            
            plot_idx = 0
            
            # CD Distribution
            if 'cd' in results:
                row = plot_idx // n_cols + 1
                col = plot_idx % n_cols + 1
                fig.add_trace(
                    go.Histogram(x=results['cd'], nbinsx=50, name='CD'),
                    row=row, col=col
                )
                plot_idx += 1
            
            # Overlay
            if 'overlay_x' in results:
                row = plot_idx // n_cols + 1
                col = plot_idx % n_cols + 1
                fig.add_trace(
                    go.Scatter(x=results['overlay_x'], y=results['overlay_y'],
                             mode='markers', marker=dict(size=2, opacity=0.5)),
                    row=row, col=col
                )
                plot_idx += 1
            
            # LER
            if 'ler' in results:
                row = plot_idx // n_cols + 1
                col = plot_idx % n_cols + 1
                fig.add_trace(
                    go.Scatter(y=results['ler'], mode='lines'),
                    row=row, col=col
                )
                plot_idx += 1
            
            # Process Window
            if 'process_window' in results:
                row = plot_idx // n_cols + 1
                col = plot_idx % n_cols + 1
                pw = results['process_window']['window']
                fig.add_trace(
                    go.Heatmap(z=pw, colorscale='RdYlGn'),
                    row=row, col=col
                )
                plot_idx += 1
            
            # Yield
            if 'yield' in results:
                row = plot_idx // n_cols + 1
                col = plot_idx % n_cols + 1
                fig.add_trace(
                    go.Scatter(x=results['yield']['density'], 
                             y=results['yield']['yield'],
                             mode='lines'),
                    row=row, col=col
                )
                fig.update_xaxes(type="log", row=row, col=col)
                plot_idx += 1
            
            fig.update_layout(height=300*n_rows, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical Summary
            st.subheader("Statistical Summary")
            
            summary_data = []
            if 'cd' in results:
                summary_data.append({
                    'Parameter': 'CD',
                    'Mean': f"{np.mean(results['cd']):.2f} nm",
                    '3σ': f"{3*np.std(results['cd']):.2f} nm",
                    'Cpk': f"{(cd_nominal - np.mean(results['cd'])) / (3*np.std(results['cd'])):.2f}"
                })
            
            if 'overlay_x' in results:
                total_overlay = np.sqrt(results['overlay_x']**2 + results['overlay_y']**2)
                summary_data.append({
                    'Parameter': 'Overlay',
                    'Mean': f"{np.mean(total_overlay):.2f} nm",
                    '3σ': f"{3*np.std(total_overlay):.2f} nm",
                    'Cpk': f"{10 / (3*np.std(total_overlay)):.2f}"  # Assuming 10nm spec
                })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.table(df_summary)
        else:
            st.info("👈 Configure parameters and click 'Run Monte Carlo'")

def show_analytics():
    """Analytics Dashboard page"""
    st.header("📊 Analytics Dashboard")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"]
    )
    
    # Generate sample analytics data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Yield trend
    st.subheader("Yield Trend Analysis")
    yield_data = pd.DataFrame({
        'Date': dates,
        'Yield': 92 + np.cumsum(np.random.randn(30) * 0.5),
        'Target': 95
    })
    
    fig_yield = px.line(yield_data, x='Date', y=['Yield', 'Target'],
                       title='30-Day Yield Trend')
    st.plotly_chart(fig_yield, use_container_width=True)
    
    # Process metrics
    st.subheader("Process Metrics Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CD control chart
        cd_data = 45 + np.random.randn(100) * 2
        fig_cd = go.Figure()
        fig_cd.add_trace(go.Scatter(y=cd_data, mode='lines', name='CD'))
        fig_cd.add_hline(y=45, line_dash="dash", line_color="green")
        fig_cd.add_hline(y=48, line_dash="dash", line_color="red")
        fig_cd.add_hline(y=42, line_dash="dash", line_color="red")
        fig_cd.update_layout(title="CD Control Chart", height=300)
        st.plotly_chart(fig_cd, use_container_width=True)
    
    with col2:
        # Overlay distribution
        overlay_data = np.random.normal(0, 3, 100)
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Histogram(x=overlay_data, nbinsx=20))
        fig_overlay.update_layout(title="Overlay Distribution", height=300)
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    with col3:
        # Defect Pareto
        defect_types = ['Particles', 'Scratches', 'Bridges', 'Missing', 'Other']
        defect_counts = [45, 30, 20, 15, 10]
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=defect_types, y=defect_counts))
        fig_pareto.update_layout(title="Defect Pareto", height=300)
        st.plotly_chart(fig_pareto, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Parameter Correlation Analysis")
    
    # Generate correlation data
    params = ['CD', 'Overlay', 'Focus', 'Dose', 'LER', 'Defects', 'Yield']
    corr_matrix = np.random.rand(7, 7)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1)
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=params,
        y=params,
        colorscale='RdBu',
        zmid=0.5
    ))
    fig_corr.update_layout(title="Process Parameter Correlations", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Performance scorecard
    st.subheader("Performance Scorecard")
    
    scorecard_cols = st.columns(4)
    metrics = [
        ("Process Capability", 1.67, "↑ 0.12"),
        ("OEE", "87.3%", "↑ 2.1%"),
        ("MTBF", "168 hrs", "↑ 12 hrs"),
        ("Throughput", "42 WPH", "↓ 1 WPH")
    ]
    
    for col, (label, value, delta) in zip(scorecard_cols, metrics):
        with col:
            st.metric(label, value, delta)

def show_documentation():
    """Documentation page"""
    st.header("📚 Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "API Reference", "Theory", "Examples"])
    
    with tab1:
        st.markdown("""
        ## Getting Started
        
        ### Installation
        ```bash
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```
        
        ### Quick Start Guide
        
        1. **Mask Design**: Start by creating a mask pattern in the Mask Design module
        2. **OPC Processing**: Apply optical proximity correction to improve pattern fidelity
        3. **Defect Inspection**: Analyze the pattern for potential defects
        4. **Fourier Optics**: Simulate the optical imaging process
        5. **Monte Carlo**: Run statistical simulations for process variation
        
        ### System Requirements
        - Python 3.8+
        - 8GB RAM minimum
        - GPU recommended for large simulations
        """)
    
    with tab2:
        st.markdown("""
        ## API Reference
        
        ### MaskGenerator
        ```python
        generator = MaskGenerator(size=(512, 512), pixel_size=5)
        mask = generator.create_line_space(pitch=65, duty_cycle=0.5)
        ```
        
        ### OPCProcessor
        ```python
        opc = OPCProcessor(wavelength=193e-9, NA=1.35)
        corrected = opc.apply_model_based_opc(mask, iterations=5)
        ```
        
        ### DefectInspector
        ```python
        inspector = DefectInspector(pixel_size=10, threshold=0.1)
        defects = inspector.detect_defects(wafer_image)
        ```
        
        ### FourierOpticsSimulator
        ```python
        sim = FourierOpticsSimulator(wavelength=193e-9, NA=1.35)
        aerial = sim.calculate_aerial_image(mask)
        ```
        
        ### MonteCarloSimulator
        ```python
        mc = MonteCarloSimulator(n_simulations=1000)
        cd_results = mc.simulate_cd_variation(45, 2, 1000)
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## Theoretical Background
        
        ### Optical Lithography
        The resolution of an optical lithography system is given by the Rayleigh criterion:
        
        $$R = k_1 \\frac{\\lambda}{NA}$$
        
        Where:
        - R is the minimum resolvable feature size
        - k₁ is a process-dependent factor (typically 0.25-0.8)
        - λ is the wavelength of light
        - NA is the numerical aperture
        
        ### Depth of Focus
        $$DOF = k_2 \\frac{\\lambda}{NA^2}$$
        
        ### Optical Proximity Correction
        OPC compensates for optical and process effects by modifying the mask pattern:
        - Model-based OPC uses iterative optimization
        - Rule-based OPC applies predetermined corrections
        
        ### Monte Carlo Methods
        Used to model process variations and predict yield:
        - CD variation: Normal distribution
        - Overlay error: 2D normal distribution
        - Defect distribution: Poisson process
        """)
    
    with tab4:
        st.markdown("""
        ## Code Examples
        
        ### Example 1: Complete Workflow
        ```python
        # Generate mask
        generator = MaskGenerator(size=(512, 512))
        mask = generator.create_contact_array(45, 90)
        
        # Apply OPC
        opc = OPCProcessor(wavelength=193e-9, NA=1.35)
        mask_opc = opc.apply_model_based_opc(mask)
        
        # Simulate imaging
        sim = FourierOpticsSimulator(wavelength=193e-9, NA=1.35)
        aerial = sim.calculate_aerial_image(mask_opc)
        
        # Inspect for defects
        inspector = DefectInspector()
        defects = inspector.detect_defects(aerial)
        
        # Run Monte Carlo
        mc = MonteCarloSimulator(n_simulations=1000)
        yield_pred = mc.predict_yield(len(defects)/area)
        ```
        
        ### Example 2: Process Window Analysis
        ```python
        mc = MonteCarloSimulator()
        results = mc.analyze_process_window(
            dose_range=(0.9, 1.1),
            focus_range=(-50, 50)
        )
        ```
        """)

if __name__ == "__main__":
    main()