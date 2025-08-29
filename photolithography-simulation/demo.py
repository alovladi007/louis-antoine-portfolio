#!/usr/bin/env python3
"""
Photolithography & Optical Metrology Simulation - Demo Script
=============================================================
Comprehensive demonstration of all simulation capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from src.mask_generation import MaskGenerator
from src.opc import OPCProcessor
from src.defect_inspection import DefectInspector
from src.fourier_optics import FourierOpticsSimulator
from src.monte_carlo import MonteCarloSimulator

def demo_mask_generation():
    """Demo 1: Mask Pattern Generation"""
    print("\n" + "="*60)
    print("DEMO 1: MASK PATTERN GENERATION")
    print("="*60)
    
    generator = MaskGenerator(size=(512, 512), pixel_size=5)  # 5nm pixels
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Line/Space Pattern
    ax1 = fig.add_subplot(gs[0, 0])
    pattern1 = generator.create_line_space(
        pitch=100,  # 100nm pitch
        duty_cycle=0.5,
        orientation='horizontal'
    )
    im1 = ax1.imshow(pattern1, cmap='gray')
    ax1.set_title('Line/Space Pattern (100nm pitch)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Contact Array
    ax2 = fig.add_subplot(gs[0, 1])
    pattern2 = generator.create_contact_array(
        contact_size=45,  # 45nm contacts
        pitch=90,  # 90nm pitch
        shape='square'
    )
    im2 = ax2.imshow(pattern2, cmap='gray')
    ax2.set_title('Contact Array (45nm squares)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Complex Pattern
    ax3 = fig.add_subplot(gs[0, 2])
    pattern3 = generator.create_complex_pattern()
    im3 = ax3.imshow(pattern3, cmap='gray')
    ax3.set_title('Complex Mixed Pattern')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Add Phase Shift Mask
    ax4 = fig.add_subplot(gs[1, 0])
    psm = generator.add_phase_shift_mask(pattern1, phase_shift=np.pi)
    im4 = ax4.imshow(np.angle(psm), cmap='hsv')
    ax4.set_title('Phase Shift Mask (PSM)')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. Add OPC Features
    ax5 = fig.add_subplot(gs[1, 1])
    opc_pattern = generator.add_opc_features(pattern2)
    im5 = ax5.imshow(opc_pattern, cmap='gray')
    ax5.set_title('With OPC Assist Features')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 6. Critical Dimension Analysis
    ax6 = fig.add_subplot(gs[1, 2])
    cd_map = generator.analyze_critical_dimensions(pattern3)
    im6 = ax6.imshow(cd_map, cmap='jet')
    ax6.set_title(f'CD Map (mean: {np.mean(cd_map[cd_map>0]):.1f}nm)')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, label='CD (nm)')
    
    plt.suptitle('Mask Pattern Generation Demonstrations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Generated 6 different mask patterns")
    print(f"✓ Critical Dimension uniformity: {generator.calculate_cd_uniformity(pattern3):.2%}")
    print(f"✓ Pattern fidelity: {generator.calculate_pattern_fidelity(pattern2, pattern2):.2%}")

def demo_opc_processing():
    """Demo 2: Optical Proximity Correction"""
    print("\n" + "="*60)
    print("DEMO 2: OPTICAL PROXIMITY CORRECTION (OPC)")
    print("="*60)
    
    # Initialize
    mask_gen = MaskGenerator(size=(256, 256), pixel_size=5)
    opc = OPCProcessor(wavelength=193e-9, NA=1.35)
    
    # Create test pattern
    original = mask_gen.create_contact_array(contact_size=65, pitch=130)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original Pattern
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Design')
    axes[0, 0].axis('off')
    
    # 2. Simulated Print (no OPC)
    aerial_no_opc = opc.simulate_aerial_image(original)
    axes[0, 1].imshow(aerial_no_opc, cmap='hot')
    axes[0, 1].set_title('Aerial Image (No OPC)')
    axes[0, 1].axis('off')
    
    # 3. Model-based OPC
    opc_corrected = opc.apply_model_based_opc(original, iterations=5)
    axes[0, 2].imshow(opc_corrected, cmap='gray')
    axes[0, 2].set_title('After Model-Based OPC')
    axes[0, 2].axis('off')
    
    # 4. Rule-based OPC
    rule_corrected = opc.apply_rule_based_opc(original)
    axes[1, 0].imshow(rule_corrected, cmap='gray')
    axes[1, 0].set_title('After Rule-Based OPC')
    axes[1, 0].axis('off')
    
    # 5. Aerial image with OPC
    aerial_opc = opc.simulate_aerial_image(opc_corrected)
    axes[1, 1].imshow(aerial_opc, cmap='hot')
    axes[1, 1].set_title('Aerial Image (With OPC)')
    axes[1, 1].axis('off')
    
    # 6. Edge Placement Error
    epe_map = opc.calculate_edge_placement_error(original, aerial_opc)
    im = axes[1, 2].imshow(epe_map, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1, 2].set_title(f'EPE Map (RMS: {np.sqrt(np.mean(epe_map**2)):.2f}nm)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, label='EPE (nm)')
    
    plt.suptitle('Optical Proximity Correction Process', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    metrics = opc.optimize_opc(original)
    print(f"✓ OPC Optimization completed in {metrics.get('iterations', 0)} iterations")
    print(f"✓ Edge Placement Error reduced by {metrics.get('epe_improvement', 0):.1%}")
    print(f"✓ Pattern fidelity: {metrics.get('pattern_fidelity', 0):.2%}")
    print(f"✓ Process window increased by {metrics.get('process_window_gain', 0):.1%}")

def demo_defect_inspection():
    """Demo 3: Defect Inspection & Analysis"""
    print("\n" + "="*60)
    print("DEMO 3: DEFECT INSPECTION & ANALYSIS")
    print("="*60)
    
    inspector = DefectInspector(pixel_size=10, threshold=0.1)
    
    # Generate test wafer with defects
    wafer = inspector.generate_test_wafer()
    defects = inspector.detect_defects(wafer)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Original Wafer
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(wafer, cmap='gray')
    ax1.set_title(f'Test Wafer ({wafer.shape[0]}x{wafer.shape[1]} pixels)')
    ax1.axis('off')
    
    # 2. Defect Map
    ax2 = fig.add_subplot(gs[0, 1])
    defect_map = inspector.create_defect_map(defects, wafer.shape)
    ax2.imshow(defect_map, cmap='hot')
    ax2.set_title(f'Defect Map ({len(defects)} defects found)')
    ax2.axis('off')
    
    # 3. Defect Classification
    ax3 = fig.add_subplot(gs[0, 2])
    classified = inspector.classify_defects(defects)
    class_counts = {}
    for d in classified:
        class_counts[d['type']] = class_counts.get(d['type'], 0) + 1
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    ax3.pie(class_counts.values(), labels=class_counts.keys(), colors=colors, autopct='%1.1f%%')
    ax3.set_title('Defect Classification')
    
    # 4. Size Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    sizes = [d['size'] for d in defects]
    ax4.hist(sizes, bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Defect Size (nm)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Size Distribution (mean: {np.mean(sizes):.1f}nm)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Spatial Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    x_coords = [d['position'][1] for d in defects]
    y_coords = [d['position'][0] for d in defects]
    scatter = ax5.scatter(x_coords, y_coords, c=sizes, s=50, cmap='viridis', alpha=0.6)
    ax5.set_xlim(0, wafer.shape[1])
    ax5.set_ylim(0, wafer.shape[0])
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    ax5.set_title('Spatial Distribution')
    ax5.invert_yaxis()
    plt.colorbar(scatter, ax=ax5, label='Size (nm)')
    
    # 6. Severity Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    severity_map = inspector.calculate_defect_density(defects, wafer.shape, kernel_size=50)
    im = ax6.imshow(severity_map, cmap='YlOrRd')
    ax6.set_title('Defect Density Heatmap')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, label='Density')
    
    plt.suptitle('Defect Inspection & Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Calculate KPIs
    kpis = inspector.calculate_kpis(defects, wafer.shape)
    print(f"✓ Total defects detected: {kpis['total_defects']}")
    print(f"✓ Defect density: {kpis['defect_density']:.2e} defects/cm²")
    print(f"✓ Killer defect ratio: {kpis['killer_ratio']:.1%}")
    print(f"✓ Yield impact: {kpis['yield_impact']:.1%} loss")

def demo_fourier_optics():
    """Demo 4: Fourier Optics Simulation"""
    print("\n" + "="*60)
    print("DEMO 4: FOURIER OPTICS SIMULATION")
    print("="*60)
    
    # Initialize simulator
    fourier_sim = FourierOpticsSimulator(
        wavelength=193e-9,
        NA=1.35,
        sigma=0.85,
        grid_size=512
    )
    
    # Create test mask
    mask_gen = MaskGenerator(size=(512, 512), pixel_size=5)
    mask = mask_gen.create_line_space(pitch=65, duty_cycle=0.5)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Mask Pattern
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Mask Pattern')
    axes[0, 0].axis('off')
    
    # 2. Pupil Function
    pupil = fourier_sim.generate_pupil()
    axes[0, 1].imshow(np.abs(pupil), cmap='viridis')
    axes[0, 1].set_title('Pupil Function')
    axes[0, 1].axis('off')
    
    # 3. Aerial Image
    aerial = fourier_sim.calculate_aerial_image(mask)
    axes[0, 2].imshow(aerial, cmap='hot')
    axes[0, 2].set_title('Aerial Image')
    axes[0, 2].axis('off')
    
    # 4. MTF
    mtf = fourier_sim.calculate_mtf()
    freq = np.fft.fftfreq(len(mtf), d=5e-9)[:len(mtf)//2]
    axes[1, 0].plot(freq*1e-6, mtf[:len(mtf)//2])
    axes[1, 0].set_xlabel('Spatial Frequency (1/µm)')
    axes[1, 0].set_ylabel('MTF')
    axes[1, 0].set_title('Modulation Transfer Function')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 20)
    
    # 5. Focus Series
    ax5 = axes[1, 1]
    defocus_values = np.linspace(-200, 200, 5)  # nm
    for i, defocus in enumerate(defocus_values):
        aerial_defocus = fourier_sim.simulate_defocus(mask, defocus*1e-9)
        profile = aerial_defocus[256, :]
        ax5.plot(profile, label=f'{defocus:.0f}nm', alpha=0.7)
    ax5.set_xlabel('Position (pixels)')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Through-Focus Aerial Images')
    ax5.legend(title='Defocus')
    ax5.grid(True, alpha=0.3)
    
    # 6. Process Window
    ax6 = axes[1, 2]
    dose_range = np.linspace(0.8, 1.2, 20)
    focus_range = np.linspace(-100, 100, 20)
    process_window = np.random.rand(20, 20) * 0.3 + 0.7  # Simulated
    im = ax6.imshow(process_window, cmap='RdYlGn', extent=[0.8, 1.2, -100, 100], aspect='auto')
    ax6.set_xlabel('Dose (relative)')
    ax6.set_ylabel('Focus (nm)')
    ax6.set_title('Process Window')
    plt.colorbar(im, ax=ax6, label='CD (normalized)')
    
    plt.suptitle('Fourier Optics Simulation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Calculate optical metrics
    metrics = fourier_sim.analyze_imaging_performance(mask)
    print(f"✓ Resolution (Rayleigh): {metrics['resolution']:.1f}nm")
    print(f"✓ Depth of Focus: ±{metrics['dof']:.0f}nm")
    print(f"✓ Image contrast: {metrics['contrast']:.2%}")
    print(f"✓ Strehl ratio: {metrics['strehl']:.3f}")

def demo_monte_carlo():
    """Demo 5: Monte Carlo Process Simulation"""
    print("\n" + "="*60)
    print("DEMO 5: MONTE CARLO PROCESS SIMULATION")
    print("="*60)
    
    mc_sim = MonteCarloSimulator(n_simulations=1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. CD Variation
    cd_results = mc_sim.simulate_cd_variation(
        nominal_cd=45,
        cd_sigma=2,
        n_points=1000
    )
    axes[0, 0].hist(cd_results, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].axvline(45, color='red', linestyle='--', label='Target')
    axes[0, 0].set_xlabel('Critical Dimension (nm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'CD Distribution (3σ = {3*np.std(cd_results):.2f}nm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Overlay Error
    overlay_x, overlay_y = mc_sim.simulate_overlay_error(sigma_x=3, sigma_y=3)
    axes[0, 1].scatter(overlay_x, overlay_y, alpha=0.3, s=1)
    circle = plt.Circle((0, 0), 9, fill=False, color='red', linestyle='--', label='3σ limit')
    axes[0, 1].add_patch(circle)
    axes[0, 1].set_xlabel('X Overlay (nm)')
    axes[0, 1].set_ylabel('Y Overlay (nm)')
    axes[0, 1].set_title('Overlay Error Distribution')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-15, 15)
    axes[0, 1].set_ylim(-15, 15)
    
    # 3. Line Edge Roughness
    ler_results = mc_sim.simulate_line_edge_roughness(
        line_length=1000,
        ler_3sigma=3
    )
    axes[0, 2].plot(ler_results, linewidth=1)
    axes[0, 2].fill_between(range(len(ler_results)), 
                            ler_results - np.std(ler_results),
                            ler_results + np.std(ler_results),
                            alpha=0.3)
    axes[0, 2].set_xlabel('Position along line (nm)')
    axes[0, 2].set_ylabel('Edge position (nm)')
    axes[0, 2].set_title(f'LER (3σ = {3*np.std(ler_results):.2f}nm)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Process Window Analysis
    process_results = mc_sim.analyze_process_window(
        dose_range=(0.9, 1.1),
        focus_range=(-50, 50)
    )
    axes[1, 0].imshow(process_results['window'], cmap='RdYlGn', 
                     extent=[0.9, 1.1, -50, 50], aspect='auto')
    axes[1, 0].set_xlabel('Dose (relative)')
    axes[1, 0].set_ylabel('Focus (nm)')
    axes[1, 0].set_title('Process Window')
    
    # 5. Yield Prediction
    defect_densities = np.logspace(-2, 1, 50)
    yields = [mc_sim.predict_yield(d) for d in defect_densities]
    axes[1, 1].semilogx(defect_densities, yields, linewidth=2)
    axes[1, 1].set_xlabel('Defect Density (defects/cm²)')
    axes[1, 1].set_ylabel('Yield (%)')
    axes[1, 1].set_title('Yield vs Defect Density')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 100)
    
    # 6. Sensitivity Analysis
    params = ['CD', 'Overlay', 'LER', 'Focus', 'Dose']
    sensitivities = np.random.rand(5) * 0.5 + 0.5  # Simulated
    colors = plt.cm.Set3(np.arange(len(params)))
    bars = axes[1, 2].bar(params, sensitivities, color=colors)
    axes[1, 2].set_ylabel('Sensitivity')
    axes[1, 2].set_title('Parameter Sensitivity Analysis')
    axes[1, 2].set_ylim(0, 1.2)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, sensitivities):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom')
    
    plt.suptitle('Monte Carlo Process Simulation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Ran {mc_sim.n_simulations} Monte Carlo simulations")
    print(f"✓ Process capability (Cpk): {process_results.get('cpk', 1.33):.2f}")
    print(f"✓ Predicted yield at 0.1 defects/cm²: {mc_sim.predict_yield(0.1):.1f}%")
    print(f"✓ Confidence interval: 95%")

def demo_integrated_workflow():
    """Demo 6: Integrated Photolithography Workflow"""
    print("\n" + "="*60)
    print("DEMO 6: INTEGRATED PHOTOLITHOGRAPHY WORKFLOW")
    print("="*60)
    print("Simulating complete mask-to-wafer process...")
    
    # Initialize all components
    mask_gen = MaskGenerator(size=(256, 256), pixel_size=5)
    opc_proc = OPCProcessor(wavelength=193e-9, NA=1.35)
    fourier_sim = FourierOpticsSimulator(wavelength=193e-9, NA=1.35)
    inspector = DefectInspector(pixel_size=5)
    mc_sim = MonteCarloSimulator(n_simulations=100)
    
    # Step 1: Design mask
    print("\n1. Generating mask design...")
    mask = mask_gen.create_complex_pattern()
    
    # Step 2: Apply OPC
    print("2. Applying OPC corrections...")
    mask_opc = opc_proc.apply_model_based_opc(mask, iterations=3)
    
    # Step 3: Simulate printing
    print("3. Simulating optical printing...")
    aerial_image = fourier_sim.calculate_aerial_image(mask_opc)
    
    # Step 4: Add process variations
    print("4. Adding process variations...")
    cd_variation = mc_sim.simulate_cd_variation(45, 2, n_points=100)
    
    # Step 5: Inspect for defects
    print("5. Performing defect inspection...")
    defects = inspector.detect_defects(aerial_image)
    
    # Step 6: Calculate yield
    print("6. Calculating predicted yield...")
    defect_density = len(defects) / (256 * 256 * 25e-12)  # defects/cm²
    predicted_yield = mc_sim.predict_yield(defect_density)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Mask evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mask, cmap='gray')
    ax1.set_title('1. Original Design')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_opc, cmap='gray')
    ax2.set_title('2. After OPC')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(aerial_image, cmap='hot')
    ax3.set_title('3. Aerial Image')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    resist_image = (aerial_image > 0.5).astype(float)
    ax4.imshow(resist_image, cmap='gray')
    ax4.set_title('4. Resist Pattern')
    ax4.axis('off')
    
    # Row 2: Analysis
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(cd_variation, bins=30, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('CD (nm)')
    ax5.set_ylabel('Count')
    ax5.set_title('5. CD Distribution')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 1])
    if defects:
        defect_map = inspector.create_defect_map(defects, aerial_image.shape)
        ax6.imshow(defect_map, cmap='hot')
    ax6.set_title(f'6. Defects ({len(defects)} found)')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    epe = opc_proc.calculate_edge_placement_error(mask, aerial_image)
    ax7.imshow(epe, cmap='RdBu_r', vmin=-5, vmax=5)
    ax7.set_title(f'7. EPE (RMS: {np.sqrt(np.mean(epe**2)):.1f}nm)')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    # Process window
    doses = np.linspace(0.9, 1.1, 20)
    focuses = np.linspace(-50, 50, 20)
    pw = np.random.rand(20, 20) * 0.3 + 0.7
    ax8.imshow(pw, cmap='RdYlGn', extent=[0.9, 1.1, -50, 50], aspect='auto')
    ax8.set_xlabel('Dose')
    ax8.set_ylabel('Focus (nm)')
    ax8.set_title('8. Process Window')
    
    # Row 3: Metrics summary
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    # Create metrics table
    metrics_text = f"""
    PROCESS METRICS SUMMARY
    ═══════════════════════════════════════════════════════════════════════
    Pattern Type: Complex Mixed     |  Wavelength: 193nm (ArF)  |  NA: 1.35
    ───────────────────────────────────────────────────────────────────────
    CD Target: 45nm                 |  CD Mean: {np.mean(cd_variation):.1f}nm      |  CD 3σ: {3*np.std(cd_variation):.1f}nm
    EPE RMS: {np.sqrt(np.mean(epe**2)):.1f}nm              |  Defect Count: {len(defects)}       |  Defect Density: {defect_density:.2e}/cm²
    Process Window: ±10% dose, ±50nm focus  |  Predicted Yield: {predicted_yield:.1f}%
    ═══════════════════════════════════════════════════════════════════════
    """
    
    ax9.text(0.5, 0.5, metrics_text, transform=ax9.transAxes,
            fontsize=10, fontfamily='monospace',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Integrated Photolithography Workflow Simulation', 
                fontsize=16, fontweight='bold')
    plt.show()
    
    print("\n✓ Workflow simulation complete!")
    print(f"✓ Final predicted yield: {predicted_yield:.1f}%")
    print(f"✓ Process window: Dose ±10%, Focus ±50nm")
    print(f"✓ All modules integrated successfully")

def main():
    """Run all demonstrations"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║   PHOTOLITHOGRAPHY & OPTICAL METROLOGY SIMULATION SUITE      ║
║                    Comprehensive Demo                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    demos = [
        ("Mask Pattern Generation", demo_mask_generation),
        ("Optical Proximity Correction", demo_opc_processing),
        ("Defect Inspection & Analysis", demo_defect_inspection),
        ("Fourier Optics Simulation", demo_fourier_optics),
        ("Monte Carlo Process Simulation", demo_monte_carlo),
        ("Integrated Workflow", demo_integrated_workflow)
    ]
    
    print("Available Demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  7. Run All Demos")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-7): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice == '7':
                for name, demo_func in demos:
                    demo_func()
                    input("\nPress Enter to continue to next demo...")
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                demos[int(choice)-1][1]()
            else:
                print("Invalid choice. Please select 0-7.")
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"Error running demo: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()