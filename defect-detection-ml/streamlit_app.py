"""
Streamlit Web Application for Defect Detection & Yield Prediction
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.data_generator import DefectDataGenerator, SyntheticWaferGenerator, DefectType
from src.cnn_models import create_model
from src.bayesian_models import MCDropoutClassifier, RareDefectDetector

# Page config
st.set_page_config(
    page_title="Defect Detection ML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: #1e3a5f;
        color: white;
    }
    h1 {
        color: #00d4ff;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = DefectDataGenerator(wafer_size=300, die_size=(10, 10))
if 'wafer_map' not in st.session_state:
    st.session_state.wafer_map = None
if 'model' not in st.session_state:
    st.session_state.model = None

def main():
    st.title("🔬 Defect Detection & Yield Prediction ML System")
    st.markdown("### Advanced Machine Learning for Semiconductor Quality Assurance")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Module",
            ["🏠 Overview", "🎲 Data Generation", "🧠 Model Testing", 
             "📊 Analytics", "🔮 Predictions", "📚 Documentation"]
        )
    
    if page == "🏠 Overview":
        show_overview()
    elif page == "🎲 Data Generation":
        show_data_generation()
    elif page == "🧠 Model Testing":
        show_model_testing()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "🔮 Predictions":
        show_predictions()
    elif page == "📚 Documentation":
        show_documentation()

def show_overview():
    """Display project overview"""
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Core Modules", "4", "Implemented")
    with col2:
        st.metric("CNN Models", "5", "Ready")
    with col3:
        st.metric("Defect Types", "10+", "Supported")
    with col4:
        st.metric("Lines of Code", "1.5k+", "Python")
    
    st.markdown("---")
    
    # Feature highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Implemented Features")
        features = [
            "Synthetic wafer data generation",
            "Multiple CNN architectures (ResNet, EfficientNet, etc.)",
            "Bayesian uncertainty quantification",
            "Active learning strategies",
            "Rare defect detection",
            "Real-time visualization"
        ]
        for feature in features:
            st.write(f"• {feature}")
    
    with col2:
        st.subheader("🎯 Key Capabilities")
        capabilities = [
            "Generate realistic wafer defect patterns",
            "Train deep learning models for classification",
            "Quantify prediction uncertainty",
            "Detect rare/anomalous defects",
            "Select informative samples for labeling",
            "Interactive data exploration"
        ]
        for cap in capabilities:
            st.write(f"• {cap}")
    
    # Quick stats
    st.markdown("---")
    st.subheader("📊 System Architecture")
    
    architecture_text = """
    ```
    defect-detection-ml/
    ├── src/
    │   ├── data_generator.py      # Synthetic data generation
    │   ├── cnn_models.py          # CNN architectures
    │   ├── bayesian_models.py     # Bayesian ML
    │   └── yield_prediction.py    # Yield forecasting
    ├── streamlit_app.py           # This interface
    ├── demo.py                    # Comprehensive demos
    └── train.py                   # Training pipeline
    ```
    """
    st.code(architecture_text, language="text")

def show_data_generation():
    """Data generation interface"""
    st.header("🎲 Synthetic Data Generation")
    
    tab1, tab2, tab3 = st.tabs(["Wafer Maps", "Defect Images", "Dataset Builder"])
    
    with tab1:
        st.subheader("Generate Wafer Defect Maps")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            defect_type = st.selectbox(
                "Defect Type",
                ["PARTICLE", "SCRATCH", "CLUSTER", "RING", "EDGE", "PATTERN"]
            )
            
            defect_density = st.slider(
                "Defect Density",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01
            )
            
            wafer_size = st.number_input(
                "Wafer Size (mm)",
                min_value=100,
                max_value=450,
                value=300,
                step=50
            )
            
            if st.button("Generate Wafer Map", type="primary"):
                with st.spinner("Generating..."):
                    generator = DefectDataGenerator(wafer_size=wafer_size, die_size=(10, 10))
                    wafer_map = generator.generate_wafer_map(
                        defect_type=DefectType[defect_type],
                        defect_density=defect_density
                    )
                    st.session_state.wafer_map = wafer_map
        
        with col2:
            if st.session_state.wafer_map:
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(st.session_state.wafer_map.data, cmap='RdYlGn_r')
                ax.set_title(f'Wafer Map - Yield: {st.session_state.wafer_map.yield_rate:.1%}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Yield Rate", f"{st.session_state.wafer_map.yield_rate:.1%}")
                col2.metric("Defect Count", st.session_state.wafer_map.defect_count)
                col3.metric("Total Dies", st.session_state.wafer_map.metadata['wafer_info']['total_dies'])
    
    with tab2:
        st.subheader("Generate Defect Images")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image_class = st.selectbox(
                "Defect Class",
                ["particle", "scratch", "void", "bridge", "none"]
            )
            
            add_noise = st.checkbox("Add Background Noise", value=True)
            
            n_samples = st.number_input(
                "Number of Samples",
                min_value=1,
                max_value=20,
                value=5
            )
            
            if st.button("Generate Images", type="primary"):
                with st.spinner("Generating images..."):
                    generator = DefectDataGenerator()
                    images = []
                    for _ in range(n_samples):
                        img = generator.generate_defect_image(
                            size=(224, 224),
                            defect_class=image_class,
                            add_noise=add_noise
                        )
                        images.append(img)
                    st.session_state.generated_images = images
        
        with col2:
            if 'generated_images' in st.session_state:
                # Display images in grid
                n_cols = 3
                n_rows = (len(st.session_state.generated_images) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 3))
                axes = axes.ravel() if n_rows > 1 else [axes]
                
                for i, img in enumerate(st.session_state.generated_images):
                    if i < len(axes):
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(f'Sample {i+1}')
                        axes[i].axis('off')
                
                # Hide empty subplots
                for i in range(len(st.session_state.generated_images), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Build Training Dataset")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Dataset Configuration**")
            
            n_samples = st.number_input(
                "Total Samples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            train_split = st.slider(
                "Training Split",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05
            )
            
            selected_classes = st.multiselect(
                "Defect Classes",
                ["particle", "scratch", "void", "bridge", "none"],
                default=["particle", "scratch", "void", "bridge", "none"]
            )
        
        with col2:
            st.write("**Dataset Summary**")
            
            if selected_classes:
                n_train = int(n_samples * train_split)
                n_test = n_samples - n_train
                samples_per_class = n_samples // len(selected_classes)
                
                st.info(f"""
                **Configuration:**
                - Training samples: {n_train}
                - Test samples: {n_test}
                - Classes: {len(selected_classes)}
                - Samples per class: ~{samples_per_class}
                """)
                
                if st.button("Generate Dataset", type="primary"):
                    with st.spinner(f"Generating {n_samples} samples..."):
                        generator = DefectDataGenerator()
                        dataset = generator.generate_dataset(
                            n_samples=n_samples,
                            defect_classes=selected_classes,
                            train_split=train_split
                        )
                        st.success(f"✅ Dataset generated successfully!")
                        st.session_state.dataset = dataset

def show_model_testing():
    """Model testing interface"""
    st.header("🧠 Model Testing & Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["Model Selection", "Inference Demo", "Uncertainty Analysis"])
    
    with tab1:
        st.subheader("Select and Load Model")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_type = st.selectbox(
                "Model Architecture",
                ["DefectCNN", "ResNet-50", "EfficientNet", "AttentionCNN", "MultiScaleCNN"]
            )
            
            model_map = {
                "DefectCNN": "defect_cnn",
                "ResNet-50": "resnet",
                "EfficientNet": "efficientnet",
                "AttentionCNN": "attention_cnn",
                "MultiScaleCNN": "multiscale_cnn"
            }
            
            if st.button("Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    kwargs = {}
                    if model_type == "ResNet-50":
                        kwargs = {'resnet_version': 'resnet50', 'pretrained': False}
                    
                    model = create_model(
                        model_map[model_type],
                        num_classes=5,
                        **kwargs
                    )
                    st.session_state.model = model
                    st.session_state.model_type = model_type
                    st.success(f"✅ {model_type} loaded successfully!")
        
        with col2:
            if st.session_state.model:
                st.write("**Model Information**")
                
                # Count parameters
                total_params = sum(p.numel() for p in st.session_state.model.parameters())
                trainable_params = sum(p.numel() for p in st.session_state.model.parameters() 
                                     if p.requires_grad)
                
                info_df = pd.DataFrame({
                    'Metric': ['Architecture', 'Total Parameters', 'Trainable Parameters', 
                              'Input Shape', 'Output Classes'],
                    'Value': [st.session_state.model_type, f"{total_params:,}", 
                             f"{trainable_params:,}", '(B, 1, 224, 224)', '5']
                })
                
                st.dataframe(info_df, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("Run Inference")
        
        if st.session_state.model:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Generate Test Input**")
                
                if st.button("Generate Random Input"):
                    test_input = torch.randn(1, 1, 224, 224)
                    st.session_state.test_input = test_input
                    
                    # Visualize input
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(test_input[0, 0].numpy(), cmap='gray')
                    ax.set_title("Test Input Image")
                    ax.axis('off')
                    st.pyplot(fig)
            
            with col2:
                if 'test_input' in st.session_state:
                    st.write("**Prediction Results**")
                    
                    with torch.no_grad():
                        if st.session_state.model_type == "AttentionCNN":
                            output, attention = st.session_state.model(st.session_state.test_input)
                        else:
                            output = st.session_state.model(st.session_state.test_input)
                    
                    probs = F.softmax(output, dim=1)[0].numpy()
                    classes = ["Particle", "Scratch", "Void", "Bridge", "None"]
                    
                    # Bar chart of predictions
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['red' if p < 0.2 else 'yellow' if p < 0.5 else 'green' 
                             for p in probs]
                    bars = ax.bar(classes, probs, color=colors, alpha=0.7)
                    ax.set_ylabel("Probability")
                    ax.set_title("Classification Results")
                    ax.set_ylim([0, 1])
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probs):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.2%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    # Predicted class
                    pred_class = classes[np.argmax(probs)]
                    confidence = np.max(probs)
                    
                    if confidence > 0.7:
                        st.success(f"**Prediction: {pred_class}** (Confidence: {confidence:.1%})")
                    elif confidence > 0.4:
                        st.warning(f"**Prediction: {pred_class}** (Confidence: {confidence:.1%})")
                    else:
                        st.error(f"**Low Confidence Prediction: {pred_class}** ({confidence:.1%})")
        else:
            st.info("Please load a model first in the 'Model Selection' tab")
    
    with tab3:
        st.subheader("Uncertainty Quantification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**MC Dropout Settings**")
            
            n_samples = st.slider(
                "Monte Carlo Samples",
                min_value=10,
                max_value=100,
                value=30,
                step=10
            )
            
            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
            
            if st.button("Run Uncertainty Analysis", type="primary"):
                with st.spinner("Running MC Dropout..."):
                    # Create MC Dropout model
                    mc_model = MCDropoutClassifier(num_classes=5, dropout_rate=dropout_rate)
                    
                    # Generate test batch
                    test_batch = torch.randn(10, 1, 224, 224)
                    
                    # Get predictions with uncertainty
                    mean_pred, uncertainty = mc_model.predict_with_uncertainty(
                        test_batch, n_samples=n_samples
                    )
                    
                    st.session_state.uncertainty_results = {
                        'mean_pred': mean_pred,
                        'uncertainty': uncertainty
                    }
        
        with col2:
            if 'uncertainty_results' in st.session_state:
                st.write("**Uncertainty Analysis Results**")
                
                uncertainty = st.session_state.uncertainty_results['uncertainty'].numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Uncertainty distribution
                ax1.hist(uncertainty, bins=20, color='purple', alpha=0.7, edgecolor='white')
                ax1.set_xlabel("Uncertainty (Mutual Information)")
                ax1.set_ylabel("Frequency")
                ax1.set_title("Uncertainty Distribution")
                ax1.axvline(uncertainty.mean(), color='yellow', linestyle='--',
                          label=f'Mean: {uncertainty.mean():.3f}')
                ax1.legend()
                
                # Sample uncertainties
                ax2.bar(range(len(uncertainty)), uncertainty, color='orange', alpha=0.7)
                ax2.set_xlabel("Sample Index")
                ax2.set_ylabel("Uncertainty")
                ax2.set_title("Per-Sample Uncertainty")
                ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='High Uncertainty Threshold')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Uncertainty", f"{uncertainty.mean():.3f}")
                col2.metric("Max Uncertainty", f"{uncertainty.max():.3f}")
                col3.metric("High Uncertainty Samples", f"{(uncertainty > 0.5).sum()}")

def show_analytics():
    """Analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Generate sample data for visualization
    np.random.seed(42)
    n_wafers = 100
    
    # Simulated data
    dates = pd.date_range(start='2024-01-01', periods=n_wafers, freq='H')
    yields = np.random.beta(9, 1, n_wafers)  # Skewed towards high yield
    defect_counts = np.random.poisson(20, n_wafers)
    defect_types = np.random.choice(['Particle', 'Scratch', 'Void', 'Bridge'], n_wafers)
    
    df = pd.DataFrame({
        'Timestamp': dates,
        'Yield': yields,
        'Defects': defect_counts,
        'Type': defect_types
    })
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_yield = df['Yield'].mean()
        st.metric("Average Yield", f"{avg_yield:.1%}", 
                 f"{(avg_yield - 0.9)*100:+.1f}%")
    
    with col2:
        avg_defects = df['Defects'].mean()
        st.metric("Avg Defects/Wafer", f"{avg_defects:.1f}",
                 f"{(20 - avg_defects):+.1f}")
    
    with col3:
        high_yield_rate = (df['Yield'] > 0.95).mean()
        st.metric("High Yield Rate", f"{high_yield_rate:.1%}")
    
    with col4:
        defect_trend = np.polyfit(range(len(df)), df['Defects'], 1)[0]
        trend_indicator = "📈" if defect_trend > 0 else "📉"
        st.metric("Defect Trend", trend_indicator,
                 f"{defect_trend:.2f}/hour")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Yield Trend Over Time")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Timestamp'], df['Yield'], color='green', alpha=0.7, linewidth=2)
        ax.fill_between(df['Timestamp'], df['Yield'], alpha=0.3, color='green')
        ax.set_ylabel("Yield Rate")
        ax.set_xlabel("Time")
        ax.set_ylim([0.8, 1.0])
        ax.grid(True, alpha=0.3)
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Target')
        ax.legend()
        
        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:00'))
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Defect Type Distribution")
        
        defect_counts = df['Type'].value_counts()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax.pie(defect_counts.values, 
                                          labels=defect_counts.index,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        
        # Make percentage text white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        st.pyplot(fig)
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("Yield vs Defect Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(df['Defects'], df['Yield'], 
                           c=pd.Categorical(df['Type']).codes,
                           cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel("Defect Count")
        ax.set_ylabel("Yield Rate")
        ax.set_title("Yield vs Defect Correlation")
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Defects'], df['Yield'], 1)
        p = np.poly1d(z)
        ax.plot(sorted(df['Defects']), p(sorted(df['Defects'])), 
               "r--", alpha=0.8, label=f'Trend (R²={np.corrcoef(df["Defects"], df["Yield"])[0,1]**2:.3f})')
        ax.legend()
        
        plt.colorbar(scatter, ax=ax, label='Defect Type')
        st.pyplot(fig)
    
    with col2:
        # Defect heatmap by hour
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day_name()
        
        pivot_table = df.pivot_table(values='Defects', index='Hour', 
                                    columns='Type', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax)
        ax.set_title("Average Defects by Hour and Type")
        ax.set_xlabel("Defect Type")
        ax.set_ylabel("Hour of Day")
        st.pyplot(fig)

def show_predictions():
    """Yield prediction interface"""
    st.header("🔮 Yield Prediction")
    
    st.info("⚠️ Yield prediction module is under development. This is a demonstration interface.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Process parameters
        temperature = st.slider("Process Temperature (°C)", 20, 200, 150)
        pressure = st.slider("Chamber Pressure (mTorr)", 100, 1000, 500)
        exposure_time = st.slider("Exposure Time (s)", 10, 100, 50)
        
        # Defect inputs
        particle_density = st.slider("Particle Density", 0.0, 0.1, 0.02)
        scratch_count = st.number_input("Scratch Count", 0, 50, 5)
        
        if st.button("Predict Yield", type="primary"):
            # Simulated prediction
            base_yield = 0.95
            yield_loss = (particle_density * 2 + scratch_count * 0.005 + 
                         abs(temperature - 150) * 0.0001 +
                         abs(pressure - 500) * 0.00001)
            
            predicted_yield = max(0.5, min(1.0, base_yield - yield_loss))
            confidence_interval = 0.02
            
            st.session_state.prediction = {
                'yield': predicted_yield,
                'ci': confidence_interval,
                'lower': predicted_yield - confidence_interval,
                'upper': predicted_yield + confidence_interval
            }
    
    with col2:
        if 'prediction' in st.session_state:
            st.subheader("Prediction Results")
            
            pred = st.session_state.prediction
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Yield", f"{pred['yield']:.1%}")
            
            with col2:
                st.metric("Lower Bound", f"{pred['lower']:.1%}")
            
            with col3:
                st.metric("Upper Bound", f"{pred['upper']:.1%}")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Gauge chart
            theta = np.linspace(0, np.pi, 100)
            r = 1
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            ax1.plot(x, y, 'gray', linewidth=2)
            ax1.fill_between(x, 0, y, color='lightgray', alpha=0.3)
            
            # Needle
            needle_angle = np.pi * (1 - pred['yield'])
            needle_x = r * 0.9 * np.cos(needle_angle)
            needle_y = r * 0.9 * np.sin(needle_angle)
            ax1.arrow(0, 0, needle_x, needle_y, head_width=0.05, 
                     head_length=0.05, fc='red', ec='red', linewidth=2)
            
            # Color zones
            colors = ['red', 'yellow', 'green']
            boundaries = [0, 0.8, 0.9, 1.0]
            for i, color in enumerate(colors):
                start = np.pi * (1 - boundaries[i])
                end = np.pi * (1 - boundaries[i+1])
                theta_zone = np.linspace(end, start, 50)
                x_zone = r * np.cos(theta_zone)
                y_zone = r * np.sin(theta_zone)
                ax1.fill_between(x_zone, 0, y_zone, color=color, alpha=0.3)
            
            ax1.set_xlim([-1.2, 1.2])
            ax1.set_ylim([-0.2, 1.2])
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title(f"Yield: {pred['yield']:.1%}")
            
            # Confidence interval
            ax2.barh(['Prediction'], [pred['yield']], xerr=pred['ci'], 
                    color='blue', alpha=0.7, error_kw={'linewidth': 2})
            ax2.set_xlim([0, 1])
            ax2.set_xlabel("Yield Rate")
            ax2.set_title("95% Confidence Interval")
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if pred['yield'] < 0.9:
                st.warning("""
                **Low Yield Alert!** Consider the following actions:
                - Reduce particle contamination
                - Inspect and clean chamber
                - Verify process parameters
                - Check for systematic defects
                """)
            else:
                st.success("""
                **Good Yield!** To maintain performance:
                - Continue current process controls
                - Monitor for trend changes
                - Document successful parameters
                """)

def show_documentation():
    """Documentation page"""
    st.header("📚 Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "API Reference", "Examples", "FAQ"])
    
    with tab1:
        st.markdown("""
        ## Getting Started
        
        ### Installation
        ```bash
        cd defect-detection-ml
        pip install -r requirements.txt
        ```
        
        ### Quick Start
        ```python
        from src.data_generator import DefectDataGenerator
        from src.cnn_models import create_model
        
        # Generate data
        generator = DefectDataGenerator()
        dataset = generator.generate_dataset(n_samples=1000)
        
        # Create model
        model = create_model('resnet', num_classes=5)
        
        # Train model
        # ... training code ...
        ```
        
        ### Running the Demo
        ```bash
        python demo.py  # Run comprehensive demo
        streamlit run streamlit_app.py  # Launch web interface
        ```
        """)
    
    with tab2:
        st.markdown("""
        ## API Reference
        
        ### Data Generation
        
        #### `DefectDataGenerator`
        - `generate_wafer_map()`: Generate wafer defect map
        - `generate_defect_image()`: Create synthetic defect image
        - `generate_dataset()`: Build complete dataset
        
        #### `SyntheticWaferGenerator`
        - `generate_wafer_image()`: Create realistic wafer image
        - `add_inspection_artifacts()`: Add realistic artifacts
        
        ### Models
        
        #### `create_model()`
        Create any supported CNN architecture:
        - `'defect_cnn'`: Custom CNN
        - `'resnet'`: ResNet-based model
        - `'efficientnet'`: EfficientNet
        - `'attention_cnn'`: CNN with attention
        - `'multiscale_cnn'`: Multi-scale CNN
        
        ### Bayesian Models
        
        #### `MCDropoutClassifier`
        - `predict_with_uncertainty()`: Get predictions with uncertainty
        
        #### `RareDefectDetector`
        - `fit()`: Train on normal samples
        - `predict()`: Detect anomalies
        """)
    
    with tab3:
        st.markdown("""
        ## Code Examples
        
        ### Example 1: Generate Training Data
        ```python
        from src.data_generator import DefectDataGenerator, DefectType
        
        generator = DefectDataGenerator(wafer_size=300)
        
        # Generate specific defect type
        wafer_map = generator.generate_wafer_map(
            defect_type=DefectType.CLUSTER,
            defect_density=0.05
        )
        
        print(f"Yield: {wafer_map.yield_rate:.1%}")
        ```
        
        ### Example 2: Train Model with Uncertainty
        ```python
        from src.bayesian_models import MCDropoutClassifier
        import torch
        
        model = MCDropoutClassifier(num_classes=5)
        
        # Training loop
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(10):
            # ... training code ...
            pass
        
        # Inference with uncertainty
        predictions, uncertainty = model.predict_with_uncertainty(
            test_data, n_samples=50
        )
        ```
        
        ### Example 3: Active Learning
        ```python
        from src.bayesian_models import ActiveLearningSelector
        
        selector = ActiveLearningSelector(strategy='uncertainty')
        
        # Select most informative samples
        indices = selector.select_samples(
            model=trained_model,
            unlabeled_data=pool,
            n_samples=10
        )
        ```
        """)
    
    with tab4:
        st.markdown("""
        ## Frequently Asked Questions
        
        **Q: How do I add a new defect type?**
        
        A: Modify the `DefectType` enum in `data_generator.py` and implement the generation logic in the corresponding method.
        
        **Q: Can I use my own data?**
        
        A: Yes! The models accept standard PyTorch tensors. Format your data as (B, 1, H, W) for grayscale images.
        
        **Q: How do I export trained models?**
        
        A: Use PyTorch's standard saving mechanism:
        ```python
        torch.save(model.state_dict(), 'model.pth')
        ```
        
        **Q: What's the recommended batch size?**
        
        A: For training: 32-64 depending on GPU memory. For inference: 1-16 for real-time applications.
        
        **Q: How do I integrate with existing systems?**
        
        A: The models can be exported to ONNX format for deployment:
        ```python
        torch.onnx.export(model, dummy_input, "model.onnx")
        ```
        """)

if __name__ == "__main__":
    main()