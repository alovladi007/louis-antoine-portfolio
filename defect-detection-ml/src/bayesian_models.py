"""
Bayesian ML Models for Rare Defect Detection

This module implements Bayesian approaches for handling imbalanced datasets
and detecting rare defect classes with uncertainty quantification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from typing import Tuple, Dict, Any, Optional, List
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BayesianCNN(PyroModule):
    """
    Bayesian CNN for defect classification with uncertainty quantification
    """
    
    def __init__(self, num_classes: int = 5, input_channels: int = 1):
        """
        Initialize Bayesian CNN
        
        Args:
            num_classes: Number of defect classes
            input_channels: Number of input channels
        """
        super().__init__()
        
        # Define architecture
        self.conv1 = PyroModule[nn.Conv2d](input_channels, 32, 3, padding=1)
        self.conv2 = PyroModule[nn.Conv2d](32, 64, 3, padding=1)
        self.conv3 = PyroModule[nn.Conv2d](64, 128, 3, padding=1)
        
        self.fc1 = PyroModule[nn.Linear](128 * 28 * 28, 256)
        self.fc2 = PyroModule[nn.Linear](256, num_classes)
        
        # Set priors for weights
        self.conv1.weight = PyroSample(
            dist.Normal(0., 0.1).expand([32, input_channels, 3, 3]).to_event(4)
        )
        self.conv1.bias = PyroSample(
            dist.Normal(0., 0.1).expand([32]).to_event(1)
        )
        
        self.conv2.weight = PyroSample(
            dist.Normal(0., 0.1).expand([64, 32, 3, 3]).to_event(4)
        )
        self.conv2.bias = PyroSample(
            dist.Normal(0., 0.1).expand([64]).to_event(1)
        )
        
        self.conv3.weight = PyroSample(
            dist.Normal(0., 0.1).expand([128, 64, 3, 3]).to_event(4)
        )
        self.conv3.bias = PyroSample(
            dist.Normal(0., 0.1).expand([128]).to_event(1)
        )
        
        self.fc1.weight = PyroSample(
            dist.Normal(0., 0.1).expand([256, 128 * 28 * 28]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0., 0.1).expand([256]).to_event(1)
        )
        
        self.fc2.weight = PyroSample(
            dist.Normal(0., 0.1).expand([num_classes, 256]).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0., 0.1).expand([num_classes]).to_event(1)
        )
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            y: Target labels (for training)
            
        Returns:
            Logits or sampled predictions
        """
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        # Sample predictions
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits


class BayesianDefectClassifier:
    """
    Bayesian classifier for defect detection with uncertainty estimation
    """
    
    def __init__(self, input_dim: int = 224*224, num_classes: int = 5,
                 hidden_dims: List[int] = [512, 256, 128]):
        """
        Initialize Bayesian classifier
        
        Args:
            input_dim: Input dimension (flattened image)
            num_classes: Number of defect classes
            hidden_dims: Hidden layer dimensions
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # Build model
        self.model = self._build_model()
        self.guide = None
        self.svi = None
        self.predictive = None
        
    def _build_model(self) -> callable:
        """Build Bayesian neural network model"""
        
        def model(x, y=None):
            # Define priors for weights and biases
            layers = []
            input_size = self.input_dim
            
            for i, hidden_size in enumerate(self.hidden_dims):
                # Weight prior
                w_loc = torch.zeros(hidden_size, input_size)
                w_scale = torch.ones(hidden_size, input_size) * 0.1
                w = pyro.sample(f"w_{i}", dist.Normal(w_loc, w_scale).to_event(2))
                
                # Bias prior
                b_loc = torch.zeros(hidden_size)
                b_scale = torch.ones(hidden_size) * 0.1
                b = pyro.sample(f"b_{i}", dist.Normal(b_loc, b_scale).to_event(1))
                
                layers.append((w, b))
                input_size = hidden_size
            
            # Output layer
            w_out_loc = torch.zeros(self.num_classes, input_size)
            w_out_scale = torch.ones(self.num_classes, input_size) * 0.1
            w_out = pyro.sample("w_out", dist.Normal(w_out_loc, w_out_scale).to_event(2))
            
            b_out_loc = torch.zeros(self.num_classes)
            b_out_scale = torch.ones(self.num_classes) * 0.1
            b_out = pyro.sample("b_out", dist.Normal(b_out_loc, b_out_scale).to_event(1))
            
            # Forward pass
            h = x.view(-1, self.input_dim)
            for w, b in layers:
                h = F.relu(F.linear(h, w, b))
            
            logits = F.linear(h, w_out, b_out)
            
            # Likelihood
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
            
            return logits
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             num_epochs: int = 100, lr: float = 0.01) -> List[float]:
        """
        Train the Bayesian classifier
        
        Args:
            X_train: Training images
            y_train: Training labels
            num_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            List of losses
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        
        # Setup guide (variational distribution)
        from pyro.infer.autoguide import AutoDiagonalNormal
        self.guide = AutoDiagonalNormal(self.model)
        
        # Setup optimizer and loss
        adam = Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            loss = self.svi.step(X_train, y_train)
            losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        # Setup predictive distribution
        self.predictive = Predictive(self.model, guide=self.guide, num_samples=100)
        
        return losses
    
    def predict_with_uncertainty(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Args:
            X_test: Test images
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if self.predictive is None:
            raise ValueError("Model must be trained first")
        
        X_test = torch.FloatTensor(X_test)
        
        # Get samples from posterior
        samples = self.predictive(X_test)
        
        # Get predictions from samples
        logits_samples = samples['obs']  # Shape: (num_samples, batch_size)
        
        # Calculate mean prediction and uncertainty
        probs = F.softmax(logits_samples, dim=-1)
        mean_probs = probs.mean(dim=0)
        
        # Predictions
        predictions = mean_probs.argmax(dim=-1).numpy()
        
        # Uncertainty (entropy of predictive distribution)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        uncertainties = entropy.numpy()
        
        return predictions, uncertainties


class RareDefectDetector:
    """
    Specialized detector for rare defect classes using few-shot learning
    and Bayesian approaches
    """
    
    def __init__(self, n_components: int = 10, contamination: float = 0.1):
        """
        Initialize rare defect detector
        
        Args:
            n_components: Number of Gaussian components for GMM
            contamination: Expected proportion of outliers
        """
        self.n_components = n_components
        self.contamination = contamination
        self.gmm = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def fit(self, X_normal: np.ndarray) -> None:
        """
        Fit the model on normal (non-defective) samples
        
        Args:
            X_normal: Normal samples (flattened images)
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Fit Gaussian Mixture Model
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        self.gmm.fit(X_scaled)
        
        # Calculate threshold based on contamination
        scores = self.gmm.score_samples(X_scaled)
        self.threshold = np.percentile(scores, self.contamination * 100)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict whether samples are rare defects
        
        Args:
            X: Samples to predict (flattened images)
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        if self.gmm is None:
            raise ValueError("Model must be fitted first")
        
        # Standardize features
        X_scaled = self.scaler.transform(X)
        
        # Calculate anomaly scores
        scores = self.gmm.score_samples(X_scaled)
        
        # Predictions (1 for anomaly/rare defect, 0 for normal)
        predictions = (scores < self.threshold).astype(int)
        
        # Normalize scores to [0, 1]
        anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        return predictions, anomaly_scores
    
    def update_with_new_defects(self, X_new_defects: np.ndarray,
                               learning_rate: float = 0.1) -> None:
        """
        Update model with newly discovered rare defects
        
        Args:
            X_new_defects: New rare defect samples
            learning_rate: Rate at which to incorporate new information
        """
        if self.gmm is None:
            raise ValueError("Model must be fitted first")
        
        # Standardize new samples
        X_scaled = self.scaler.transform(X_new_defects)
        
        # Partial fit (simplified approach)
        # In practice, you might want to use online learning methods
        old_weights = self.gmm.weights_ * (1 - learning_rate)
        
        # Fit a new GMM on new data
        new_gmm = GaussianMixture(
            n_components=min(len(X_new_defects), self.n_components),
            covariance_type='full',
            random_state=42
        )
        new_gmm.fit(X_scaled)
        
        # Combine models (simplified)
        # This is a basic approach; more sophisticated methods exist
        self.gmm.weights_ = old_weights
        # Note: In practice, you'd also update means and covariances


class MCDropoutClassifier(nn.Module):
    """
    CNN with Monte Carlo Dropout for uncertainty estimation
    """
    
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.5):
        """
        Initialize MC Dropout classifier
        
        Args:
            num_classes: Number of defect classes
            dropout_rate: Dropout rate for MC sampling
        """
        super(MCDropoutClassifier, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using MC Dropout
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = F.softmax(self.forward(x), dim=-1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty (predictive entropy)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=-1)
        
        # Mutual information (epistemic uncertainty)
        expected_entropy = -(predictions * torch.log(predictions + 1e-8)).sum(dim=-1).mean(dim=0)
        mutual_info = entropy - expected_entropy
        
        return mean_pred, mutual_info


class ActiveLearningSelector:
    """
    Active learning strategy for selecting informative samples
    """
    
    def __init__(self, strategy: str = 'uncertainty'):
        """
        Initialize active learning selector
        
        Args:
            strategy: Selection strategy ('uncertainty', 'diversity', 'hybrid')
        """
        self.strategy = strategy
    
    def select_samples(self, model: nn.Module, unlabeled_data: torch.Tensor,
                      n_samples: int = 10) -> np.ndarray:
        """
        Select most informative samples for labeling
        
        Args:
            model: Trained model with uncertainty estimation
            unlabeled_data: Pool of unlabeled samples
            n_samples: Number of samples to select
            
        Returns:
            Indices of selected samples
        """
        if self.strategy == 'uncertainty':
            return self._uncertainty_sampling(model, unlabeled_data, n_samples)
        elif self.strategy == 'diversity':
            return self._diversity_sampling(model, unlabeled_data, n_samples)
        elif self.strategy == 'hybrid':
            return self._hybrid_sampling(model, unlabeled_data, n_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _uncertainty_sampling(self, model: nn.Module, data: torch.Tensor,
                            n_samples: int) -> np.ndarray:
        """Select samples with highest uncertainty"""
        if isinstance(model, MCDropoutClassifier):
            _, uncertainties = model.predict_with_uncertainty(data)
            uncertainties = uncertainties.numpy()
        else:
            # Use entropy for standard models
            with torch.no_grad():
                logits = model(data)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                uncertainties = entropy.numpy()
        
        # Select top uncertain samples
        indices = np.argsort(uncertainties)[-n_samples:]
        return indices
    
    def _diversity_sampling(self, model: nn.Module, data: torch.Tensor,
                          n_samples: int) -> np.ndarray:
        """Select diverse samples using clustering"""
        # Extract features
        with torch.no_grad():
            if hasattr(model, 'features'):
                features = model.features(data)
                features = features.view(features.size(0), -1).numpy()
            else:
                # Use model outputs as features
                features = model(data).numpy()
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(features)
        
        # Select closest sample to each cluster center
        indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features - center, axis=1)
            indices.append(np.argmin(distances))
        
        return np.array(indices)
    
    def _hybrid_sampling(self, model: nn.Module, data: torch.Tensor,
                        n_samples: int) -> np.ndarray:
        """Combine uncertainty and diversity sampling"""
        # Get uncertainty scores
        uncertain_indices = self._uncertainty_sampling(
            model, data, n_samples * 2
        )
        
        # Select diverse samples from uncertain ones
        uncertain_data = data[uncertain_indices]
        diverse_indices_local = self._diversity_sampling(
            model, uncertain_data, n_samples
        )
        
        # Map back to original indices
        final_indices = uncertain_indices[diverse_indices_local]
        
        return final_indices


# Example usage
if __name__ == "__main__":
    # Test Bayesian classifier
    print("Testing Bayesian Defect Classifier...")
    
    # Generate synthetic data
    n_samples = 100
    input_dim = 224 * 224
    num_classes = 5
    
    X_train = np.random.randn(n_samples, input_dim)
    y_train = np.random.randint(0, num_classes, n_samples)
    
    # Create and train Bayesian classifier
    bayesian_clf = BayesianDefectClassifier(
        input_dim=input_dim,
        num_classes=num_classes
    )
    
    # Note: Training would require Pyro setup
    # losses = bayesian_clf.train(X_train, y_train, num_epochs=10)
    
    # Test MC Dropout classifier
    print("\nTesting MC Dropout Classifier...")
    
    mc_dropout_model = MCDropoutClassifier(num_classes=5)
    test_input = torch.randn(4, 1, 224, 224)
    
    # Get predictions with uncertainty
    mean_pred, uncertainty = mc_dropout_model.predict_with_uncertainty(
        test_input, n_samples=10
    )
    
    print(f"Mean predictions shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Predicted classes: {mean_pred.argmax(dim=-1)}")
    print(f"Uncertainties: {uncertainty}")
    
    # Test rare defect detector
    print("\nTesting Rare Defect Detector...")
    
    # Generate normal samples
    X_normal = np.random.randn(500, 1000)
    
    # Create and fit detector
    detector = RareDefectDetector(n_components=5)
    detector.fit(X_normal)
    
    # Test on new samples (including anomalies)
    X_test = np.random.randn(50, 1000)
    X_test[45:] = np.random.randn(5, 1000) * 3  # Anomalies
    
    predictions, scores = detector.predict(X_test)
    print(f"Detected {np.sum(predictions)} rare defects")
    print(f"Anomaly scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Test active learning
    print("\nTesting Active Learning Selector...")
    
    selector = ActiveLearningSelector(strategy='uncertainty')
    unlabeled_data = torch.randn(100, 1, 224, 224)
    
    selected_indices = selector.select_samples(
        mc_dropout_model, unlabeled_data, n_samples=10
    )
    print(f"Selected sample indices: {selected_indices}")