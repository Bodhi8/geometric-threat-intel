"""
Dimensionality reduction module for projecting embeddings to 3D.

Supports multiple projection methods:
- UMAP with security-specific metrics
- t-SNE for local structure refinement
- PaCMAP for balanced preservation
- Custom geodesic-aware projections
"""

import numpy as np
from typing import Optional, Literal


class ThreatPlaneProjector:
    """
    Projects high-dimensional embeddings to 3D threat plane.
    
    Preserves security-relevant distances and structures while
    creating an interpretable 3D representation.
    
    Example:
        >>> projector = ThreatPlaneProjector(method="umap")
        >>> projection = projector.fit_transform(embeddings)
        >>> trajectories = projector.predict_trajectory(projection, start_idx, steps=10)
    """
    
    def __init__(
        self,
        method: Literal["umap", "tsne", "pacmap", "custom"] = "umap",
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        geodesic_weight: float = 0.1,
        **kwargs
    ):
        self.method = method
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.geodesic_weight = geodesic_weight
        self.kwargs = kwargs
        
        self.reducer = None
        self._fitted = False
    
    def fit(self, embeddings: np.ndarray) -> "ThreatPlaneProjector":
        """Fit the projection model to embeddings."""
        self.reducer = self._create_reducer()
        self.reducer.fit(embeddings)
        self._fitted = True
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to 3D."""
        if not self._fitted:
            raise ValueError("Projector not fitted. Call fit() first.")
        return self.reducer.transform(embeddings)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.reducer = self._create_reducer()
        projection = self.reducer.fit_transform(embeddings)
        self._fitted = True
        return projection
    
    def _create_reducer(self):
        """Create the dimensionality reduction model."""
        if self.method == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError("UMAP required: pip install umap-learn")
            
            return umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self._get_metric(),
                random_state=self.random_state,
                **self.kwargs
            )
        
        elif self.method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(
                n_components=self.n_components,
                perplexity=min(30, self.n_neighbors),
                random_state=self.random_state,
                **self.kwargs
            )
        
        elif self.method == "pacmap":
            try:
                import pacmap
            except ImportError:
                raise ImportError("PaCMAP required: pip install pacmap")
            
            return pacmap.PaCMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                **self.kwargs
            )
        
        elif self.method == "custom":
            return GeodesicAwareProjector(
                n_components=self.n_components,
                geodesic_weight=self.geodesic_weight,
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_metric(self):
        """Get the distance metric for UMAP."""
        if self.metric == "security_weighted":
            return self._security_weighted_distance
        return self.metric
    
    @staticmethod
    def _security_weighted_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        Security-weighted distance metric.
        
        Weights dimensions by their relevance to attack path feasibility.
        """
        # Placeholder: In practice, weight dimensions based on
        # feature importance from attack path analysis
        weights = np.ones(len(u))
        return np.sqrt(np.sum(weights * (u - v) ** 2))
    
    def predict_trajectory(
        self,
        projection: np.ndarray,
        start_idx: int,
        steps: int = 10,
        velocity_field: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict attack trajectory from a starting point.
        
        Uses either:
        - Learned velocity field (Neural ODE)
        - Local gradient descent toward high-risk regions
        
        Args:
            projection: 3D coordinates of all points
            start_idx: Index of starting point
            steps: Number of trajectory steps
            velocity_field: Optional pre-computed velocity field
            
        Returns:
            Array of shape (steps, 3) representing trajectory
        """
        trajectory = [projection[start_idx]]
        current = projection[start_idx].copy()
        
        for _ in range(steps - 1):
            if velocity_field is not None:
                # Use velocity field
                velocity = self._interpolate_velocity(current, velocity_field)
            else:
                # Simple gradient toward nearest high-risk point
                velocity = self._compute_risk_gradient(current, projection)
            
            current = current + velocity * 0.1
            trajectory.append(current.copy())
        
        return np.array(trajectory)
    
    def _interpolate_velocity(
        self,
        point: np.ndarray,
        velocity_field: np.ndarray
    ) -> np.ndarray:
        """Interpolate velocity at a point from the velocity field."""
        # Placeholder: Would use trilinear interpolation or neural network
        return np.random.randn(3) * 0.1
    
    def _compute_risk_gradient(
        self,
        point: np.ndarray,
        projection: np.ndarray
    ) -> np.ndarray:
        """Compute gradient toward high-risk regions."""
        # Find nearest neighbors
        distances = np.linalg.norm(projection - point, axis=1)
        nearest = np.argsort(distances)[1:6]  # Exclude self
        
        # Compute weighted direction toward neighbors
        directions = projection[nearest] - point
        gradient = np.mean(directions, axis=0)
        
        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient = gradient / norm
        
        return gradient


class GeodesicAwareProjector:
    """
    Custom projection that preserves geodesic distances.
    
    Uses stress majorization with geodesic distance constraints
    to ensure attack paths map to navigable 3D trajectories.
    """
    
    def __init__(
        self,
        n_components: int = 3,
        geodesic_weight: float = 0.1,
        max_iter: int = 300,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.geodesic_weight = geodesic_weight
        self.max_iter = max_iter
        self.random_state = random_state
        
        self._embedding = None
    
    def fit(self, X: np.ndarray, geodesic_distances: Optional[np.ndarray] = None):
        """Fit the projection with optional geodesic constraints."""
        # Initialize with PCA
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._embedding = pca.fit_transform(X)
        
        if geodesic_distances is not None:
            # Refine with stress majorization
            self._embedding = self._stress_majorization(
                X, geodesic_distances, self._embedding
            )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new points (out-of-sample extension)."""
        # Use nearest neighbor interpolation
        raise NotImplementedError("Out-of-sample transform not implemented")
    
    def fit_transform(self, X: np.ndarray, geodesic_distances: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, geodesic_distances)
        return self._embedding
    
    def _stress_majorization(
        self,
        X: np.ndarray,
        geodesic_distances: np.ndarray,
        init: np.ndarray
    ) -> np.ndarray:
        """Refine embedding using stress majorization."""
        embedding = init.copy()
        
        # Compute pairwise Euclidean distances in high-dim space
        from scipy.spatial.distance import pdist, squareform
        high_dim_dist = squareform(pdist(X))
        
        # Blend with geodesic distances
        target_dist = (
            (1 - self.geodesic_weight) * high_dim_dist +
            self.geodesic_weight * geodesic_distances
        )
        
        # Iterative stress majorization
        for iteration in range(self.max_iter):
            low_dim_dist = squareform(pdist(embedding))
            
            # Compute stress
            stress = np.sum((target_dist - low_dim_dist) ** 2)
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Stress = {stress:.4f}")
            
            # Update using Guttman transform
            B = np.zeros_like(target_dist)
            mask = low_dim_dist > 0
            B[mask] = -target_dist[mask] / low_dim_dist[mask]
            B[np.arange(len(B)), np.arange(len(B))] = -np.sum(B, axis=1)
            
            embedding = B @ embedding / len(embedding)
        
        return embedding
