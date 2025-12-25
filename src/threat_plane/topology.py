"""
Topological analysis module for the Threat Plane.

Implements:
- Persistent homology for multi-scale feature extraction
- Betti number computation
- Mapper algorithm for structure discovery
- Anomaly detection in geometric space
- Clustering algorithms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class PersistenceDiagram:
    """Represents a persistence diagram with birth/death pairs."""
    dimension: int
    pairs: np.ndarray  # Shape (n_features, 2) for birth/death times
    
    def get_persistent_features(self, min_persistence: float = 0.1) -> np.ndarray:
        """Get features with persistence above threshold."""
        persistence = self.pairs[:, 1] - self.pairs[:, 0]
        mask = persistence > min_persistence
        return self.pairs[mask]


class TopologicalAnalyzer:
    """
    Topological analysis of the 3D threat plane.
    
    Extracts topological features that have security interpretations:
    - H₀: Connected components (threat clusters)
    - H₁: Loops (cyclic attack dependencies)
    - H₂: Voids (defensive coverage regions)
    
    Example:
        >>> analyzer = TopologicalAnalyzer(projection)
        >>> persistence = analyzer.compute_persistence()
        >>> betti = analyzer.compute_betti_numbers()
        >>> clusters = analyzer.cluster(method="hdbscan")
    """
    
    def __init__(self, projection: np.ndarray, labels: Optional[np.ndarray] = None):
        self.projection = projection
        self.labels = labels
        
        self._persistence_diagrams: Dict[int, PersistenceDiagram] = {}
        self._mapper_graph = None
        self._clusters = None
    
    def compute_persistence(
        self,
        max_dimension: int = 2,
        max_edge_length: float = None
    ) -> Dict[int, PersistenceDiagram]:
        """
        Compute persistent homology of the point cloud.
        
        Uses Vietoris-Rips filtration to track topological features
        across multiple scales.
        
        Args:
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge length for Rips complex
            
        Returns:
            Dictionary mapping dimension to PersistenceDiagram
        """
        try:
            import gudhi
        except ImportError:
            raise ImportError("GUDHI required: pip install gudhi")
        
        # Create Rips complex
        rips = gudhi.RipsComplex(
            points=self.projection,
            max_edge_length=max_edge_length or self._estimate_max_edge()
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract persistence diagrams
        for dim in range(max_dimension + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            # Filter out infinite deaths for analysis
            finite_pairs = pairs[pairs[:, 1] != np.inf] if len(pairs) > 0 else np.array([])
            
            self._persistence_diagrams[dim] = PersistenceDiagram(
                dimension=dim,
                pairs=finite_pairs if len(finite_pairs) > 0 else np.zeros((0, 2))
            )
        
        return self._persistence_diagrams
    
    def compute_betti_numbers(
        self,
        threshold: float = None
    ) -> Dict[int, int]:
        """
        Compute Betti numbers at a given threshold.
        
        Args:
            threshold: Filtration value (default: median persistence)
            
        Returns:
            Dictionary mapping dimension to Betti number
        """
        if not self._persistence_diagrams:
            self.compute_persistence()
        
        if threshold is None:
            # Use median persistence as threshold
            all_pairs = np.vstack([
                pd.pairs for pd in self._persistence_diagrams.values()
                if len(pd.pairs) > 0
            ])
            if len(all_pairs) > 0:
                threshold = np.median(all_pairs[:, 1] - all_pairs[:, 0])
            else:
                threshold = 0.5
        
        betti = {}
        for dim, pd in self._persistence_diagrams.items():
            if len(pd.pairs) == 0:
                betti[dim] = 0
            else:
                # Count features alive at threshold
                alive = (pd.pairs[:, 0] <= threshold) & (pd.pairs[:, 1] > threshold)
                betti[dim] = np.sum(alive)
        
        return betti
    
    def compute_mapper(
        self,
        filter_function: str = "density",
        n_cubes: int = 10,
        overlap: float = 0.5,
        clusterer: str = "dbscan"
    ):
        """
        Compute Mapper graph for structure discovery.
        
        Creates a simplified graph that captures the essential
        skeleton of the threat landscape.
        
        Args:
            filter_function: Function to filter point cloud
            n_cubes: Number of intervals for covering
            overlap: Overlap percentage between intervals
            clusterer: Clustering algorithm for each interval
        """
        try:
            import kmapper as km
        except ImportError:
            raise ImportError("KeplerMapper required: pip install kmapper")
        
        mapper = km.KeplerMapper()
        
        # Choose filter function
        if filter_function == "density":
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(bandwidth=0.5)
            kde.fit(self.projection)
            lens = kde.score_samples(self.projection)
        elif filter_function == "eccentricity":
            lens = np.mean(
                np.linalg.norm(self.projection - self.projection[:, None], axis=2),
                axis=1
            )
        elif filter_function == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            lens = pca.fit_transform(self.projection).flatten()
        else:
            raise ValueError(f"Unknown filter function: {filter_function}")
        
        # Create cover and graph
        self._mapper_graph = mapper.map(
            lens,
            self.projection,
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
            clusterer=self._get_clusterer(clusterer)
        )
        
        return self._mapper_graph
    
    def detect_anomalies(
        self,
        method: Literal["lof", "isolation_forest", "geodesic"] = "lof",
        contamination: float = 0.1,
        **kwargs
    ) -> np.ndarray:
        """
        Detect anomalous points in the threat plane.
        
        Args:
            method: Anomaly detection algorithm
            contamination: Expected proportion of anomalies
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        if method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(
                n_neighbors=kwargs.get("n_neighbors", 20),
                contamination=contamination
            )
            # LOF returns negative scores, so negate
            scores = -lof.fit_predict(self.projection)
            return lof.negative_outlier_factor_
        
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get("random_state", 42)
            )
            iso.fit(self.projection)
            return -iso.score_samples(self.projection)
        
        elif method == "geodesic":
            return self._geodesic_anomaly_scores()
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def cluster(
        self,
        method: Literal["hdbscan", "dbscan", "spectral", "kmeans"] = "hdbscan",
        **kwargs
    ) -> Dict[int, np.ndarray]:
        """
        Cluster threats in the threat plane.
        
        Args:
            method: Clustering algorithm
            
        Returns:
            Dictionary mapping cluster ID to array of point indices
        """
        if method == "hdbscan":
            try:
                import hdbscan
            except ImportError:
                raise ImportError("HDBSCAN required: pip install hdbscan")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=kwargs.get("min_cluster_size", 5),
                min_samples=kwargs.get("min_samples", 3)
            )
            labels = clusterer.fit_predict(self.projection)
        
        elif method == "dbscan":
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5)
            )
            labels = clusterer.fit_predict(self.projection)
        
        elif method == "spectral":
            from sklearn.cluster import SpectralClustering
            n_clusters = kwargs.get("n_clusters", 6)
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                affinity="nearest_neighbors",
                random_state=kwargs.get("random_state", 42)
            )
            labels = clusterer.fit_predict(self.projection)
        
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            n_clusters = kwargs.get("n_clusters", 6)
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=kwargs.get("random_state", 42)
            )
            labels = clusterer.fit_predict(self.projection)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self._clusters = labels
        
        # Convert to dictionary
        clusters = {}
        for label in np.unique(labels):
            if label != -1:  # Exclude noise
                clusters[label] = np.where(labels == label)[0]
        
        return clusters
    
    def get_cluster_nodes(self, cluster_id: int) -> np.ndarray:
        """Get node indices belonging to a cluster."""
        if self._clusters is None:
            raise ValueError("Clustering not performed. Call cluster() first.")
        return np.where(self._clusters == cluster_id)[0]
    
    def _estimate_max_edge(self) -> float:
        """Estimate max edge length for Rips complex."""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(self.projection)
        distances, _ = nn.kneighbors()
        
        return np.percentile(distances[:, -1], 95) * 2
    
    def _get_clusterer(self, name: str):
        """Get clusterer for Mapper."""
        if name == "dbscan":
            from sklearn.cluster import DBSCAN
            return DBSCAN(eps=0.5, min_samples=3)
        elif name == "kmeans":
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=3)
        else:
            raise ValueError(f"Unknown clusterer: {name}")
    
    def _geodesic_anomaly_scores(self) -> np.ndarray:
        """
        Compute geodesic-based anomaly scores.
        
        Points that are Euclidean-close but geodesically-distant
        indicate unexpected connections (threat bridges).
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.sparse.csgraph import shortest_path
        from sklearn.neighbors import kneighbors_graph
        
        # Build k-NN graph
        k = min(10, len(self.projection) - 1)
        knn_graph = kneighbors_graph(
            self.projection, k, mode='distance', include_self=False
        )
        
        # Compute geodesic distances
        geodesic_dist = shortest_path(knn_graph, directed=False)
        
        # Compute Euclidean distances
        euclidean_dist = squareform(pdist(self.projection))
        
        # Anomaly score: ratio of geodesic to Euclidean distance
        # High ratio means the point is isolated in the graph structure
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = geodesic_dist / (euclidean_dist + 1e-10)
            ratio[np.isinf(ratio)] = 0
        
        # Average ratio for each point
        scores = np.mean(ratio, axis=1)
        
        return scores
