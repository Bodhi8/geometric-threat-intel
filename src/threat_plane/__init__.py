"""
Threat Plane: A Geometric Framework for Cyber Risk Visualization

This package provides tools for:
- Building Security Knowledge Graphs
- Training Graph Neural Network embeddings
- Projecting to 3D threat planes
- Topological analysis of threat landscapes
- Interactive visualization
"""

from .graph import SecurityKnowledgeGraph
from .embedding import ThreatEmbedding
from .projection import ThreatPlaneProjector
from .topology import TopologicalAnalyzer
from .visualization import ThreatPlaneVisualizer

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "SecurityKnowledgeGraph",
    "ThreatEmbedding", 
    "ThreatPlaneProjector",
    "TopologicalAnalyzer",
    "ThreatPlaneVisualizer",
]


class ThreatPlane:
    """
    Main interface for the Threat Plane framework.
    
    Combines all components into a unified workflow for:
    1. Building a Security Knowledge Graph
    2. Training GNN embeddings
    3. Projecting to 3D
    4. Analyzing topology
    5. Visualization
    
    Example:
        >>> tp = ThreatPlane()
        >>> tp.ingest_data("vulnerabilities.json", "assets.csv")
        >>> tp.build_graph()
        >>> tp.train(epochs=100)
        >>> tp.project(method="umap")
        >>> tp.analyze_topology()
        >>> tp.visualize()
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.graph = SecurityKnowledgeGraph()
        self.embedder = None
        self.projector = None
        self.topology = None
        self.visualizer = None
        
        self._embeddings = None
        self._projection = None
        self._topology_features = None
    
    def ingest_vulnerabilities(self, path, format="json"):
        """Ingest vulnerability scan data."""
        self.graph.ingest_vulnerabilities(path, format)
        return self
    
    def ingest_assets(self, path, format="csv"):
        """Ingest asset inventory data."""
        self.graph.ingest_assets(path, format)
        return self
    
    def ingest_threat_intel(self, path, format="stix"):
        """Ingest threat intelligence feeds."""
        self.graph.ingest_threat_intel(path, format)
        return self
    
    def build_graph(self):
        """Construct the Security Knowledge Graph."""
        self.graph.resolve_entities()
        self.graph.infer_relationships()
        self.graph.compute_attack_paths()
        return self
    
    def train(self, epochs=100, lr=0.001, hidden_dim=128):
        """Train the GNN embedding model."""
        self.embedder = ThreatEmbedding(
            self.graph,
            hidden_dim=hidden_dim,
            num_layers=4
        )
        self._embeddings = self.embedder.train(epochs=epochs, lr=lr)
        return self
    
    def project(self, method="umap", **kwargs):
        """Project embeddings to 3D threat plane."""
        self.projector = ThreatPlaneProjector(method=method, **kwargs)
        self._projection = self.projector.fit_transform(self._embeddings)
        return self
    
    def analyze_topology(self):
        """Compute topological features of the threat plane."""
        self.topology = TopologicalAnalyzer(self._projection)
        self._topology_features = {
            "persistence_diagram": self.topology.compute_persistence(),
            "betti_numbers": self.topology.compute_betti_numbers(),
            "mapper_graph": self.topology.compute_mapper()
        }
        return self
    
    def detect_anomalies(self, method="lof", **kwargs):
        """Detect anomalous points in the threat plane."""
        return self.topology.detect_anomalies(method=method, **kwargs)
    
    def predict_trajectories(self, start_node, steps=10):
        """Predict attack trajectories from a starting point."""
        return self.projector.predict_trajectory(
            self._projection,
            start_node,
            steps=steps
        )
    
    def cluster_threats(self, method="hdbscan", **kwargs):
        """Cluster threats in the threat plane."""
        return self.topology.cluster(method=method, **kwargs)
    
    def visualize(self, output="interactive"):
        """Render the threat plane visualization."""
        self.visualizer = ThreatPlaneVisualizer(
            self._projection,
            self.graph,
            self._topology_features
        )
        if output == "interactive":
            return self.visualizer.render_interactive()
        elif output == "static":
            return self.visualizer.render_static()
        elif output == "html":
            return self.visualizer.export_html()
    
    def get_attack_paths_through(self, cluster_id):
        """Get all attack paths passing through a cluster."""
        return self.graph.get_paths_through_nodes(
            self.topology.get_cluster_nodes(cluster_id)
        )
    
    def compute_cluster_risks(self):
        """Compute aggregate risk scores for each cluster."""
        clusters = self.cluster_threats()
        return {
            cid: self.graph.compute_cluster_risk(nodes)
            for cid, nodes in clusters.items()
        }
