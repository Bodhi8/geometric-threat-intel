"""
Visualization module for the Threat Plane.

Provides:
- Interactive 3D visualizations using Plotly
- Static exports for reports
- HTML exports for sharing
- Integration with Three.js for web deployment
"""

import numpy as np
from typing import Dict, List, Optional
import json


class ThreatPlaneVisualizer:
    """
    3D visualization of the threat plane.
    
    Renders the threat landscape with:
    - Color-coded threat clusters
    - Attack path trajectories
    - Topology overlays (loops, voids)
    - Interactive exploration
    
    Example:
        >>> viz = ThreatPlaneVisualizer(projection, graph, topology)
        >>> viz.render_interactive()
        >>> viz.export_html("threat_plane.html")
    """
    
    def __init__(
        self,
        projection: np.ndarray,
        graph=None,
        topology_features: Optional[Dict] = None,
        labels: Optional[np.ndarray] = None,
        node_metadata: Optional[List[Dict]] = None
    ):
        self.projection = projection
        self.graph = graph
        self.topology_features = topology_features or {}
        self.labels = labels
        self.node_metadata = node_metadata or [{}] * len(projection)
        
        self._colors = self._generate_colors()
    
    def render_interactive(self, **kwargs):
        """
        Render an interactive 3D visualization.
        
        Uses Plotly for in-notebook/browser visualization.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly required: pip install plotly")
        
        # Create traces for nodes
        node_trace = go.Scatter3d(
            x=self.projection[:, 0],
            y=self.projection[:, 1],
            z=self.projection[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=self._colors,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=self._get_hover_text(),
            hoverinfo='text'
        )
        
        traces = [node_trace]
        
        # Add attack paths if available
        if self.graph and hasattr(self.graph, '_attack_paths'):
            path_traces = self._create_path_traces()
            traces.extend(path_traces)
        
        # Create figure
        fig = go.Figure(data=traces)
        
        fig.update_layout(
            title="Threat Plane Visualization",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
                bgcolor='rgb(10, 10, 26)',
                xaxis=dict(gridcolor='rgb(50, 50, 80)', zerolinecolor='rgb(50, 50, 80)'),
                yaxis=dict(gridcolor='rgb(50, 50, 80)', zerolinecolor='rgb(50, 50, 80)'),
                zaxis=dict(gridcolor='rgb(50, 50, 80)', zerolinecolor='rgb(50, 50, 80)'),
            ),
            paper_bgcolor='rgb(10, 10, 26)',
            font=dict(color='white'),
            showlegend=True,
            **kwargs
        )
        
        return fig
    
    def render_static(self, filepath: str = "threat_plane.png", **kwargs):
        """Export static image of the threat plane."""
        fig = self.render_interactive(**kwargs)
        fig.write_image(filepath)
        return filepath
    
    def export_html(self, filepath: str = "threat_plane.html"):
        """Export as standalone HTML file."""
        fig = self.render_interactive()
        fig.write_html(filepath, include_plotlyjs=True)
        return filepath
    
    def export_threejs_json(self, filepath: str = "threat_data.json"):
        """
        Export data for Three.js visualization.
        
        Creates a JSON file that can be loaded by the
        interactive Three.js demo.
        """
        data = {
            "nodes": [],
            "edges": [],
            "clusters": [],
            "attackPaths": []
        }
        
        # Export nodes
        for i, (pos, meta) in enumerate(zip(self.projection, self.node_metadata)):
            data["nodes"].append({
                "id": meta.get("id", f"node_{i}"),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "cluster": int(self.labels[i]) if self.labels is not None else 0,
                "type": meta.get("type", "unknown"),
                "name": meta.get("name", f"Node {i}"),
                "severity": float(meta.get("severity", 0.5))
            })
        
        # Export edges
        if self.graph:
            for edge in self.graph.edges[:100]:  # Limit for performance
                data["edges"].append({
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type.value
                })
        
        # Export cluster info
        if self.labels is not None:
            for cluster_id in np.unique(self.labels):
                if cluster_id == -1:
                    continue
                mask = self.labels == cluster_id
                center = np.mean(self.projection[mask], axis=0)
                data["clusters"].append({
                    "id": int(cluster_id),
                    "center": center.tolist(),
                    "size": int(np.sum(mask)),
                    "color": self._cluster_colors.get(cluster_id, "#888888")
                })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def _generate_colors(self) -> np.ndarray:
        """Generate colors based on labels or severity."""
        if self.labels is not None:
            # Color by cluster
            unique_labels = np.unique(self.labels)
            color_map = {
                label: f'hsl({int(i * 360 / len(unique_labels))}, 70%, 50%)'
                for i, label in enumerate(unique_labels)
            }
            return [color_map.get(l, 'gray') for l in self.labels]
        else:
            # Color by severity from metadata
            severities = [m.get('severity', 0.5) for m in self.node_metadata]
            return severities
    
    def _get_hover_text(self) -> List[str]:
        """Generate hover text for each node."""
        texts = []
        for i, meta in enumerate(self.node_metadata):
            text = f"ID: {meta.get('id', i)}<br>"
            text += f"Type: {meta.get('type', 'unknown')}<br>"
            text += f"Name: {meta.get('name', f'Node {i}')}<br>"
            if 'severity' in meta:
                text += f"Severity: {meta['severity']:.2f}"
            texts.append(text)
        return texts
    
    def _create_path_traces(self):
        """Create traces for attack paths."""
        import plotly.graph_objects as go
        
        traces = []
        # Simplified: would need node ID to index mapping
        return traces
    
    @property
    def _cluster_colors(self) -> Dict[int, str]:
        """Default cluster color palette."""
        palette = [
            '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3',
            '#dda0dd', '#87ceeb', '#98fb98', '#ffa07a'
        ]
        if self.labels is None:
            return {}
        return {
            label: palette[i % len(palette)]
            for i, label in enumerate(np.unique(self.labels))
            if label != -1
        }
