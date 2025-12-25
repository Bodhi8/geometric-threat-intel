"""
Basic Usage Example for the Threat Plane Framework

This example demonstrates the core workflow:
1. Create synthetic security data
2. Build a Security Knowledge Graph
3. Generate embeddings using GNN
4. Project to 3D threat plane
5. Analyze topology
6. Visualize results
"""

import numpy as np
from threat_plane import ThreatPlane
from threat_plane.graph import SecurityKnowledgeGraph, Node, Edge, NodeType, EdgeType


def create_synthetic_data():
    """Create synthetic security data for demonstration."""
    
    # Create Security Knowledge Graph
    skg = SecurityKnowledgeGraph()
    
    # Add assets
    assets = [
        ("web-server-01", {"hostname": "web-server-01", "ip": "10.0.1.10", "internet_facing": True}),
        ("web-server-02", {"hostname": "web-server-02", "ip": "10.0.1.11", "internet_facing": True}),
        ("app-server-01", {"hostname": "app-server-01", "ip": "10.0.2.10", "internet_facing": False}),
        ("db-server-01", {"hostname": "db-server-01", "ip": "10.0.3.10", "internet_facing": False, "is_database": True}),
        ("dc-01", {"hostname": "dc-01", "ip": "10.0.4.10", "is_domain_controller": True}),
        ("workstation-01", {"hostname": "workstation-01", "ip": "10.0.5.10"}),
        ("workstation-02", {"hostname": "workstation-02", "ip": "10.0.5.11"}),
    ]
    
    for asset_id, attrs in assets:
        skg.add_node(Node(
            id=asset_id,
            node_type=NodeType.ASSET,
            attributes=attrs,
            criticality=0.8 if attrs.get("is_domain_controller") or attrs.get("is_database") else 0.5
        ))
    
    # Add vulnerabilities
    vulns = [
        ("CVE-2024-0001", {"cvss_score": 9.8, "severity": "critical", "affected_assets": ["web-server-01"]}),
        ("CVE-2024-0002", {"cvss_score": 7.5, "severity": "high", "affected_assets": ["web-server-02"]}),
        ("CVE-2024-0003", {"cvss_score": 8.1, "severity": "high", "affected_assets": ["app-server-01"]}),
        ("CVE-2024-0004", {"cvss_score": 6.5, "severity": "medium", "affected_assets": ["db-server-01"]}),
        ("CVE-2024-0005", {"cvss_score": 9.0, "severity": "critical", "affected_assets": ["dc-01"]}),
    ]
    
    for vuln_id, attrs in vulns:
        skg.add_node(Node(
            id=vuln_id,
            node_type=NodeType.VULNERABILITY,
            attributes=attrs,
            criticality=attrs["cvss_score"] / 10.0
        ))
        
        # Create EXPOSES edges
        for asset_id in attrs["affected_assets"]:
            skg.add_edge(Edge(
                source=asset_id,
                target=vuln_id,
                edge_type=EdgeType.EXPOSES,
                weight=attrs["cvss_score"] / 10.0
            ))
    
    # Add lateral movement paths
    lateral_moves = [
        ("web-server-01", "app-server-01"),
        ("web-server-02", "app-server-01"),
        ("app-server-01", "db-server-01"),
        ("app-server-01", "dc-01"),
        ("workstation-01", "dc-01"),
        ("workstation-02", "dc-01"),
    ]
    
    for source, target in lateral_moves:
        skg.add_edge(Edge(
            source=source,
            target=target,
            edge_type=EdgeType.LATERAL_MOVE,
            weight=0.7
        ))
    
    return skg


def main():
    print("=" * 60)
    print("Threat Plane Framework - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create synthetic data
    print("\n[1] Creating synthetic security data...")
    skg = create_synthetic_data()
    print(f"    Created graph with {len(skg.nodes)} nodes and {len(skg.edges)} edges")
    
    # Step 2: Compute attack paths
    print("\n[2] Computing attack paths...")
    skg.compute_attack_paths(max_depth=5)
    critical_paths = skg.get_critical_paths(top_k=5)
    print(f"    Found {len(skg._attack_paths)} total paths")
    print("    Top 5 critical paths:")
    for path, score in critical_paths:
        print(f"      {' → '.join(path)} (risk: {score:.2f})")
    
    # Step 3: Generate embeddings (simplified - random for demo)
    print("\n[3] Generating node embeddings...")
    n_nodes = len(skg.nodes)
    embeddings = np.random.randn(n_nodes, 64)  # Placeholder
    print(f"    Generated {n_nodes} embeddings of dimension 64")
    
    # Step 4: Project to 3D
    print("\n[4] Projecting to 3D threat plane...")
    try:
        from threat_plane.projection import ThreatPlaneProjector
        projector = ThreatPlaneProjector(method="umap", n_neighbors=min(5, n_nodes-1))
        projection = projector.fit_transform(embeddings)
        print(f"    Projection shape: {projection.shape}")
    except ImportError:
        print("    UMAP not installed, using PCA instead...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projection = pca.fit_transform(embeddings)
        print(f"    Projection shape: {projection.shape}")
    
    # Step 5: Analyze topology
    print("\n[5] Analyzing topology...")
    try:
        from threat_plane.topology import TopologicalAnalyzer
        analyzer = TopologicalAnalyzer(projection)
        
        # Compute clusters
        clusters = analyzer.cluster(method="kmeans", n_clusters=3)
        print(f"    Found {len(clusters)} clusters")
        for cid, indices in clusters.items():
            print(f"      Cluster {cid}: {len(indices)} nodes")
        
        # Detect anomalies
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=min(3, n_nodes-1))
        anomaly_labels = lof.fit_predict(projection)
        n_anomalies = np.sum(anomaly_labels == -1)
        print(f"    Detected {n_anomalies} anomalous points")
        
    except Exception as e:
        print(f"    Topology analysis skipped: {e}")
    
    # Step 6: Visualize
    print("\n[6] Generating visualization...")
    try:
        from threat_plane.visualization import ThreatPlaneVisualizer
        
        # Create metadata for nodes
        node_metadata = [
            {"id": node.id, "type": node.node_type.value, "severity": node.criticality}
            for node in skg.nodes.values()
        ]
        
        viz = ThreatPlaneVisualizer(projection, skg, node_metadata=node_metadata)
        fig = viz.render_interactive()
        
        # Save to HTML
        output_path = "threat_plane_demo.html"
        fig.write_html(output_path)
        print(f"    Saved interactive visualization to {output_path}")
        
    except ImportError:
        print("    Plotly not installed, skipping visualization")
    except Exception as e:
        print(f"    Visualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
