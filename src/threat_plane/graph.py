"""
Security Knowledge Graph module.

Builds and maintains the heterogeneous graph representation of 
organizational security state including assets, vulnerabilities,
threat actors, techniques, and controls.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class NodeType(Enum):
    ASSET = "asset"
    VULNERABILITY = "vulnerability"
    THREAT_ACTOR = "threat_actor"
    TECHNIQUE = "technique"
    CONTROL = "control"
    IDENTITY = "identity"


class EdgeType(Enum):
    EXPOSES = "exposes"
    EXPLOITS = "exploits"
    USES = "uses"
    TARGETS = "targets"
    PROTECTS = "protects"
    MITIGATES = "mitigates"
    AUTHENTICATES_AS = "authenticates_as"
    LATERAL_MOVE = "lateral_move"
    DATA_FLOW = "data_flow"


@dataclass
class Node:
    """Represents a node in the Security Knowledge Graph."""
    id: str
    node_type: NodeType
    attributes: Dict = field(default_factory=dict)
    criticality: float = 0.5
    
    def __hash__(self):
        return hash(self.id)


@dataclass 
class Edge:
    """Represents an edge in the Security Knowledge Graph."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    temporal: Optional[float] = None  # Timestamp if temporal edge
    attributes: Dict = field(default_factory=dict)


class SecurityKnowledgeGraph:
    """
    Heterogeneous graph representing organizational security state.
    
    Integrates multiple data sources into a unified graph structure
    with typed nodes and edges for security analysis.
    
    Example:
        >>> skg = SecurityKnowledgeGraph()
        >>> skg.ingest_vulnerabilities("vuln_scan.json")
        >>> skg.ingest_assets("cmdb.csv")
        >>> skg.compute_attack_paths()
        >>> paths = skg.get_critical_paths(top_k=10)
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, Set[str]] = {}
        self._attack_paths: List[List[str]] = []
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = set()
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        if edge.source not in self.adjacency:
            self.adjacency[edge.source] = set()
        self.adjacency[edge.source].add(edge.target)
    
    def ingest_vulnerabilities(self, path: str, format: str = "json") -> None:
        """
        Ingest vulnerability scan data.
        
        Supports:
        - Qualys, Tenable, Rapid7 JSON exports
        - NVD CVE JSON feeds
        - Custom JSON/CSV formats
        """
        if format == "json":
            with open(path) as f:
                data = json.load(f)
            
            for vuln in data.get("vulnerabilities", data):
                node = Node(
                    id=vuln.get("cve_id", vuln.get("id")),
                    node_type=NodeType.VULNERABILITY,
                    attributes={
                        "cvss": vuln.get("cvss_score", 0),
                        "severity": vuln.get("severity", "unknown"),
                        "description": vuln.get("description", ""),
                        "affected_assets": vuln.get("affected_assets", []),
                    },
                    criticality=vuln.get("cvss_score", 5.0) / 10.0
                )
                self.add_node(node)
                
                # Create EXPOSES edges to affected assets
                for asset_id in vuln.get("affected_assets", []):
                    self.add_edge(Edge(
                        source=asset_id,
                        target=node.id,
                        edge_type=EdgeType.EXPOSES,
                        weight=node.criticality
                    ))
    
    def ingest_assets(self, path: str, format: str = "csv") -> None:
        """
        Ingest asset inventory data.
        
        Supports:
        - ServiceNow CMDB exports
        - Axonius asset data
        - Custom CSV/JSON formats
        """
        import csv
        
        if format == "csv":
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node = Node(
                        id=row.get("asset_id", row.get("hostname")),
                        node_type=NodeType.ASSET,
                        attributes={
                            "hostname": row.get("hostname"),
                            "ip": row.get("ip_address"),
                            "os": row.get("operating_system"),
                            "owner": row.get("owner"),
                            "business_unit": row.get("business_unit"),
                            "data_classification": row.get("data_classification"),
                        },
                        criticality=self._compute_asset_criticality(row)
                    )
                    self.add_node(node)
    
    def ingest_threat_intel(self, path: str, format: str = "stix") -> None:
        """
        Ingest threat intelligence feeds.
        
        Supports:
        - STIX 2.1 bundles
        - MISP events
        - OpenCTI exports
        - ATT&CK Navigator layers
        """
        # Implementation would parse threat intel and create
        # THREAT_ACTOR and TECHNIQUE nodes with relationships
        pass
    
    def ingest_controls(self, path: str) -> None:
        """Ingest security control data (firewalls, EDR, policies)."""
        pass
    
    def resolve_entities(self) -> None:
        """
        Perform entity resolution to merge duplicate nodes.
        
        Uses fuzzy matching on hostnames, IPs, and identifiers
        to consolidate entities that refer to the same real-world object.
        """
        # Implementation would use similarity hashing and
        # graph-based entity resolution
        pass
    
    def infer_relationships(self) -> None:
        """
        Infer implicit relationships from observed patterns.
        
        Examples:
        - Network connectivity from flow data
        - Lateral movement paths from authentication logs
        - Data flows from DLP events
        """
        pass
    
    def compute_attack_paths(self, max_depth: int = 10) -> None:
        """
        Compute all viable attack paths through the graph.
        
        Uses weighted graph traversal considering:
        - Vulnerability exploitability
        - Asset criticality
        - Control effectiveness
        - Network reachability
        """
        # Find entry points (internet-facing assets with vulns)
        entry_points = self._find_entry_points()
        
        # Find crown jewels (high-value targets)
        targets = self._find_high_value_targets()
        
        # Compute paths using modified Dijkstra
        self._attack_paths = []
        for entry in entry_points:
            for target in targets:
                paths = self._find_paths(entry, target, max_depth)
                self._attack_paths.extend(paths)
    
    def get_critical_paths(self, top_k: int = 10) -> List[Tuple[List[str], float]]:
        """Return the top-k highest risk attack paths."""
        scored_paths = [
            (path, self._score_path(path))
            for path in self._attack_paths
        ]
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths[:top_k]
    
    def get_paths_through_nodes(self, node_ids: Set[str]) -> List[List[str]]:
        """Get all attack paths passing through specified nodes."""
        return [
            path for path in self._attack_paths
            if any(node in path for node in node_ids)
        ]
    
    def compute_cluster_risk(self, node_ids: Set[str]) -> float:
        """Compute aggregate risk score for a cluster of nodes."""
        if not node_ids:
            return 0.0
        
        # Aggregate criticality of nodes
        criticalities = [
            self.nodes[nid].criticality 
            for nid in node_ids 
            if nid in self.nodes
        ]
        
        # Consider paths through cluster
        paths_through = self.get_paths_through_nodes(node_ids)
        path_risk = sum(self._score_path(p) for p in paths_through)
        
        return sum(criticalities) + path_risk * 0.1
    
    def to_pyg_data(self):
        """Convert to PyTorch Geometric HeteroData format."""
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError("PyTorch Geometric required: pip install torch-geometric")
        
        data = HeteroData()
        
        # Group nodes by type
        node_type_indices = {}
        for node_type in NodeType:
            nodes_of_type = [n for n in self.nodes.values() if n.node_type == node_type]
            if nodes_of_type:
                node_type_indices[node_type] = {n.id: i for i, n in enumerate(nodes_of_type)}
                # Create feature tensor (placeholder - implement actual features)
                data[node_type.value].x = torch.randn(len(nodes_of_type), 64)
        
        # Create edge indices by type
        for edge_type in EdgeType:
            edges_of_type = [e for e in self.edges if e.edge_type == edge_type]
            if edges_of_type:
                # Determine source/target node types and create edge_index
                # This is simplified - real implementation needs type inference
                pass
        
        return data
    
    def _compute_asset_criticality(self, asset_data: Dict) -> float:
        """Compute criticality score for an asset."""
        score = 0.5  # Base score
        
        # Adjust based on data classification
        classification = asset_data.get("data_classification", "").lower()
        if "confidential" in classification or "pii" in classification:
            score += 0.3
        if "public" in classification:
            score -= 0.2
        
        # Adjust based on role
        if asset_data.get("is_domain_controller"):
            score += 0.4
        if asset_data.get("is_database"):
            score += 0.2
            
        return min(1.0, max(0.0, score))
    
    def _find_entry_points(self) -> List[str]:
        """Find potential attack entry points."""
        return [
            node.id for node in self.nodes.values()
            if node.node_type == NodeType.ASSET
            and node.attributes.get("internet_facing", False)
        ]
    
    def _find_high_value_targets(self) -> List[str]:
        """Find high-value target assets."""
        return [
            node.id for node in self.nodes.values()
            if node.node_type == NodeType.ASSET
            and node.criticality > 0.7
        ]
    
    def _find_paths(self, start: str, end: str, max_depth: int) -> List[List[str]]:
        """Find all paths between two nodes up to max_depth."""
        paths = []
        stack = [(start, [start])]
        
        while stack:
            node, path = stack.pop()
            if len(path) > max_depth:
                continue
            if node == end:
                paths.append(path)
                continue
            
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        
        return paths
    
    def _score_path(self, path: List[str]) -> float:
        """Score an attack path by aggregating risk factors."""
        if not path:
            return 0.0
        
        score = 0.0
        for node_id in path:
            if node_id in self.nodes:
                score += self.nodes[node_id].criticality
        
        # Shorter paths are more dangerous
        length_penalty = 1.0 / (1.0 + len(path) * 0.1)
        
        return score * length_penalty
