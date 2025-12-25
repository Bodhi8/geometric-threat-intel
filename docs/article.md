# The Threat Plane

## A Geometric Framework for Modeling and Visualizing Organizational Cyber Risk in Three-Dimensional Space

*A research methodology for data scientists, security researchers, and practitioners seeking to advance threat modeling through topological and machine learning approaches.*

---

**Abstract**: This paper introduces the Threat Plane—a novel geometric framework that reconceptualizes organizational cyber risk as a navigable three-dimensional manifold. By synthesizing graph neural networks, topological data analysis, advanced dimensionality reduction techniques, and real-time telemetry integration, we propose a unified methodology for projecting the high-dimensional attack surface onto an interpretable 3D space. This representation enables security teams to identify emergent threat clusters, predict attack trajectories, and quantify defensive coverage gaps through spatial reasoning. We present both the theoretical foundations and practical implementation pathways for organizations seeking to operationalize geometric threat intelligence.

---

## 1. Introduction: Beyond Flat Threat Models

Traditional threat modeling approaches—from STRIDE to attack trees to kill chains—share a fundamental limitation: they represent inherently multidimensional risk landscapes as linear sequences or hierarchical structures. While these frameworks have served the security community well, they struggle to capture the complex interdependencies, emergent behaviors, and dynamic evolution that characterize modern threat environments. An adversary does not traverse a tree; they navigate a manifold.

The concept of the **Threat Plane** emerges from a simple but powerful observation: if we can embed the entirety of an organization's exposed risk surface into a geometric space, we unlock an entirely new vocabulary for threat analysis. Proximity becomes vulnerability correlation. Curvature indicates attack amplification potential. Geodesics trace optimal adversary paths. Voids represent defensive coverage. The threat landscape becomes literally navigable.

This paper synthesizes advances across multiple domains—graph representation learning, topological data analysis, information geometry, and interactive visualization—to propose a comprehensive framework for constructing, analyzing, and operationalizing threat planes. We move beyond visualization as an afterthought and instead position geometric representation as the foundational analytical primitive.

---

## 2. Theoretical Foundations

### 2.1 The Threat Manifold Hypothesis

We begin with a foundational claim: the space of all possible security states for an organization forms a high-dimensional manifold **M** embedded in ℝⁿ, where *n* corresponds to the dimensionality of the feature space encoding assets, vulnerabilities, configurations, and threat intelligence. The **Threat Plane** is then a lower-dimensional projection *π: M → ℝ³* that preserves essential topological and geometric properties relevant to security analysis.

```
π: M ⊂ ℝⁿ → T ⊂ ℝ³  subject to  d_T(π(x), π(y)) ≈ d_M(x, y)
```

This preservation constraint is critical. We require that the projection approximately maintains geodesic distances on the manifold—points that are "close" in security-relevant terms should remain proximate in the threat plane. This is not merely a visualization convenience; it is the foundation that allows spatial reasoning to yield valid security insights.

### 2.2 Information-Theoretic Grounding

The threat plane can be formalized through the lens of information geometry. Each point in the space represents a probability distribution over possible attack outcomes given the current security state. The Fisher information metric induces a natural Riemannian structure on this space, and the threat plane becomes a 3D slice through this statistical manifold.

Formally, let *p(attack | state)* denote the conditional probability of attack success given a security configuration. The Fisher information matrix **G** defines local distances in the state space, and our projection must respect this metric structure to ensure that similar risk profiles cluster together in the visualization.

```
G_ij(θ) = E[∂ log p(x|θ)/∂θ_i · ∂ log p(x|θ)/∂θ_j]
```

### 2.3 Topological Considerations

Beyond metric preservation, we care deeply about topological invariants. The Betti numbers of the threat manifold—counts of connected components, loops, and voids—carry security meaning:

| Betti Number | Topological Feature | Security Interpretation |
|--------------|---------------------|------------------------|
| **H₀** | Connected Components | Clusters of correlated risks |
| **H₁** | Loops / Cycles | Cyclic attack dependencies |
| **H₂** | Voids / Cavities | Regions of defensive coverage |

Persistent homology provides the mathematical machinery to extract these features across multiple scales. By constructing a filtration of simplicial complexes from our threat data and tracking the birth and death of topological features, we obtain a multi-scale signature of the threat landscape's structure.

---

## 3. Constructing the Threat Graph

### 3.1 Heterogeneous Security Knowledge Graphs

The threat plane is not constructed directly from raw telemetry but from a structured representation we call the **Security Knowledge Graph (SKG)**. This heterogeneous graph integrates multiple entity types and relationship classes into a unified representation.

**Node Types:**
- Assets (hosts, services, applications, data stores)
- Vulnerabilities (CVEs, misconfigurations, weak credentials)
- Threat Actors (APT groups, criminal organizations, insider threats)
- Techniques (MITRE ATT&CK tactics and techniques)
- Controls (firewalls, EDR, access policies)
- Identity entities (users, service accounts, roles)

**Edge Types:**
- EXPOSES (asset → vulnerability)
- EXPLOITS (technique → vulnerability)
- USES (actor → technique)
- TARGETS (actor → asset)
- PROTECTS (control → asset)
- MITIGATES (control → vulnerability)
- AUTHENTICATES_AS (identity → asset)
- LATERAL_MOVE (asset → asset)
- DATA_FLOW (asset → asset)

This graph structure captures not just what vulnerabilities exist but how they interconnect, which actors might exploit them, and what defensive measures are in place.

### 3.2 Temporal Attack Graphs

Static graphs miss the dynamic nature of threat evolution. We augment the SKG with temporal edges that encode *attack progression probability*—the likelihood that compromise of node *u* leads to compromise of node *v* within time window *Δt*. These probabilities are learned from historical incident data, red team exercises, and threat intelligence feeds.

```
P(v_t+Δt | u_t) = σ(f_θ(h_u, h_v, e_uv, Δt))
```

Where *h_u* and *h_v* are node embeddings, *e_uv* is the edge feature vector, and *f_θ* is a learned scoring function.

---

## 4. Embedding into the Threat Plane

### 4.1 Graph Neural Network Architectures

Transforming the security knowledge graph into the threat plane requires learning meaningful node embeddings that capture both local neighborhood structure and global graph topology. We employ a multi-layer Graph Attention Network (GAT) architecture enhanced with several innovations for the security domain.

**Heterogeneous Message Passing:** Standard GNNs assume homogeneous node and edge types. Our architecture uses relation-specific transformation matrices *W_r* for each edge type *r*, allowing the network to learn distinct propagation patterns for different relationship semantics.

```
h_v^(l+1) = σ(Σ_{r∈R} Σ_{u∈N_r(v)} α_uv^r W_r^(l) h_u^(l))
```

**Risk-Aware Attention:** Attention coefficients are modulated by node criticality scores derived from asset valuation and vulnerability severity. High-value targets receive amplified attention during message aggregation.

**Temporal Positional Encoding:** For temporal attack graphs, we incorporate sinusoidal positional encodings based on edge timestamps, enabling the network to distinguish between historical and recent relationships.

### 4.2 Advanced Dimensionality Reduction

With node embeddings in hand (typically 64-256 dimensions), we project to 3D using techniques that balance local and global structure preservation.

**UMAP with Custom Metrics:** Uniform Manifold Approximation and Projection (UMAP) serves as our primary projection engine, configured with security-specific distance metrics. Rather than Euclidean distance in embedding space, we use a composite metric that weights dimensions by their contribution to attack path feasibility.

```
d_security(u, v) = √(Σ_i w_i(u_i - v_i)²)  where  w_i = I(dim_i ∈ attack_relevant)
```

**t-SNE for Local Cluster Refinement:** After UMAP projection, we apply localized t-SNE optimization to refine cluster boundaries.

**PaCMAP for Balanced Preservation:** Pairwise Controlled Manifold Approximation (PaCMAP) offers excellent balance between local and global structure through its three-stage optimization.

### 4.3 Preserving Attack Path Geodesics

A critical requirement is that attack paths in the original graph correspond to navigable trajectories in the threat plane. We enforce this through a geodesic regularization term:

```
L_geo = Σ_{paths p} ||d_T(π(p_start), π(p_end)) - Σ_{edges e ∈ p} cost(e)||²
```

---

## 5. Topological Analysis of the Threat Plane

### 5.1 Persistent Homology for Threat Detection

Once embedded in 3D, we apply persistent homology to extract topological features at multiple scales. The Vietoris-Rips complex built from threat plane points yields persistence diagrams that encode the lifespan of topological features.

**H₀ (Connected Components):** Long-lived components represent distinct threat clusters—groups of assets, vulnerabilities, and attack paths that form isolated risk domains.

**H₁ (Loops):** Persistent one-dimensional holes indicate cyclic attack dependencies—situations where compromise of A enables compromise of B, which enables compromise of C, which in turn facilitates compromise of A. These feedback loops warrant priority remediation.

**H₂ (Voids):** Two-dimensional cavities represent regions of defensive coverage—configurations of controls that create protected zones within the risk space.

### 5.2 Mapper Algorithm for Structure Discovery

The Mapper algorithm provides a complementary lens on threat plane structure. By covering the plane with overlapping regions, clustering within each region, and connecting clusters that share points, Mapper produces a simplified graph that captures the essential skeleton of the threat landscape.

---

## 6. Machine Learning on the Threat Plane

### 6.1 Anomaly Detection in Geometric Space

**Local Outlier Factor (LOF):** Points with low local density indicate unusual risk configurations—potentially novel attack vectors or emerging threat patterns.

**Isolation Forest on 3D Coordinates:** Random partitioning efficiently identifies outliers as points requiring fewer splits to isolate.

**Geodesic Anomaly Scoring:** Points that are Euclidean-close but geodesically-distant indicate threat bridges—unexpected connections between separated risk domains.

### 6.2 Attack Trajectory Prediction

**Recurrent Neural ODE:** We model adversary movement as a continuous-time dynamical system, where a neural network parameterizes the velocity field on the threat plane.

```
dz/dt = f_θ(z, t, context)  where  z ∈ ℝ³ is the threat plane position
```

**Transformer-based Sequence Modeling:** For discrete attack steps, we tokenize threat plane regions and train a transformer to predict the next region given the attack history.

### 6.3 Clustering for Threat Family Discovery

**HDBSCAN for Variable-Density Clusters:** Handles dense cores surrounded by sparser peripheries while identifying noise points.

**Spectral Clustering on the Threat Graph:** Applied to the graph Laplacian, with results projected onto the threat plane to validate cluster coherence.

---

## 7. Novel Advances: Pushing the State of the Art

### 7.1 Dynamic Threat Plane Evolution

**Anchor-Based Incremental Embedding:** Stable, high-criticality nodes serve as anchors with fixed positions. New nodes are embedded relative to these anchors.

**Temporal Morphing:** Optimal transport theory provides the framework for computing minimal-cost deformations between successive states, enabling smooth animations.

### 7.2 Hyperbolic Embeddings for Hierarchical Threats

When the security knowledge graph exhibits strong hierarchical structure, hyperbolic geometry offers superior embedding fidelity. The Poincaré ball model embeds hierarchical trees with exponentially growing capacity near the boundary.

### 7.3 Multi-Modal Threat Fusion

A **Multi-Modal Contrastive Learning** framework aligns representations across modalities—structured vulnerability scans, unstructured threat intelligence reports, network flow telemetry—into a unified threat plane.

### 7.4 Causal Structure Discovery

The PC algorithm and its variants, applied to observational data from security telemetry, identify candidate causal relationships. Causal edges are overlaid on the threat plane as directed arrows, distinguishing correlation from true attack enablement.

### 7.5 Adversarial Robustness of the Threat Plane

**Certified Robust Embeddings:** Using randomized smoothing, we bound the maximum displacement of any point under bounded input perturbations.

**Ensemble Embedding:** Multiple threat planes constructed with different configurations. Disagreement triggers human review.

---

## 8. Practical Implementation Framework

### 8.1 Data Pipeline Architecture

| Layer | Function | Technologies |
|-------|----------|--------------|
| **Ingestion** | Streaming telemetry | Apache Kafka |
| **Graph Store** | Knowledge graph | Neo4j, TigerGraph |
| **Compute** | GNN training | PyTorch Geometric, DGL |
| **Projection** | Dimensionality reduction | cuML (GPU-accelerated) |
| **Visualization** | 3D rendering | Three.js, WebGL |

### 8.2 Integration with Security Workflows

**Vulnerability Prioritization:** Identify critical junctions where multiple attack paths converge.

**Threat Hunting:** Use the threat plane to guide hypothesis-driven hunting along predicted trajectories.

**Red Team Planning:** Plan engagement strategies by identifying high-value paths through the risk landscape.

---

## 9. Conclusion: Toward Geometric Threat Intelligence

The Threat Plane represents a paradigm shift in how we conceptualize and interact with organizational cyber risk. By embedding the high-dimensional attack surface into a navigable three-dimensional space, we transform threat analysis from symbolic reasoning about discrete entities into spatial reasoning about continuous landscapes.

This geometric perspective unlocks new analytical capabilities: topological analysis reveals structural vulnerabilities invisible to traditional methods, machine learning on the manifold enables predictive threat intelligence, and interactive visualization makes the risk landscape accessible to diverse stakeholders.

As threat landscapes grow more complex and adversaries more sophisticated, geometric intuition will become an increasingly valuable tool in the defender's arsenal. The threat plane offers a path toward that future—a future where we don't just list our risks, but truly understand the shape of our exposure.

---

*The code, datasets, and interactive demos are available in this repository. We invite the security research community to build upon this foundation and explore the vast uncharted territory of geometric threat intelligence.*
