import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';

// Threat Plane 3D Visualization
// An interactive exploration of the geometric threat landscape

export default function ThreatPlaneVisualization() {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const rotationRef = useRef({ x: 0.3, y: 0 });
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedCluster, setSelectedCluster] = useState('all');
  const [showAttackPaths, setShowAttackPaths] = useState(true);
  const [showTopology, setShowTopology] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [viewMode, setViewMode] = useState('default');
  const nodesRef = useRef([]);
  const pathsRef = useRef([]);
  const timeRef = useRef(0);

  // Generate synthetic threat data
  const threatData = useMemo(() => {
    const clusters = [
      { name: 'Network Infrastructure', color: '#ff6b6b', center: [-2, 0, -1], risk: 'critical' },
      { name: 'Identity & Access', color: '#4ecdc4', center: [2, 1, 0], risk: 'high' },
      { name: 'Application Layer', color: '#ffe66d', center: [0, -1.5, 2], risk: 'medium' },
      { name: 'Data Stores', color: '#95e1d3', center: [-1, 2, 1], risk: 'high' },
      { name: 'Endpoint Systems', color: '#dda0dd', center: [1.5, -0.5, -2], risk: 'critical' },
      { name: 'Cloud Services', color: '#87ceeb', center: [0, 0.5, 0], risk: 'medium' }
    ];

    const nodes = [];
    const seededRandom = (seed) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    clusters.forEach((cluster, ci) => {
      const nodeCount = 15 + Math.floor(seededRandom(ci * 100) * 20);
      for (let i = 0; i < nodeCount; i++) {
        const seed = ci * 1000 + i;
        const radius = 0.8 + seededRandom(seed) * 0.8;
        const theta = seededRandom(seed + 1) * Math.PI * 2;
        const phi = seededRandom(seed + 2) * Math.PI;
        
        nodes.push({
          id: `node-${ci}-${i}`,
          cluster: ci,
          clusterName: cluster.name,
          color: cluster.color,
          x: cluster.center[0] + radius * Math.sin(phi) * Math.cos(theta),
          y: cluster.center[1] + radius * Math.sin(phi) * Math.sin(theta),
          z: cluster.center[2] + radius * Math.cos(phi),
          size: 0.04 + seededRandom(seed + 3) * 0.06,
          severity: seededRandom(seed + 4),
          type: ['asset', 'vulnerability', 'threat', 'control'][Math.floor(seededRandom(seed + 5) * 4)],
          name: `${cluster.name.split(' ')[0]}-${['CVE', 'ASSET', 'CTRL', 'THR'][Math.floor(seededRandom(seed + 6) * 4)]}-${Math.floor(seededRandom(seed + 7) * 9999)}`
        });
      }
    });

    // Generate attack paths between clusters
    const paths = [];
    const pathPairs = [[0, 4], [4, 1], [1, 3], [2, 5], [5, 0], [3, 2], [0, 1], [4, 3]];
    pathPairs.forEach(([from, to], idx) => {
      const fromNodes = nodes.filter(n => n.cluster === from);
      const toNodes = nodes.filter(n => n.cluster === to);
      if (fromNodes.length && toNodes.length) {
        const startNode = fromNodes[Math.floor(seededRandom(idx * 50) * fromNodes.length)];
        const endNode = toNodes[Math.floor(seededRandom(idx * 51) * toNodes.length)];
        paths.push({
          id: `path-${idx}`,
          from: startNode,
          to: endNode,
          risk: seededRandom(idx * 52),
          name: `Attack Vector ${idx + 1}`
        });
      }
    });

    return { nodes, clusters, paths };
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, containerRef.current.clientWidth / containerRef.current.clientHeight, 0.1, 1000);
    camera.position.set(5, 3, 5);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0a0a1a, 1);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0xff6b6b, 1, 20);
    pointLight1.position.set(5, 5, 5);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x4ecdc4, 0.8, 20);
    pointLight2.position.set(-5, 3, -5);
    scene.add(pointLight2);

    // Grid helper (threat plane surface)
    const gridHelper = new THREE.GridHelper(8, 20, 0x1a1a3e, 0x1a1a3e);
    gridHelper.position.y = -2.5;
    scene.add(gridHelper);

    // Create nodes
    const nodeGeometry = new THREE.SphereGeometry(1, 16, 16);
    nodesRef.current = [];

    threatData.nodes.forEach(node => {
      const material = new THREE.MeshPhongMaterial({
        color: node.color,
        emissive: node.color,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.85
      });
      
      const mesh = new THREE.Mesh(nodeGeometry, material);
      mesh.position.set(node.x, node.y, node.z);
      mesh.scale.setScalar(node.size);
      mesh.userData = node;
      scene.add(mesh);
      nodesRef.current.push(mesh);
    });

    // Create attack paths
    pathsRef.current = [];
    threatData.paths.forEach(path => {
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(path.from.x, path.from.y, path.from.z),
        new THREE.Vector3(
          (path.from.x + path.to.x) / 2 + (Math.random() - 0.5) * 0.5,
          Math.max(path.from.y, path.to.y) + 0.5,
          (path.from.z + path.to.z) / 2 + (Math.random() - 0.5) * 0.5
        ),
        new THREE.Vector3(path.to.x, path.to.y, path.to.z)
      );
      
      const points = curve.getPoints(50);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      
      const material = new THREE.LineBasicMaterial({
        color: 0xff4444,
        transparent: true,
        opacity: 0.4,
        linewidth: 2
      });
      
      const line = new THREE.Line(geometry, material);
      line.userData = path;
      scene.add(line);
      pathsRef.current.push({ line, curve, path });
    });

    // Animate
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      timeRef.current += 0.01 * animationSpeed;

      // Rotate scene based on mouse
      rotationRef.current.y += 0.002 * animationSpeed;
      scene.rotation.y = rotationRef.current.y + mouseRef.current.x * 0.5;
      scene.rotation.x = rotationRef.current.x + mouseRef.current.y * 0.3;

      // Animate nodes
      nodesRef.current.forEach((mesh, i) => {
        const node = mesh.userData;
        const pulse = Math.sin(timeRef.current * 2 + i * 0.5) * 0.1 + 1;
        mesh.scale.setScalar(node.size * pulse);
        
        // Gentle floating motion
        mesh.position.y = node.y + Math.sin(timeRef.current + i) * 0.02;
      });

      // Animate attack path particles
      pathsRef.current.forEach((pathObj, i) => {
        const t = (timeRef.current * 0.3 + i * 0.2) % 1;
        pathObj.line.material.opacity = showAttackPaths ? 0.4 : 0;
      });

      renderer.render(scene, camera);
    };

    animate();

    // Mouse interaction
    const handleMouseMove = (e) => {
      const rect = containerRef.current.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      mouseRef.current.y = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    };

    containerRef.current.addEventListener('mousemove', handleMouseMove);

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener('resize', handleResize);
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [threatData, animationSpeed, showAttackPaths]);

  // Filter nodes by cluster
  useEffect(() => {
    nodesRef.current.forEach(mesh => {
      const visible = selectedCluster === 'all' || mesh.userData.cluster === parseInt(selectedCluster);
      mesh.visible = visible;
      if (mesh.material) {
        mesh.material.opacity = visible ? 0.85 : 0.1;
      }
    });
  }, [selectedCluster]);

  const stats = useMemo(() => {
    const critical = threatData.nodes.filter(n => n.severity > 0.8).length;
    const high = threatData.nodes.filter(n => n.severity > 0.6 && n.severity <= 0.8).length;
    const medium = threatData.nodes.filter(n => n.severity > 0.4 && n.severity <= 0.6).length;
    const low = threatData.nodes.filter(n => n.severity <= 0.4).length;
    return { critical, high, medium, low, total: threatData.nodes.length, paths: threatData.paths.length };
  }, [threatData]);

  return (
    <div style={{
      width: '100%',
      height: '100vh',
      background: 'linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%)',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      color: '#e0e0e0',
      overflow: 'hidden',
      position: 'relative'
    }}>
      {/* Header */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        padding: '24px 32px',
        background: 'linear-gradient(180deg, rgba(10,10,26,0.95) 0%, transparent 100%)',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 style={{
              fontSize: '28px',
              fontWeight: 300,
              letterSpacing: '4px',
              margin: 0,
              color: '#fff',
              textTransform: 'uppercase'
            }}>
              <span style={{ color: '#ff6b6b' }}>◆</span> THREAT PLANE
            </h1>
            <p style={{
              fontSize: '11px',
              letterSpacing: '2px',
              color: '#6b6b8d',
              margin: '8px 0 0 0',
              textTransform: 'uppercase'
            }}>
              Geometric Risk Surface Visualization | Real-time Manifold Analysis
            </p>
          </div>
          <div style={{
            display: 'flex',
            gap: '24px',
            fontSize: '11px',
            letterSpacing: '1px'
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: '#ff6b6b', fontSize: '24px', fontWeight: 600 }}>{stats.critical}</div>
              <div style={{ color: '#6b6b8d', textTransform: 'uppercase' }}>Critical</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: '#ffe66d', fontSize: '24px', fontWeight: 600 }}>{stats.high}</div>
              <div style={{ color: '#6b6b8d', textTransform: 'uppercase' }}>High</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: '#4ecdc4', fontSize: '24px', fontWeight: 600 }}>{stats.medium}</div>
              <div style={{ color: '#6b6b8d', textTransform: 'uppercase' }}>Medium</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: '#95e1d3', fontSize: '24px', fontWeight: 600 }}>{stats.low}</div>
              <div style={{ color: '#6b6b8d', textTransform: 'uppercase' }}>Low</div>
            </div>
          </div>
        </div>
      </div>

      {/* 3D Canvas Container */}
      <div 
        ref={containerRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          cursor: 'grab'
        }} 
      />

      {/* Left Panel - Cluster Legend */}
      <div style={{
        position: 'absolute',
        left: '24px',
        top: '50%',
        transform: 'translateY(-50%)',
        background: 'rgba(10,10,26,0.85)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '8px',
        padding: '20px',
        zIndex: 10,
        minWidth: '200px'
      }}>
        <div style={{
          fontSize: '10px',
          letterSpacing: '2px',
          color: '#6b6b8d',
          marginBottom: '16px',
          textTransform: 'uppercase'
        }}>
          Threat Clusters
        </div>
        <div 
          onClick={() => setSelectedCluster('all')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            padding: '8px 12px',
            marginBottom: '4px',
            borderRadius: '4px',
            cursor: 'pointer',
            background: selectedCluster === 'all' ? 'rgba(255,255,255,0.1)' : 'transparent',
            transition: 'background 0.2s'
          }}
        >
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #ff6b6b, #4ecdc4)'
          }} />
          <span style={{ fontSize: '12px' }}>All Clusters</span>
        </div>
        {threatData.clusters.map((cluster, i) => (
          <div 
            key={i}
            onClick={() => setSelectedCluster(i.toString())}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '8px 12px',
              marginBottom: '4px',
              borderRadius: '4px',
              cursor: 'pointer',
              background: selectedCluster === i.toString() ? 'rgba(255,255,255,0.1)' : 'transparent',
              transition: 'background 0.2s'
            }}
          >
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: cluster.color,
              boxShadow: `0 0 10px ${cluster.color}40`
            }} />
            <span style={{ fontSize: '12px' }}>{cluster.name}</span>
          </div>
        ))}
      </div>

      {/* Right Panel - Controls */}
      <div style={{
        position: 'absolute',
        right: '24px',
        top: '50%',
        transform: 'translateY(-50%)',
        background: 'rgba(10,10,26,0.85)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '8px',
        padding: '20px',
        zIndex: 10,
        minWidth: '180px'
      }}>
        <div style={{
          fontSize: '10px',
          letterSpacing: '2px',
          color: '#6b6b8d',
          marginBottom: '16px',
          textTransform: 'uppercase'
        }}>
          Visualization Controls
        </div>
        
        <label style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '10px',
          marginBottom: '12px',
          cursor: 'pointer',
          fontSize: '12px'
        }}>
          <input 
            type="checkbox" 
            checked={showAttackPaths}
            onChange={(e) => setShowAttackPaths(e.target.checked)}
            style={{ accentColor: '#ff6b6b' }}
          />
          Attack Paths
        </label>
        
        <label style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '10px',
          marginBottom: '16px',
          cursor: 'pointer',
          fontSize: '12px'
        }}>
          <input 
            type="checkbox" 
            checked={showTopology}
            onChange={(e) => setShowTopology(e.target.checked)}
            style={{ accentColor: '#4ecdc4' }}
          />
          Topology Overlay
        </label>

        <div style={{ marginBottom: '8px', fontSize: '11px', color: '#6b6b8d' }}>
          Animation Speed
        </div>
        <input 
          type="range" 
          min="0" 
          max="3" 
          step="0.5"
          value={animationSpeed}
          onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
          style={{ 
            width: '100%',
            accentColor: '#ffe66d'
          }}
        />

        <div style={{
          marginTop: '20px',
          paddingTop: '16px',
          borderTop: '1px solid rgba(255,255,255,0.1)',
          fontSize: '10px',
          color: '#6b6b8d'
        }}>
          <div style={{ marginBottom: '8px' }}>
            <span style={{ color: '#fff' }}>{stats.total}</span> nodes mapped
          </div>
          <div>
            <span style={{ color: '#ff6b6b' }}>{stats.paths}</span> attack vectors
          </div>
        </div>
      </div>

      {/* Bottom Info Bar */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: '16px 32px',
        background: 'linear-gradient(0deg, rgba(10,10,26,0.95) 0%, transparent 100%)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        zIndex: 10
      }}>
        <div style={{ 
          fontSize: '10px', 
          letterSpacing: '1px',
          color: '#6b6b8d'
        }}>
          PROJECTION: UMAP-3D | METRIC: SECURITY-WEIGHTED EUCLIDEAN | GNN: 4-LAYER HETEROGENEOUS GAT
        </div>
        <div style={{ 
          fontSize: '10px',
          letterSpacing: '1px',
          color: '#4ecdc4'
        }}>
          ◉ MANIFOLD STABLE | TOPOLOGY: H₀=6 H₁=3 H₂=1
        </div>
      </div>

      {/* Floating Equation */}
      <div style={{
        position: 'absolute',
        bottom: '80px',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(10,10,26,0.7)',
        backdropFilter: 'blur(5px)',
        border: '1px solid rgba(255,255,255,0.05)',
        borderRadius: '4px',
        padding: '8px 16px',
        fontSize: '12px',
        fontFamily: "'Cambria Math', 'Times New Roman', serif",
        fontStyle: 'italic',
        color: '#8b8bab',
        zIndex: 10
      }}>
        π: M ⊂ ℝⁿ → T ⊂ ℝ³ | d_T(π(x), π(y)) ≈ d_M(x, y)
      </div>
    </div>
  );
}
