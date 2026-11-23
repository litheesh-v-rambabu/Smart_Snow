"""
road_graph.py
=============
Builds the Buffalo road network graph.
- Primary path: real OSMnx data (requires `pip install osmnx`)
- Fallback path: synthetic grid graph that approximates Buffalo's topology
                 (used for testing and offline development)

Priority classes mirror real OSM highway attribute values:
    5 = motorway / trunk   (highway, interstate)
    4 = primary            (main arterials)
    3 = secondary          (collector roads)
    2 = tertiary           (local connectors)
    1 = residential        (neighbourhood streets)
"""

import os
import pickle
import networkx as nx
import numpy as np

# Optional: only needed for real OSM download
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

CACHE_PATH = os.path.join(os.path.dirname(__file__), "../data/buffalo_graph.pkl")

# ---------------------------------------------------------------------------
# Synthetic grid priority design (edge-based, not node-based)
# ---------------------------------------------------------------------------
# For a 10×10 grid (rows 0-9, cols 0-9), roads are classified by position
# to mirror Buffalo's real street hierarchy.  Target distribution over 180 edges:
#
#   P5 highway / hospital  ~10%  18 edges  — main corridors (row 5, col 5)
#   P4 primary             ~10%  18 edges  — arterials      (row 0, col 0)
#   P3 secondary           ~10%  18 edges  — collector rds  (row 3, row 7)
#   P2 tertiary            ~10%  18 edges  — local connectors (col 3, col 7)
#   P1 residential         ~60%  108 edges — all other streets
#
# Assignment rules (checked in priority order):
#   Horizontal edge (same row):
#       row 5        → P5  (I-190 / Niagara corridor)
#       row 0        → P4  (north arterial)
#       row 3, 7     → P3  (secondary collectors)
#       all others   → P1
#   Vertical edge (same col):
#       col 5        → P5  (Main St / hospital spine)
#       col 0        → P4  (west arterial / Elmwood Ave)
#       col 3, 7     → P2  (tertiary connectors)
#       all others   → P1
# ---------------------------------------------------------------------------

# Row/col rules — evaluated top-to-bottom, first match wins
_H_PRIORITY = {5: 5, 0: 4, 3: 3, 7: 3}   # horizontal: row → priority
_V_PRIORITY = {5: 5, 0: 4, 3: 2, 7: 2}   # vertical:   col → priority


def _assign_priority(highway_type):
    """Map OSM highway attribute to integer priority class (1–5)."""
    if isinstance(highway_type, list):
        highway_type = highway_type[0]
    mapping = {
        "motorway": 5, "motorway_link": 5,
        "trunk": 5,    "trunk_link": 5,
        "primary": 4,  "primary_link": 4,
        "secondary": 3,"secondary_link": 3,
        "tertiary": 2, "tertiary_link": 2,
    }
    return mapping.get(str(highway_type), 1)


def build_synthetic_graph(n_nodes=100, seed=42):
    """
    Build a synthetic grid graph approximating a small Buffalo sub-network.
    Used when OSMnx is unavailable or for fast unit testing.

    Nodes carry:
        x, y   : 2-D position (metres, UTM-like)

    Edges carry:
        length         : road segment length in metres
        priority_class : 1–5 integer
        snow_depth     : float (inches), starts at 0
        steps_uncleared: int counter for urgency-decay reward
        traffic_density: float 0–1
    """
    rng = np.random.default_rng(seed)
    rows = cols = int(np.ceil(np.sqrt(n_nodes)))

    G = nx.grid_2d_graph(rows, cols)
    # Relabel to integers
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Assign (x, y) positions
    for i, (r, c) in enumerate(nx.grid_2d_graph(rows, cols).nodes()):
        G.nodes[i]["x"] = c * 200.0   # 200 m spacing
        G.nodes[i]["y"] = r * 200.0

    # Assign edge attributes
    for u, v in G.edges():
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        length = np.hypot(xv - xu, yv - yu)

        # ── Edge-based priority assignment ────────────────────────
        # Recover (row, col) from node integer index
        ru, cu = divmod(u, cols)
        rv, cv = divmod(v, cols)

        if ru == rv:
            # Horizontal edge — classified by row
            priority = _H_PRIORITY.get(ru, 1)
        else:
            # Vertical edge — classified by column
            # (cu == cv for a valid grid vertical edge)
            priority = _V_PRIORITY.get(cu, 1)

        G[u][v]["length"] = length
        G[u][v]["priority_class"] = priority
        G[u][v]["snow_depth"] = 0.0
        G[u][v]["steps_uncleared"] = 0
        G[u][v]["traffic_density"] = 0.0

    # Keep only nodes that exist after grid construction
    G = G.subgraph(list(range(min(n_nodes, rows * cols)))).copy()
    return G


def load_osmnx_graph(place="Buffalo, New York, USA",
                     network_type="drive",
                     cache_path=CACHE_PATH):
    """
    Load real Buffalo road network via OSMnx.
    Caches to disk to avoid repeated API calls.
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("osmnx is not installed. Run: pip install osmnx")

    if os.path.exists(cache_path):
        print(f"[road_graph] Loading cached graph from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[road_graph] Downloading Buffalo graph from OpenStreetMap …")
    G_raw = ox.graph_from_place(place, network_type=network_type)
    G_proj = ox.project_graph(G_raw)   # UTM projection for accurate distances

    # Annotate edges
    for u, v, data in G_proj.edges(data=True):
        data["priority_class"]  = _assign_priority(data.get("highway", "residential"))
        data["snow_depth"]      = 0.0
        data["steps_uncleared"] = 0
        data["traffic_density"] = 0.0

    # Simplify to undirected for plow routing
    G_undirected = ox.convert.to_undirected(G_proj)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(G_undirected, f)
    print(f"[road_graph] Saved to {cache_path}")
    return G_undirected


def get_graph(use_real=False, n_nodes=100, seed=42):
    """
    Convenience loader.
    use_real=True  → download / load cached OSMnx Buffalo graph
    use_real=False → fast synthetic grid (default for development)
    """
    if use_real:
        return load_osmnx_graph()
    return build_synthetic_graph(n_nodes=n_nodes, seed=seed)


def get_edge_list(G):
    """Return sorted list of (u, v) edge tuples for stable indexing."""
    return sorted(G.edges())


def get_zone_partition(G, n_zones=10):
    """
    Partition graph nodes into zones using greedy modularity communities
    (falls back to simple index-based partition when community detection
    is unavailable).

    Returns: dict  {node_id: zone_id}
    """
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G.to_undirected()))
        partition = {}
        for zone_id, community in enumerate(communities[:n_zones]):
            for node in community:
                partition[node] = zone_id % n_zones
        # Assign any remaining nodes
        for node in G.nodes():
            if node not in partition:
                partition[node] = 0
        return partition
    except Exception:
        nodes = list(G.nodes())
        return {n: i * n_zones // len(nodes) for i, n in enumerate(nodes)}


if __name__ == "__main__":
    G = get_graph(use_real=False)
    print(f"Synthetic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    priorities = [d["priority_class"] for _, _, d in G.edges(data=True)]
    from collections import Counter
    print("Priority distribution:", dict(Counter(priorities)))
