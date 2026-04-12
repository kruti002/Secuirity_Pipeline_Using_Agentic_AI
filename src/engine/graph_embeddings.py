import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse.linalg import svds
import joblib
import os

class GraphSVDModel:
    """
    Production-grade structural embedding model using Sparse SVD.
    Captures Tripartite User-Subnet-Device relations with weighted edges.
    """
    def __init__(self, dimensions=16):
        self.dimensions = dimensions
        self.wv = {}
        self.vector_size = dimensions

    def fit(self, G):
        print(f"Factorizing weighted adjacency matrix (dim={self.dimensions})...")
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build Sparse Adjacency Matrix with Frequency Weights
        adj = nx.adjacency_matrix(G, nodelist=nodes, weight='weight')
        adj = adj.astype(float)
        
        # SVD Factorization
        # Using k=self.dimensions to find the top latent structural components
        u, s, vt = svds(adj, k=self.dimensions)
        
        # SVD Instability Fix: SVDS is not guaranteed to return sorted values
        # We sort them descending so that USER_V_0 is always the most significant component
        idx = np.argsort(-s)
        s = s[idx]
        u = u[:, idx]
        
        # Map back to embedding dictionary (scaled by singular values)
        for node, i in node_to_idx.items():
            self.wv[node] = u[i] * np.sqrt(s)
            
        return self

def generate_graph_embeddings(df, dimensions=16):
    """
    Builds a full weighted identity-infrastructure graph.
    Uses SVD for high-performance structural embedding.
    """
    print(f"\n[1/2] Building weighted graph from {len(df)} events...")
    df = df.rename(columns={'User ID': 'User_ID'})
    
    # Pre-process columns
    if 'IP_Subnet' not in df.columns:
        df['IP_Subnet'] = df['Country']
    if 'Device_Combo' not in df.columns:
        df['Device_Combo'] = 'Unknown'

    G = nx.Graph()
    
    # 1. ACCUMULATE WEIGHTS (Frequency based)
    # This ensures repeated interactions strengthen structural similarity
    for row in df.itertuples():
        u = f"USER_{getattr(row, 'User_ID', 'Unknown')}"
        s = f"SUB_{getattr(row, 'IP_Subnet', 'Unknown')}"
        d = f"DEV_{getattr(row, 'Device_Combo', 'Unknown')}"
        
        # Weighted edges: User-Subnet, User-Device, Subnet-Device
        for node1, node2 in [(u, s), (u, d), (s, d)]:
            if G.has_edge(node1, node2):
                G[node1][node2]['weight'] += 1.0
            else:
                G.add_edge(node1, node2, weight=1.0)
        
    print(f"      Graph Ready: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    
    # 2. RUN SVD
    model = GraphSVDModel(dimensions=dimensions)
    model.fit(G)
    
    return model

def extract_embedding_features(df, model):
    """
    Full Triple-Vector Row Representation:
    - User/Subnet/Device embeddings
    - 3-way Cosine Proximity Metrics
    - Out-of-Vocabulary (OOV) Flags
    """
    rows_embs = []
    df = df.rename(columns={'User ID': 'User_ID', 'IP Address': 'IP_Address'})
    
    for row in df.itertuples():
        u_node = f"USER_{getattr(row, 'User_ID', 'Unknown')}"
        s_node = f"SUB_{getattr(row, 'IP_Subnet', 'Unknown')}"
        d_node = f"DEV_{getattr(row, 'Device_Combo', 'Unknown')}"
        
        # Vector fetching + Missing Node Flag
        def get_v_data(n):
            if n in model.wv:
                return model.wv[n], 0
            return np.zeros(model.vector_size), 1
            
        u_v, u_miss = get_v_data(u_node)
        s_v, s_miss = get_v_data(s_node)
        d_v, d_miss = get_v_data(d_node)
        
        # Cosine Proximity Calculations
        def cos_sim(v1, v2):
            norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.dot(v1, v2) / norm if norm > 0 else 0.0
            
        # 3-way similarities
        sim_us = cos_sim(u_v, s_v)
        sim_ud = cos_sim(u_v, d_v)
        sim_sd = cos_sim(s_v, d_v)
        
        # Combine everything into a 1D row feature array
        row_feat = np.concatenate([
            u_v, s_v, d_v,                  # Dimensional Vectors (48 feats)
            [sim_us, sim_ud, sim_sd],       # Relational Proxity (3 feats)
            [u_miss, s_miss, d_miss]        # OOV Safety Flags (3 feats)
        ])
        rows_embs.append(row_feat)
        
    dim = model.vector_size
    cols = [f"USER_V_{i}" for i in range(dim)] + \
           [f"SUB_V_{i}" for i in range(dim)]  + \
           [f"DEV_V_{i}" for i in range(dim)]  + \
           ['SIM_USER_SUB', 'SIM_USER_DEV', 'SIM_SUB_DEV'] + \
           ['USER_MISSING', 'SUB_MISSING', 'DEV_MISSING']
           
    return pd.DataFrame(rows_embs, columns=cols, index=df.index)

if __name__ == "__main__":
    train_path = 'data/splits/train.csv'
    if os.path.exists(train_path):
        print("Training Max-Horizon Graph SVD Model (4.2M rows)...")
        # Load full training set for structural learning
        df = pd.read_csv(train_path) 
        
        # Ensure helper columns exist for tripartite construction
        if 'IP_Subnet' not in df.columns:
            df['IP_Subnet'] = df['Country']
            
        emb_model = generate_graph_embeddings(df, dimensions=16)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(emb_model, 'models/graph_svd_model.joblib')
        print("Saved: models/graph_svd_model.joblib")
