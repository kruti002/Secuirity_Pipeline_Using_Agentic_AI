import os
import sys
import time
import psutil
import joblib
import pandas as pd
import numpy as np
import heapq
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.features import BehavioralFeatureEngine, drop_unneeded_columns_for_ml
from engine.graph_features import GraphFeatureEngine
from engine.graph_embeddings import extract_embedding_features, GraphSVDModel

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

class LargeScaleInferenceOrchestrator:
    def __init__(self, model_bundle_path, svd_model_path, output_path, stop_at_chunk=None):
        print(f"[{datetime.now()}] Initializing Large-Scale Inference Orchestrator...")
        
        # 1. Load Models
        self.bundle = joblib.load(model_bundle_path)
        self.model = self.bundle['model']
        self.svd_model = joblib.load(svd_model_path)
        self.output_path = output_path
        self.stop_at_chunk = stop_at_chunk
        
        # 2. Initialize Engines
        self.beh_engine = BehavioralFeatureEngine()
        self.graph_engine = GraphFeatureEngine()
        
        # 3. State for Global Stats
        self.top_alerts = []  # Heap of (score, timestamp, user, row)
        self.max_top_alerts = 5000
        
        self.total_processed = 0
        self.start_time = time.time()
        
        # Incremental Metrics (TP, FP, TN, FN)
        self.metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        self.threshold = self.bundle.get('threshold', 0.5)

        # Clear output file and write header
        if os.path.exists(output_path):
            os.remove(output_path)
        
    def process_dataset(self, csv_path, chunksize=100000):
        print(f"[{datetime.now()}] Starting stream from {csv_path} (chunksize={chunksize})")
        
        # Phase A: Data Streaming
        chunks = pd.read_csv(
            csv_path, 
            chunksize=chunksize, 
            parse_dates=['Login Timestamp'],
            # Use specific types for memory if possible
            dtype={
                'User ID': str,
                'Country': str,
                'ASN': str,
                'Login Successful': 'boolean',
                'Is Account Takeover': 'boolean'
            }
        )
        
        for i, chunk in enumerate(chunks):
            if self.stop_at_chunk and i >= self.stop_at_chunk:
                print(f"Reached stop_at_chunk={self.stop_at_chunk}. Stopping.")
                break
                
            chunk_start = time.time()

            # Memory Safety check
            import gc
            gc.collect() 
            mem_pct = psutil.virtual_memory().percent
            if mem_pct > 92:
                print(f"[{datetime.now()}] CRITICAL: System RAM at {mem_pct}%. Stopping to prevent OOM.")
                break

            # 1. Pre-process
            chunk = chunk.sort_values('Login Timestamp')
            
            # Engines like itertuples will see 'Login Successful' as 'Login_Successful'
            # But the ML model wants the original name.
            chunk['Login Hour'] = chunk['Login Timestamp'].dt.hour
            
            # Ensure bool columns are int/float for engines and model
            if 'Login Successful' in chunk.columns:
                chunk['Login Successful'] = chunk['Login Successful'].fillna(True).astype(int)
            if 'Is Account Takeover' in chunk.columns:
                chunk['label_int'] = chunk['Is Account Takeover'].fillna(False).astype(int)

            # Engines expect these internal names for itertuples safety
            chunk_proc = chunk.copy()
            # But we must NOT rename 'Login Successful' in the final DF used for model.predict()
            # If the engine needs 'User_ID' specifically:
            chunk_proc = chunk_proc.rename(columns={
                'User ID': 'User_ID',
                'Login Timestamp': 'Login_Timestamp'
            })

            # 2. Feature Generation (Stateful)
            # Behavioral
            chunk_beh = self.beh_engine.process_chunk(chunk_proc)
            
            # Graph Stateful
            # Note: process_chunk in GraphFeatureEngine expects 'IP_Subnet' and 'Device_Combo'
            # (BehavioralEngine already adds them to the dataframe it returns)
            chunk_graph = self.graph_engine.process_chunk(chunk_beh)
            
            # Graph Embeddings (Static SVD)
            chunk_svd = extract_embedding_features(chunk_graph, self.svd_model)
            
            # Combine all features
            final_df = pd.concat([chunk_graph.reset_index(drop=True), chunk_svd.reset_index(drop=True)], axis=1)
            
            # 3. Model Scoring
            # Align features
            X = final_df[self.bundle['numerical_cols'] + self.bundle['categorical_cols']]
            
            # Predict
            scores = self.model.predict_proba(X)[:, 1]
            final_df['risk_score'] = scores
            
            # 4. Output Handling
            self._handle_output_incremental(final_df, i == 0)
            
            # 5. Update Metrics
            if 'label_int' in final_df.columns:
                preds = (scores >= self.threshold).astype(int)
                labels = final_df['label_int'].values
                self.metrics['tp'] += np.sum((preds == 1) & (labels == 1))
                self.metrics['fp'] += np.sum((preds == 1) & (labels == 0))
                self.metrics['tn'] += np.sum((preds == 0) & (labels == 0))
                self.metrics['fn'] += np.sum((preds == 0) & (labels == 1))

            # 6. Global Stats & Memory
            self.total_processed += len(chunk)
            elapsed = time.time() - self.start_time
            chunk_time = time.time() - chunk_start
            
            print(f"Chunk {i:3d} | Rows: {self.total_processed:8d} | "
                  f"RAM_Usage: {get_memory_usage():7.1f} MB | {mem_pct:2.0f}% Sys | "
                  f"ChunkTime: {chunk_time:5.2f}s | TotalTime: {elapsed/60:5.2f}m")
            
            # Log state sizes for Engine 1
            if i % 10 == 0:
                self._log_engine_state()

        self._export_final_reports()

    def _handle_output_incremental(self, df, is_first):
        # Save compact scored results to disk
        cols_to_save = ['Login_Timestamp', 'User_ID', 'risk_score']
        if 'label_int' in df.columns:
            cols_to_save.append('label_int')
            
        df[cols_to_save].to_csv(
            self.output_path, 
            mode='a', 
            index=False, 
            header=is_first
        )
        
        # Maintain Top Alerts Heap (keep only top N)
        # Use itertuples for speed
        # Note: Select using exact column names in DF, then rename for safe itertuples access
        select_map = {
            'risk_score': 'risk_score',
            'Login_Timestamp': 'Login_Timestamp',
            'User_ID': 'User_ID',
            'Country': 'Country',
            'Device Type': 'Device_Type',
            'Login Successful': 'Login_Successful'
        }
        # Filter for only existing columns in case some are missing (e.g. debugging)
        available_cols = [c for c in select_map.keys() if c in df.columns]
        relevant_data = df[available_cols].rename(columns={c: select_map[c] for c in available_cols})
        
        for row in relevant_data.itertuples(index=False):
            if len(self.top_alerts) < self.max_top_alerts:
                heapq.heappush(self.top_alerts, (row.risk_score, row.Login_Timestamp, row.User_ID, row.Country, row.Device_Type, row.Login_Successful))
            else:
                if row.risk_score > self.top_alerts[0][0]:
                    heapq.heapreplace(self.top_alerts, (row.risk_score, row.Login_Timestamp, row.User_ID, row.Country, row.Device_Type, row.Login_Successful))

    def _log_engine_state(self):
        # Monitoring state growth as requested
        beh_users = len(self.beh_engine.user_login_count)
        graph_subnets = len(self.graph_engine.subnet_total_count)
        print(f"   [State] Users Tracked: {beh_users:,} | Subnets Tracked: {graph_subnets:,}")

    def _export_final_reports(self):
        print(f"\n[{datetime.now()}] Exporting final reports...")
        
        # 1. Dashboard Alerts
        top_df = pd.DataFrame(self.top_alerts, columns=['risk_score', 'timestamp', 'user', 'country', 'device', 'success'])
        top_df = top_df.sort_values('risk_score', ascending=False)
        top_df.to_csv('reports/dashboard_top_alerts.csv', index=False)
        print(f"   Saved {len(top_df)} top alerts to reports/dashboard_top_alerts.csv")
        
        # 2. Performance Summary
        if self.metrics['tp'] + self.metrics['fn'] > 0:
            tp, fp, tn, fn = self.metrics['tp'], self.metrics['fp'], self.metrics['tn'], self.metrics['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            summary = (
                f"Full Scale Inference Metrics:\n"
                f"-----------------------------\n"
                f"Total Rows: {self.total_processed}\n"
                f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall:    {recall:.4f}\n"
                f"F1 Score:  {f1:.4f}\n"
            )
            print(summary)
            with open('reports/inference_summary.txt', 'w') as f:
                f.write(summary)
        
        print(f"DONE. Scored archive at {self.output_path}")

if __name__ == "__main__":
    orchestrator = LargeScaleInferenceOrchestrator(
        model_bundle_path='models/supervised_model_bundle.joblib',
        svd_model_path='models/graph_svd_model.joblib',
        output_path='reports/full_dataset_scored_results.csv',
        stop_at_chunk=None # Set to small number for testing
    )
    
    # Run test on 5M rows (50 chunks)
    print("\nStarting PHASE 3: Stress Test (5M rows)...")
    orchestrator.stop_at_chunk = 50
    orchestrator.process_dataset('data/rba-dataset.csv', chunksize=100000)
    
    # If successful, user can run full dataset by removing stop_at_chunk
    
    # If successful, user can run full dataset by removing stop_at_chunk
