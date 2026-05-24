from copy import deepcopy
from src.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace
        
        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}
        
        self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")
        
    def insert_text(self, text_list):
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]      
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        if not texts_to_encode:
            print(f"[{self.namespace}] All {len(all_hash_ids)} texts already indexed.")
            return

        print(f"[{self.namespace}] Embedding {len(texts_to_encode)} new texts "
              f"({len(all_hash_ids) - len(texts_to_encode)} already cached)...")

        # Encode and persist incrementally — save after every batch so that
        # interrupting mid-run preserves all work done so far.
        for i in tqdm(range(0, len(texts_to_encode), self.batch_size),
                      desc=f"[{self.namespace}] embedding", unit="batch"):
            batch_ids   = missing_ids[i : i + self.batch_size]
            batch_texts = texts_to_encode[i : i + self.batch_size]
            batch_embs  = self.embedding_model.encode(
                batch_texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=self.batch_size, is_query=False
            )
            self._upsert(batch_ids, batch_texts, list(batch_embs))

    def _upsert(self, hash_ids, texts, embeddings):
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
        
        self._save_data()

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings
        })
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)
      
    def get_hash_id_to_text(self):
        return deepcopy(self.hash_id_to_text)
    
    def encode_texts(self, texts):
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size, is_query=False)
    
    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings