import os
from tqdm import tqdm
import pandas as pd
import numpy as np


def merge_results_by_distance(ids_1, distances_1, ids_2, distances_2):
    """Merge and sort results by distance (lowest first)"""
    combined = list(zip(ids_1 + ids_2, distances_1 + distances_2))
    combined.sort(key=lambda x: x[1])  # Sort by distance
    return [x[0] for x in combined], [x[1] for x in combined]

def find_gold_ranks(retrieved_ids, gold_ids, not_found_value=-1):
    """Find rank of each gold ID in retrieved list. Returns -1 if not found."""
    ranks = []
    for gold_id in gold_ids:
        try:
            rank = retrieved_ids.index(gold_id)
            ranks.append(rank)
        except ValueError:
            ranks.append(not_found_value)
    return ranks

def compute_recall_metrics(retrieval_results, k_values=[1, 3, 5, 10], gold_ranks_key='gold_ranks'):
    """Compute recall@k metrics from retrieval results"""
    metrics = {f'recall@{k}': [] for k in k_values}
    
    for result in retrieval_results:
        gold_ranks = result[gold_ranks_key]
        
        for k in k_values:
            # Count how many gold documents are found within top-k
            found_count = sum(1 for rank in gold_ranks if 0 <= rank < k)
            recall = found_count / len(gold_ranks)  # Total gold documents = 2
            metrics[f'recall@{k}'].append(recall)
    
    return metrics

def compute_recall_metrics_dataframe(df, k_values=[1, 3, 5], gold_ranks_column='gold_ranks'):
    """Compute recall@k metrics from retrieval results"""
    metrics = {f'recall@{k}': [] for k in k_values}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gold_ranks = row[gold_ranks_column]
        
        for k in k_values:
            # Count how many gold documents are found within top-k
            found_count = sum(1 for rank in gold_ranks if 0 <= rank < k)
            recall = found_count / len(gold_ranks)  # Total gold documents = 2
            metrics[f'recall@{k}'].append(recall)
    
    return metrics

def compute_retrieval_metrics(results_df, k_values):
    """Compute and display retrieval metrics"""
    recall_metrics = compute_recall_metrics_dataframe(results_df, k_values=k_values)
    accuracy_metrics = compute_accuracy_at_k(results_df, k_values=k_values)
    mrr_metrics = compute_mrr_at_k(results_df, k_values=k_values)
    
    metrics_summary = {}
    for k in k_values:
        metrics_summary[f'recall@{k}'] = np.mean(recall_metrics[f'recall@{k}'])
        metrics_summary[f'accuracy@{k}'] = np.mean(accuracy_metrics[f'accuracy@{k}'])
        metrics_summary[f'mrr@{k}'] = np.mean(mrr_metrics[f'mrr@{k}'])
    
    return metrics_summary

def compute_mrr_at_k(df, k_values=[1, 3, 5, 10], gold_ranks_column='gold_ranks'):
    """
    Compute Mean Reciprocal Rank (MRR) @k metrics from retrieval results
    
    Args:
        df: DataFrame with 'gold_ranks' column
        k_values: List of k values to compute MRR for
        
    Returns:
        dict: Dictionary with MRR@k metrics
    """
    metrics = {f'mrr@{k}': [] for k in k_values}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing MRR"):
        gold_ranks = row[gold_ranks_column]
        
        for k in k_values:
            reciprocal_ranks = []
            for rank in gold_ranks:
                if 0 <= rank < k:
                    reciprocal_ranks.append(1.0 / (rank + 1))  # rank is 0-indexed
                else:
                    reciprocal_ranks.append(0.0)
            
            # MRR is the mean of reciprocal ranks
            mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
            metrics[f'mrr@{k}'].append(mrr)
    
    return metrics

def compute_accuracy_at_k(df, k_values=[1, 3, 5, 10],gold_ranks_column='gold_ranks'):
    """
    Compute Accuracy@k metrics - 1 if ALL gold documents are retrieved within top-k, else 0
    
    Args:
        df: DataFrame with 'gold_ranks' column
        k_values: List of k values to compute accuracy for
        
    Returns:
        dict: Dictionary with accuracy@k metrics
    """
    metrics = {f'accuracy@{k}': [] for k in k_values}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Accuracy"):
        gold_ranks = row[gold_ranks_column]
        
        for k in k_values:
            # Check if ALL gold documents are found within top-k
            all_found = all(0 <= rank < k for rank in gold_ranks)
            accuracy = 1.0 if all_found else 0.0
            metrics[f'accuracy@{k}'].append(accuracy)
    
    return metrics





class RAGRetriever:
    def __init__(self, collections, n_results=5, metadata_filter=None, collection_specific_filters=None):
        """
        Initialize RAG retriever.
        
        Args:
            collections: List of ChromaDB collections to query
            n_results: Number of results to retrieve per collection
            metadata_filter: Optional dict for filtering by metadata applied to ALL collections (ChromaDB 'where' clause)
                           Single condition: {'sommaire': 'yes'}
                           Multiple conditions (AND): {'$and': [{'chamber': 'no'}, {'vote': 'no'}]}
                           Multiple conditions (OR): {'$or': [{'sommaire': 'yes'}, {'year': '1887'}]}
                           Complex: {'$and': [{'year': '1887'}, {'$or': [{'sommaire': 'yes'}, {'chamber': 'yes'}]}]}
            collection_specific_filters: Optional dict mapping collection indices or names to specific filters
                           Example by index: {2: {'vote': 'no'}}  # Only filter collection at index 2
                           Example by name: {'debattre_1887_10000_hierarchical_cohere': {'vote': 'no'}}
                           If provided, overrides metadata_filter for specified collections
        """
        self.collections = collections
        self.n_results = n_results
        self.metadata_filter = metadata_filter
        self.collection_specific_filters = collection_specific_filters or {}
    
    def retrieve(self, query):
        """Retrieve documents from all collections and merge by distance"""
        # Collect all results
        all_results = []
        
        for idx, collection in enumerate(self.collections):
            # Determine which filter to use for this collection
            filter_to_use = None
            
            # Check if there's a collection-specific filter by index
            if idx in self.collection_specific_filters:
                filter_to_use = self.collection_specific_filters[idx]
            # Check if there's a collection-specific filter by name
            elif collection.name in self.collection_specific_filters:
                filter_to_use = self.collection_specific_filters[collection.name]
            # Otherwise use the global filter
            else:
                filter_to_use = self.metadata_filter
            
            result = collection.query(
                query_texts=[query], 
                n_results=self.n_results,
                where=filter_to_use
            )
            
            # Combine id, distance, document for each result
            for i in range(len(result['ids'][0])):
                all_results.append({
                    'id': result['ids'][0][i],
                    'distance': result['distances'][0][i],
                    'document': result['documents'][0][i]
                })
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        
        # Extract sorted components
        return {
            'retrieved_ids': [r['id'] for r in all_results],
            'distances': [r['distance'] for r in all_results],
            'documents': [r['document'] for r in all_results]
        }
    
class RAGRetriever_force_multicollection:
    def __init__(self, collections, n_results=5):
        self.collections = collections
        self.n_results = n_results
    
    def retrieve(self, query):
        """Retrieve documents from all collections and merge by distance
        
        Retrieves an equal number of documents from each collection,
        distributing n_results evenly across all collections.
        """
        # Calculate how many documents to retrieve from each collection
        num_collections = len(self.collections)
        docs_per_collection = self.n_results // num_collections
        remainder = self.n_results % num_collections
        
        # Collect all results
        all_results = []
        
        for idx, collection in enumerate(self.collections):
            # Distribute remainder among first collections
            n_docs_this_collection = docs_per_collection + (1 if idx < remainder else 0)
            
            if n_docs_this_collection == 0:
                continue
                
            result = collection.query(query_texts=[query], n_results=n_docs_this_collection)
            
            # Combine id, distance, document for each result
            for i in range(len(result['ids'][0])):
                all_results.append({
                    'id': result['ids'][0][i],
                    'distance': result['distances'][0][i],
                    'document': result['documents'][0][i]
                })
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        
        # Extract sorted components
        return {
            'retrieved_ids': [r['id'] for r in all_results],
            'distances': [r['distance'] for r in all_results],
            'documents': [r['document'] for r in all_results]
        }


class RAGRetriever_with_classification:
    """
    RAG Retriever that uses a classification model to route queries to appropriate collections.
    
    This retriever first classifies the question to predict which source combination
    is most appropriate, then queries only the relevant collections.
    
    Args:
        collections_dict: Dictionary mapping collection labels to collection objects
                         e.g., {'Le Gaulois': collection1, "L'Intransigeant": collection2, 'Les Débats': collection3}
        classifier_model: Trained sklearn classifier that predicts source combinations
        embedding_function: Function to embed queries for classification (e.g., cohere.embed)
        n_results: Number of documents to retrieve per collection
    """
    def __init__(self, collections_dict, class_to_collections, classifier_model, embedding_function, n_results=5):
        self.collections_dict = collections_dict
        self.classifier_model = classifier_model
        self.embedding_function = embedding_function
        self.n_results = n_results
        
        # Mapping from predicted class to collection labels
        # Predicted classes are like: "Le Gaulois + L'Intransigeant"
        self.class_to_collections = class_to_collections
    
    def retrieve(self, query):
        """Retrieve documents after classifying the query to select appropriate collections"""
        # Step 1: Embed the query
        query_embedding = self.embedding_function([query])
        
        # Step 2: Classify to predict source combination
        predicted_class = self.classifier_model.predict(query_embedding)[0]
        
        # Step 3: Get the collections to query based on prediction
        collections_to_query = self.class_to_collections.get(
            predicted_class, 
            list(self.collections_dict.keys())  # Default to all collections if class not found
        )
        
        # Step 4: Retrieve from selected collections only
        all_results = []
        
        for collection_label in collections_to_query:
            if collection_label not in self.collections_dict:
                continue
                
            collection = self.collections_dict[collection_label]
            result = collection.query(query_texts=[query], n_results=self.n_results)
            
            # Combine id, distance, document for each result
            for i in range(len(result['ids'][0])):
                all_results.append({
                    'id': result['ids'][0][i],
                    'distance': result['distances'][0][i],
                    'document': result['documents'][0][i],
                    'source': collection_label
                })
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        
        # Extract sorted components
        return {
            'retrieved_ids': [r['id'] for r in all_results],
            'distances': [r['distance'] for r in all_results],
            'documents': [r['document'] for r in all_results],
            'sources': [r['source'] for r in all_results],
            'predicted_class': predicted_class
        }


class RAGRetriever_with_classification_force_multicollection:
    """
    RAG Retriever that combines classification-based routing with forced multi-collection retrieval.
    
    This retriever:
    1. Classifies the question to predict which source combination is most appropriate
    2. Queries only the selected collections
    3. Retrieves an equal number of documents from each selected collection (forced distribution)
    
    Args:
        collections_dict: Dictionary mapping collection labels to collection objects
                         e.g., {'Le Gaulois': collection1, "L'Intransigeant": collection2, 'Les Débats': collection3}
        class_to_collections: Mapping from predicted class to list of collection labels
        classifier_model: Trained sklearn classifier that predicts source combinations
        embedding_function: Function to embed queries for classification (e.g., cohere.embed)
        n_results: Total number of documents to retrieve (distributed evenly across selected collections)
    """
    def __init__(self, collections_dict, class_to_collections, classifier_model, embedding_function, n_results=5):
        self.collections_dict = collections_dict
        self.classifier_model = classifier_model
        self.embedding_function = embedding_function
        self.n_results = n_results
        
        # Mapping from predicted class to collection labels
        self.class_to_collections = class_to_collections
    
    def retrieve(self, query):
        """Retrieve documents after classifying the query and distributing retrieval evenly"""
        # Step 1: Embed the query
        query_embedding = self.embedding_function([query])
        
        # Step 2: Classify to predict source combination
        predicted_class = self.classifier_model.predict(query_embedding)[0]
        
        # Step 3: Get the collections to query based on prediction
        collections_to_query = self.class_to_collections.get(
            predicted_class, 
            list(self.collections_dict.keys())  # Default to all collections if class not found
        )
        
        # Step 4: Calculate how many documents to retrieve from each collection
        num_collections = len(collections_to_query)
        docs_per_collection = self.n_results // num_collections
        remainder = self.n_results % num_collections
        
        # Step 5: Retrieve from selected collections with equal distribution
        all_results = []
        
        for idx, collection_label in enumerate(collections_to_query):
            if collection_label not in self.collections_dict:
                continue
            
            # Distribute remainder among first collections
            n_docs_this_collection = docs_per_collection + (1 if idx < remainder else 0)
            
            if n_docs_this_collection == 0:
                continue
                
            collection = self.collections_dict[collection_label]
            result = collection.query(query_texts=[query], n_results=n_docs_this_collection)
            
            # Combine id, distance, document for each result
            for i in range(len(result['ids'][0])):
                all_results.append({
                    'id': result['ids'][0][i],
                    'distance': result['distances'][0][i],
                    'document': result['documents'][0][i],
                    'source': collection_label
                })
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        
        # Extract sorted components
        return {
            'retrieved_ids': [r['id'] for r in all_results],
            'distances': [r['distance'] for r in all_results],
            'documents': [r['document'] for r in all_results],
            'sources': [r['source'] for r in all_results],
            'predicted_class': predicted_class
        }
    
    

def evaluate_retrieval(questions_df, retriever, k_top=3):
    """Perform retrieval and evaluation"""
    
    results = []
    
    print("Performing retrieval...")
    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        question = row['question']
        gold_ids = row['gold_ids']
        
        # Retrieve documents
        retrieval_result = retriever.retrieve(question)
         
        # Find ranks of gold documents
        gold_ranks = find_gold_ranks(
            retrieval_result['retrieved_ids'], 
            gold_ids, 
            not_found_value=-1
        )
        
        # Check if both gold IDs are in top-k
        dual_status = "both_found" if all(0 <= rank < k_top for rank in gold_ranks) else "incomplete"
        
        # Create result row
        result_row = row.to_dict()
        result_row.update({
            'retrieved_ids': retrieval_result['retrieved_ids'],
            'retrieval_distances': retrieval_result['distances'],
            'gold_ranks': gold_ranks,
            'dual_retrieval_status': dual_status,
            "retrieved_documents": retrieval_result['documents']
        })
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def evaluate_and_print_metrics(df, dataset_name=None, gold_ranks_column='gold_ranks',k_values=[2, 3, 5, 10]):
    """
    Compute and print all evaluation metrics for a given dataframe
    
    Args:
        df: DataFrame with retrieval results
        dataset_name: Optional name for the dataset (for display purposes)
    """
    if dataset_name is None:
        dataset_name = "dataset"
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS FOR: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Compute recall metrics
    print("\nComputing recall metrics...")
    recall_metrics = compute_recall_metrics_dataframe(df, k_values=k_values, gold_ranks_column=gold_ranks_column)
    
    print(f"\nRecall Metrics for {len(df)} questions:")
    for metric_name, values in recall_metrics.items():
        mean_recall = np.mean(values)
        print(f"  {metric_name}: {mean_recall:.3f}")
    
    # Compute accuracy metrics
    accuracy_metrics = compute_accuracy_at_k(df, k_values=k_values, gold_ranks_column=gold_ranks_column)
    
    print(f"\nAccuracy Metrics for {len(df)} questions:")
    for metric_name, values in accuracy_metrics.items():
        mean_accuracy = np.mean(values)
        print(f"  {metric_name}: {mean_accuracy:.3f}")
    
    # Compute MRR metrics
    mrr_metrics = compute_mrr_at_k(df, k_values=k_values, gold_ranks_column=gold_ranks_column)
    
    print(f"\nMRR Metrics for {len(df)} questions:")
    for metric_name, values in mrr_metrics.items():
        mean_mrr = np.mean(values)
        print(f"  {metric_name}: {mean_mrr:.3f}")
    
    return {
        'recall': recall_metrics,
        'accuracy': accuracy_metrics, 
        'mrr': mrr_metrics
    }


class RAGRetriever_all_collections_with_classification:
    """
    Universal RAG Retriever that retrieves from all collections and saves classification.
    
    This retriever retrieves a large number of documents from each collection and saves
    the classification prediction. Different retrieval strategies can then be applied
    post-hoc without re-computing retrieval.
    
    Args:
        collections_dict: Dictionary mapping collection labels to collection objects
        classifier_model: Trained sklearn classifier that predicts source combinations
        embedding_function: Function to embed queries for classification
        n_results_per_collection: Number of documents to retrieve from each collection
    """
    def __init__(self, collections_dict, classifier_model, embedding_function, n_results_per_collection=15):
        self.collections_dict = collections_dict
        self.classifier_model = classifier_model
        self.embedding_function = embedding_function
        self.n_results_per_collection = n_results_per_collection
    
    def retrieve(self, query):
        """Retrieve documents from all collections and save classification"""
        # Step 1: Embed the query and classify
        query_embedding = self.embedding_function([query])
        predicted_class = self.classifier_model.predict(query_embedding)[0]
        
        # Step 2: Retrieve from ALL collections
        all_results = []
        
        for collection_label, collection in self.collections_dict.items():
            result = collection.query(query_texts=[query], n_results=self.n_results_per_collection)
            
            # Combine id, distance, document, source for each result
            for i in range(len(result['ids'][0])):
                all_results.append({
                    'id': result['ids'][0][i],
                    'distance': result['distances'][0][i],
                    'document': result['documents'][0][i],
                    'source': collection_label
                })
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        
        # Extract sorted components
        return {
            'retrieved_ids': [r['id'] for r in all_results],
            'distances': [r['distance'] for r in all_results],
            'documents': [r['document'] for r in all_results],
            'sources': [r['source'] for r in all_results],
            'predicted_class': predicted_class
        }


def apply_query_rerouting(df, class_to_collections, n_results=10):
    """
    Apply query rerouting strategy to pre-retrieved results.
    
    Filters retrieved documents to only include those from collections 
    predicted by the classifier, then takes top n_results by distance.
    
    Args:
        df: DataFrame with columns: retrieved_ids, distances, sources, predicted_class
        class_to_collections: Mapping from predicted class to list of collection labels
        n_results: Number of top results to keep
        
    Returns:
        DataFrame with additional columns for rerouted results
    """
    import copy
    
    results = []
    for _, row in df.iterrows():
        predicted_class = row['predicted_class']
        allowed_collections = class_to_collections.get(predicted_class, list(set(row['sources'])))
        
        # Filter to only include documents from allowed collections
        filtered_indices = [i for i, source in enumerate(row['sources']) 
                           if source in allowed_collections]
        
        # Take top n_results
        filtered_indices = filtered_indices[:n_results]
        
        rerouted_ids = [row['retrieved_ids'][i] for i in filtered_indices]
        rerouted_distances = [row['distances'][i] for i in filtered_indices]
        rerouted_docs = [row['documents'][i] for i in filtered_indices]
        rerouted_sources = [row['sources'][i] for i in filtered_indices]
        
        # Compute gold ranks
        gold_ranks = find_gold_ranks(rerouted_ids, row['gold_ids'], not_found_value=-1)
        
        result_row = row.to_dict()
        result_row.update({
            'retrieved_ids_rerouted': rerouted_ids,
            'distances_rerouted': rerouted_distances,
            'documents_rerouted': rerouted_docs,
            'sources_rerouted': rerouted_sources,
            'gold_ranks': gold_ranks
        })
        results.append(result_row)
    
    return pd.DataFrame(results)


def apply_forced_multicollection(df, class_to_collections, n_results=10):
    """
    Apply forced multicollection strategy to pre-retrieved results.
    
    First applies query rerouting, then ensures equal distribution across 
    selected collections by taking an equal number of documents from each.
    
    Args:
        df: DataFrame with columns: retrieved_ids, distances, sources, predicted_class
        class_to_collections: Mapping from predicted class to list of collection labels
        n_results: Total number of results to keep (distributed evenly)
        
    Returns:
        DataFrame with additional columns for forced multi results
    """
    results = []
    for _, row in df.iterrows():
        predicted_class = row['predicted_class']
        allowed_collections = class_to_collections.get(predicted_class, list(set(row['sources'])))
        
        # Group documents by collection
        collection_docs = {coll: [] for coll in allowed_collections}
        for i, source in enumerate(row['sources']):
            if source in allowed_collections:
                collection_docs[source].append({
                    'id': row['retrieved_ids'][i],
                    'distance': row['distances'][i],
                    'document': row['documents'][i],
                    'source': source,
                    'index': i
                })
        
        # Calculate docs per collection
        num_collections = len([coll for coll in allowed_collections if collection_docs[coll]])
        if num_collections == 0:
            num_collections = 1
        docs_per_collection = n_results // num_collections
        remainder = n_results % num_collections
        
        # Select documents evenly from each collection
        selected_docs = []
        for idx, coll in enumerate(allowed_collections):
            n_docs = docs_per_collection + (1 if idx < remainder else 0)
            selected_docs.extend(collection_docs[coll][:n_docs])
        
        # Sort by distance
        selected_docs.sort(key=lambda x: x['distance'])
        
        forced_ids = [d['id'] for d in selected_docs]
        forced_distances = [d['distance'] for d in selected_docs]
        forced_docs = [d['document'] for d in selected_docs]
        forced_sources = [d['source'] for d in selected_docs]
        
        # Compute gold ranks
        gold_ranks = find_gold_ranks(forced_ids, row['gold_ids'], not_found_value=-1)
        
        result_row = row.to_dict()
        result_row.update({
            'retrieved_ids_forced': forced_ids,
            'distances_forced': forced_distances,
            'documents_forced': forced_docs,
            'sources_forced': forced_sources,
            'gold_ranks': gold_ranks
        })
        results.append(result_row)
    
    return pd.DataFrame(results)


def apply_no_rerouting(df, n_results=10):
    """
    Apply no rerouting - just take top n_results by distance from all collections.
    
    Args:
        df: DataFrame with columns: retrieved_ids, distances, sources
        n_results: Number of top results to keep
        
    Returns:
        DataFrame with gold_ranks column
    """
    results = []
    for _, row in df.iterrows():
        # Take top n_results
        top_ids = row['retrieved_ids'][:n_results]
        top_distances = row['distances'][:n_results]
        top_docs = row['documents'][:n_results]
        top_sources = row['sources'][:n_results]
        
        # Compute gold ranks
        gold_ranks = find_gold_ranks(top_ids, row['gold_ids'], not_found_value=-1)
        
        result_row = row.to_dict()
        result_row.update({
            'gold_ranks': gold_ranks
        })
        results.append(result_row)
    
    return pd.DataFrame(results)


def apply_forced_equal_from_all_collections(df, n_results=10):
    """
    Apply forced equal distribution from ALL collections (no query rerouting).
    
    Retrieves an equal number of documents from each collection regardless of 
    the classification prediction. This ensures balanced representation from all sources.
    
    Args:
        df: DataFrame with columns: retrieved_ids, distances, sources
        n_results: Total number of results to keep (distributed evenly across ALL collections)
        
    Returns:
        DataFrame with gold_ranks column
    """
    results = []
    for _, row in df.iterrows():
        # Get all unique collections in the retrieved documents
        all_collections = list(set(row['sources']))
        
        # Group documents by collection
        collection_docs = {coll: [] for coll in all_collections}
        for i, source in enumerate(row['sources']):
            collection_docs[source].append({
                'id': row['retrieved_ids'][i],
                'distance': row['distances'][i],
                'document': row['documents'][i],
                'source': source,
                'index': i
            })
        
        # Calculate docs per collection (equal distribution)
        num_collections = len(all_collections)
        docs_per_collection = n_results // num_collections
        remainder = n_results % num_collections
        
        # Select documents evenly from each collection
        selected_docs = []
        for idx, coll in enumerate(sorted(all_collections)):  # Sort for consistency
            n_docs = docs_per_collection + (1 if idx < remainder else 0)
            selected_docs.extend(collection_docs[coll][:n_docs])
        
        # Sort by distance
        selected_docs.sort(key=lambda x: x['distance'])
        
        forced_equal_ids = [d['id'] for d in selected_docs]
        forced_equal_distances = [d['distance'] for d in selected_docs]
        forced_equal_docs = [d['document'] for d in selected_docs]
        forced_equal_sources = [d['source'] for d in selected_docs]
        
        # Compute gold ranks
        gold_ranks = find_gold_ranks(forced_equal_ids, row['gold_ids'], not_found_value=-1)
        
        result_row = row.to_dict()
        result_row.update({
            'retrieved_ids_forced_equal': forced_equal_ids,
            'distances_forced_equal': forced_equal_distances,
            'documents_forced_equal': forced_equal_docs,
            'sources_forced_equal': forced_equal_sources,
            'gold_ranks': gold_ranks
        })
        results.append(result_row)
    
    return pd.DataFrame(results)


# ============================================================================
# JSONL CACHING UTILITIES
# ============================================================================

def get_existing_ids(filepath):
    """Get set of unique_ids already processed in a JSONL file."""
    import json
    existing_ids = set()
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_ids.add(json.loads(line).get("unique_id", ""))
        except Exception as e:
            print(f"Error reading existing file: {e}")
    return existing_ids


def retrieve_for_strategy(
    strategy_id, debats_col, questions_df, output_path,
    newspaper_collections, classifier_model, embedding_function,
    n_per_col=15, max_retries=5
):
    """Run universal retrieval with a specific Les Débats collection.

    Saves incrementally to JSONL so partial progress is preserved.
    Skips questions whose unique_id is already in the output file.

    Args:
        strategy_id: label for tqdm progress bar
        debats_col: ChromaDB collection for Les Débats (strategy-specific)
        questions_df: DataFrame with question, unique_id, gold_ids, etc.
        output_path: JSONL file to write results to
        newspaper_collections: dict mapping label -> ChromaDB collection (newspapers)
        classifier_model: trained sklearn classifier for query routing
        embedding_function: callable(texts) -> np.ndarray for classification
        n_per_col: number of documents to retrieve per collection
        max_retries: maximum retries on API errors
    """
    import json
    import time

    collections_dict = {
        **newspaper_collections,
        "Les Débats": debats_col,
    }

    retriever = RAGRetriever_all_collections_with_classification(
        collections_dict=collections_dict,
        classifier_model=classifier_model,
        embedding_function=embedding_function,
        n_results_per_collection=n_per_col,
    )

    existing_ids = get_existing_ids(output_path)
    remaining = questions_df[~questions_df["unique_id"].isin(existing_ids)]
    if len(remaining) == 0:
        print(f"  All {len(questions_df)} questions already retrieved")
        return

    print(f"  {len(existing_ids)} cached, {len(remaining)} remaining")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc=strategy_id):
            question = row["question"]

            for attempt in range(max_retries):
                try:
                    retrieval_result = retriever.retrieve(question)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"\n  Retry {attempt+1}/{max_retries} after error: {e!r:.80s}... waiting {wait}s")
                        time.sleep(wait)
                    else:
                        raise

            result_row = row.to_dict()
            result_row.update({
                "retrieved_ids": retrieval_result["retrieved_ids"],
                "distances": retrieval_result["distances"],
                "documents": retrieval_result["documents"],
                "sources": retrieval_result["sources"],
                "predicted_class": retrieval_result["predicted_class"],
            })
            f.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            f.flush()


# ============================================================================
# RERANKING FUNCTIONS (metadata-based post-hoc reranking)
# ============================================================================

# Document type sets used by reranking
BOOST_DOC_TYPES = {
    "debate", "legal_text", "law_presentation", "law_adoption",
    "petition", "sommaire", "session_opening",
}
PENALIZE_DOC_TYPES = {
    "vote_list", "absence_list", "fragment", "chamber_header",
    "scrutin", "excuse", "conge",
}


def rerank_type_boost(ids, distances, metadatas, sources, alpha=0.008):
    """Boost debate/legal types by reducing their distance, penalize noisy types."""
    adjusted = []
    for did, dist, meta, src in zip(ids, distances, metadatas, sources):
        bonus = 0.0
        if meta and meta.get("doc_type") in BOOST_DOC_TYPES:
            bonus = -alpha
        elif meta and meta.get("doc_type") in PENALIZE_DOC_TYPES:
            bonus = alpha
        adjusted.append((did, dist + bonus, meta, src))
    adjusted.sort(key=lambda x: x[1])
    return (
        [x[0] for x in adjusted],
        [x[1] for x in adjusted],
        [x[2] for x in adjusted],
        [x[3] for x in adjusted],
    )


def rerank_speaker_boost(ids, distances, metadatas, sources, alpha=0.006):
    """Boost documents with multiple speakers (likely substantive debates)."""
    adjusted = []
    for did, dist, meta, src in zip(ids, distances, metadatas, sources):
        bonus = 0.0
        if meta:
            sc = meta.get("speaker_count", 0)
            if isinstance(sc, int) and sc >= 2:
                bonus = -alpha
            elif isinstance(sc, int) and sc == 0:
                bonus = alpha * 0.5
        adjusted.append((did, dist + bonus, meta, src))
    adjusted.sort(key=lambda x: x[1])
    return (
        [x[0] for x in adjusted],
        [x[1] for x in adjusted],
        [x[2] for x in adjusted],
        [x[3] for x in adjusted],
    )


def rerank_combined(ids, distances, metadatas, sources, alpha_type=0.006, alpha_speaker=0.004):
    """Combined reranking: type boost + speaker boost."""
    adjusted = []
    for did, dist, meta, src in zip(ids, distances, metadatas, sources):
        bonus = 0.0
        if meta:
            dt = meta.get("doc_type", "")
            if dt in BOOST_DOC_TYPES:
                bonus -= alpha_type
            elif dt in PENALIZE_DOC_TYPES:
                bonus += alpha_type
            sc = meta.get("speaker_count", 0)
            if isinstance(sc, int) and sc >= 2:
                bonus -= alpha_speaker
            elif isinstance(sc, int) and sc == 0:
                bonus += alpha_speaker * 0.5
        adjusted.append((did, dist + bonus, meta, src))
    adjusted.sort(key=lambda x: x[1])
    return (
        [x[0] for x in adjusted],
        [x[1] for x in adjusted],
        [x[2] for x in adjusted],
        [x[3] for x in adjusted],
    )


# ============================================================================
# BM25 RETRIEVER
# ============================================================================

class BM25Retriever:
    """BM25 multi-collection retriever using bm25s.

    Mirrors the RAGRetriever interface so the same evaluation pipeline works.
    BM25 scores are higher=better (unlike Cohere distances which are lower=better).
    The returned 'retrieved_ids' are sorted by descending score, so find_gold_ranks
    works identically to the Cohere path.
    """

    def __init__(self, retrievers_dict: dict, stopwords: str = "fr", n_results: int = 10):
        """
        Args:
            retrievers_dict: {collection_name: bm25s.BM25} loaded with load_corpus=True.
                             Corpus items must be dicts with 'id' and 'text' keys.
            stopwords: Language for bm25s stopword filtering (default: 'fr').
            n_results: Number of results to return per query (default: 10).
        """
        import bm25s as _bm25s
        self._bm25s = _bm25s
        self.retrievers_dict = retrievers_dict
        self.stopwords = stopwords
        self.n_results = n_results

    def retrieve(self, query: str) -> dict:
        """Query all BM25 indexes and merge results by descending score.

        Returns:
            dict with keys:
                - retrieved_ids: list[str] sorted by descending BM25 score.
                - scores: list[float] BM25 scores (higher = more relevant).
                - documents: list[str] document texts.
        """
        query_tokens = self._bm25s.tokenize([query], stopwords=self.stopwords)
        all_results = []

        for retriever in self.retrievers_dict.values():
            n_docs = min(self.n_results, len(retriever.corpus))
            if n_docs == 0:
                continue
            docs, scores = retriever.retrieve(query_tokens, corpus=retriever.corpus, k=n_docs)
            for doc, score in zip(docs[0], scores[0]):
                all_results.append({
                    "id": str(doc["id"]),
                    "score": float(score),
                    "text": doc["text"],
                })

        # Higher score = more relevant
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[: self.n_results]

        return {
            "retrieved_ids": [r["id"] for r in all_results],
            "scores": [r["score"] for r in all_results],
            "documents": [r["text"] for r in all_results],
        }


# ============================================================================
# SHARED RETRIEVAL FUNCTION
# ============================================================================

def perform_retrieval(questions_df, retriever, output_file):
    """Perform retrieval for multi-hop questions and append results to a JSONL file.

    Works with any retriever that implements `.retrieve(query)` and returns a dict
    with 'retrieved_ids' and 'documents' keys.  RAGRetriever also returns 'distances';
    BM25Retriever returns 'scores'.  Both are saved when present.

    unique_id format: ``{source_1}|{source_2}|{id_1}|{id_2}|{question_index}|{row_number}``
    The row number (0-based position in the concatenated DataFrame) guarantees uniqueness
    even when the same document pair appears across multiple datasets.

    Args:
        questions_df: DataFrame produced by create_questions_dataset.
        retriever: Any retriever with a .retrieve(query) -> dict interface.
        output_file: Path to the output JSONL file (appended to).
    """
    import json as _json

    existing_ids = get_existing_ids(output_file)
    print(f"Found {len(existing_ids)} existing retrieval results")

    questions_df = questions_df.copy().reset_index(drop=True)
    questions_df["unique_id"] = questions_df.apply(
        lambda row: f"{row['source_1']}|{row['source_2']}|{row['id_1']}|{row['id_2']}|{row['question_index']}|{row.name}",
        axis=1,
    )

    n_results = getattr(retriever, "n_results", 10)
    remaining_df = questions_df[~questions_df["unique_id"].isin(existing_ids)]

    if len(remaining_df) == 0:
        print("All questions already retrieved!")
        return

    print(f"Retrieving for {len(remaining_df)} questions...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):
        question = row["question"]
        gold_ids = row["gold_ids"]

        retrieval_result = retriever.retrieve(question)

        retrieved_ids = retrieval_result["retrieved_ids"]
        retrieved_documents = retrieval_result["documents"]

        gold_ranks = find_gold_ranks(retrieved_ids, gold_ids, not_found_value=-1)
        dual_status = (
            "both_found"
            if all(0 <= rank < n_results for rank in gold_ranks)
            else "incomplete"
        )

        result_record = {
            "unique_id": row["unique_id"],
            "id_1": row["id_1"],
            "id_2": row["id_2"],
            "question": question,
            "llm_answer": row["llm_answer"],
            "question_type": row["question_type"],
            "source_1": row["source_1"],
            "source_2": row["source_2"],
            "doc_1": row["doc_1"],
            "doc_2": row["doc_2"],
            "gold_ids": gold_ids,
            "retrieved_ids": retrieved_ids,
            "retrieved_documents": retrieved_documents,
            "gold_ranks": [int(r) for r in gold_ranks],
            "dual_retrieval_status": dual_status,
        }

        # Persist retrieval scores/distances depending on retriever type
        if "distances" in retrieval_result:
            result_record["retrieval_distances"] = [float(d) for d in retrieval_result["distances"]]
        if "scores" in retrieval_result:
            result_record["retrieval_scores"] = [float(s) for s in retrieval_result["scores"]]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(_json.dumps(result_record, ensure_ascii=False) + "\n")

    print(f"Retrieval complete! Results saved to {output_file}")

    # ------------------------------------------------------------------
    # Also persist a CSV for easy inspection (list columns as JSON strings)
    # ------------------------------------------------------------------
    csv_file = (
        output_file[: -len(".jsonl")] + ".csv"
        if output_file.endswith(".jsonl")
        else output_file + ".csv"
    )
    rows = []
    with open(output_file, "r", encoding="utf-8") as _fh:
        for _line in _fh:
            if _line.strip():
                rows.append(_json.loads(_line))
    if rows:
        _df_csv = pd.DataFrame(rows)
        _list_cols = [
            c for c in _df_csv.columns if _df_csv[c].apply(lambda x: isinstance(x, list)).any()
        ]
        for _col in _list_cols:
            _df_csv[_col] = _df_csv[_col].apply(_json.dumps)
        _df_csv.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"Results also saved as CSV → {csv_file}")


# Create the dataset
