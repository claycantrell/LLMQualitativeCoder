#!/usr/bin/env python3
# cluster_qualitative_codes_with_visualization.py

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from jsonschema import ValidationError, validate as json_validate

import hdbscan
import umap

# =======================
# Configuration Parameters
# =======================

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "app.log"  # Set to None or "" to disable file logging

# Paths
OUTPUT_DIR = "outputs"
JSON_FILE_PATH = "outputs/amazon_output_20250107_121529.json"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
INSTRUCTION = "Focus on the underlying product-related suggestions"
BATCH_SIZE = 32

# Clustering Configuration
CLUSTERING_METHOD = "hdbscan"  # Options: "hierarchical", "hdbscan"
N_COMPONENTS = 5  # Set to None or 0 to skip dimensionality reduction (UMAP), else e.g., 50
DISTANCE_THRESHOLD = 0.0005        # Relevant for hierarchical clustering
LINKAGE_METHOD = "average"         # Options: single, complete, average, ward (for hierarchical)
HDBSCAN_MIN_CLUSTER_SIZE = 3       # Relevant for HDBSCAN
HDBSCAN_MIN_SAMPLES = 1            # Relevant for HDBSCAN

# Labeling Configuration
LABELING_CRITERIA = "Group codes into parent themes that reflect underlying product-related suggestions."
LLM_MODEL = "gpt-4o-mini"

# Preprocessing / Embedding Caching
# Added a new mode "codes_only_name" that embeds only the code name (no justification).
PREPROCESSING_MODE = "meaning_unit"  # Options: "codes", "meaning_unit", "combined", "codes_only_name"
REUSE_EMBEDDINGS = False         # If True, reuse embeddings if available
EMBEDDING_FILE = None            # Custom embeddings file path; if None, auto-generated

# Visualization Configuration
CLUSTERS_JSON_PATH = None  # If None, will use the output clusters file based on PREPROCESSING_MODE

# =======================
# Logging Setup
# =======================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging with specified level and optional log file.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5))

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# =======================
# JSON Schema Validation
# =======================

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate JSON data against a provided schema.
    """
    try:
        json_validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        logging.error(f"JSON schema validation error: {e}")
        sys.exit(1)

# =======================
# Data Processing
# =======================

def replace_nan_with_null(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replace NaN values in the data with null.
    """
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, float) and np.isnan(value):
                entry[key] = None
    return data

def preprocess_data(data: Dict[str, Any], mode: str = "codes") -> List[Dict[str, Any]]:
    """
    Preprocess JSON data to extract code details, meaning units, or both.
    Depending on 'mode':
      - 'codes': includes code_name and code_justification
      - 'codes_only_name': includes only code_name
      - 'meaning_unit': includes meaning_unit_string (only those with non-empty assigned_code_list)
      - 'combined': merges meaning_unit_string + code_name + code_justification
    """
    meaning_units = data.get('meaning_units', [])
    if not meaning_units:
        logging.error("No 'meaning_units' found in the JSON data.")
        sys.exit(1)

    df_meaning = pd.json_normalize(meaning_units, sep='_')

    if mode == "codes":
        if 'assigned_code_list' not in df_meaning.columns:
            logging.error("'assigned_code_list' key is missing in the data.")
            sys.exit(1)

        df_codes = df_meaning.explode('assigned_code_list')
        df_filtered_codes = pd.concat(
            [df_codes.drop(['assigned_code_list'], axis=1),
             df_codes['assigned_code_list'].apply(pd.Series)],
            axis=1
        )

        if not {'code_name', 'code_justification'}.issubset(df_filtered_codes.columns):
            logging.error("Required keys 'code_name' and/or 'code_justification' are missing.")
            sys.exit(1)

        df_selected = df_filtered_codes[['code_name', 'code_justification']]
        selected_data = df_selected.to_dict(orient='records')
        selected_data = replace_nan_with_null(selected_data)
        filtered_code_details = [cd for cd in selected_data if cd.get('code_name')]

        if not filtered_code_details:
            logging.error("No valid code details found after filtering.")
            sys.exit(1)

        logging.info(f"Preprocessed and filtered {len(filtered_code_details)} code details.")
        return filtered_code_details

    elif mode == "codes_only_name":
        """
        Same as 'codes', but we only retain 'code_name'.
        """
        if 'assigned_code_list' not in df_meaning.columns:
            logging.error("'assigned_code_list' key is missing in the data.")
            sys.exit(1)

        df_codes = df_meaning.explode('assigned_code_list')
        df_filtered_codes = pd.concat(
            [df_codes.drop(['assigned_code_list'], axis=1),
             df_codes['assigned_code_list'].apply(pd.Series)],
            axis=1
        )

        # For "codes_only_name", we only need 'code_name'.
        if 'code_name' not in df_filtered_codes.columns:
            logging.error("'code_name' is missing in the data.")
            sys.exit(1)

        df_selected = df_filtered_codes[['code_name']]
        selected_data = df_selected.to_dict(orient='records')
        selected_data = replace_nan_with_null(selected_data)
        filtered_code_details = [cd for cd in selected_data if cd.get('code_name')]

        if not filtered_code_details:
            logging.error("No valid code details found after filtering.")
            sys.exit(1)

        logging.info(f"Preprocessed and filtered {len(filtered_code_details)} code names (no justifications).")
        return filtered_code_details

    elif mode == "meaning_unit":
        # -------------------------------------------------------
        # Filter meaning_units that have a non-empty assigned_code_list,
        # but ONLY embed meaning_unit_string later (codes not embedded).
        # Return both meaning_unit_string and assigned_code_list
        # so you still have code details for labeling.
        # -------------------------------------------------------
        if ('meaning_unit_string' not in df_meaning.columns or 
                'assigned_code_list' not in df_meaning.columns):
            logging.error("'meaning_unit_string' and/or 'assigned_code_list' keys are missing in the data.")
            sys.exit(1)

        # Filter only meaning_units with a non-empty assigned_code_list
        df_filtered = df_meaning[df_meaning['assigned_code_list'].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )]

        # Keep meaning_unit_string and assigned_code_list
        df_selected = df_filtered[['meaning_unit_string', 'assigned_code_list']]
        selected_data = df_selected.to_dict(orient='records')
        selected_data = replace_nan_with_null(selected_data)

        # Filter out any that have empty or None meaning_unit_string
        filtered_meaning_units = [mu for mu in selected_data if mu.get('meaning_unit_string')]

        if not filtered_meaning_units:
            logging.error("No valid meaning units found after filtering.")
            sys.exit(1)

        logging.info(
            f"Preprocessed and filtered {len(filtered_meaning_units)} meaning units "
            "with non-empty assigned_code_list."
        )
        return filtered_meaning_units

    elif mode == "combined":
        if 'assigned_code_list' not in df_meaning.columns or 'meaning_unit_string' not in df_meaning.columns:
            logging.error("'assigned_code_list' and/or 'meaning_unit_string' keys are missing in the data.")
            sys.exit(1)

        # Combine meaning_unit_string with its assigned codes, ensuring assigned_code_list is non-empty
        combined_data = []
        for _, row in df_meaning.iterrows():
            meaning_unit = row.get('meaning_unit_string', '')
            assigned_codes = row.get('assigned_code_list', [])
            if not meaning_unit:
                continue  # Skip if meaning_unit_string is empty
            if not isinstance(assigned_codes, list) or not assigned_codes:
                continue  # Skip if assigned_code_list is empty or not a list
            for code in assigned_codes:
                code_name = code.get('code_name', '').strip()
                code_justification = code.get('code_justification', '').strip()
                if not code_name:
                    continue  # Skip codes without a name
                combined_text = (
                    f"Meaning Unit: {meaning_unit}\n"
                    f"Code Name: {code_name}\n"
                    f"Code Justification: {code_justification}"
                )
                combined_entry = {
                    "combined_text": combined_text,
                    "code_name": code_name,
                    "code_justification": code_justification,
                    "meaning_unit_string": meaning_unit  # Add meaning_unit_string separately
                }
                combined_data.append(combined_entry)

        combined_data = replace_nan_with_null(combined_data)
        filtered_combined = [cd for cd in combined_data if cd.get('combined_text')]

        if not filtered_combined:
            logging.error("No valid combined entries found after filtering.")
            sys.exit(1)

        logging.info(f"Preprocessed and filtered {len(filtered_combined)} combined entries.")
        return filtered_combined

    else:
        logging.error(f"Invalid preprocessing mode: {mode}. Choose 'codes', 'codes_only_name', 'meaning_unit', or 'combined'.")
        sys.exit(1)

# =======================
# Embedding Functions
# =======================

def embed_texts(
    texts: List[str],
    client,
    embedding_model: str,
    batch_size: int,
    instruction: Optional[str] = None
) -> List[List[float]]:
    """
    Embed a list of texts using the specified embedding model with optional instruction.
    """
    embeddings = []

    if instruction:
        texts = [f"Instruction: {instruction} Text: {text}" for text in texts]

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch = texts[i:i + batch_size]
        batch_number = i // batch_size + 1
        logging.debug(f"Processing batch {batch_number}...")
        try:
            response = client.embeddings.create(
                input=batch,
                model=embedding_model
            )
            batch_embeddings = [res.embedding for res in response.data]
            embeddings.extend(batch_embeddings)
            logging.debug(f"Batch {batch_number} embedded successfully.")
        except Exception as e:
            logging.error(f"Error embedding batch {batch_number}: {e}")
            continue

    logging.info(f"Generated embeddings for {len(embeddings)} texts.")
    return embeddings

# =======================
# Clustering Functions
# =======================

def hierarchical_cluster_codes(
    embeddings: List[List[float]],
    code_data: List[Dict[str, Any]],
    distance_threshold: float = 0.5,
    linkage_method: str = 'average',
    n_components: Optional[int] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Perform optional dimensionality reduction and hierarchical clustering.
    """
    X = np.array(embeddings, dtype='float32')

    # Optional dimensionality reduction via UMAP
    if n_components and n_components > 0:
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            X = reducer.fit_transform(X)
            logging.info(f"UMAP reduced dimensions to {n_components} components.")
        except Exception as e:
            logging.error(f"Error during UMAP dimensionality reduction: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping dimensionality reduction for hierarchical clustering.")

    # Normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_normalized = X / norms

    # Hierarchical clustering with linkage
    try:
        Z = linkage(X_normalized, method=linkage_method, metric='cosine')
        logging.debug("Linkage matrix computed successfully.")
    except Exception as e:
        logging.error(f"Error computing linkage matrix: {e}")
        sys.exit(1)

    # Plot dendrogram (optional for hierarchical)
    try:
        # Extract labels based on preprocessing mode
        labels = []
        for item in code_data:
            if 'code_name' in item:
                labels.append(item.get('code_name', ''))
            elif 'meaning_unit_string' in item:
                labels.append(item.get('meaning_unit_string', ''))
            else:
                labels.append('')
        plot_dendrogram(Z, labels=labels)
    except Exception as e:
        logging.error(f"Error plotting dendrogram: {e}")

    # Assign cluster labels based on distance threshold
    try:
        labels = fcluster(Z, t=distance_threshold, criterion='distance')
        logging.info(f"Assigned cluster labels with distance threshold {distance_threshold}.")
    except Exception as e:
        logging.error(f"Error assigning cluster labels: {e}")
        sys.exit(1)

    clusters = {}
    for label, code in zip(labels, code_data):
        clusters.setdefault(label, []).append(code)

    logging.info(f"Formed {len(clusters)} clusters (hierarchical).")
    return clusters

def hdbscan_cluster_codes(
    embeddings: List[List[float]],
    code_data: List[Dict[str, Any]],
    min_cluster_size: int = 5,
    min_samples: int = 1,
    n_components: Optional[int] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Perform optional dimensionality reduction and HDBSCAN clustering.
    """
    X = np.array(embeddings, dtype='float32')

    # Optional dimensionality reduction via UMAP
    if n_components and n_components > 0:
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            X = reducer.fit_transform(X)
            logging.info(f"UMAP reduced dimensions to {n_components} components.")
        except Exception as e:
            logging.error(f"Error during UMAP dimensionality reduction: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping dimensionality reduction for HDBSCAN.")

    # Normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_normalized = X / norms

    # Perform HDBSCAN clustering
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(X_normalized)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"HDBSCAN found {n_clusters} clusters.")
    except Exception as e:
        logging.error(f"Error during HDBSCAN clustering: {e}")
        sys.exit(1)

    # Build cluster dictionary excluding noise points
    clusters = {}
    for label, code in zip(labels, code_data):
        if label == -1:
            continue  # Exclude noise
        clusters.setdefault(label, []).append(code)

    logging.info(f"Formed {len(clusters)} clusters (HDBSCAN), excluding noise.")
    return clusters

def plot_dendrogram(Z: np.ndarray, labels: List[str]):
    """
    Plot and save a dendrogram for hierarchical clustering.
    """
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Codes' if any(labels) else 'Items')
    plt.ylabel('Distance')
    plt.tight_layout()
    try:
        dendro_path = Path(OUTPUT_DIR) / 'dendrogram.png'
        plt.savefig(dendro_path)
        logging.info(f"Dendrogram saved to '{dendro_path}'.")
        plt.close()
    except Exception as e:
        logging.error(f"Error saving dendrogram: {e}")

# =======================
# Cluster Labeling Functions
# =======================

def label_clusters_with_llm(
    clusters: Dict[int, List[Dict[str, Any]]],
    client,
    criteria: str,
    llm_model: str,
    mode: str
) -> Dict[str, Dict[str, Any]]:
    """
    Use an LLM to label each cluster with a parent theme.
    """
    labeled_clusters = {}
    for cluster_id, items in clusters.items():
        cluster_texts = []
        if mode == "codes":
            for item in items:
                code_name = item.get('code_name', '')
                code_justification = item.get('code_justification', '')
                cluster_texts.append(f"- Code Name: {code_name}\n  Code Justification: {code_justification}")
        elif mode == "codes_only_name":
            for item in items:
                code_name = item.get('code_name', '')
                # No justification included in the prompt
                cluster_texts.append(f"- Code Name: {code_name}")
        elif mode == "meaning_unit":
            for item in items:
                # We have meaning_unit_string and assigned_code_list,
                # but we only embedded the meaning_unit_string. Still, we can display code details here:
                mu = item.get('meaning_unit_string', '')
                code_list = item.get('assigned_code_list', [])
                # Build a short text that includes the code names + justifications
                codes_text = "; ".join(
                    f"{c.get('code_name', '')}: {c.get('code_justification', '')}"
                    for c in code_list if c
                )
                cluster_texts.append(
                    f"- Meaning Unit: {mu}\n  Assigned Codes: {codes_text}"
                )
        elif mode == "combined":
            for item in items:
                # Extract separate fields
                meaning_unit = item.get('meaning_unit_string', '')
                code_name = item.get('code_name', '')
                code_justification = item.get('code_justification', '')
                cluster_texts.append(
                    f"- Meaning Unit: {meaning_unit}\n"
                    f"  Code Name: {code_name}\n"
                    f"  Code Justification: {code_justification}"
                )
        else:
            logging.error(f"Invalid mode for labeling: {mode}")
            sys.exit(1)

        prompt = f"""
You are a researcher analyzing qualitative data. These items have been grouped into a cluster.
Your task: Provide a concise theme label that describes the overarching topic or theme of this cluster.
The theme should reflect: {criteria}

Here are the items in the cluster:
{chr(10).join(cluster_texts)}

Respond with a single short phrase or sentence that best captures the theme of this cluster.
Ensure the label is specific and directly related to the provided items.
"""

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.5,
                n=1
            )
            theme_response = response.choices[0].message.content.strip()
            if not theme_response:
                raise ValueError("Empty response received from LLM.")
            logging.debug(f"Cluster {cluster_id} labeled as: {theme_response}")
        except Exception as e:
            logging.warning(f"Error labeling cluster {cluster_id}: {e}")
            theme_response = "Unlabeled"

        # Exclude 'combined_text' from the output items and keep other fields
        cleaned_items = []
        for item in items:
            cleaned_item = {}
            if 'code_name' in item:
                cleaned_item['code_name'] = item['code_name']
            if 'code_justification' in item:
                cleaned_item['code_justification'] = item['code_justification']
            if 'meaning_unit_string' in item:
                cleaned_item['meaning_unit_string'] = item['meaning_unit_string']
            if 'assigned_code_list' in item:
                cleaned_item['assigned_code_list'] = item['assigned_code_list']
            cleaned_items.append(cleaned_item)

        labeled_clusters[str(cluster_id)] = {
            "theme_label": theme_response,
            "items": cleaned_items
        }

    logging.info("Clusters have been labeled with themes.")
    return labeled_clusters

# =======================
# JSON Handling Functions
# =======================

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logging.info(f"Loaded data from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        sys.exit(1)

def save_json(data: Any, file_path: Path):
    """
    Save data to a JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data saved to '{file_path}'.")
    except IOError as e:
        logging.error(f"Error writing to file {file_path}: {e}")
        sys.exit(1)

# =======================
# Embedding Caching Functions
# =======================

def get_embeddings_file_path(json_file_path: str) -> Path:
    """
    Generate a corresponding embeddings file path based on the JSON file path.
    """
    json_path = Path(json_file_path)
    embeddings_filename = json_path.stem + "_embeddings.npy"
    return Path(OUTPUT_DIR) / embeddings_filename

def load_embeddings_file(embeddings_file_path: Path) -> List[List[float]]:
    """
    Load embeddings from a .npy file.
    """
    try:
        embeddings = np.load(embeddings_file_path, allow_pickle=True)
        logging.info(f"Loaded embeddings from '{embeddings_file_path}'.")
        return embeddings.tolist()
    except FileNotFoundError:
        logging.error(f"Embeddings file '{embeddings_file_path}' not found.")
        return []
    except Exception as e:
        logging.error(f"Error loading embeddings from '{embeddings_file_path}': {e}")
        return []

def save_embeddings_file(embeddings: List[List[float]], embeddings_file_path: Path):
    """
    Save embeddings to a .npy file.
    """
    try:
        np.save(embeddings_file_path, np.array(embeddings))
        logging.info(f"Embeddings saved to '{embeddings_file_path}'.")
    except Exception as e:
        logging.error(f"Error saving embeddings to '{embeddings_file_path}': {e}")
        sys.exit(1)

# =======================
# Visualization Functions
# =======================

def extract_cluster_data(data: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Extract cluster labels and item counts from the JSON data.

    Parameters:
        data (dict): The JSON data containing clusters.

    Returns:
        list of tuples: Each tuple contains (theme_label, item_count).
    """
    clusters = data.get('clusters', {})
    cluster_data = []

    for cluster_id, cluster_info in clusters.items():
        theme_label = cluster_info.get('theme_label', f"Cluster {cluster_id}")
        items = cluster_info.get('items', [])
        item_count = len(items)
        cluster_data.append((theme_label, item_count))

    # Sort clusters from largest to smallest based on item count
    cluster_data.sort(key=lambda x: x[1], reverse=True)

    return cluster_data

def plot_bar_chart(cluster_data: List[Tuple[str, int]]):
    """
    Plot a horizontal bar chart of clusters.

    Parameters:
        cluster_data (list of tuples): Each tuple contains (theme_label, item_count).
    """
    if not cluster_data:
        logging.warning("No cluster data available to plot.")
        return

    # Unzip the cluster data
    labels, counts = zip(*cluster_data)

    # Create a horizontal bar chart
    plt.figure(figsize=(12, max(8, len(labels) * 0.4)))
    bars = plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Number of Items')
    plt.title('Cluster Item Counts')

    # Invert y-axis to have the largest cluster on top
    plt.gca().invert_yaxis()

    # Add counts to the bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_width() + max(counts)*0.01,
            bar.get_y() + bar.get_height()/2,
            str(count),
            va='center'
        )

    plt.tight_layout()
    try:
        chart_path = Path(OUTPUT_DIR) / 'cluster_item_counts.png'
        plt.savefig(chart_path)
        logging.info(f"Bar chart saved to '{chart_path}'.")
        plt.close()
    except Exception as e:
        logging.error(f"Error saving bar chart: {e}")

# =======================
# Main Function
# =======================

def main():
    """
    Main function to orchestrate the clustering of qualitative codes and visualize the results.
    """
    # Setup logging
    setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)

    # Ensure root directory is set for custom modules
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))

    # Import custom modules with error handling
    try:
        from qual_functions import client  # Must provide the 'client' for embeddings and LLM calls
    except ImportError as e:
        logging.error(f"Custom module import failed: {e}")
        sys.exit(1)

    # Ensure the output directory exists
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directory '{OUTPUT_DIR}' is ready.")

    # Load JSON data
    data = load_json_file(JSON_FILE_PATH)

    # Define JSON schema
    schema = {
        "type": "object",
        "properties": {
            "meaning_units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "meaning_unit_id": {"type": "number"},
                        "meaning_unit_string": {"type": "string"},
                        "assigned_code_list": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "code_name": {"type": "string"},
                                    "code_justification": {"type": "string"}
                                },
                                "required": ["code_name", "code_justification"]
                            }
                        }
                    },
                    "required": [
                        "meaning_unit_id",
                        "meaning_unit_string",
                        "assigned_code_list"
                    ]
                }
            }
        },
        "required": ["meaning_units"]
    }

    # Validate JSON schema
    validate_json_schema(data, schema)

    # Preprocess data
    filtered_data = preprocess_data(data, mode=PREPROCESSING_MODE)

    # Prepare texts for embedding
    if PREPROCESSING_MODE == "codes":
        texts_to_embed = [
            f"Code Name: {cd.get('code_name', '')}\nCode Justification: {cd.get('code_justification', '')}"
            for cd in filtered_data
        ]
    elif PREPROCESSING_MODE == "codes_only_name":
        texts_to_embed = [
            cd.get('code_name', '')
            for cd in filtered_data
        ]
    elif PREPROCESSING_MODE == "meaning_unit":
        # ---------------------------------------------------
        # Even though we kept the codes in the dictionary,
        # we ONLY embed the meaning_unit_string here.
        # ---------------------------------------------------
        texts_to_embed = [
            mu.get('meaning_unit_string', '')
            for mu in filtered_data
        ]
    elif PREPROCESSING_MODE == "combined":
        texts_to_embed = [
            cd.get('combined_text', '')
            for cd in filtered_data
        ]
    else:
        logging.error(f"Unsupported preprocessing mode: {PREPROCESSING_MODE}")
        sys.exit(1)

    logging.info(f"Preparing to embed {len(texts_to_embed)} texts.")

    # Determine embeddings file path
    if EMBEDDING_FILE:
        embeddings_file_path = Path(EMBEDDING_FILE)
    else:
        embeddings_file_path = get_embeddings_file_path(JSON_FILE_PATH)

    # Reuse or generate embeddings
    if REUSE_EMBEDDINGS and embeddings_file_path.exists():
        embeddings = load_embeddings_file(embeddings_file_path)
        if len(embeddings) != len(texts_to_embed):
            logging.warning("Existing embeddings do not match the current data. Recalculating embeddings.")
            embeddings = embed_texts(texts_to_embed, client, EMBEDDING_MODEL, BATCH_SIZE, INSTRUCTION)
            save_embeddings_file(embeddings, embeddings_file_path)
    else:
        embeddings = embed_texts(texts_to_embed, client, EMBEDDING_MODEL, BATCH_SIZE, INSTRUCTION)
        save_embeddings_file(embeddings, embeddings_file_path)

    if not embeddings:
        logging.error("No embeddings were generated. Exiting.")
        sys.exit(1)

    if len(embeddings) != len(filtered_data):
        logging.error("Mismatch between number of embeddings and filtered data.")
        sys.exit(1)

    logging.info("Embeddings successfully generated.")
    embeddings_array = np.array(embeddings)
    logging.info(f"Embeddings shape: {embeddings_array.shape}")

    # Perform clustering
    if CLUSTERING_METHOD == "hierarchical":
        try:
            clusters = hierarchical_cluster_codes(
                embeddings=embeddings,
                code_data=filtered_data,
                distance_threshold=DISTANCE_THRESHOLD,
                linkage_method=LINKAGE_METHOD,
                n_components=N_COMPONENTS  # Use UMAP if set
            )
        except Exception as e:
            logging.error(f"Hierarchical clustering failed: {e}")
            sys.exit(1)
    elif CLUSTERING_METHOD == "hdbscan":
        try:
            clusters = hdbscan_cluster_codes(
                embeddings=embeddings,
                code_data=filtered_data,
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                n_components=N_COMPONENTS  # Use UMAP if set
            )
        except Exception as e:
            logging.error(f"HDBSCAN clustering failed: {e}")
            sys.exit(1)
    else:
        logging.error(f"Unsupported clustering method: {CLUSTERING_METHOD}")
        sys.exit(1)

    if not clusters:
        logging.error("No clusters were formed. Exiting.")
        sys.exit(1)

    # Calculate Silhouette Score
    try:
        if CLUSTERING_METHOD == "hierarchical":
            cluster_labels = []
            for label, items in clusters.items():
                cluster_labels.extend([label] * len(items))
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(embeddings_array, cluster_labels, metric='euclidean')
                logging.info(f"Silhouette Score (Hierarchical): {silhouette_avg:.4f}")
            else:
                logging.warning("Only one cluster found with hierarchical clustering. Silhouette Score not computed.")
                silhouette_avg = None
        else:  # HDBSCAN
            hdbscan_labels = []
            for label, items in clusters.items():
                hdbscan_labels.extend([label] * len(items))
            unique_labels = set(hdbscan_labels)
            if len(unique_labels) > 1:
                silhouette_avg = silhouette_score(embeddings_array, hdbscan_labels, metric='euclidean')
                logging.info(f"Silhouette Score (HDBSCAN): {silhouette_avg:.4f}")
            else:
                logging.warning("Only one cluster found with HDBSCAN. Silhouette Score not computed.")
                silhouette_avg = None
    except Exception as e:
        logging.error(f"Error calculating Silhouette Score: {e}")
        silhouette_avg = None

    # Label clusters using LLM
    labeled_clusters = label_clusters_with_llm(
        clusters=clusters,
        client=client,
        criteria=LABELING_CRITERIA,
        llm_model=LLM_MODEL,
        mode=PREPROCESSING_MODE
    )

    # Final output structure
    output_data = {
        "silhouette_score": silhouette_avg,
        "clusters": labeled_clusters
    }

    # Save to JSON
    if PREPROCESSING_MODE == "codes":
        clusters_output_file = output_dir / 'clusters_with_themes.json'
    elif PREPROCESSING_MODE == "codes_only_name":
        clusters_output_file = output_dir / 'clusters_with_themes_codes_only_name.json'
    elif PREPROCESSING_MODE == "meaning_unit":
        clusters_output_file = output_dir / 'clusters_with_themes_meaning_units.json'
    elif PREPROCESSING_MODE == "combined":
        clusters_output_file = output_dir / 'clusters_with_themes_combined.json'
    else:
        logging.error(f"Unsupported preprocessing mode for output: {PREPROCESSING_MODE}")
        sys.exit(1)

    save_json(output_data, clusters_output_file)

    logging.info("Clustering process completed successfully.")

    # =======================
    # Visualization
    # =======================

    # Determine the path to the clusters JSON for visualization
    global CLUSTERS_JSON_PATH
    if CLUSTERS_JSON_PATH is None:
        CLUSTERS_JSON_PATH = clusters_output_file
    else:
        CLUSTERS_JSON_PATH = Path(CLUSTERS_JSON_PATH)

    # Load clusters data
    clusters_data = load_json_file(str(CLUSTERS_JSON_PATH))

    # Extract cluster labels and counts
    cluster_data = extract_cluster_data(clusters_data)

    # Plot the bar chart
    plot_bar_chart(cluster_data)

    logging.info("Visualization completed successfully.")

# =======================
# Entry Point
# =======================

if __name__ == "__main__":
    main()
