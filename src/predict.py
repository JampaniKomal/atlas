#!/usr/bin/env python3
# =============================================================================
# ATLAS: MASTER PREDICTION SCRIPT
# =============================================================================
# This script serves as the primary engine for the ATLAS system's analysis pipeline.
# It is designed to be a single, callable module that can process a FASTA file from
# start to finish. It integrates the logic of both the "Filter" (classification)
# and "Explorer" (clustering) pipelines in a memory-safe and efficient manner.
#
# USAGE FROM COMMAND LINE:
#
#   To run an analysis, execute the script with the `--input_fasta` argument:
#
#   python src/predict.py --input_fasta "path/to/your/file.fasta"
#
#   The script will print a detailed report to the console and also save a copy
#   as a .txt file in the 'reports' folder.
#
# ENVIRONMENT REQUIREMENTS:
#
#   This script MUST be run from within the project's Conda environment
#   (e.g., 'atlas' or 'atlas-cpu'). This environment manages all necessary
#   dependencies (TensorFlow, Biopython, etc.).
#
# GPU vs. CPU SUPPORT:
#
#   The script is fully compatible with both GPU and CPU systems. The 'tensorflow'
#   library will automatically use an available NVIDIA GPU for faster processing.
#   If a GPU is not detected, it will gracefully fall back to the CPU.
#
# =============================================================================

# --- Core Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import pickle
import argparse
import sys
from collections import Counter
import gc
import io
import uuid
import logging
import time

# --- Machine Learning & Data Processing Imports ---
from tensorflow.keras.models import load_model
from scipy.sparse import csc_matrix
import tensorflow as tf
import hdbscan
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import normalize
from tqdm import tqdm

# --- Configuration ---
# All file paths are relative to the project root for portability.
project_root = Path(__file__).parent.parent
MODELS_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---

def get_kmer_counts(sequence: str, k: int) -> dict:
    """
    Calculates k-mer counts for a single DNA sequence.

    This function iterates through the sequence, extracting all k-mers
    (sub-sequences of length k) and counting their occurrences. It ignores
    any k-mer that contains an 'N', which typically represents an unknown
    nucleotide.

    Args:
        sequence (str): The DNA sequence string.
        k (int): The size of the k-mer.

    Returns:
        dict: A dictionary where keys are k-mers and values are their counts.
    """
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            counts[kmer] += 1
    return dict(counts)

def sequence_to_kmers(sequence_str: str, k: int) -> list:
    """Converts a DNA sequence string into a list of its k-mers."""
    return [sequence_str[i:i+k] for i in range(len(sequence_str) - k + 1)]

# --- Classifier Class ---

class TaxonClassifier:
    """
    A wrapper class to load and use a trained Keras model and its artifacts.

    This class encapsulates the model, DictVectorizer, and LabelEncoder,
    ensuring they are all loaded correctly for a specific marker (e.g., 16s, coi).
    """
    def __init__(self, marker_name: str, kmer_size: int, verbose: bool):
        self.name = marker_name
        self.kmer_size = kmer_size
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = self._load_artifacts(verbose)

    def _load_artifacts(self, verbose: bool) -> bool:
        """
        Loads the model, vectorizer, and label encoder from disk.

        Args:
            verbose (bool): If True, prints verbose loading messages.

        Returns:
            bool: True if all artifacts were successfully loaded, False otherwise.
        """
        if verbose:
            logging.info(f"  - Attempting to load '{self.name}' model...")
        try:
            model_path = MODELS_DIR / f"{self.name}_genus_classifier.keras"
            vectorizer_path = MODELS_DIR / f"{self.name}_genus_vectorizer.pkl"
            encoder_path = MODELS_DIR / f"{self.name}_genus_label_encoder.pkl"
            
            if not model_path.exists() or not vectorizer_path.exists() or not encoder_path.exists():
                if verbose:
                    logging.warning(f"  - Skipping {self.name}: Required model artifacts not found.")
                return False
            
            self.model = load_model(model_path)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            if verbose:
                logging.info(f"  - Successfully loaded '{self.name}' model.")
            return True
        except Exception as e:
            logging.error(f"Error loading {self.name} model: {e}")
            return False

    def predict(self, sequence: str, confidence_threshold: float = 0.8) -> tuple[str, float]:
        """
        Predicts the taxonomic genus for a single DNA sequence.

        Args:
            sequence (str): The DNA sequence to classify.
            confidence_threshold (float): The minimum probability required to
                                          consider a prediction valid.

        Returns:
            tuple[str, float]: The predicted genus (or None) and the
                               prediction probability.
        """
        if not self.is_loaded:
            return None, 0.0
            
        kmer_counts = get_kmer_counts(sequence, self.kmer_size)
        
        if not kmer_counts:
            return None, 0.0

        # Transform the k-mer counts into the numerical vector the model expects
        vectorized_sequence = self.vectorizer.transform([kmer_counts])
        
        # Predict the probabilities for all classes
        prediction_probabilities = self.model.predict(vectorized_sequence, verbose=0)[0]
        
        # Get the highest probability and its corresponding class index
        top_prob = np.max(prediction_probabilities)
        top_class_index = np.argmax(prediction_probabilities)
        
        # Return the prediction only if it meets the confidence threshold
        if top_prob >= confidence_threshold:
            predicted_label = self.label_encoder.inverse_transform([top_class_index])[0]
            return predicted_label, top_prob
        
        # Otherwise, return None for the label
        return None, top_prob

# --- System & Explorer Functions ---

def check_gpu_status() -> str:
    """Checks and returns the GPU availability status."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU is available: {gpus[0].name}"
    else:
        return "GPU not found. Running on CPU."

def explorer_step_1_vectorize(sequences: list) -> np.ndarray:
    """
    Vectorizes unclassified sequences using a pre-trained Doc2Vec model.

    These are, by definition, sequences the Filter models couldn't classify,
    so their IDs were never part of the Doc2Vec model's training corpus.
    infer_vector() is the correct API for embedding genuinely new documents
    with a trained Doc2Vec model (looking them up via model.dv[...] only
    works for documents that were actually present at training time).

    Args:
        sequences (list): A list of Bio.SeqRecord objects.

    Returns:
        np.ndarray: A normalized matrix of sequence vectors.
    """
    KMER_SIZE = 6
    doc2vec_model_path = MODELS_DIR / "explorer_doc2vec.model"

    if not doc2vec_model_path.exists():
        return np.array([])

    doc2vec_model = Doc2Vec.load(str(doc2vec_model_path))

    sequence_vectors = np.array([
        doc2vec_model.infer_vector(sequence_to_kmers(str(s.seq), KMER_SIZE))
        for s in sequences
    ])
    sequence_vectors = normalize(sequence_vectors)

    return sequence_vectors

def explorer_step_2_cluster(sequence_vectors: np.ndarray) -> np.ndarray:
    """
    Clusters sequence vectors using HDBSCAN.

    Args:
        sequence_vectors (np.ndarray): The matrix of sequence vectors.

    Returns:
        np.ndarray: An array of cluster labels for each sequence.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(sequence_vectors)
    return cluster_labels

def explorer_step_3_interpret(sequences: list, sequence_vectors: np.ndarray, cluster_labels: np.ndarray) -> str:
    """
    Generates a report for the discovered clusters.

    It finds a representative sequence for each cluster and summarizes the findings.
    Note: This version does not perform BLAST, as that is a slow, online process.

    Args:
        sequences (list): The list of original Bio.SeqRecord objects.
        sequence_vectors (np.ndarray): The sequence vectors.
        cluster_labels (np.ndarray): The cluster assignments.

    Returns:
        str: A human-readable report string.
    """
    report_lines = []
    unique_cluster_ids = sorted(np.unique(cluster_labels))
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1)

    if not unique_cluster_ids:
        return "No significant clusters were discovered in the unclassified sequences."

    for cluster_id in unique_cluster_ids:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = sequence_vectors[cluster_indices]
        
        centroid = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        rep_index_in_cluster = np.argmin(distances)
        
        representative_sequence = sequences[cluster_indices[rep_index_in_cluster]]
        
        report_lines.append(f"Cluster ID: {cluster_id}")
        report_lines.append(f"  - Size: {len(cluster_indices)} sequences")
        report_lines.append(f"  - Representative Sequence ID: {representative_sequence.id}")
        report_lines.append(f"  - Note: A manual BLAST search on this representative sequence ID is recommended to hypothesize a taxonomic identity.\n")

    return "\n".join(report_lines)

# =============================================================================
# --- Main Analysis Function ---
# =============================================================================
def run_analysis(input_fasta_path: Path, report_name: str = None, verbose: bool = False):
    """
    The main analysis pipeline.

    This is the core function of the script. It processes the input FASTA file
    in two stages:
    1.  **Filter**: It attempts to classify each sequence using the four
        pre-trained neural network models (16s, 18s, coi, its).
    2.  **Explorer**: Any sequences that could not be classified are then
        passed to the Explorer pipeline for novel taxa discovery via
        clustering.

    Args:
        input_fasta_path (Path): The Path object to the input FASTA file.
        report_name (str): The desired name for the output report file.
        verbose (bool): If True, prints detailed logging messages.
        
    Returns:
        dict: A dictionary containing the analysis results, including a
              formatted report, and structured data for potential further use.
    """
    
    start_time = time.time()
    
    # --- 1. Check GPU Status ---
    gpu_status = check_gpu_status()

    # --- 2. Load and Run Filter Models Sequentially ---
    potential_classifiers = [
        ("16s", 6), ("18s", 6), ("coi", 8), ("its", 7)
    ]
    
    classified_results = Counter()
    unclassified_sequences = []

    try:
        input_sequences = list(SeqIO.parse(input_fasta_path, "fasta"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse FASTA file: {e}"}
    
    unclassified_sequences = input_sequences.copy()

    if verbose:
        logging.info("\n[  STAGE 1: KNOWN TAXA CLASSIFICATION  ]")
        logging.info("---------------------------------------------")

    # The sequential loading and unloading of models is critical for memory management.
    for name, kmer_size in potential_classifiers:
        classifier = TaxonClassifier(name, kmer_size, verbose)
        if classifier.is_loaded:
            if verbose:
                logging.info(f"  - Classifying with {name} model...")
            newly_classified = []
            sequences_to_reprocess = []
            
            for seq_record in unclassified_sequences:
                label, prob = classifier.predict(str(seq_record.seq))
                if label:
                    classified_results[label] += 1
                    newly_classified.append(seq_record)
                else:
                    sequences_to_reprocess.append(seq_record)
            
            unclassified_sequences = sequences_to_reprocess
            if verbose:
                logging.info(f"    - Found {len(newly_classified)} known sequences.")
        
        # Explicitly unload the model and free memory
        del classifier
        tf.keras.backend.clear_session()
        gc.collect()

    # --- 3. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = ""
    if unclassified_sequences:
        if verbose:
            logging.info("\n[  STAGE 2: NOVEL TAXA EXPLORATION  ]")
            logging.info("------------------------------------------")
            logging.info(f"  - {len(unclassified_sequences)} sequences remain unclassified. Starting explorer pipeline.")
        try:
            sequence_vectors = explorer_step_1_vectorize(unclassified_sequences)
            cluster_labels = explorer_step_2_cluster(sequence_vectors)
            explorer_report_content = explorer_step_3_interpret(unclassified_sequences, sequence_vectors, cluster_labels)
        except Exception as e:
            explorer_report_content = f"Error during explorer pipeline: {e}"
    else:
        if verbose:
            logging.info("\n[  STAGE 2: NOVEL TAXA EXPLORATION  ]")
            logging.info("------------------------------------------")
            logging.info("  - All sequences were classified by the Filter models. Skipping explorer.")
        explorer_report_content = "All sequences were classified by the Filter models. No need for the Explorer pipeline."
    
    # --- 4. Generate Final Report ---
    sorted_results = sorted(classified_results.items(), key=lambda item: item[1], reverse=True)
    classified_results_str = 'No known organisms were identified.' if not classified_results else '\n'.join([f"- {genus}: {count} sequences" for genus, count in sorted_results])

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    final_report_text = f"""
+-------------------------------------------------------------+
|          ATLAS: AI Taxonomic Learning & Analysis System     |
|                         FINAL REPORT                        |
+-------------------------------------------------------------+

[  ENVIRONMENT & INPUT  ]
+-------------------------------------------------------------+
| GPU Status: {gpu_status}
| Input File: {Path(input_fasta_path).name}
| Total Sequences Analyzed: {len(input_sequences)}
| Analysis Time: {elapsed_time:.2f} seconds
+-------------------------------------------------------------+

[  PART 1: FILTER RESULTS  ]
+-------------------------------------------------------------+
| (Classification of known organisms)
{classified_results_str}
+-------------------------------------------------------------+

[  PART 2: EXPLORER RESULTS  ]
+-------------------------------------------------------------+
| (Discovery of novel or unclassified taxa)
{explorer_report_content}
+-------------------------------------------------------------+
"""
    
    # --- 5. Save Report to File ---
    report_filename = (report_name if report_name else f"ATLAS_REPORT_{Path(input_fasta_path).stem}_{uuid.uuid4().hex}") + ".txt"
    report_path = REPORTS_DIR / report_filename

    try:
        with open(report_path, "w") as f:
            f.write(final_report_text.strip())
        if verbose:
            logging.info(f"\n[SUCCESS] Analysis complete. Report saved to: {report_path}")
    except Exception as e:
        logging.error(f"\n[ERROR] Failed to save report file: {e}")
    
    return {
        "status": "success",
        "report_content": final_report_text.strip(),
        "classified_results": {k: v for k, v in classified_results.items()},
    }

# --- ASCII Art Header for "ATLAS" (matches cli/__main__.py) ---
ATLAS_ASCII = r"""
    _  _____ _        _    ____
   / \|_   _| |      / \  / ___|
  / _ \ | | | |     / _ \ \___ \
 / ___ \| | | |___ / ___ \ ___) |
/_/   \_\_| |_____/_/   \_\____/

 A.T.L.A.S. Command-Line Interface
---------------------------------------
"""

# =============================================================================
# --- Main Execution Block (for CLI) ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ATLAS: AI Taxonomic Learning & Analysis System. Processes a FASTA file through filter and explorer pipelines.",
        usage="python -m cli --input_fasta <path_to_file> [--report-name <name>] [--verbose]"
    )
    parser.add_argument(
        '--input_fasta', 
        type=Path, 
        help="Path to the input FASTA file for analysis."
    )
    parser.add_argument(
        '--report-name',
        type=str,
        help="A custom name for the output report file (e.g., 'my_report')."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose output to show detailed analysis steps."
    )
    args = parser.parse_args()
    
    # Set logging level based on the --verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    fasta_path = args.input_fasta

    # If no argument is provided, switch to interactive mode
    if not fasta_path:
        print(ATLAS_ASCII)
        logging.warning("Welcome to the ATLAS analysis tool.")
        logging.warning("This program will process a FASTA file and generate a taxonomic report.")
        logging.warning("To proceed, please enter the path to your FASTA file.")
        logging.warning("Example: data/raw/your_file.fasta\n")
        input_path = input("Enter FASTA file path: ")
        fasta_path = Path(input_path.strip())

    if not fasta_path.is_file():
        logging.error(f"\n[ERROR] File not found: {fasta_path}")
        logging.error("Please check the path and try again.")
        sys.exit(1)

    logging.info(f"\nProcessing '{fasta_path}'...")
    logging.info("This may take a few minutes, please wait...")

    # Call the core analysis function from the predict.py module
    result = run_analysis(
        input_fasta_path=fasta_path,
        report_name=args.report_name,
        verbose=args.verbose
    )

    # Print the report from the returned dictionary
    if result["status"] == "success":
        print("\n" + result["report_content"])
    else:
        logging.error(f"\n[ERROR] An error occurred during the analysis:")
        logging.error(f"  > {result['message']}")
        sys.exit(1)
