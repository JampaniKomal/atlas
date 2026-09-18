# =============================================================================
# ATLAS - 16S PIPELINE - SCRIPT 1: PREPARE DATA
# =============================================================================
# This script converts the raw SILVA FASTA file into clean, numerical,
# and model-ready training and testing sets.
#
# WORKFLOW:
# 1.  Reads the source FASTA file (either a sample or the full dataset).
# 2.  Filters sequences to keep only Bacteria and Archaea.
# 3.  Parses the taxonomy string for each record.
# 4.  Cleans the data by removing records with missing genus labels and
#     genera with too few members (less than 3).
# 5.  Engineers k-mer features from the DNA sequences.
# 6.  Vectorizes features and labels into numerical matrices.
# 7.  Splits the data into training and testing sets.
# 8.  Saves all final artifacts (.npz, .npy, .pkl) to disk.
# =============================================================================

# --- Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
import pickle
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# --- FIX: Add the missing import for saving sparse matrices ---
from scipy.sparse import save_npz

# --- Configuration ---
# This section contains all the key settings for the script.

# Set the project root path
project_root = Path(__file__).parent.parent.parent

# Define directories
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- SCRIPT BEHAVIOR SWITCH ---
# Set to True to run on the small 10k sample for quick tests.
# To run on the full dataset, comment out the first line and uncomment the second.
USE_SAMPLE = True
# USE_SAMPLE = False

# --- Data File Paths ---
FULL_SILVA_PATH = RAW_DATA_DIR / "SILVA_138.1_SSURef_NR99_tax_silva.fasta"
SAMPLE_SILVA_PATH = RAW_DATA_DIR / "SILVA_sample_10k.fasta"

# --- Feature and Model Parameters ---
KMER_SIZE = 6
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3  # Strategic decision to ensure data quality
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- Helper Functions ---

def parse_silva_taxonomy(taxonomy_str):
    """Parses a SILVA taxonomy string into a dictionary of ranks."""
    DISCARD_RANKS = {'uncultured', 'unidentified', 'metagenome'}
    parsed_ranks = {'kingdom': None, 'phylum': None, 'class': None, 'order': None, 'family': None, 'genus': None, 'species': None}
    ranks = [rank.strip() for rank in taxonomy_str.split(';') if rank.strip() and rank.strip().lower() not in DISCARD_RANKS]
    if not ranks: return parsed_ranks
    if len(ranks) > 0: parsed_ranks['kingdom'] = ranks[0]
    if len(ranks) > 1: parsed_ranks['phylum'] = ranks[1]
    if len(ranks) > 2: parsed_ranks['class'] = ranks[2]
    if len(ranks) > 3: parsed_ranks['order'] = ranks[3]
    if len(ranks) > 4: parsed_ranks['family'] = ranks[4]
    if len(ranks) > 5: parsed_ranks['genus'] = ranks[5]
    if len(ranks) > 6: parsed_ranks['species'] = ranks[6]
    return parsed_ranks

def get_kmer_counts(sequence, k):
    """Calculates the k-mer counts for a given DNA sequence."""
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            counts[kmer] += 1
    return dict(counts)

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":

    # --- Step 1: Select Input File ---
    if USE_SAMPLE:
        input_fasta_path = SAMPLE_SILVA_PATH
        print(f"--- Running in SAMPLE mode on: {input_fasta_path.name} ---")
        if not input_fasta_path.exists():
            print("Sample file not found. Please run the `01_Refining_16S_Preparation.ipynb` notebook to create it.")
            exit()
    else:
        input_fasta_path = FULL_SILVA_PATH
        print(f"--- Running in FULL DATASET mode on: {input_fasta_path.name} ---")
        if not input_fasta_path.exists():
            print("Full SILVA file not found. Please download it and place it in the `data/raw` directory.")
            exit()

    # --- Step 2: Filter for Prokaryotes ---
    print("\nStep 2: Filtering for Bacteria & Archaea...")
    prokaryote_records = []
    with open(input_fasta_path, "r") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), desc="Filtering sequences"):
            if "bacteria" in record.description.lower() or "archaea" in record.description.lower():
                prokaryote_records.append(record)
    print(f"Found {len(prokaryote_records)} prokaryote sequences.")

    # --- Step 3: Parse Taxonomy ---
    print("\nStep 3: Parsing taxonomy...")
    parsed_data = []
    for record in tqdm(prokaryote_records, desc="Parsing taxonomy"):
        accession, taxonomy_str = record.description.split(' ', 1)
        taxonomy_dict = parse_silva_taxonomy(taxonomy_str)
        taxonomy_dict['id'] = record.id
        taxonomy_dict['sequence'] = str(record.seq)
        parsed_data.append(taxonomy_dict)
    df = pd.DataFrame(parsed_data)
    print(f"Created DataFrame with {len(df)} rows.")

    # --- Step 4: Clean and Filter DataFrame ---
    print("\nStep 4: Cleaning and filtering data...")
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    print(f"Final DataFrame has {len(df_filtered)} sequences after cleaning.")

    # --- Step 5: Feature Engineering (K-mer Counting) ---
    print(f"\nStep 5: Engineering {KMER_SIZE}-mer features...")
    df_filtered['kmer_counts'] = list(tqdm((get_kmer_counts(seq, KMER_SIZE) for seq in df_filtered['sequence']), total=len(df_filtered), desc="Calculating k-mers"))

    # --- Step 6: Vectorize Features and Labels ---
    print("\nStep 6: Vectorizing data...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(df_filtered['kmer_counts'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered[TARGET_RANK])
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape:   {y.shape}")

    # --- Step 7: Split Data ---
    print("\nStep 7: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")

    # --- Step 8: Save Artifacts ---
    print("\nStep 8: Saving all artifacts to disk...")
    save_npz(PROCESSED_DATA_DIR / "X_train_16s.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_16s.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_16s.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_16s.npy", y_test)
    with open(MODELS_DIR / "16s_genus_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "16s_genus_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("All artifacts saved successfully.")
    print("\n--- DATA PREPARATION COMPLETE ---")

