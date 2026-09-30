# 16S Pipeline: Data Preparation Workflow

This document details the refined step-by-step workflow for preparing the 16S (prokaryotic) data, as developed in the `01_Refining_16S_Preparation.ipynb` notebook.

## Step 1: Create a Development Sample

**Input:** The full SILVA FASTA file (`SILVA_138.1_SSURef_NR99_tax_silva.fasta`).

**Action:** Extract the first 10,000 records to create a smaller, faster `SILVA_sample_10k.fasta` for development.

## Step 2: Filter for Prokaryotes

**Input:** The sample FASTA file.

**Action:** Iterate through the sequences and keep only those whose descriptions contain "Bacteria" or "Archaea".

## Step 3: Parse Taxonomy

**Input:** The filtered list of prokaryote records.

**Action:** For each record, parse the semi-colon delimited taxonomy string into a structured format (Kingdom, Phylum, Class, etc.). Store the results in a pandas DataFrame.

## Step 4: Clean and Filter the DataFrame

**Action 1 (Remove Missing Targets):** Drop any rows from the DataFrame that do not have a value in the genus column.

**Action 2 (Remove Singletons):** Remove any genus that is only represented by a single sequence to ensure data quality for model training.

## Step 5: Feature Engineering (K-mer Counting)

**Input:** The cleaned DataFrame.

**Action:** For each sequence, calculate the frequency of all possible 6-mers (sub-sequences of length 6). Store these counts in a new `kmer_counts` column.

## Step 6: Vectorize Features and Labels

**Action 1 (Vectorize Features):** Use `DictVectorizer` to convert the `kmer_counts` column into a large, sparse numerical matrix (X).

**Action 2 (Encode Labels):** Use `LabelEncoder` to convert the text-based genus column into a numerical vector of labels (y).

## Step 7: Split Data

**Input:** The final X and y matrices.

**Action:** Split the data into an 80% training set and a 20% testing set using a stratified split to maintain class proportions.

## Step 8: Save Artifacts

**Action:** Save the final training/testing data splits (`.npz`, `.npy`) and the fitted vectorizer and `label_encoder` (`.pkl`) to disk for use in the model training phase.