# ATLAS Development Log: 18S Pipeline Refinement

This document details the development and strategic decisions made during the refinement of the 18S (Eukaryote) pipeline.

## 1. Development Workflow

The successful "notebook-first" methodology established during the 16S refinement was re-applied. All development was conducted in `03_Refining_18S_Preparation.ipynb` and `04_Refining_18S_Training.ipynb` before being converted to the final production scripts.

## 2. Key Challenge: Eukaryotic Taxonomy Parsing

The central challenge of this pipeline was handling the complex and inconsistent taxonomic strings found in the SILVA database for Eukaryotes.

**Problem:** The simple, position-based parser from the 16S pipeline was insufficient. Eukaryotic lineages contain numerous non-standard, intermediate ranks (e.g., "Sar", "Opisthokonta", "Metazoa") that disrupt a simple hierarchical structure.

**Solution:** A new, more robust function, `parse_eukaryote_taxonomy`, was developed. This function incorporates two key strategies:

- **A DISCARD_RANKS Set:** A comprehensive set of known non-standard ranks was compiled. The parser first strips these useless terms from the taxonomy string.

- **Intelligent Assignment:** After cleaning the string, the function uses heuristics to assign the primary ranks. It assumes the first rank is "Kingdom" and then works from the end of the string backwards, using the presence of a space to differentiate between a species and a genus. This makes the parser resilient to missing intermediate ranks.

## 3. Data Staging and Pre-processing

A key efficiency improvement was introduced in this pipeline.

**Decision:** To avoid re-scanning the entire multi-gigabyte SILVA database on every run, the script first creates a dedicated `SILVA_eukaryotes_only.fasta` file.

**Implementation:** The `01_prepare_data.py` script now checks for the existence of this file. If it is not found, the script performs the one-time, lengthy filtering process to create it. On all subsequent runs, the script reads directly from this much smaller, pre-filtered file, drastically reducing execution time.

## 4. Memory Management and Final Evaluation

The memory issues encountered during the 16S training were proactively managed.

**Challenge:** The 18S dataset is larger and more complex (more features and classes), increasing the risk of memory-related kernel crashes.

**Solution:** The robust workflow of separating the final evaluation and visualization steps was adopted from the outset. The final cell in the training notebook and the final block in the `02_train_model.py` script are dedicated, self-contained units that clear memory before loading the saved model for a final, stable evaluation.

## 5. Final Outcome

The 18S refinement was successful. We produced a pair of robust production scripts and a trained classifier with a baseline accuracy of 64.09% on the sample test set. This lower accuracy (compared to 16S) is an expected and realistic result, reflecting the significantly higher complexity of eukaryotic classification.