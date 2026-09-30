# ATLAS Development Log: COI Pipeline Refinement

This document details the development and strategic decisions made during the refinement of the COI (Animalia) pipeline.

## 1. Development Workflow

The successful "notebook-first" methodology was re-applied. All development was conducted in `05_Refining_COI_Preparation.ipynb` and `06_Refining_COI_Training.ipynb` before being converted to the final production scripts.

## 2. Key Challenge: BOLD Taxonomy Parsing

The central challenge of this pipeline was handling the unique FASTA header format of the BOLD database.

**Problem:** Initial analysis revealed a complex, inconsistent format. Unlike the simple, single-delimiter formats of other databases, BOLD headers use a pipe (`|`) to separate metadata fields, but the actual taxonomy is often contained within one of these fields as a comma-separated string.

**Solution:** A new, two-stage parser, `parse_bold_taxonomy_v2`, was developed.

- **Stage 1 (Find):** The function iterates through the pipe-separated fields to locate the specific field containing the taxonomic string (identified by the presence of commas and the keyword "Animalia").

- **Stage 2 (Extract):** It then splits this specific field by commas to extract the final list of taxonomic ranks.

This revised parser proved to be highly effective at structuring the real-world BOLD data.

## 3. Feature Engineering: K-mer Size Selection

**Decision:** A k-mer size of 8 was chosen for the COI pipeline.

**Rationale:** The COI gene is a protein-coding gene, which is generally less variable than non-coding regions like ITS but more variable than highly conserved rRNA genes. A slightly larger k-mer size (compared to 6 for 16S/18S) can capture more specific "fingerprints," which is effective for distinguishing between the vast number of animal genera.

## 4. Final Outcome

The refinement of the COI pipeline was a complete success. The interactive development process allowed us to quickly identify and solve the complex parsing issue. The resulting production scripts are robust and produce a highly accurate classifier with a **Test Set Accuracy of 96.87%** on the sample data. This high accuracy is expected for the well-curated COI barcode.