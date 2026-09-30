# ATLAS Development Log: ITS Pipeline Refinement

This document details the development, challenges, and strategic decisions made during the creation of the fourth and final "Filter" pipeline for the ITS (Fungi) marker.

## 1. Development Workflow

The successful "notebook-first" methodology was applied. All development was conducted in `07_Refining_ITS_Preparation.ipynb` and `08_Refining_ITS_Training.ipynb` before being converted into the final production scripts in `src/pipeline_its/`.

## 2. Key Challenges and Solutions

The ITS pipeline presented two unique and critical technical challenges that were identified and solved through the interactive notebook workflow.

### Challenge: UNITE Database Taxonomy Parsing

The central challenge was handling the unique FASTA header format of the UNITE database.

**Problem:** The format is `ACCESSION|...|k__Fungi;p__Ascomycota...`. An initial parser was built assuming the taxonomy was the second pipe-separated element (`split('|')[1]`), which resulted in a DataFrame of None values.

**Analysis:** Interactive inspection in the notebook revealed the taxonomy string was consistently the last element in the header.

**Solution:** A revised parser, `parse_unite_taxonomy_v2`, was created. This function correctly targets the last element (`split('|')[-1]`) before splitting by semicolons and parsing the `k__` prefixes. This proved to be a robust solution.

### Challenge: Kernel Crash During Final Evaluation

A memory-related issue, previously seen in the 16S pipeline, re-emerged with greater severity.

**Problem:** The Jupyter kernel crashed consistently when `model.evaluate()` was called on the test set.

**Analysis:** This was diagnosed as GPU VRAM exhaustion. The combination of a high feature count (18,837 k-mers) and a large number of classes (634 genera) made the evaluation step extremely memory-intensive.

**Solution:** The problem was solved by explicitly setting a smaller batch_size in the `model.evaluate(..., batch_size=16)` call. This forces TensorFlow to process the test set in smaller, more manageable chunks, preventing the memory overflow and allowing the evaluation to complete successfully.

## 3. Feature Engineering: K-mer Size Selection

**Decision:** A k-mer size of 7 was chosen for the ITS pipeline.

**Rationale:** The ITS region is a non-coding spacer and is known to be more variable than the rRNA genes (16S/18S) but has different sequence properties than the protein-coding COI gene. A k-mer of 7 provides a good balance, capturing more specific signatures than k=6 without creating an unnecessarily large feature space.

## 4. Final Outcome

The refinement of the ITS pipeline was successful. The interactive development process was crucial for debugging both the novel parser and the critical memory issue. The final production scripts are robust and produce a classifier with a **Test Set Accuracy of 62.19%** on the sample data.

This accuracy, while lower than that of the 16S and COI pipelines, is an expected and realistic result. It reflects the immense diversity and complexity of the Fungal kingdom and the high variability of the ITS marker gene, making it an inherently more challenging classification task.