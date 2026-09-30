# A.T.L.A.S.: Project Overview

This document outlines the vision and architecture for the third version of the A.T.L.A.S. project.

## Core Philosophy

A.T.L.A.S. is built on a hybrid development model that combines the strengths of interactive development with the stability of production scripts.

**Research & Development (R&D) in Jupyter Notebooks**: All new features, pipeline refinements, and model experiments are first developed in Jupyter Notebooks (`/notebooks`). This allows for rapid, cell-by-cell testing, data visualization, and interactive feedback.

**Production Pipelines in Python Scripts**: Once a workflow has been proven and finalized in a notebook, its logic is converted into a clean, efficient, and reproducible Python script (`.py`). These scripts will eventually form the final, automated pipelines.

## Project Structure

The project is organized into the following key directories:

```
atlas/
├── data/
│   ├── raw/          # Raw, original database files (e.g., SILVA.fasta)
│   └── processed/    # Clean, numerical data ready for training (.npz, .npy)
│
├── docs/             # All project documentation and guides
│
├── models/           # Saved artifacts like trained models (.keras) and encoders (.pkl)
│
├── notebooks/        # Jupyter notebooks for interactive development and experimentation
│
└── src/              # (Future) Final, production-ready Python (.py) scripts
```
