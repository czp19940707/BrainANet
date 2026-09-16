# BrainANet: An Anatomical Brain Representation Network with Large-Scale Self-Supervised Learning for Brain Disease Diagnosis

This repository provides the official implementation and supplementary material for **BrainANet**, an anatomical brain representation framework for constructing individualized anatomical brain networks from T1-weighted MRI.

BrainANet learns adaptive regional representations using a self-supervised patch-level Swin Transformer pretrained on large-scale T1-weighted MRI data. Pairwise cosine similarity between regional representations is then used to characterize interregional anatomical similarity and construct individualized anatomical brain networks for downstream GNN-based brain disease classification.

## Supplementary Material

The file [`SupplementaryMaterials.pdf`](./SupplementaryMaterials.pdf) contains the official supplementary material accompanying our paper published in the **IEEE Journal of Biomedical and Health Informatics (JBHI)**.

**Paper:** *BrainANet: An Anatomical Brain Representation Network with Large-Scale Self-Supervised Learning for Brain Disease Diagnosis*
**DOI:** 10.1109/JBHI.2026.3734496
