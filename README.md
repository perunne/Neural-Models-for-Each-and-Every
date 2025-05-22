# Neural-Models-for-Each-and-Every

This repository accompanies my paper "Neural Models as Psychosemantic Models for _Each_ and _Every_", submitted as my final assessment for PLIN0072 Seminar in Computational Linguistics. It generates and evaluates visual stimuli to study how framing by the universal quantifiers _each_ and _every_ affects performance on psychosemantic visual identification tasks by neural vision models. The model used in this experiment is CLIP ViT/B-32 (Contrastive Language-Image Pre-Training; Radford et al., 2021).

## Requirements

The experiments were run using:
- Python 3.7+
- PyTorch 1.8+
- OpenCLIP
- CUDA 11.3 (for GPU acceleration)
- Standard scientific Python stack (NumPy, pandas, matplotlib, scikit-learn)

## Pipeline Overview

### 1. Stimulus Generation
```
python stimuli_generation.py
```

### 2. Metadata Validation
```
python dataset_validation.py
```

### 3. CLIP Evaluation
```
python model.py
```

### 4. Results Processing
```
python process_results.py
```

## Key Features

- **Human-paradigm replication**: Implements the dual-judgment trial structure from Knowlton et al. (2021).
- **Controlled manipulations**: 
  - Quantifier framing ("each" vs "every")
  - Color category violations
  - Subtle within-category hue changes
- **Processing constraints**:
  - Model depth truncation (6/12 layers)
  - Resolution reduction (112px vs 224px)
- **Multi-stage evaluation**:
  - Quantifier verification (sentence-image matching)
  - Change detection (image-image comparison)
- **Threshold-based classification**: Uses cosine similarity scores for binary judgments

## AI Usage Statement

I acknowledge the usage of DeepSeek V3 in proofreading and LaTeX formatting. I acknowledge the usage of GitHub copilot (GPT-4o) in general coding assistance, particularly in debugging and in optimisation.

## Citations

Knowlton, T. Z., Halberda, J., Pietroski, P., & Lidz, J. (2023). Individuals versus ensembles and ‘each’ versus ‘every’: Linguistic framing affects performance in a change detection task. Glossa Psycholinguistics, 2(1). https://doi.org/10.5070/G6011181

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision (arXiv:2103.00020). arXiv. https://doi.org/10.48550/arXiv.2103.00020
