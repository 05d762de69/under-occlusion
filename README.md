# Occlusion Project

A reproducible research repository for studying **shape completion and robustness under occlusion** in convolutional neural networks, with confirmatory and exploratory analyses across model behavior, generated completions, and human ratings.

## Overview

This project investigates how different visual recognition models behave under occlusion, with a particular focus on the contrast between **shape-biased** and **texture-biased** CNNs.

The project is organized into three main parts:

### Part I. Robustness under attribution-guided occlusion

Tests how strongly model confidence changes when informative image regions are occluded.

- **Primary outcome:** target logit drop
- **Main comparison:** shape-biased vs. texture-biased models
- **Manipulations:**
  - targeted vs. random occlusion placement
  - number of occluders
  - occluder size

### Part II. Hypothesis distribution under occlusion

Studies the distribution of generated shape completions after occlusion.

- **Primary outcome:** debiased entropy of BGMM cluster weights
- **Goal:** quantify whether models exhibit concentrated or diffuse completion hypotheses
- **Pipeline:**
  1. generate candidate completions
  2. extract model embeddings
  3. cluster completions with a Bayesian Gaussian Mixture Model
  4. debias cluster mass to account for generator bias
  5. analyze entropy and related summary measures

### Part III. Human ratings

Evaluates the plausibility and perceived complexity of model-derived contour completions.

- **Primary outcome:** plausibility rating
- **Secondary outcome:** complexity rating
- **Stimuli:** cluster mean contour visualizations derived from model-generated completions

---

## Research goals

This repository supports a broader research program on:

- visual robustness under occlusion
- probabilistic shape completion in neural networks
- differences between shape-biased and texture-biased representations
- the relationship between model-internal hypotheses and human judgments

---

## Repository structure

The exact structure may evolve, but the repository is intended to follow a layout close to this:

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── notebooks/
│   ├── part1_robustness/
│   ├── part2_hypothesis_distribution/
│   └── part3_human_ratings/
├── scripts/
│   
├── results/
│   ├── figures/
│   ├── tables/
│   └── intermediate/
└── stimuli/
    ├── cases/
    ├── generated/
    └── rating_exports/
```


## Data availability

The data/ folder is not included in this repository because of its size and the presence of large intermediate artifacts.

Data are available upon reasonable request.
