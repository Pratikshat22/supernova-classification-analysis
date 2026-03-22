# Supernova Classification Using Deep Learning

I worked on classifying different types of supernovae using machine learning. The dataset includes six types: Type Ia, Type II-P, Type II-L, Type Ib, Type Ic, and Type IIn. The goal was to see if a neural network could learn to distinguish them based on physical parameters like peak magnitude, rise time, decay time, expansion velocity, and nickel mass.

## What I Did

I took supernova data (synthetic, based on real physics) and trained a deep neural network with 6 classes. The model had 184,646 parameters and achieved about 60% accuracy on the test set. Not perfect, but enough to see patterns.

I also made an interactive dashboard with 12 figures showing:
- Light curves (animated)
- Hubble diagram
- Expansion animation
- 3D parameter space
- Spectral fingerprints
- t-SNE clustering
- Nickel mass distributions
- Confusion matrix
- Expansion velocities

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 60.70% |
| Hubble Constant (H₀) | 99.93 ± 2.5 km/s/Mpc |
| Number of Types | 6 |
| Dataset Size | 10,000 samples |

## Key Findings

- Type Ia supernovae have a stretch factor around 1.0 (standardizable candles)
- Light curves for Type Ia are narrower than Type II
- Hubble diagram shows acceleration (consistent with dark energy)
- Nickel-56 mass correlates with peak brightness
- t-SNE reveals natural clustering of types
- The model struggles most with Type II-L and Type II-P (similar light curve shapes)

## Interactive Dashboard

The dashboard has 12 interactive figures. Open `supernova_analysis.html` in any browser to explore:

| Figure | What It Shows |
|--------|---------------|
| 1.1 | Supernova type distribution |
| 1.2 | Light curves (animated) |
| 1.3 | Hubble diagram |
| 1.4 | Expansion animation |
| 1.5 | Model training history |
| 1.6 | 3D parameter space |
| 1.7 | Spectral fingerprints |
| 1.8 | Cosmic distance ladder |
| 1.9 | Confusion matrix |
| 1.10 | Nickel mass distribution |
| 1.11 | Expansion velocities |
| 1.12 | t-SNE clustering |

**Live Dashboard:**  
https://Pratikshat22.github.io/supernova-classification-analysis/supernova_analysis.html

## Files

- `supernova_analysis.html` — interactive dashboard
- `analysis_script.py` — Python code
- `README.md` — this file
- `requirements.txt` — dependencies

## Requirements

```bash
pip install -r requirements.txt
