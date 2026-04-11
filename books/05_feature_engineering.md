# Feature Engineering

## Current Position

The current codebase relies primarily on learned visual representations rather than heavy manual feature engineering.

## Practical Feature Spaces

- RGB image tensors
- single sampled video frames
- contiguous video clips
- contiguous sequences from derived frame folders

## Research Notes

Potential feature engineering or representation studies may include:

- frame-only vs native video comparison
- temporal clip length sensitivity
- frequency-domain or compression-artifact features
- face-cropped vs full-frame ablations
- image resize-policy studies for oversized source images in `ai-generated-images-vs-real-images`

These should be documented as explicit ablations, not mixed into baseline claims.
