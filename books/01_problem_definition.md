# Problem Definition

## Objective

Build a research-grade deepfake detection pipeline that supports:

- image-only detection
- video-only detection
- optional derived frame-folder experiments

The goal is to produce robust, reproducible experiments rather than inflated benchmark numbers from weak split choices.

## Problem Framing

Deepfakes appear across heterogeneous media:

- static AI-generated or manipulated images
- manipulated or synthetic videos
- videos that can also be analyzed through extracted frame folders

These are related but not identical learning problems. The project therefore treats image and video experiments as separate default tracks.

## Success Criteria

- correct and defensible dataset boundaries
- reproducible splits
- clear separation of image and video protocols
- paper-ready reporting of preprocessing and evaluation

## Non-Goals

- default pooled image+video training
- undocumented split leakage
- relying on legacy scripts outside the active pipeline
