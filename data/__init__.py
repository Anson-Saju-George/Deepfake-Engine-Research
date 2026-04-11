"""Data package for dataset loading, validation, cleanup, and analysis tools.

Primary research path:
- image datasets -> image-only training
- raw video datasets -> video-only training
  - `single` mode is the spatial baseline
  - `sequence` mode is the spatial+temporal path

Optional support remains for derived frame-folder datasets, but frame
materialization is auxiliary rather than the default training workflow.
"""
