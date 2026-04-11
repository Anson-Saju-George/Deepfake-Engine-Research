# Iteration / Continuous Improvement

## Iteration Strategy

The project should iterate in a disciplined order:

1. verify raw dataset truth
2. verify split correctness
3. verify preprocessing correctness
4. establish baseline models
5. evaluate and diagnose errors
6. tune selectively
7. expand ablations only after baseline stability

## Near-Term Iteration Themes

- continue image-family expansion after ViT
- migrate video trainer execution cleanly into the new tree
- benchmark image-only baselines
- benchmark video-only baselines
- formalize frame-only training presets if frame experiments become central

## Improvement Philosophy

- prefer defensible methodology over optimistic but weak benchmarking
- keep a written record of rejected paths and why they were rejected
- make paper-writing easier by documenting decisions while they are fresh
