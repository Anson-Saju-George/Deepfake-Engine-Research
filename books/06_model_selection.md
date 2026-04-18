# Model Selection

## What This Stage Means

Model selection decides which architecture families deserve compute, why they are in scope, and in what order they should be studied.

For this repository, model selection is not just `pick a popular backbone`. It is meant to answer:

- what representational bias the family brings
- what comparison question it answers
- why it belongs in the paper
- why it is worth its computational cost

## Active Image Families

Current active image families in the new tree:

- ViT
- ConvNeXt
- Swin
- DeiT
- ConvNeXtV2
- MaxViT
- EVA

Current image experiment ordering:

- `IMG-EXP-01..03`: ViT
- `IMG-EXP-04..06`: ConvNeXt
- `IMG-EXP-07..08`: Swin
- `IMG-EXP-09`: DeiT
- `IMG-EXP-10..11`: ConvNeXtV2
- `IMG-EXP-12`: MaxViT
- `IMG-EXP-13..14`: EVA

Why the active image surface changed:

- EfficientNet and ResNet are no longer part of the active image plan
- the current image phase prioritizes backbones above the stronger modern parameter floor and better-aligned research comparison families

## Active Video Research Families

Current video registry families are grouped by category.

Spatial categories include:

- Xception
- ConvNeXt
- ConvNeXtV2
- Swin
- ViT
- EVA
- MaxViT

Temporal categories include:

- ConvNeXt
- ConvNeXtV2
- Swin
- MaxViT

Spatiotemporal categories include:

- Xception hybrid
- ConvNeXt hybrid
- Swin hybrid
- MaxViT hybrid

Reserved but inactive native-video targets include:

- Video Swin
- TimeSformer
- MViT

Important current execution truth:

- only the image-style video backbones are active in the current timm-backed runner surface
- native-video transformer IDs remain reserved, not validated

## What The Completed Results Already Say

Current evidence from saved runs is stronger than the original wish-list.

Image-side pattern so far:

- ConvNeXt-Base is the strongest completed saved image result
- Swin-Base is close, but not ahead
- ViT runs trail the strongest ConvNeXt and Swin runs
- bigger did not automatically mean better, because ConvNeXt-Large did not beat ConvNeXt-Base

Video-side pattern so far:

- ConvNeXt is the strongest validated family
- Swin underperformed the ConvNeXt baselines in the completed saved video runs
- MaxViT improved over weak early hybrid baselines, but still trails the best ConvNeXt clip-based result
- the completed evidence does not currently support a claim that transformer-heavy families are better than CNN-style backbones on this repo

What this means for model selection:

- the selection logic was reasonable
- but the completed evidence now narrows the strongest active story to ConvNeXt-led image and clip-based video modeling
- native-video transformer families remain a future methodological extension rather than part of the current validated claim set

## Why These Families Were Chosen

### ConvNeXt and ConvNeXtV2

Why they matter:

- strong modern CNN baselines
- useful for texture-heavy and forensic-style cues
- provide a stable convolutional comparison surface against transformer families

### Swin

Why it matters:

- hierarchical transformer with local window attention
- useful midpoint between pure convolution and pure global transformer behavior
- strong candidate for both image and clip-based video experiments

### ViT and EVA

Why they matter:

- pure transformer comparison family
- useful for testing global patch-attention style reasoning
- EVA adds stronger pretrained transformer initialization than plain ViT

### DeiT

Why it matters:

- data-efficient transformer baseline
- gives a controlled base-scale transformer comparison without requiring the largest transformer family first

### MaxViT

Why it matters:

- hybrid local and global attention structure
- useful for testing whether mixed inductive bias outperforms pure CNN or pure transformer alternatives

### Xception, Video Swin, TimeSformer, and MViT

Why they matter on the video side:

- Xception provides a legacy deepfake baseline with strong historical relevance
- Video Swin provides native video-windowed attention
- TimeSformer provides explicit space-time transformer comparison
- MViT provides multiscale native-video transformer behavior

## Selection Principles

The active repo follows these selection rules:

- keep image and video model selection conceptually separate
- preserve raw-video-first methodology for the main video path
- compare strong CNN, hierarchical transformer, pure transformer, and hybrid-attention families deliberately
- use staged parameter scaling rather than arbitrary model choice
- keep the current image optimization stack fixed before optimizer-specific ablations

## What The Thesis Or Paper Should Say

The write-up should state explicitly that:

- the active image tree was narrowed to a stronger research-facing model surface
- EfficientNet and ResNet were removed from the active image plan
- the video registry is now organized by category and ordered by paradigm, family, and parameter scale
- the current model-selection logic exists to make CNN vs transformer vs hybrid comparisons easier to defend
- the completed evidence currently favors ConvNeXt-style backbones over the finished transformer-family runs in this repository
