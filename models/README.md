# Hulu glyph model

The supplied model contains numeric HOG descriptors of normalized badge glyphs and negative examples, plus similarity thresholds. Raw camera images, image coordinates, private manifests and session metadata are not included. The preset targets Hulu's light Ad badge; it is not a general classifier for every streaming service.

Reused-image chronological regression: 53/57 ads detected, four missed ads, 18/21 no-ad frames cleared, three uncertain no-ad frames, zero false positives. These 78 images came from one camera session and were used repeatedly during development. This is not independent field accuracy. Live unmute latency after the latest changes has not yet been measured.
