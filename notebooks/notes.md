






job post finding automated from local area in claude cowork
* search also people i should write to get involved as consultant
* search local and for spec roles





cv rewriting more concise and first page only direct Q&A :

less R&D or complex catch words that sound scary and more straight interpretation explanation about work and how i go about it - early quick&dirty prototype or mockup (AI) then iterate once expectations/feasibility is clearer (sci lit can give indications but not always are useful/reproducible)


* in summary: much more concise with just a few key sentences about my "mantra" like (1 bullet on short sentences):

- i believe R&D activities should always lead to actionable endpoints, if not clear what they are upfront they should be progressively and deliberately defined 

- interpretation trumps accuracy: no matter the modelling approach, i think it matters more to have outcomes that can be interpreted/validated/tested than just getting great results that no-one can understand how

- R&D approach: some project/problems are loosely defined so kitchen-sink attempts to solve are fine to get started but domain expertise is central in finding a valid solution more effectively (or guide more/less promising directions of search)

* what i bring to the table

* what kind of projects i've worked with success on

* technical section and next page goes education and other jobs...






next steps for production showcase as open tool from browser or app ?








## Vision pipeline — status 2026-04-19

Full status + next steps: `vision/VISION_PROJECT_PLAN.md` § "Current Status"

### Tier 1 results (two-layer MLP head, DINOv2-base CLS, manual QA on hymenium)

| Feature           | Test Acc | wF1  | Notes                                      |
|-------------------|----------|------|--------------------------------------------|
| hymenium_type     | 90.5%    | 0.90 | best head; ridges↔gills main confusion     |
| overall_body_form | 85.6%    | 0.85 | agaricoid/boletoid strong; rare forms weak  |
| cap_shape         | 74.8%    | 0.74 | convex dominates; subtle shapes unlearnable |
| cap_color         | 43.8%    | 0.46 | too noisy — color varies within species     |

### Bugs fixed
- double balancing (sampler + class weights) → removed class weights, kept sampler
- phantom classes (0-image classes staying in class_names) → always sync with manifest

### Open questions

1. Is DINOv2's CLS token the right representation? I start to think is not best one, working by patches can
be more interesting but seems going in a direction like ViT rather, do you think we have margin to improve on
the representation before going on full ViT (basically extending properly the vlm annotation to extract also
features from pics)? Would it make sense to extend embedding set to be not just final output of DINOv2 but
maybe we can also consider intermediate activations from it?

2. noisy labeling - I assume labeling is ok but very possible that images are not clear or simple enough to
interpret for feature extraction as is but not sure we have clear way to improve further besides changing
model or putting attention boxes around what region we should focus on before embedding?

3. cap_color likely needs different approach: white balance preprocessing, color-specific features, or VLM
4. manual QA for other features besides hymenium_type?                       





