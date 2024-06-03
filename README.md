# NEURAL STYLE TRANSFER: DIGITAL ART

This code uses a pretrained VGG19 neural network for transfer the style of an image (S) to a "content" image C. The generated image G preserves the content of C while applying the style of S. How this is done depends on the selection of layers for content and style, the number of epochs and the learning rate. There is a lot of room for experimenting. This proces is computation intensive and is recommended to use an accelerator device. In this case I use a standard consumer GPU with CUDA. Some rexperiment results you can see next:

### Van Gogh with sandstone style
<img title="Van Gogh Sandstone" alt="van gogh sandstone" src="/output/vg_goes_sandstone_2k_lr1e-3/all.jpg">

### Van Gogh with water drops style
<img title="Van Gogh Drops" alt="van gogh dorps" src="/output/vg_goes_drops_8k_lr2e-4/all.jpg">

### Van Gogh with stones style
<img title="Van Gogh Stones" alt="van gogh stones" src="/output/vg_goes_stones_10k_lr2e-3/all.jpg">
