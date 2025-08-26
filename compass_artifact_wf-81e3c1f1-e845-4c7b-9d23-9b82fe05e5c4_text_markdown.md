# Deep Learning Study Guide

Based on analysis of all three instructors' course materials, here are focused study paragraphs emphasizing core principles for your January 17, 2025 exam.

## Alexander Meyer's Course: CNNs and Classical Deep Learning

### Convolutional Neural Networks: The foundation of spatial learning

**Core principle**: CNNs work because they exploit **spatial locality and translation invariance** through three key mechanisms. Local connectivity means neurons only connect to nearby pixels, capturing local features like edges. Weight sharing applies the same filter across the entire image, drastically reducing parameters while enforcing translation invariance. Hierarchical feature learning builds complexity—lower layers detect edges and textures, while deeper layers recognize complex objects. The mathematical beauty is in convolution: `(f * g)[n] = Σ f[m] * g[n-m]`, where output size follows `(W - F + 2P)/S + 1`.

**Exam insight**: Expect questions about **why** CNNs outperform fully connected networks for images. The answer centers on inductive biases—CNNs assume spatial structure exists and shouldn't be ignored, unlike MLPs that treat pixels as independent features.

### Style Transfer: Separating content from artistic expression

**Core principle**: Gatys et al.'s breakthrough separates **content** (semantic meaning) from **style** (artistic patterns) using different layers of a pretrained VGG19 network. Content representations come from high-level features (conv4_2), capturing "what" is in the image. Style representations use **Gram matrices** across multiple layers, capturing texture correlations that define "how" something looks. The Gram matrix `G_ij = Σ_k F_ik * F_jk` captures feature correlations—high values mean features consistently appear together, defining style.

**Key equations**: Content loss `L_content = 1/2 * Σ(F_l - P_l)²` preserves semantic meaning, while style loss `L_style = 1/4N²M² * Σ(G_l - A_l)²` transfers artistic patterns. Total loss `L_total = α*L_content + β*L_style` balances both objectives.

**Exam focus**: Understanding **why** Gram matrices capture style (they measure feature co-occurrence patterns) and why different CNN layers are used for content versus style representation.

### YOLO: Real-time object detection revolution

**Core principle**: YOLO treats object detection as a **single regression problem**, seeing the entire image at once rather than using sliding windows. It divides images into an S×S grid, with each cell predicting B bounding boxes and class probabilities. This global context enables real-time performance while maintaining reasonable accuracy—a crucial trade-off in computer vision.

**Exam insight**: Focus on **speed versus accuracy trade-offs**. YOLO sacrifices some accuracy for massive speed gains by making detection predictions in one forward pass rather than thousands.

### GANs: Adversarial learning dynamics

**Core principle**: GANs create a **competitive game** between generator (creates fake data) and discriminator (detects fakes). This adversarial training pushes both networks to improve—the generator learns realistic data generation, while the discriminator becomes better at detection. The minimax objective `min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]` formalizes this competition.

**Practical example**: "Everybody Dance Now" demonstrates GANs' power in **pose-to-person mapping**—learning to generate realistic human appearance from skeletal poses, enabling motion transfer between people in videos.

## Nicolas Bonneel's Course: Advanced Probabilistic Models

### Optimal Transport: The mathematics of transformation

**Core principle**: Optimal transport solves the fundamental problem of **reshaping one probability distribution into another at minimum cost**. Think of moving a pile of sand into a different shape using minimal effort—this intuition captures the Wasserstein distance, which measures how much "work" is needed to transform distributions. Unlike KL divergence, Wasserstein distance respects the underlying geometry of the space.

**Key equations**: The Kantorovich formulation `W(f,g) = min_P ∫ c(x,y)P(x,y)dxdy` finds the optimal coupling between distributions. For cost `c(x,y) = ||x-y||^p`, this gives the Wasserstein-p distance: `W_p(f,g)^{1/p}`.

**Practical applications**: Color transfer (reshaping color histograms), texture synthesis (interpolating texture distributions), and style transfer all leverage optimal transport's ability to preserve structure while transforming distributions.

### Diffusion Models: Learning to reverse entropy

**Core principle**: Diffusion models learn generation by **modeling two complementary processes**. The forward process systematically destroys structure by adding Gaussian noise: `q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t I)`. The reverse process learns to reconstruct data by iteratively removing noise: `p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))`.

**Mathematical insight**: The reparameterization trick `x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε_t` allows direct sampling at any timestep. The simplified loss `L_simple = E[||ε - ε_θ(x_t,t)||²]` trains the model to predict noise rather than images directly—a more stable learning objective.

**Connection to optimal transport**: Recent research shows DDIM sampling corresponds to optimal transport maps between data and noise distributions, connecting these seemingly different frameworks.

## Julie Digne's Course: 3D and Geometric Deep Learning

### Neural Radiance Fields: Continuous scene representation

**Core principle**: NeRF represents scenes as **continuous 5D functions** that capture how light varies across space and viewing direction. The key insight is that 3D scenes can be encoded as neural networks that map coordinates `(x,y,z,θ,φ)` to color and density `(r,g,b,σ)`. This enables infinite-resolution view synthesis from a finite set of training images.

**Volume rendering equation**: `C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt` where `T(t) = exp(-∫₀ᵗ σ(r(s)) ds)` represents transmittance. This differentiable rendering enables end-to-end optimization.

**Critical insight**: **Positional encoding** maps coordinates to high-frequency space, enabling the network to represent fine details. Without it, neural networks suffer from spectral bias and produce blurry results.

### 3D Gaussian Splatting: Real-time neural rendering

**Core principle**: Instead of implicit neural functions, Gaussian splatting represents scenes with **millions of 3D Gaussians** (ellipsoids) that can be rasterized in real-time. Each Gaussian is defined by position μ and covariance matrix `Σ = RSRᵀ` controlling shape and orientation.

**Speed advantage**: The key breakthrough is **differentiable rasterization**—3D Gaussians project to 2D screen space with covariance `Σ' = JΣJᵀ`, enabling GPU-accelerated rendering at >100 FPS while maintaining NeRF-quality visuals.

**Exam focus**: Understanding the **speed-quality trade-off**—Gaussian splatting achieves real-time performance by using explicit primitives instead of neural network evaluation for every pixel.

### Implicit Neural Representations: Continuous function learning

**Core principle**: Traditional representations (voxels, meshes) discretize continuous phenomena, losing detail and requiring massive storage. **Implicit representations use neural networks as continuous functions** mapping coordinates to properties—enabling infinite resolution in compact form.

**Mathematical formulation**: `f_θ(x) → properties` where x represents coordinates and θ are network parameters. For signed distance functions: `f(x,y,z) → distance to surface`. For occupancy: `f(x,y,z) → {inside, outside}`.

**Advantage**: A single neural network can represent arbitrarily high-resolution geometry, while a 1024³ voxel grid requires gigabytes of memory.

### Generative Models Integration: From 2D to 3D

**Core principle**: Modern 3D generation combines **2D generative models with 3D representations**. Diffusion models excel at generating 2D images, while NeRF and Gaussians excel at 3D consistency. The challenge is maintaining **multi-view consistency**—ensuring generated views are geometrically coherent.

**Current approaches**: Score Distillation Sampling (SDS) uses 2D diffusion models to guide 3D optimization, while direct 3D diffusion operates on point clouds or voxels. The field is rapidly evolving toward efficient, controllable 3D generation.

## Key Exam Strategies

**For open-ended questions, focus on**:
- **Why methods work** (mathematical intuition) rather than implementation details
- **Trade-offs and limitations** of different approaches
- **Connections between topics** (e.g., how optimal transport relates to diffusion models)
- **Practical applications** and when to use which method

**Common question patterns**:
- "Explain the fundamental principle behind..." (focus on core mathematical insight)
- "Compare and contrast..." (highlight key trade-offs)
- "How would you modify this approach for..." (demonstrate understanding of principles)
- "What are the limitations of..." (show critical analysis)

**Mathematical focus**: Understand the **intuition** behind key equations rather than memorizing formulas. Be able to explain why certain loss functions are chosen and what they optimize for.

The exam emphasizes conceptual understanding over implementation details—focus on grasping the core principles that make each method work and be prepared to explain these insights clearly in your own words.
