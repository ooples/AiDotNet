# Issue #416: Junior Developer Implementation Guide
## Generative Adversarial Networks (GANs)

---

## Table of Contents
1. [Understanding GANs](#understanding-gans)
2. [Understanding GAN Training](#understanding-gan-training)
3. [GAN Architectures](#gan-architectures)
4. [Advanced GAN Techniques](#advanced-gan-techniques)
5. [Architecture Overview](#architecture-overview)
6. [Phase 1: Core GAN Framework](#phase-1-core-gan-framework)
7. [Phase 2: Vanilla GAN](#phase-2-vanilla-gan)
8. [Phase 3: DCGAN](#phase-3-dcgan)
9. [Phase 4: WGAN](#phase-4-wgan)
10. [Phase 5: StyleGAN](#phase-5-stylegan)
11. [Testing Strategy](#testing-strategy)
12. [Common Pitfalls](#common-pitfalls)

---

## Understanding GANs

### What Is a GAN?

A **Generative Adversarial Network (GAN)** is a framework for training generative models through adversarial competition between two neural networks:

1. **Generator (G)**: Creates fake data (images, audio, etc.)
2. **Discriminator (D)**: Distinguishes real data from fake data

### The Adversarial Game

```
Generator Goal: Fool the discriminator by creating realistic fake data
Discriminator Goal: Correctly classify real vs fake data

Training: They compete, both improving until generator creates perfect fakes
```

### Intuitive Analogy

**Generator = Counterfeiter**: Tries to create fake money that looks real

**Discriminator = Police**: Tries to detect fake money

As the counterfeiter gets better, the police must improve their detection. Eventually, the counterfeiter creates perfect fakes.

### Mathematical Formulation

The GAN objective is a minimax game:

```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

Where:
- x = real data samples
- z = random noise (latent vector)
- G(z) = generated fake samples
- D(x) = discriminator's output for real data (should be close to 1)
- D(G(z)) = discriminator's output for fake data (should be close to 0)
```

**Discriminator maximizes** V(D, G):
- Wants D(x) close to 1 (real data classified as real)
- Wants D(G(z)) close to 0 (fake data classified as fake)

**Generator minimizes** V(D, G):
- Wants D(G(z)) close to 1 (fake data classified as real)

### Why GANs Matter

1. **Unsupervised Learning**: No labeled data required
2. **High-Quality Generation**: Produces photorealistic images
3. **Diverse Applications**:
   - Image synthesis (faces, landscapes, art)
   - Image-to-image translation (day to night, sketch to photo)
   - Data augmentation
   - Super-resolution
   - Video generation

---

## Understanding GAN Training

### Alternating Optimization

GAN training alternates between updating the discriminator and generator:

```
For each training iteration:
  1. Update Discriminator:
     - Sample real data x
     - Sample noise z, generate fake data G(z)
     - Compute discriminator loss: L_D = -log D(x) - log(1 - D(G(z)))
     - Update D's weights to minimize L_D

  2. Update Generator:
     - Sample noise z, generate fake data G(z)
     - Compute generator loss: L_G = -log D(G(z))
     - Update G's weights to minimize L_G
```

### Training Dynamics

```csharp
// Pseudocode for GAN training
for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in dataLoader)
    {
        // Step 1: Train Discriminator
        var realData = batch.Data;
        var noise = SampleNoise(batchSize, latentDim);
        var fakeData = generator.Forward(noise);

        var realOutput = discriminator.Forward(realData);
        var fakeOutput = discriminator.Forward(fakeData.Detach());  // Don't backprop through generator

        var dLoss = -Mean(Log(realOutput)) - Mean(Log(1 - fakeOutput));
        discriminator.BackwardAndUpdate(dLoss);

        // Step 2: Train Generator
        noise = SampleNoise(batchSize, latentDim);
        fakeData = generator.Forward(noise);
        fakeOutput = discriminator.Forward(fakeData);  // Backprop through discriminator

        var gLoss = -Mean(Log(fakeOutput));
        generator.BackwardAndUpdate(gLoss);
    }
}
```

### Training Challenges

#### 1. Mode Collapse

**Problem**: Generator produces limited variety of outputs

```
Generator discovers one "good" fake that fools discriminator
Generator starts outputting only that fake repeatedly
Diversity is lost
```

**Solutions**:
- Minibatch discrimination
- Unrolled optimization
- Use WGAN loss

#### 2. Vanishing Gradients

**Problem**: When discriminator is too strong, generator gradients vanish

```
If D(G(z)) ≈ 0 (discriminator confidently rejects fakes):
  Gradient of -log(D(G(z))) approaches zero
  Generator stops learning
```

**Solutions**:
- Use non-saturating loss: Maximize log D(G(z)) instead of minimizing log(1 - D(G(z)))
- Use WGAN loss
- Carefully tune discriminator/generator update ratio

#### 3. Training Instability

**Problem**: Training oscillates, never converges

**Solutions**:
- Use spectral normalization
- Update discriminator more frequently than generator
- Use learning rate scheduling
- Add gradient penalty (WGAN-GP)

---

## GAN Architectures

### 1. Vanilla GAN (2014)

**Architecture**: Fully connected neural networks

**Generator**:
```
Input: Random noise z (e.g., 100-dim)
Layer 1: Linear(100 → 256) + LeakyReLU
Layer 2: Linear(256 → 512) + LeakyReLU
Layer 3: Linear(512 → 1024) + LeakyReLU
Output: Linear(1024 → 784) + Tanh  // 28x28 = 784 for MNIST
```

**Discriminator**:
```
Input: Image (784-dim flattened)
Layer 1: Linear(784 → 512) + LeakyReLU
Layer 2: Linear(512 → 256) + LeakyReLU
Output: Linear(256 → 1) + Sigmoid  // Probability real vs fake
```

**Use Case**: Simple datasets (MNIST, small images)

### 2. DCGAN (Deep Convolutional GAN, 2015)

**Key Innovation**: Replace fully connected layers with convolutional layers

**Generator** (Transposed Convolutions):
```
Input: Random noise z (100-dim)
Layer 1: Linear(100 → 4×4×1024) + Reshape to (4, 4, 1024)
Layer 2: ConvTranspose2d(1024 → 512, kernel=4, stride=2) + BatchNorm + ReLU
  Output: (8, 8, 512)
Layer 3: ConvTranspose2d(512 → 256, kernel=4, stride=2) + BatchNorm + ReLU
  Output: (16, 16, 256)
Layer 4: ConvTranspose2d(256 → 128, kernel=4, stride=2) + BatchNorm + ReLU
  Output: (32, 32, 128)
Output: ConvTranspose2d(128 → 3, kernel=4, stride=2) + Tanh
  Output: (64, 64, 3) RGB image
```

**Discriminator** (Strided Convolutions):
```
Input: RGB image (64, 64, 3)
Layer 1: Conv2d(3 → 128, kernel=4, stride=2) + LeakyReLU
  Output: (32, 32, 128)
Layer 2: Conv2d(128 → 256, kernel=4, stride=2) + BatchNorm + LeakyReLU
  Output: (16, 16, 256)
Layer 3: Conv2d(256 → 512, kernel=4, stride=2) + BatchNorm + LeakyReLU
  Output: (8, 8, 512)
Layer 4: Conv2d(512 → 1024, kernel=4, stride=2) + BatchNorm + LeakyReLU
  Output: (4, 4, 1024)
Output: Conv2d(1024 → 1, kernel=4) + Flatten + Sigmoid
  Output: Scalar probability
```

**DCGAN Guidelines**:
1. Replace pooling with strided convolutions
2. Use batch normalization (except generator output and discriminator input)
3. Remove fully connected layers
4. Use ReLU in generator (except output uses Tanh)
5. Use LeakyReLU in discriminator

**Use Case**: Natural images (CelebA, ImageNet)

### 3. WGAN (Wasserstein GAN, 2017)

**Key Innovation**: Replace adversarial loss with Wasserstein distance

**Original GAN Loss** (Jensen-Shannon Divergence):
```
Discriminator: Maximize log D(x) + log(1 - D(G(z)))
Generator: Minimize log(1 - D(G(z)))
```

**WGAN Loss** (Wasserstein Distance):
```
Critic (replaces discriminator): Maximize E[C(x)] - E[C(G(z))]
Generator: Minimize -E[C(G(z))]

Where C is the critic (no sigmoid, outputs raw score)
```

**Key Differences**:
1. **No sigmoid** in critic output
2. **Lipschitz constraint**: Critic must be 1-Lipschitz continuous
   - Original WGAN: Weight clipping (clip weights to [-0.01, 0.01])
   - WGAN-GP: Gradient penalty (penalize gradient norm != 1)

**WGAN-GP Gradient Penalty**:
```
Interpolate between real and fake: x_hat = ε * x + (1 - ε) * G(z)
Compute gradients: ∇_x_hat C(x_hat)
Penalty: λ * (||∇_x_hat C(x_hat)||_2 - 1)²
```

**Benefits**:
- More stable training
- Meaningful loss curves (lower = better)
- Reduced mode collapse

### 4. StyleGAN (2018-2019)

**Key Innovation**: Style-based generator with progressive growing

**Architecture Components**:

1. **Mapping Network**: Maps latent code z to intermediate latent w
```
z (512-dim) → FC(512 → 512) × 8 layers → w (512-dim)
```

2. **Synthesis Network**: Generates image from w using adaptive instance normalization (AdaIN)
```
For each resolution (4×4, 8×8, ..., 1024×1024):
  1. Apply learned constant or upsample previous layer
  2. Apply style (w) via AdaIN:
     AdaIN(x, w) = (x - μ(x)) / σ(x) * scale(w) + bias(w)
  3. Add noise
  4. Apply convolution
```

**Style Mixing**: Use different w vectors at different resolutions for diverse outputs

**Adaptive Instance Normalization (AdaIN)**:
```csharp
AdaIN(x, style):
  μ_x = Mean(x, axis=(height, width))
  σ_x = StdDev(x, axis=(height, width))
  scale = LearnedTransform(style)
  bias = LearnedTransform(style)
  return (x - μ_x) / σ_x * scale + bias
```

**Progressive Growing**: Train incrementally at increasing resolutions
```
Train 4×4 → Add 8×8 layers → Train 8×8 → Add 16×16 layers → ...
```

**Benefits**:
- Photorealistic high-resolution images (1024×1024)
- Controllable generation via style mixing
- Disentangled latent space

---

## Advanced GAN Techniques

### 1. Spectral Normalization

**Purpose**: Stabilize discriminator training by constraining Lipschitz constant

**Method**: Normalize weights by their spectral norm (largest singular value)
```csharp
W_normalized = W / σ(W)

Where σ(W) is computed via power iteration
```

### 2. Self-Attention

**Purpose**: Capture long-range dependencies in images (e.g., eyes should align with face structure)

**Self-Attention Module**:
```csharp
Query = Conv1x1(x)
Key = Conv1x1(x)
Value = Conv1x1(x)

Attention = Softmax((Query × Key^T) / sqrt(d))
Output = Attention × Value
```

### 3. Conditional GANs (cGAN)

**Purpose**: Generate images conditioned on additional information (labels, text)

**Modification**:
```
Generator: G(z, y) where y is condition (e.g., class label)
Discriminator: D(x, y) where y is condition

Loss: Same as GAN but conditioned on y
```

### 4. CycleGAN (Unpaired Image-to-Image Translation)

**Purpose**: Translate images between domains without paired examples (e.g., horse ↔ zebra)

**Architecture**:
```
G: X → Y (horse → zebra)
F: Y → X (zebra → horse)
D_X: Discriminates real/fake X
D_Y: Discriminates real/fake Y

Cycle-consistency loss: ||F(G(x)) - x|| + ||G(F(y)) - y||
```

---

## Architecture Overview

### Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│                   User Application                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    GANTrainer                           │
│  - AlternatingOptimization()                            │
│  - Discriminator/Generator update logic                 │
│  - Loss computation                                     │
└─────────────────────────────────────────────────────────┘
         │                                    │
         ↓                                    ↓
┌────────────────────┐            ┌────────────────────┐
│    Generator       │            │   Discriminator    │
│  - VanillaGen      │            │  - VanillaDisc     │
│  - DCGANGenerator  │            │  - DCGANDisc       │
│  - WGANGenerator   │            │  - WGANCritic      │
│  - StyleGANGen     │            │  - StyleGANDisc    │
└────────────────────┘            └────────────────────┘
         │                                    │
         ↓                                    ↓
┌─────────────────────────────────────────────────────────┐
│                   Loss Functions                        │
│  - BinaryCrossEntropyLoss                               │
│  - WassersteinLoss                                      │
│  - GradientPenalty                                      │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/NeuralNetworks/GANs/
├── Core/
│   ├── IGAN.cs                       # GAN interface
│   ├── GANTrainer.cs                 # Training orchestrator
│   ├── GANConfig.cs                  # Configuration
│   └── Losses/
│       ├── GANLoss.cs                # Base GAN loss
│       ├── BCEGANLoss.cs             # Binary cross-entropy loss
│       ├── WassersteinLoss.cs        # WGAN loss
│       └── GradientPenalty.cs        # WGAN-GP gradient penalty
│
├── Architectures/
│   ├── Vanilla/
│   │   ├── VanillaGenerator.cs      # Fully connected generator
│   │   └── VanillaDiscriminator.cs  # Fully connected discriminator
│   ├── DCGAN/
│   │   ├── DCGANGenerator.cs        # Convolutional generator
│   │   └── DCGANDiscriminator.cs    # Convolutional discriminator
│   ├── WGAN/
│   │   ├── WGANGenerator.cs         # WGAN generator
│   │   └── WGANCritic.cs            # WGAN critic (no sigmoid)
│   └── StyleGAN/
│       ├── StyleGANGenerator.cs     # Style-based generator
│       ├── StyleGANDiscriminator.cs # StyleGAN discriminator
│       ├── MappingNetwork.cs        # z → w mapping
│       ├── SynthesisNetwork.cs      # w → image synthesis
│       └── AdaIN.cs                 # Adaptive instance norm
│
├── Layers/
│   ├── SpectralNormalization.cs     # Spectral norm wrapper
│   ├── SelfAttention.cs             # Self-attention layer
│   ├── PixelNormalization.cs        # Pixel-wise normalization
│   └── MinibatchStddev.cs           # Minibatch stddev layer
│
└── Utils/
    ├── NoiseGenerator.cs            # Generate latent noise
    ├── ImageSaver.cs                # Save generated images
    └── FID.cs                       # Frechet Inception Distance metric
```

---

## Phase 1: Core GAN Framework

### Step 1: Define GAN Interface

**File**: `src/NeuralNetworks/GANs/Core/IGAN.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Core;

/// <summary>
/// Interface for GAN architectures.
/// </summary>
public interface IGAN
{
    /// <summary>
    /// Generator network.
    /// </summary>
    IModel Generator { get; }

    /// <summary>
    /// Discriminator network.
    /// </summary>
    IModel Discriminator { get; }

    /// <summary>
    /// Generate fake samples from noise.
    /// </summary>
    Tensor<double> Generate(Tensor<double> noise);

    /// <summary>
    /// Compute discriminator output for real/fake samples.
    /// </summary>
    Tensor<double> Discriminate(Tensor<double> samples);

    /// <summary>
    /// Compute discriminator loss.
    /// </summary>
    double ComputeDiscriminatorLoss(Tensor<double> realSamples, Tensor<double> fakeSamples);

    /// <summary>
    /// Compute generator loss.
    /// </summary>
    double ComputeGeneratorLoss(Tensor<double> fakeSamples);
}
```

### Step 2: Define GAN Configuration

**File**: `src/NeuralNetworks/GANs/Core/GANConfig.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Core;

/// <summary>
/// Configuration for GAN training.
/// </summary>
public class GANConfig
{
    /// <summary>
    /// Dimension of latent noise vector.
    /// </summary>
    public int LatentDim { get; set; } = 100;

    /// <summary>
    /// Number of training epochs.
    /// </summary>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Batch size for training.
    /// </summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Learning rate for generator.
    /// </summary>
    public double GeneratorLearningRate { get; set; } = 0.0002;

    /// <summary>
    /// Learning rate for discriminator.
    /// </summary>
    public double DiscriminatorLearningRate { get; set; } = 0.0002;

    /// <summary>
    /// Number of discriminator updates per generator update.
    /// </summary>
    public int DiscriminatorUpdatesPerGenerator { get; set; } = 1;

    /// <summary>
    /// Whether to use label smoothing for discriminator.
    /// </summary>
    public bool UseLabelSmoothing { get; set; } = true;

    /// <summary>
    /// Smoothing factor for real labels (e.g., 0.9 instead of 1.0).
    /// </summary>
    public double LabelSmoothingFactor { get; set; } = 0.9;

    /// <summary>
    /// Save generated images every N epochs.
    /// </summary>
    public int SaveImagesEveryNEpochs { get; set; } = 10;

    /// <summary>
    /// Directory to save generated images.
    /// </summary>
    public string OutputDirectory { get; set; } = "./gan_output";
}
```

### Step 3: Implement Binary Cross-Entropy Loss

**File**: `src/NeuralNetworks/GANs/Core/Losses/BCEGANLoss.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Core.Losses;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Binary cross-entropy loss for GANs.
/// </summary>
public class BCEGANLoss
{
    /// <summary>
    /// Compute binary cross-entropy: -[y * log(x) + (1-y) * log(1-x)]
    /// </summary>
    public static double Compute(Tensor<double> predictions, Tensor<double> targets)
    {
        double loss = 0.0;
        int count = 0;

        for (int i = 0; i < predictions.FlattenedLength; i++)
        {
            double pred = Math.Clamp(predictions.GetFlat(i), 1e-7, 1 - 1e-7);  // Numerical stability
            double target = targets.GetFlat(i);

            loss += -(target * Math.Log(pred) + (1 - target) * Math.Log(1 - pred));
            count++;
        }

        return loss / count;
    }

    /// <summary>
    /// Compute discriminator loss for real and fake samples.
    /// </summary>
    public static double DiscriminatorLoss(
        Tensor<double> realOutput,
        Tensor<double> fakeOutput,
        bool useLabelSmoothing = true,
        double smoothingFactor = 0.9)
    {
        int batchSize = realOutput.Dimensions[0];

        // Real labels (1.0 or smoothed 0.9)
        var realLabels = new Tensor<double>(batchSize);
        double realLabel = useLabelSmoothing ? smoothingFactor : 1.0;
        for (int i = 0; i < batchSize; i++)
            realLabels[i] = realLabel;

        // Fake labels (0.0)
        var fakeLabels = new Tensor<double>(batchSize);
        for (int i = 0; i < batchSize; i++)
            fakeLabels[i] = 0.0;

        double realLoss = Compute(realOutput, realLabels);
        double fakeLoss = Compute(fakeOutput, fakeLabels);

        return realLoss + fakeLoss;
    }

    /// <summary>
    /// Compute generator loss.
    /// Generator wants discriminator to output 1.0 for fake samples.
    /// </summary>
    public static double GeneratorLoss(Tensor<double> fakeOutput)
    {
        int batchSize = fakeOutput.Dimensions[0];

        // Generator wants discriminator to output 1.0
        var realLabels = new Tensor<double>(batchSize);
        for (int i = 0; i < batchSize; i++)
            realLabels[i] = 1.0;

        return Compute(fakeOutput, realLabels);
    }
}
```

### Step 4: Implement Noise Generator

**File**: `src/NeuralNetworks/GANs/Utils/NoiseGenerator.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Utils;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Utility for generating random noise for GAN training.
/// </summary>
public class NoiseGenerator
{
    private readonly Random _rng;

    public NoiseGenerator(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Generate random noise from standard normal distribution.
    /// </summary>
    /// <param name="batchSize">Number of samples</param>
    /// <param name="latentDim">Dimension of latent vector</param>
    public Tensor<double> GenerateNoise(int batchSize, int latentDim)
    {
        var noise = new Tensor<double>(batchSize, latentDim);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < latentDim; j++)
            {
                noise[i, j] = SampleNormal(0, 1);
            }
        }

        return noise;
    }

    /// <summary>
    /// Generate random noise from uniform distribution [-1, 1].
    /// </summary>
    public Tensor<double> GenerateUniformNoise(int batchSize, int latentDim)
    {
        var noise = new Tensor<double>(batchSize, latentDim);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < latentDim; j++)
            {
                noise[i, j] = _rng.NextDouble() * 2 - 1;  // [-1, 1]
            }
        }

        return noise;
    }

    /// <summary>
    /// Sample from standard normal distribution using Box-Muller transform.
    /// </summary>
    private double SampleNormal(double mean, double stdDev)
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
```

### Step 5: Implement GAN Trainer

**File**: `src/NeuralNetworks/GANs/Core/GANTrainer.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Core;

using AiDotNet.NeuralNetworks.GANs.Core.Losses;
using AiDotNet.NeuralNetworks.GANs.Utils;
using AiDotNet.LinearAlgebra;

/// <summary>
/// Trainer for GAN models using alternating optimization.
/// </summary>
public class GANTrainer
{
    private readonly IGAN _gan;
    private readonly IOptimizer _generatorOptimizer;
    private readonly IOptimizer _discriminatorOptimizer;
    private readonly GANConfig _config;
    private readonly NoiseGenerator _noiseGenerator;

    public GANTrainer(
        IGAN gan,
        IOptimizer generatorOptimizer,
        IOptimizer discriminatorOptimizer,
        GANConfig config,
        int? seed = null)
    {
        _gan = gan;
        _generatorOptimizer = generatorOptimizer;
        _discriminatorOptimizer = discriminatorOptimizer;
        _config = config;
        _noiseGenerator = new NoiseGenerator(seed);
    }

    /// <summary>
    /// Train the GAN on real data.
    /// </summary>
    public void Train(IDataLoader<Tensor<double>> realDataLoader)
    {
        Directory.CreateDirectory(_config.OutputDirectory);

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            double avgDiscLoss = 0.0;
            double avgGenLoss = 0.0;
            int batchCount = 0;

            foreach (var batch in realDataLoader)
            {
                int batchSize = batch.Dimensions[0];

                // Train discriminator
                for (int d = 0; d < _config.DiscriminatorUpdatesPerGenerator; d++)
                {
                    double discLoss = TrainDiscriminatorStep(batch, batchSize);
                    avgDiscLoss += discLoss;
                }

                // Train generator
                double genLoss = TrainGeneratorStep(batchSize);
                avgGenLoss += genLoss;

                batchCount++;
            }

            avgDiscLoss /= batchCount * _config.DiscriminatorUpdatesPerGenerator;
            avgGenLoss /= batchCount;

            Console.WriteLine($"Epoch {epoch + 1}/{_config.Epochs} - D Loss: {avgDiscLoss:F4}, G Loss: {avgGenLoss:F4}");

            // Save generated images
            if ((epoch + 1) % _config.SaveImagesEveryNEpochs == 0)
            {
                SaveGeneratedImages(epoch + 1);
            }
        }
    }

    private double TrainDiscriminatorStep(Tensor<double> realBatch, int batchSize)
    {
        // Generate fake samples
        var noise = _noiseGenerator.GenerateNoise(batchSize, _config.LatentDim);
        var fakeBatch = _gan.Generate(noise);

        // Get discriminator outputs
        var realOutput = _gan.Discriminate(realBatch);
        var fakeOutput = _gan.Discriminate(fakeBatch);  // Detach from generator graph

        // Compute discriminator loss
        double discLoss = BCEGANLoss.DiscriminatorLoss(
            realOutput,
            fakeOutput,
            _config.UseLabelSmoothing,
            _config.LabelSmoothingFactor);

        // Backprop and update discriminator
        // (Actual implementation would call discriminator.Backward() and optimizer.Step())
        _discriminatorOptimizer.Step();

        return discLoss;
    }

    private double TrainGeneratorStep(int batchSize)
    {
        // Generate fake samples
        var noise = _noiseGenerator.GenerateNoise(batchSize, _config.LatentDim);
        var fakeBatch = _gan.Generate(noise);

        // Get discriminator output (gradients flow through discriminator to generator)
        var fakeOutput = _gan.Discriminate(fakeBatch);

        // Compute generator loss
        double genLoss = BCEGANLoss.GeneratorLoss(fakeOutput);

        // Backprop and update generator
        // (Actual implementation would call generator.Backward() and optimizer.Step())
        _generatorOptimizer.Step();

        return genLoss;
    }

    private void SaveGeneratedImages(int epoch)
    {
        // Generate a grid of images for visualization
        int numImages = 16;
        var noise = _noiseGenerator.GenerateNoise(numImages, _config.LatentDim);
        var generatedImages = _gan.Generate(noise);

        string filename = Path.Combine(_config.OutputDirectory, $"epoch_{epoch}.png");
        ImageSaver.SaveImageGrid(generatedImages, filename, gridSize: 4);

        Console.WriteLine($"Saved generated images to {filename}");
    }
}
```

---

## Phase 2: Vanilla GAN

### Step 1: Implement Vanilla Generator

**File**: `src/NeuralNetworks/GANs/Architectures/Vanilla/VanillaGenerator.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.Vanilla;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Vanilla GAN generator with fully connected layers.
/// </summary>
public class VanillaGenerator : IModel
{
    private readonly int _latentDim;
    private readonly int _outputDim;
    private readonly List<ILayer> _layers;

    public VanillaGenerator(int latentDim, int outputDim)
    {
        _latentDim = latentDim;
        _outputDim = outputDim;
        _layers = new List<ILayer>();

        // Architecture: 100 → 256 → 512 → 1024 → 784 (28x28 MNIST)
        _layers.Add(new FullyConnectedLayer(latentDim, 256));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        _layers.Add(new FullyConnectedLayer(256, 512));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        _layers.Add(new FullyConnectedLayer(512, 1024));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        _layers.Add(new FullyConnectedLayer(1024, outputDim));
        _layers.Add(new TanhLayer());  // Output in [-1, 1]
    }

    public Tensor<double> Forward(Tensor<double> input)
    {
        var output = input;

        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        var gradInput = gradOutput;

        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            gradInput = _layers[i].Backward(gradInput);
        }

        return gradInput;
    }

    public List<Parameter> GetParameters()
    {
        var parameters = new List<Parameter>();
        foreach (var layer in _layers)
            parameters.AddRange(layer.GetParameters());
        return parameters;
    }
}
```

### Step 2: Implement Vanilla Discriminator

**File**: `src/NeuralNetworks/GANs/Architectures/Vanilla/VanillaDiscriminator.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.Vanilla;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Vanilla GAN discriminator with fully connected layers.
/// </summary>
public class VanillaDiscriminator : IModel
{
    private readonly int _inputDim;
    private readonly List<ILayer> _layers;

    public VanillaDiscriminator(int inputDim)
    {
        _inputDim = inputDim;
        _layers = new List<ILayer>();

        // Architecture: 784 → 512 → 256 → 1
        _layers.Add(new FullyConnectedLayer(inputDim, 512));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));
        _layers.Add(new DropoutLayer(dropoutRate: 0.3));

        _layers.Add(new FullyConnectedLayer(512, 256));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));
        _layers.Add(new DropoutLayer(dropoutRate: 0.3));

        _layers.Add(new FullyConnectedLayer(256, 1));
        _layers.Add(new SigmoidLayer());  // Output probability [0, 1]
    }

    public Tensor<double> Forward(Tensor<double> input)
    {
        var output = input;

        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        var gradInput = gradOutput;

        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            gradInput = _layers[i].Backward(gradInput);
        }

        return gradInput;
    }

    public List<Parameter> GetParameters()
    {
        var parameters = new List<Parameter>();
        foreach (var layer in _layers)
            parameters.AddRange(layer.GetParameters());
        return parameters;
    }
}
```

### Step 3: Implement Vanilla GAN

**File**: `src/NeuralNetworks/GANs/Architectures/Vanilla/VanillaGAN.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.Vanilla;

using AiDotNet.NeuralNetworks.GANs.Core;
using AiDotNet.NeuralNetworks.GANs.Core.Losses;
using AiDotNet.LinearAlgebra;

/// <summary>
/// Vanilla GAN implementation.
/// </summary>
public class VanillaGAN : IGAN
{
    public IModel Generator { get; private set; }
    public IModel Discriminator { get; private set; }

    public VanillaGAN(int latentDim, int imageDim)
    {
        Generator = new VanillaGenerator(latentDim, imageDim);
        Discriminator = new VanillaDiscriminator(imageDim);
    }

    public Tensor<double> Generate(Tensor<double> noise)
    {
        return Generator.Forward(noise);
    }

    public Tensor<double> Discriminate(Tensor<double> samples)
    {
        return Discriminator.Forward(samples);
    }

    public double ComputeDiscriminatorLoss(Tensor<double> realSamples, Tensor<double> fakeSamples)
    {
        var realOutput = Discriminate(realSamples);
        var fakeOutput = Discriminate(fakeSamples);

        return BCEGANLoss.DiscriminatorLoss(realOutput, fakeOutput);
    }

    public double ComputeGeneratorLoss(Tensor<double> fakeSamples)
    {
        var fakeOutput = Discriminate(fakeSamples);
        return BCEGANLoss.GeneratorLoss(fakeOutput);
    }
}
```

### Testing Vanilla GAN

**File**: `tests/UnitTests/GANs/VanillaGANTests.cs`

```csharp
namespace AiDotNet.Tests.GANs;

using Xunit;
using AiDotNet.NeuralNetworks.GANs.Architectures.Vanilla;
using AiDotNet.LinearAlgebra;

public class VanillaGANTests
{
    [Fact]
    public void VanillaGAN_Generate_ReturnsCorrectShape()
    {
        // Arrange
        int latentDim = 100;
        int imageDim = 784;  // 28x28 MNIST
        var gan = new VanillaGAN(latentDim, imageDim);

        var noise = new Tensor<double>(16, latentDim);  // Batch of 16
        for (int i = 0; i < noise.FlattenedLength; i++)
            noise.SetFlat(i, new Random().NextDouble());

        // Act
        var generated = gan.Generate(noise);

        // Assert
        Assert.Equal(2, generated.Dimensions.Length);
        Assert.Equal(16, generated.Dimensions[0]);  // Batch size
        Assert.Equal(imageDim, generated.Dimensions[1]);  // Image dimension
    }

    [Fact]
    public void VanillaGAN_Discriminate_ReturnsProbs()
    {
        // Arrange
        var gan = new VanillaGAN(latentDim: 100, imageDim: 784);
        var samples = new Tensor<double>(16, 784);

        // Act
        var output = gan.Discriminate(samples);

        // Assert
        Assert.Equal(2, output.Dimensions.Length);
        Assert.Equal(16, output.Dimensions[0]);
        Assert.Equal(1, output.Dimensions[1]);

        // All outputs should be in [0, 1] (sigmoid)
        for (int i = 0; i < output.Dimensions[0]; i++)
        {
            double prob = output[i, 0];
            Assert.True(prob >= 0.0 && prob <= 1.0);
        }
    }
}
```

---

## Phase 3: DCGAN

### Step 1: Implement Transposed Convolution Layer

**File**: `src/NeuralNetworks/Layers/ConvTranspose2dLayer.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Layers;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Transposed convolution layer (deconvolution) for upsampling.
/// </summary>
public class ConvTranspose2dLayer : ILayer
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    private Tensor<double> _weights;  // Shape: (outChannels, inChannels, kernelSize, kernelSize)
    private Vector<double> _bias;     // Shape: (outChannels,)

    private Tensor<double> _input;

    public ConvTranspose2dLayer(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Initialize weights with Xavier initialization
        _weights = new Tensor<double>(outChannels, inChannels, kernelSize, kernelSize);
        double scale = Math.Sqrt(2.0 / (inChannels * kernelSize * kernelSize));
        var rng = new Random();
        for (int i = 0; i < _weights.FlattenedLength; i++)
            _weights.SetFlat(i, rng.NextGaussian(0, scale));

        _bias = new Vector<double>(outChannels);
    }

    public Tensor<double> Forward(Tensor<double> input)
    {
        _input = input;

        // Input shape: (batch, inChannels, height, width)
        int batch = input.Dimensions[0];
        int inHeight = input.Dimensions[2];
        int inWidth = input.Dimensions[3];

        // Output shape calculation
        int outHeight = (inHeight - 1) * _stride - 2 * _padding + _kernelSize;
        int outWidth = (inWidth - 1) * _stride - 2 * _padding + _kernelSize;

        var output = new Tensor<double>(batch, _outChannels, outHeight, outWidth);

        // Perform transposed convolution
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < _outChannels; oc++)
            {
                for (int ic = 0; ic < _inChannels; ic++)
                {
                    for (int i = 0; i < inHeight; i++)
                    {
                        for (int j = 0; j < inWidth; j++)
                        {
                            double inputVal = input[b, ic, i, j];

                            // Apply kernel
                            for (int ki = 0; ki < _kernelSize; ki++)
                            {
                                for (int kj = 0; kj < _kernelSize; kj++)
                                {
                                    int outRow = i * _stride + ki - _padding;
                                    int outCol = j * _stride + kj - _padding;

                                    if (outRow >= 0 && outRow < outHeight &&
                                        outCol >= 0 && outCol < outWidth)
                                    {
                                        output[b, oc, outRow, outCol] += inputVal * _weights[oc, ic, ki, kj];
                                    }
                                }
                            }
                        }
                    }
                }

                // Add bias
                for (int i = 0; i < outHeight; i++)
                    for (int j = 0; j < outWidth; j++)
                        output[b, oc, i, j] += _bias[oc];
            }
        }

        return output;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        // Compute gradients for weights, bias, and input
        // (Simplified for brevity)
        return new Tensor<double>(_input.Dimensions);
    }

    public List<Parameter> GetParameters()
    {
        return new List<Parameter>
        {
            new Parameter { Value = _weights, Name = "weights" },
            new Parameter { Value = _bias, Name = "bias" }
        };
    }
}
```

### Step 2: Implement DCGAN Generator

**File**: `src/NeuralNetworks/GANs/Architectures/DCGAN/DCGANGenerator.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.DCGAN;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// DCGAN generator with transposed convolutional layers.
/// Generates 64x64 RGB images.
/// </summary>
public class DCGANGenerator : IModel
{
    private readonly int _latentDim;
    private readonly List<ILayer> _layers;

    public DCGANGenerator(int latentDim)
    {
        _latentDim = latentDim;
        _layers = new List<ILayer>();

        // Project and reshape: 100 → (4, 4, 1024)
        _layers.Add(new FullyConnectedLayer(latentDim, 4 * 4 * 1024));
        _layers.Add(new ReshapeLayer(4, 4, 1024));
        _layers.Add(new BatchNormLayer(1024));
        _layers.Add(new ReLULayer());

        // Upsample to 8x8
        _layers.Add(new ConvTranspose2dLayer(1024, 512, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new BatchNormLayer(512));
        _layers.Add(new ReLULayer());

        // Upsample to 16x16
        _layers.Add(new ConvTranspose2dLayer(512, 256, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new BatchNormLayer(256));
        _layers.Add(new ReLULayer());

        // Upsample to 32x32
        _layers.Add(new ConvTranspose2dLayer(256, 128, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new BatchNormLayer(128));
        _layers.Add(new ReLULayer());

        // Upsample to 64x64 with 3 channels (RGB)
        _layers.Add(new ConvTranspose2dLayer(128, 3, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new TanhLayer());  // Output in [-1, 1]
    }

    public Tensor<double> Forward(Tensor<double> input)
    {
        var output = input;
        foreach (var layer in _layers)
            output = layer.Forward(output);
        return output;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        var gradInput = gradOutput;
        for (int i = _layers.Count - 1; i >= 0; i--)
            gradInput = _layers[i].Backward(gradInput);
        return gradInput;
    }

    public List<Parameter> GetParameters()
    {
        var parameters = new List<Parameter>();
        foreach (var layer in _layers)
            parameters.AddRange(layer.GetParameters());
        return parameters;
    }
}
```

### Step 3: Implement DCGAN Discriminator

**File**: `src/NeuralNetworks/GANs/Architectures/DCGAN/DCGANDiscriminator.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.DCGAN;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// DCGAN discriminator with strided convolutional layers.
/// Processes 64x64 RGB images.
/// </summary>
public class DCGANDiscriminator : IModel
{
    private readonly List<ILayer> _layers;

    public DCGANDiscriminator()
    {
        _layers = new List<ILayer>();

        // Downsample from 64x64 to 32x32
        _layers.Add(new Conv2dLayer(3, 128, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        // Downsample to 16x16
        _layers.Add(new Conv2dLayer(128, 256, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new BatchNormLayer(256));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        // Downsample to 8x8
        _layers.Add(new Conv2dLayer(256, 512, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new BatchNormLayer(512));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        // Downsample to 4x4
        _layers.Add(new Conv2dLayer(512, 1024, kernelSize: 4, stride: 2, padding: 1));
        _layers.Add(new BatchNormLayer(1024));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        // Final classification
        _layers.Add(new Conv2dLayer(1024, 1, kernelSize: 4, stride: 1, padding: 0));
        _layers.Add(new FlattenLayer());
        _layers.Add(new SigmoidLayer());
    }

    public Tensor<double> Forward(Tensor<double> input)
    {
        var output = input;
        foreach (var layer in _layers)
            output = layer.Forward(output);
        return output;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        var gradInput = gradOutput;
        for (int i = _layers.Count - 1; i >= 0; i--)
            gradInput = _layers[i].Backward(gradInput);
        return gradInput;
    }

    public List<Parameter> GetParameters()
    {
        var parameters = new List<Parameter>();
        foreach (var layer in _layers)
            parameters.AddRange(layer.GetParameters());
        return parameters;
    }
}
```

---

## Phase 4: WGAN

### Step 1: Implement Wasserstein Loss

**File**: `src/NeuralNetworks/GANs/Core/Losses/WassersteinLoss.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Core.Losses;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Wasserstein loss for WGAN.
/// </summary>
public class WassersteinLoss
{
    /// <summary>
    /// Compute critic loss: -(E[C(x)] - E[C(G(z))])
    /// Critic wants to maximize difference between real and fake scores.
    /// </summary>
    public static double CriticLoss(Tensor<double> realOutput, Tensor<double> fakeOutput)
    {
        double realMean = 0.0;
        for (int i = 0; i < realOutput.FlattenedLength; i++)
            realMean += realOutput.GetFlat(i);
        realMean /= realOutput.FlattenedLength;

        double fakeMean = 0.0;
        for (int i = 0; i < fakeOutput.FlattenedLength; i++)
            fakeMean += fakeOutput.GetFlat(i);
        fakeMean /= fakeOutput.FlattenedLength;

        return -(realMean - fakeMean);
    }

    /// <summary>
    /// Compute generator loss: -E[C(G(z))]
    /// Generator wants to maximize critic's score for fake samples.
    /// </summary>
    public static double GeneratorLoss(Tensor<double> fakeOutput)
    {
        double fakeMean = 0.0;
        for (int i = 0; i < fakeOutput.FlattenedLength; i++)
            fakeMean += fakeOutput.GetFlat(i);
        fakeMean /= fakeOutput.FlattenedLength;

        return -fakeMean;
    }
}
```

### Step 2: Implement Gradient Penalty (WGAN-GP)

**File**: `src/NeuralNetworks/GANs/Core/Losses/GradientPenalty.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Core.Losses;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Gradient penalty for WGAN-GP (enforces Lipschitz constraint).
/// </summary>
public class GradientPenalty
{
    /// <summary>
    /// Compute gradient penalty: λ * (||∇C(x_hat)||_2 - 1)²
    /// </summary>
    public static double Compute(
        IModel critic,
        Tensor<double> realSamples,
        Tensor<double> fakeSamples,
        double lambda = 10.0)
    {
        int batchSize = realSamples.Dimensions[0];

        // Interpolate between real and fake samples
        var epsilon = new Tensor<double>(batchSize);
        var rng = new Random();
        for (int i = 0; i < batchSize; i++)
            epsilon[i] = rng.NextDouble();

        var interpolated = new Tensor<double>(realSamples.Dimensions);
        for (int i = 0; i < realSamples.FlattenedLength; i++)
        {
            double real = realSamples.GetFlat(i);
            double fake = fakeSamples.GetFlat(i);
            int batchIdx = i / (realSamples.FlattenedLength / batchSize);
            interpolated.SetFlat(i, epsilon[batchIdx] * real + (1 - epsilon[batchIdx]) * fake);
        }

        // Compute critic output for interpolated samples
        var criticOutput = critic.Forward(interpolated);

        // Compute gradients with respect to interpolated samples
        var gradients = critic.Backward(criticOutput);  // Simplified: assumes backward computes gradients

        // Compute L2 norm of gradients
        double gradientNorm = 0.0;
        for (int i = 0; i < gradients.FlattenedLength; i++)
        {
            double grad = gradients.GetFlat(i);
            gradientNorm += grad * grad;
        }
        gradientNorm = Math.Sqrt(gradientNorm / gradients.FlattenedLength);

        // Penalty: (||gradient|| - 1)²
        double penalty = Math.Pow(gradientNorm - 1.0, 2);

        return lambda * penalty;
    }
}
```

### Step 3: Implement WGAN Critic (No Sigmoid)

**File**: `src/NeuralNetworks/GANs/Architectures/WGAN/WGANCritic.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.WGAN;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// WGAN critic (discriminator without sigmoid output).
/// </summary>
public class WGANCritic : IModel
{
    private readonly List<ILayer> _layers;

    public WGANCritic(int inputDim)
    {
        _layers = new List<ILayer>();

        // Similar to Vanilla Discriminator but no sigmoid
        _layers.Add(new FullyConnectedLayer(inputDim, 512));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        _layers.Add(new FullyConnectedLayer(512, 256));
        _layers.Add(new LeakyReLULayer(alpha: 0.2));

        _layers.Add(new FullyConnectedLayer(256, 1));
        // NO SIGMOID! Critic outputs raw score
    }

    public Tensor<double> Forward(Tensor<double> input)
    {
        var output = input;
        foreach (var layer in _layers)
            output = layer.Forward(output);
        return output;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        var gradInput = gradOutput;
        for (int i = _layers.Count - 1; i >= 0; i--)
            gradInput = _layers[i].Backward(gradInput);
        return gradInput;
    }

    public List<Parameter> GetParameters()
    {
        var parameters = new List<Parameter>();
        foreach (var layer in _layers)
            parameters.AddRange(layer.GetParameters());
        return parameters;
    }
}
```

---

## Phase 5: StyleGAN

**(Simplified implementation - full StyleGAN is very complex)**

### Step 1: Implement Adaptive Instance Normalization (AdaIN)

**File**: `src/NeuralNetworks/GANs/Architectures/StyleGAN/AdaIN.cs`

```csharp
namespace AiDotNet.NeuralNetworks.GANs.Architectures.StyleGAN;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Adaptive Instance Normalization layer for StyleGAN.
/// </summary>
public class AdaINLayer : ILayer
{
    private readonly int _numFeatures;
    private Tensor<double> _scale;
    private Tensor<double> _bias;

    public AdaINLayer(int numFeatures)
    {
        _numFeatures = numFeatures;
    }

    /// <summary>
    /// Apply AdaIN: (x - μ(x)) / σ(x) * scale(style) + bias(style)
    /// </summary>
    public Tensor<double> Forward(Tensor<double> content, Tensor<double> style)
    {
        // content shape: (batch, channels, height, width)
        // style shape: (batch, styleChannels)

        int batch = content.Dimensions[0];
        int channels = content.Dimensions[1];
        int height = content.Dimensions[2];
        int width = content.Dimensions[3];

        // Compute learned scale and bias from style
        _scale = ComputeScale(style);  // Shape: (batch, channels)
        _bias = ComputeBias(style);    // Shape: (batch, channels)

        var output = new Tensor<double>(content.Dimensions);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                // Compute mean and std for this channel
                double mean = 0.0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        mean += content[b, c, h, w];
                mean /= (height * width);

                double variance = 0.0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        variance += Math.Pow(content[b, c, h, w] - mean, 2);
                variance /= (height * width);
                double std = Math.Sqrt(variance + 1e-5);

                // Normalize and apply style
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double normalized = (content[b, c, h, w] - mean) / std;
                        output[b, c, h, w] = normalized * _scale[b, c] + _bias[b, c];
                    }
                }
            }
        }

        return output;
    }

    private Tensor<double> ComputeScale(Tensor<double> style)
    {
        // Learned linear transformation: style → scale
        // (Simplified: in real StyleGAN, this is a learned FC layer)
        return style;
    }

    private Tensor<double> ComputeBias(Tensor<double> style)
    {
        // Learned linear transformation: style → bias
        return style;
    }

    public Tensor<double> Backward(Tensor<double> gradOutput)
    {
        // (Simplified)
        return gradOutput;
    }

    public List<Parameter> GetParameters()
    {
        return new List<Parameter>();
    }
}
```

---

## Testing Strategy

### Unit Tests

1. **Generator Tests**: Test output shapes, value ranges
2. **Discriminator Tests**: Test classification outputs
3. **Loss Tests**: Test loss computations
4. **Training Tests**: Test alternating optimization

### Integration Tests

**File**: `tests/IntegrationTests/GANs/GANTrainingTests.cs`

```csharp
namespace AiDotNet.Tests.Integration.GANs;

using Xunit;
using AiDotNet.NeuralNetworks.GANs.Architectures.Vanilla;
using AiDotNet.NeuralNetworks.GANs.Core;

public class GANTrainingTests
{
    [Fact]
    public void VanillaGAN_TrainOnMNIST_GeneratesImages()
    {
        // Arrange
        var gan = new VanillaGAN(latentDim: 100, imageDim: 784);
        var genOptimizer = new Adam(learningRate: 0.0002);
        var discOptimizer = new Adam(learningRate: 0.0002);
        var config = new GANConfig { Epochs = 10, BatchSize = 64 };

        var trainer = new GANTrainer(gan, genOptimizer, discOptimizer, config, seed: 42);

        var mnistLoader = LoadMNISTData();  // Mock data loader

        // Act
        trainer.Train(mnistLoader);

        // Assert
        // Check that generated images exist
        Assert.True(Directory.Exists(config.OutputDirectory));

        var generatedFiles = Directory.GetFiles(config.OutputDirectory, "*.png");
        Assert.NotEmpty(generatedFiles);
    }
}
```

---

## Common Pitfalls

### 1. Mode Collapse

**Symptom**: Generator produces identical outputs

**Solution**:
```csharp
// Use minibatch discrimination or switch to WGAN
var wganConfig = new GANConfig
{
    DiscriminatorUpdatesPerGenerator = 5  // Train critic more frequently
};
```

### 2. Vanishing Gradients

**Symptom**: Generator loss stays high, doesn't improve

**Solution**: Use non-saturating loss
```csharp
// Instead of: -log(1 - D(G(z)))
// Use: log(D(G(z)))  (maximizes discriminator output for fakes)
```

### 3. Training Instability

**Symptom**: Losses oscillate wildly

**Solution**: Use learning rate scheduling
```csharp
var scheduler = new ExponentialLRScheduler(optimizer, gamma: 0.99);
for (int epoch = 0; epoch < epochs; epoch++)
{
    TrainEpoch();
    scheduler.Step();
}
```

### 4. Discriminator Too Strong

**Symptom**: Discriminator accuracy = 100%, generator doesn't learn

**Solution**: Update discriminator less frequently
```csharp
config.DiscriminatorUpdatesPerGenerator = 1;  // Instead of 5
```

### 5. Wrong Input Normalization

**Symptom**: Generator outputs don't look realistic

**Solution**: Normalize real images to match generator output range
```csharp
// If generator uses Tanh (output in [-1, 1]):
realImages = (realImages - 0.5) * 2;  // Scale [0, 1] → [-1, 1]
```

---

## Summary

This guide covered:

1. **GAN Fundamentals**: Adversarial training, loss functions
2. **Vanilla GAN**: Fully connected architecture for simple datasets
3. **DCGAN**: Convolutional architecture for natural images
4. **WGAN**: Wasserstein loss for stable training
5. **StyleGAN**: Advanced style-based generation

**Key Takeaways**:
- GANs train two competing networks (generator and discriminator)
- Training is inherently unstable; careful tuning required
- DCGAN guidelines improve stability for image generation
- WGAN provides more stable training via Wasserstein distance
- StyleGAN enables high-quality, controllable generation

**Next Steps**:
- Implement conditional GANs (cGAN)
- Add CycleGAN for unpaired image translation
- Implement progressive growing for high-resolution generation
- Add evaluation metrics (FID, Inception Score)
