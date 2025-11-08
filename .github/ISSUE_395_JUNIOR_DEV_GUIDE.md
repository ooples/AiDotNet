# Junior Developer Implementation Guide: Issue #395
## Diffusion Models (DDPM, Stable Diffusion)

### Overview
This guide will walk you through implementing Diffusion Models for AiDotNet. Diffusion models are state-of-the-art generative models that create high-quality images by learning to reverse a gradual noising process. They power tools like Stable Diffusion, DALL-E 2, and Midjourney.

---

## Understanding Diffusion Models

### What Are Diffusion Models?

Diffusion models work like a "reverse movie" of image destruction:

1. **Forward Process (Diffusion)**: Take a real image and gradually add Gaussian noise over many timesteps until it becomes pure random noise. This is a fixed mathematical process (no learning involved).

2. **Reverse Process (Denoising)**: Train a neural network to reverse this process - start with random noise and gradually remove it step by step to generate a realistic image.

**Real-World Analogy**:
- **Forward**: Like a photograph slowly fading and becoming grainy over 1000 days
- **Reverse**: Training an AI to restore the original photograph from the grainy version, one day at a time

### Why Diffusion Models Are Powerful

1. **High Quality**: Generate photorealistic images with fine details
2. **Stable Training**: More stable than GANs (no mode collapse, adversarial issues)
3. **Flexible**: Can be conditioned on text, images, or other inputs
4. **Interpretable**: The gradual denoising process is easier to understand than GAN's single-step generation

### Key Concepts

#### 1. Forward Diffusion Process (Adding Noise)

```
Given a clean image x₀:
- At timestep t, add Gaussian noise to create x_t
- The amount of noise is controlled by a noise schedule (β₁, β₂, ..., β_T)
- At t=T (final timestep), x_T is pure Gaussian noise

Mathematical formulation:
x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε

Where:
- ε ~ N(0, I) is random Gaussian noise
- ᾱ_t = ∏(1 - β_s) for s=1 to t
- β_t is the noise schedule at timestep t
```

**Key Insight**: We can sample x_t directly from x₀ (no need to go through all intermediate steps)!

#### 2. Noise Schedules

Controls how fast noise is added:

**Linear Schedule**:
```
β_t increases linearly from β_start to β_end
Example: β_start = 0.0001, β_end = 0.02, T = 1000
```

**Cosine Schedule** (Better for images):
```
Uses a cosine function for smoother transitions
Better preserves signal at early timesteps
```

**Why it matters**: The schedule affects:
- Training stability
- Generation quality
- How many steps are needed for good results

#### 3. Reverse Diffusion (Denoising)

```
Goal: Learn to predict the noise added at each timestep

Given noisy image x_t and timestep t:
- Neural network predicts: ε_θ(x_t, t)
- Remove predicted noise to get x_{t-1}
- Repeat until we reach x₀ (clean image)

Training objective (simplified):
L = E[||ε - ε_θ(x_t, t)||²]

We train the network to predict the noise, then subtract it!
```

#### 4. DDPM (Denoising Diffusion Probabilistic Models)

**Original formulation by Ho et al. (2020)**:

**Sampling Algorithm**:
```
Start with x_T ~ N(0, I) (random noise)

For t = T down to 1:
    1. Predict noise: ε = ε_θ(x_t, t)
    2. Compute mean: μ_t = (1/√(1-β_t)) · (x_t - (β_t/√(1-ᾱ_t)) · ε)
    3. Sample: x_{t-1} = μ_t + σ_t · z, where z ~ N(0, I)
    4. (σ_t is the variance at timestep t)

Return x₀
```

**Key Properties**:
- Requires T steps (typically 1000) for high quality
- Stochastic sampling (randomness at each step)
- Slow but high quality

#### 5. DDIM (Denoising Diffusion Implicit Models)

**Improvement by Song et al. (2021)**:

**Key Innovation**: Deterministic sampling with fewer steps

**Sampling Algorithm**:
```
Start with x_T ~ N(0, I)

For t = T, T-skip, T-2×skip, ..., 0:
    1. Predict noise: ε = ε_θ(x_t, t)
    2. Predict x₀: x̂₀ = (x_t - √(1-ᾱ_t) · ε) / √(ᾱ_t)
    3. Compute x_{t-skip}: x_{t-skip} = √(ᾱ_{t-skip}) · x̂₀ + √(1-ᾱ_{t-skip}) · ε

Return x₀
```

**Advantages**:
- Can skip steps (e.g., use 50 steps instead of 1000)
- Deterministic (same noise → same image)
- 10-50x faster than DDPM with similar quality

#### 6. Stable Diffusion

**Modern architecture by Rombach et al. (2022)**:

**Key Innovation**: Work in latent space instead of pixel space

**Architecture**:
```
Text → CLIP Text Encoder → Text Embeddings
    ↓
Image → VAE Encoder → Latent z (compressed 8x)
    ↓
Latent Diffusion (U-Net with cross-attention to text)
    ↓
VAE Decoder → Generated Image

Benefits:
- 8x compression: 512×512 image → 64×64 latent
- Much faster and more memory-efficient
- Can condition on text, images, etc.
```

**Components**:
1. **VAE (Variational Autoencoder)**: Compress images to latent space
2. **U-Net**: Predict noise in latent space (with text cross-attention)
3. **Text Encoder**: CLIP or T5 to encode text prompts
4. **Scheduler**: DDIM, DDPM, or others for sampling

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── IDiffusionModel.cs           # Main diffusion interface
│   ├── INoiseScheduler.cs           # Noise schedule interface
│   ├── ITimeEmbedding.cs            # Timestep embedding interface
│   └── IUNet.cs                     # U-Net interface
├── Models/
│   └── Generative/
│       └── Diffusion/
│           ├── DiffusionModelBase.cs      # Base diffusion model
│           ├── DDPMModel.cs               # DDPM implementation
│           ├── DDIMSampler.cs             # DDIM sampling
│           └── StableDiffusion.cs         # Stable Diffusion
├── Diffusion/
│   ├── Schedulers/
│   │   ├── NoiseScheduler.cs        # Noise schedule base
│   │   ├── LinearSchedule.cs        # Linear beta schedule
│   │   ├── CosineSchedule.cs        # Cosine schedule
│   │   └── DDIMScheduler.cs         # DDIM-specific scheduler
│   ├── UNet/
│   │   ├── UNetModel.cs             # U-Net architecture
│   │   ├── ResidualBlock.cs         # ResNet-style blocks
│   │   ├── AttentionBlock.cs        # Self-attention
│   │   ├── CrossAttentionBlock.cs   # Cross-attention (for text)
│   │   └── TimeEmbedding.cs         # Sinusoidal time embeddings
│   └── VAE/
│       ├── VAEEncoder.cs            # Image → latent
│       └── VAEDecoder.cs            # Latent → image
```

### Class Hierarchy
```
IDiffusionModel<T>
    ↓ implements IGenerativeModel<T>
    ↓
DiffusionModelBase<T> (abstract)
    ├── DDPMModel<T>         # Original DDPM
    └── StableDiffusion<T>   # Latent diffusion

INoiseScheduler<T>
    ├── LinearSchedule<T>    # Linear beta schedule
    └── CosineSchedule<T>    # Cosine schedule

IUNet<T>
    └── UNetModel<T>         # U-Net with attention
```

---

## Step-by-Step Implementation

### Step 1: Core Interfaces

#### File: `src/Interfaces/INoiseScheduler.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a noise scheduler for diffusion models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// The noise scheduler controls how noise is added during the forward process
/// and removed during the reverse process.
///
/// Key concepts:
/// - **Beta (β_t)**: Amount of noise added at timestep t
/// - **Alpha (α_t)**: 1 - β_t (signal retained)
/// - **Alpha bar (ᾱ_t)**: Cumulative product of alphas
///
/// The scheduler pre-computes these values for efficient sampling.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface INoiseScheduler<T>
{
    /// <summary>
    /// Gets the total number of timesteps.
    /// </summary>
    int NumTimesteps { get; }

    /// <summary>
    /// Gets the beta value at a specific timestep.
    /// </summary>
    /// <param name="t">The timestep (0 to NumTimesteps-1).</param>
    /// <returns>Beta value at timestep t.</returns>
    T GetBeta(int t);

    /// <summary>
    /// Gets the alpha value at a specific timestep (α_t = 1 - β_t).
    /// </summary>
    T GetAlpha(int t);

    /// <summary>
    /// Gets the cumulative alpha product at a specific timestep.
    /// ᾱ_t = ∏(α_s) for s=0 to t
    /// </summary>
    T GetAlphaBar(int t);

    /// <summary>
    /// Adds noise to a clean sample at a given timestep.
    /// x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε
    /// </summary>
    /// <param name="x0">Clean sample.</param>
    /// <param name="noise">Random Gaussian noise.</param>
    /// <param name="t">Timestep.</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>Noisy sample at timestep t.</returns>
    Tensor<T> AddNoise(Tensor<T> x0, Tensor<T> noise, int t, INumericOperations<T> ops);

    /// <summary>
    /// Removes noise from a sample (one denoising step).
    /// </summary>
    /// <param name="xt">Noisy sample at timestep t.</param>
    /// <param name="predictedNoise">Noise predicted by the model.</param>
    /// <param name="t">Current timestep.</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>Denoised sample at timestep t-1.</returns>
    Tensor<T> RemoveNoise(
        Tensor<T> xt,
        Tensor<T> predictedNoise,
        int t,
        INumericOperations<T> ops);
}
```

#### File: `src/Interfaces/IDiffusionModel.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a diffusion model for image generation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Diffusion models generate images by:
/// 1. Starting with random noise
/// 2. Gradually removing noise over many steps
/// 3. Using a neural network to predict the noise at each step
///
/// The model can be:
/// - **Unconditional**: Generate random images from the training distribution
/// - **Conditional**: Generate images based on text, class labels, or other inputs
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IDiffusionModel<T> : IGenerativeModel<T>
{
    /// <summary>
    /// Gets the noise scheduler.
    /// </summary>
    INoiseScheduler<T> Scheduler { get; }

    /// <summary>
    /// Gets the number of diffusion timesteps.
    /// </summary>
    int NumTimesteps { get; }

    /// <summary>
    /// Predicts the noise in a noisy sample at a given timestep.
    /// </summary>
    /// <param name="xt">Noisy sample at timestep t.</param>
    /// <param name="t">Timestep (0 to NumTimesteps-1).</param>
    /// <param name="condition">Optional conditioning (text embeddings, class labels, etc.).</param>
    /// <returns>Predicted noise tensor.</returns>
    Tensor<T> PredictNoise(Tensor<T> xt, int t, Tensor<T>? condition = null);

    /// <summary>
    /// Performs one denoising step.
    /// </summary>
    /// <param name="xt">Noisy sample at timestep t.</param>
    /// <param name="t">Current timestep.</param>
    /// <param name="condition">Optional conditioning.</param>
    /// <returns>Less noisy sample at timestep t-1.</returns>
    Tensor<T> DenoisingStep(Tensor<T> xt, int t, Tensor<T>? condition = null);

    /// <summary>
    /// Generates samples from random noise.
    /// </summary>
    /// <param name="shape">Shape of samples to generate [batch, channels, height, width].</param>
    /// <param name="condition">Optional conditioning.</param>
    /// <param name="numInferenceSteps">Number of denoising steps (can be less than NumTimesteps).</param>
    /// <returns>Generated samples.</returns>
    Tensor<T> Sample(int[] shape, Tensor<T>? condition = null, int? numInferenceSteps = null);

    /// <summary>
    /// Trains the diffusion model on a batch of images.
    /// </summary>
    /// <param name="images">Training images.</param>
    /// <param name="condition">Optional conditioning for conditional generation.</param>
    /// <returns>Training loss.</returns>
    T TrainStep(Tensor<T> images, Tensor<T>? condition = null);
}
```

### Step 2: Noise Schedulers

#### File: `src/Diffusion/Schedulers/NoiseScheduler.cs`

```csharp
namespace AiDotNet.Diffusion.Schedulers;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Base class for noise schedulers in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class NoiseScheduler<T> : INoiseScheduler<T>
{
    protected readonly Vector<T> _betas;
    protected readonly Vector<T> _alphas;
    protected readonly Vector<T> _alphaBars;
    protected readonly Vector<T> _sqrtAlphaBars;
    protected readonly Vector<T> _sqrtOneMinusAlphaBars;
    protected readonly int _numTimesteps;
    protected readonly INumericOperations<T> _ops;

    /// <summary>
    /// Initializes a new instance of the <see cref="NoiseScheduler{T}"/> class.
    /// </summary>
    /// <param name="numTimesteps">Total number of diffusion timesteps.</param>
    /// <param name="ops">Numeric operations provider.</param>
    protected NoiseScheduler(int numTimesteps, INumericOperations<T> ops)
    {
        Guard.Positive(numTimesteps, nameof(numTimesteps));
        Guard.NotNull(ops, nameof(ops));

        _numTimesteps = numTimesteps;
        _ops = ops;

        _betas = new Vector<T>(numTimesteps);
        _alphas = new Vector<T>(numTimesteps);
        _alphaBars = new Vector<T>(numTimesteps);
        _sqrtAlphaBars = new Vector<T>(numTimesteps);
        _sqrtOneMinusAlphaBars = new Vector<T>(numTimesteps);

        InitializeSchedule();
        ComputeDerivedValues();
    }

    /// <inheritdoc/>
    public int NumTimesteps => _numTimesteps;

    /// <inheritdoc/>
    public T GetBeta(int t)
    {
        Guard.InRange(t, 0, _numTimesteps - 1, nameof(t));
        return _betas[t];
    }

    /// <inheritdoc/>
    public T GetAlpha(int t)
    {
        Guard.InRange(t, 0, _numTimesteps - 1, nameof(t));
        return _alphas[t];
    }

    /// <inheritdoc/>
    public T GetAlphaBar(int t)
    {
        Guard.InRange(t, 0, _numTimesteps - 1, nameof(t));
        return _alphaBars[t];
    }

    /// <inheritdoc/>
    public Tensor<T> AddNoise(
        Tensor<T> x0,
        Tensor<T> noise,
        int t,
        INumericOperations<T> ops)
    {
        Guard.NotNull(x0, nameof(x0));
        Guard.NotNull(noise, nameof(noise));
        Guard.NotNull(ops, nameof(ops));
        Guard.InRange(t, 0, _numTimesteps - 1, nameof(t));

        if (!x0.Shape.SequenceEqual(noise.Shape))
        {
            throw new ArgumentException(
                $"x0 and noise must have the same shape. Got x0: [{string.Join(", ", x0.Shape)}], noise: [{string.Join(", ", noise.Shape)}]",
                nameof(noise));
        }

        // x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε
        var sqrtAlphaBar = _sqrtAlphaBars[t];
        var sqrtOneMinusAlphaBar = _sqrtOneMinusAlphaBars[t];

        var result = new Tensor<T>(x0.Shape);
        for (int i = 0; i < x0.Data.Length; i++)
        {
            var signalPart = ops.Multiply(sqrtAlphaBar, x0.Data[i]);
            var noisePart = ops.Multiply(sqrtOneMinusAlphaBar, noise.Data[i]);
            result.Data[i] = ops.Add(signalPart, noisePart);
        }

        return result;
    }

    /// <inheritdoc/>
    public abstract Tensor<T> RemoveNoise(
        Tensor<T> xt,
        Tensor<T> predictedNoise,
        int t,
        INumericOperations<T> ops);

    /// <summary>
    /// Initializes the beta schedule. Must be implemented by subclasses.
    /// </summary>
    protected abstract void InitializeSchedule();

    /// <summary>
    /// Computes derived values (alphas, alpha_bars, etc.) from betas.
    /// </summary>
    private void ComputeDerivedValues()
    {
        // α_t = 1 - β_t
        for (int t = 0; t < _numTimesteps; t++)
        {
            _alphas[t] = _ops.Subtract(_ops.One, _betas[t]);
        }

        // ᾱ_t = ∏(α_s) for s=0 to t
        T cumulativeProduct = _ops.One;
        for (int t = 0; t < _numTimesteps; t++)
        {
            cumulativeProduct = _ops.Multiply(cumulativeProduct, _alphas[t]);
            _alphaBars[t] = cumulativeProduct;
        }

        // Pre-compute square roots for efficiency
        for (int t = 0; t < _numTimesteps; t++)
        {
            _sqrtAlphaBars[t] = _ops.Sqrt(_alphaBars[t]);
            var oneMinusAlphaBar = _ops.Subtract(_ops.One, _alphaBars[t]);
            _sqrtOneMinusAlphaBars[t] = _ops.Sqrt(oneMinusAlphaBar);
        }
    }
}
```

#### File: `src/Diffusion/Schedulers/LinearSchedule.cs`

```csharp
namespace AiDotNet.Diffusion.Schedulers;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements a linear noise schedule for diffusion models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Linear schedule increases noise uniformly from β_start to β_end.
///
/// Example: With β_start = 0.0001 and β_end = 0.02:
/// - At t=0: very little noise added (β = 0.0001)
/// - At t=T/2: medium noise (β ≈ 0.01)
/// - At t=T: maximum noise (β = 0.02)
///
/// This was used in the original DDPM paper.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LinearSchedule<T> : NoiseScheduler<T>
{
    private readonly double _betaStart;
    private readonly double _betaEnd;

    /// <summary>
    /// Initializes a new instance of the <see cref="LinearSchedule{T}"/> class.
    /// </summary>
    /// <param name="numTimesteps">Total number of timesteps.</param>
    /// <param name="betaStart">Starting beta value (e.g., 0.0001).</param>
    /// <param name="betaEnd">Ending beta value (e.g., 0.02).</param>
    /// <param name="ops">Numeric operations provider.</param>
    public LinearSchedule(
        int numTimesteps,
        double betaStart,
        double betaEnd,
        INumericOperations<T> ops)
        : base(numTimesteps, ops)
    {
        Guard.Positive(betaStart, nameof(betaStart));
        Guard.Positive(betaEnd, nameof(betaEnd));

        if (betaStart >= betaEnd)
        {
            throw new ArgumentException(
                $"betaStart ({betaStart}) must be less than betaEnd ({betaEnd})",
                nameof(betaStart));
        }

        _betaStart = betaStart;
        _betaEnd = betaEnd;
    }

    /// <inheritdoc/>
    protected override void InitializeSchedule()
    {
        // Linear interpolation from betaStart to betaEnd
        for (int t = 0; t < _numTimesteps; t++)
        {
            double fraction = (double)t / (_numTimesteps - 1);
            double beta = _betaStart + fraction * (_betaEnd - _betaStart);
            _betas[t] = _ops.FromDouble(beta);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> RemoveNoise(
        Tensor<T> xt,
        Tensor<T> predictedNoise,
        int t,
        INumericOperations<T> ops)
    {
        Guard.NotNull(xt, nameof(xt));
        Guard.NotNull(predictedNoise, nameof(predictedNoise));
        Guard.NotNull(ops, nameof(ops));
        Guard.InRange(t, 0, _numTimesteps - 1, nameof(t));

        // DDPM sampling formula:
        // μ_t = (1/√(α_t)) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t))
        // x_{t-1} = μ_t + σ_t · z, where z ~ N(0, I)

        var beta = _betas[t];
        var alpha = _alphas[t];
        var alphaBar = _alphaBars[t];
        var sqrtOneMinusAlphaBar = _sqrtOneMinusAlphaBars[t];

        // Compute mean
        var result = new Tensor<T>(xt.Shape);
        var sqrtAlpha = ops.Sqrt(alpha);
        var coeff = ops.Divide(beta, sqrtOneMinusAlphaBar);

        for (int i = 0; i < xt.Data.Length; i++)
        {
            var noiseTerm = ops.Multiply(coeff, predictedNoise.Data[i]);
            var mean = ops.Subtract(xt.Data[i], noiseTerm);
            mean = ops.Divide(mean, sqrtAlpha);

            // Add variance if not at the last step
            if (t > 0)
            {
                // Compute variance: σ_t² = β_t
                var sigma = ops.Sqrt(beta);
                var random = new Random();
                var z = SampleGaussian(random);
                var noise = ops.Multiply(sigma, ops.FromDouble(z));
                result.Data[i] = ops.Add(mean, noise);
            }
            else
            {
                result.Data[i] = mean;
            }
        }

        return result;
    }

    private static double SampleGaussian(Random random)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
```

### Step 3: Time Embeddings

#### File: `src/Diffusion/UNet/TimeEmbedding.cs`

```csharp
namespace AiDotNet.Diffusion.UNet;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements sinusoidal time embeddings for diffusion models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Time embeddings tell the U-Net what timestep it's processing.
///
/// Why we need them:
/// - Different timesteps require different denoising strategies
/// - At t=1000 (pure noise): need aggressive denoising
/// - At t=10 (almost clean): need gentle refinement
///
/// Sinusoidal embeddings:
/// - Use sine and cosine functions of different frequencies
/// - Similar to positional encodings in transformers
/// - Allow the model to interpolate between timesteps
///
/// Example: For timestep t=500 and embedding dimension 256:
/// - pos_enc[0] = sin(500 / 10000^(0/128))
/// - pos_enc[1] = cos(500 / 10000^(0/128))
/// - pos_enc[2] = sin(500 / 10000^(2/128))
/// - ... and so on
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TimeEmbedding<T>
{
    private readonly int _embedDim;
    private readonly int _maxPeriod;
    private readonly Matrix<T> _mlpWeights1;
    private readonly Vector<T> _mlpBias1;
    private readonly Matrix<T> _mlpWeights2;
    private readonly Vector<T> _mlpBias2;

    /// <summary>
    /// Initializes a new instance of the <see cref="TimeEmbedding{T}"/> class.
    /// </summary>
    /// <param name="embedDim">Dimension of the time embedding.</param>
    /// <param name="mlpDim">Hidden dimension of the MLP (typically 4 × embed_dim).</param>
    /// <param name="maxPeriod">Maximum period for sinusoidal encoding (default: 10000).</param>
    /// <param name="ops">Numeric operations provider.</param>
    public TimeEmbedding(int embedDim, int mlpDim, int maxPeriod, INumericOperations<T> ops)
    {
        Guard.Positive(embedDim, nameof(embedDim));
        Guard.Positive(mlpDim, nameof(mlpDim));
        Guard.Positive(maxPeriod, nameof(maxPeriod));
        Guard.NotNull(ops, nameof(ops));

        if (embedDim % 2 != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension must be even, got {embedDim}",
                nameof(embedDim));
        }

        _embedDim = embedDim;
        _maxPeriod = maxPeriod;

        // MLP to transform sinusoidal embeddings
        _mlpWeights1 = new Matrix<T>(mlpDim, embedDim);
        _mlpBias1 = new Vector<T>(mlpDim);
        _mlpWeights2 = new Matrix<T>(embedDim, mlpDim);
        _mlpBias2 = new Vector<T>(embedDim);

        InitializeWeights(ops);
    }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbedDim => _embedDim;

    /// <summary>
    /// Computes time embeddings for a batch of timesteps.
    /// </summary>
    /// <param name="timesteps">Timesteps to embed [batch_size].</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>Time embeddings [batch_size, embed_dim].</returns>
    public Tensor<T> Forward(int[] timesteps, INumericOperations<T> ops)
    {
        Guard.NotNull(timesteps, nameof(timesteps));
        Guard.NotNull(ops, nameof(ops));

        int batchSize = timesteps.Length;

        // Compute sinusoidal embeddings
        var sinusoidalEmbeds = new Tensor<T>(new[] { batchSize, _embedDim });

        for (int b = 0; b < batchSize; b++)
        {
            int t = timesteps[b];
            var embedding = ComputeSinusoidalEmbedding(t, ops);

            for (int i = 0; i < _embedDim; i++)
            {
                sinusoidalEmbeds[b, i] = embedding[i];
            }
        }

        // Apply MLP: Linear → SiLU → Linear
        var hidden = ApplyLinear(sinusoidalEmbeds, _mlpWeights1, _mlpBias1, ops);
        hidden = ApplySiLU(hidden, ops);
        var output = ApplyLinear(hidden, _mlpWeights2, _mlpBias2, ops);

        return output;
    }

    private Vector<T> ComputeSinusoidalEmbedding(int timestep, INumericOperations<T> ops)
    {
        var embedding = new Vector<T>(_embedDim);
        int halfDim = _embedDim / 2;

        // Compute frequencies: 1 / (max_period ^ (2i / embed_dim))
        for (int i = 0; i < halfDim; i++)
        {
            double exponent = -Math.Log(_maxPeriod) * (2.0 * i) / _embedDim;
            double freq = Math.Exp(exponent);

            // Sine component
            embedding[i] = ops.FromDouble(Math.Sin(timestep * freq));

            // Cosine component
            embedding[halfDim + i] = ops.FromDouble(Math.Cos(timestep * freq));
        }

        return embedding;
    }

    private Tensor<T> ApplyLinear(
        Tensor<T> input,
        Matrix<T> weights,
        Vector<T> bias,
        INumericOperations<T> ops)
    {
        // input: [batch, in_features]
        // weights: [out_features, in_features]
        // output: [batch, out_features]

        var shape = input.Shape;
        int batch = shape[0];
        int inFeatures = shape[1];
        int outFeatures = weights.Rows;

        var output = new Tensor<T>(new[] { batch, outFeatures });

        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < outFeatures; o++)
            {
                T sum = bias[o];
                for (int i = 0; i < inFeatures; i++)
                {
                    var prod = ops.Multiply(input[b, i], weights[o, i]);
                    sum = ops.Add(sum, prod);
                }
                output[b, o] = sum;
            }
        }

        return output;
    }

    private Tensor<T> ApplySiLU(Tensor<T> input, INumericOperations<T> ops)
    {
        // SiLU (Swish): x * sigmoid(x)
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Data.Length; i++)
        {
            var x = input.Data[i];
            // sigmoid(x) = 1 / (1 + e^(-x))
            var negX = ops.Negate(x);
            var expNegX = ops.Exp(negX);
            var sigmoid = ops.Divide(ops.One, ops.Add(ops.One, expNegX));
            output.Data[i] = ops.Multiply(x, sigmoid);
        }

        return output;
    }

    private void InitializeWeights(INumericOperations<T> ops)
    {
        var random = new Random(42);

        // Xavier initialization for MLP weights
        double stddev1 = Math.Sqrt(2.0 / (_embedDim + _mlpWeights1.Rows));
        InitializeMatrix(_mlpWeights1, random, stddev1, ops);

        double stddev2 = Math.Sqrt(2.0 / (_mlpWeights1.Rows + _embedDim));
        InitializeMatrix(_mlpWeights2, random, stddev2, ops);

        // Zero bias
        for (int i = 0; i < _mlpBias1.Length; i++)
        {
            _mlpBias1[i] = ops.Zero;
        }

        for (int i = 0; i < _mlpBias2.Length; i++)
        {
            _mlpBias2[i] = ops.Zero;
        }
    }

    private void InitializeMatrix(
        Matrix<T> matrix,
        Random random,
        double stddev,
        INumericOperations<T> ops)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                matrix[i, j] = ops.FromDouble(stddev * z0);
            }
        }
    }
}
```

### Step 4: U-Net Architecture (Simplified)

#### File: `src/Interfaces/IUNet.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a U-Net architecture for diffusion models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// U-Net is a neural network shaped like the letter "U":
///
/// Structure:
/// ```
/// Input → Encoder (downsampling) → Bottleneck → Decoder (upsampling) → Output
///         ↓                                           ↑
///         Skip connections ────────────────────────────
/// ```
///
/// Why U-Net for diffusion:
/// - **Encoder**: Compress image to capture semantic information
/// - **Bottleneck**: Process at lowest resolution with attention
/// - **Decoder**: Reconstruct details with help from skip connections
/// - **Skip connections**: Preserve fine details from encoder
///
/// For diffusion, U-Net predicts the noise added to the image.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IUNet<T>
{
    /// <summary>
    /// Predicts noise in a noisy image at a given timestep.
    /// </summary>
    /// <param name="noisyImage">Noisy image [batch, channels, height, width].</param>
    /// <param name="timestep">Timestep(s) [batch] or single value.</param>
    /// <param name="condition">Optional conditioning (text embeddings, class labels, etc.).</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>Predicted noise [batch, channels, height, width].</returns>
    Tensor<T> Forward(
        Tensor<T> noisyImage,
        int[] timestep,
        Tensor<T>? condition,
        INumericOperations<T> ops);

    /// <summary>
    /// Gets the input channels.
    /// </summary>
    int InChannels { get; }

    /// <summary>
    /// Gets the output channels.
    /// </summary>
    int OutChannels { get; }

    /// <summary>
    /// Gets the base number of channels (increases with depth).
    /// </summary>
    int BaseChannels { get; }

    /// <summary>
    /// Gets the number of downsampling/upsampling stages.
    /// </summary>
    int NumLevels { get; }
}
```

### Step 5: DDPM Model

#### File: `src/Models/Generative/Diffusion/DDPMModel.cs`

```csharp
namespace AiDotNet.Models.Generative.Diffusion;

using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements the Denoising Diffusion Probabilistic Model (DDPM).
/// </summary>
/// <remarks>
/// <para><b>Paper</b>: "Denoising Diffusion Probabilistic Models"
/// by Ho et al. (NeurIPS 2020)
///
/// <b>Key Contributions</b>:
/// 1. Simplified training objective: predict noise instead of x₀
/// 2. Linear or cosine noise schedule
/// 3. High-quality image generation with simple architecture
///
/// <b>Training Process</b>:
/// 1. Sample a clean image x₀ from training data
/// 2. Sample timestep t uniformly from [0, T]
/// 3. Sample noise ε ~ N(0, I)
/// 4. Create noisy image: x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε
/// 5. Predict noise: ε_θ(x_t, t)
/// 6. Compute loss: L = ||ε - ε_θ(x_t, t)||²
/// 7. Update model parameters
///
/// <b>Sampling Process</b>:
/// 1. Start with x_T ~ N(0, I) (random noise)
/// 2. For t = T down to 1:
///    - Predict noise: ε = ε_θ(x_t, t)
///    - Compute mean: μ_t (see scheduler)
///    - Sample: x_{t-1} = μ_t + σ_t·z
/// 3. Return x₀
///
/// <b>For Beginners</b>:
/// DDPM showed that diffusion models can generate high-quality images
/// by learning to reverse a gradual noising process. The key insight is
/// that predicting noise is easier than predicting the clean image directly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DDPMModel<T> : IDiffusionModel<T>
{
    private readonly IUNet<T> _unet;
    private readonly INoiseScheduler<T> _scheduler;
    private readonly INumericOperations<T> _ops;
    private readonly int _imageSize;
    private readonly int _channels;

    /// <summary>
    /// Initializes a new instance of the <see cref="DDPMModel{T}"/> class.
    /// </summary>
    /// <param name="unet">The U-Net architecture for noise prediction.</param>
    /// <param name="scheduler">The noise scheduler.</param>
    /// <param name="imageSize">Size of images (assumed square).</param>
    /// <param name="channels">Number of image channels (e.g., 3 for RGB).</param>
    /// <param name="ops">Numeric operations provider.</param>
    public DDPMModel(
        IUNet<T> unet,
        INoiseScheduler<T> scheduler,
        int imageSize,
        int channels,
        INumericOperations<T> ops)
    {
        Guard.NotNull(unet, nameof(unet));
        Guard.NotNull(scheduler, nameof(scheduler));
        Guard.Positive(imageSize, nameof(imageSize));
        Guard.Positive(channels, nameof(channels));
        Guard.NotNull(ops, nameof(ops));

        _unet = unet;
        _scheduler = scheduler;
        _imageSize = imageSize;
        _channels = channels;
        _ops = ops;
    }

    /// <inheritdoc/>
    public INoiseScheduler<T> Scheduler => _scheduler;

    /// <inheritdoc/>
    public int NumTimesteps => _scheduler.NumTimesteps;

    /// <inheritdoc/>
    public Tensor<T> PredictNoise(Tensor<T> xt, int t, Tensor<T>? condition = null)
    {
        Guard.NotNull(xt, nameof(xt));
        Guard.InRange(t, 0, NumTimesteps - 1, nameof(t));

        // Get batch size from input
        int batch = xt.Shape[0];

        // Create timestep array (same timestep for all samples in batch)
        var timesteps = new int[batch];
        for (int i = 0; i < batch; i++)
        {
            timesteps[i] = t;
        }

        // Predict noise using U-Net
        return _unet.Forward(xt, timesteps, condition, _ops);
    }

    /// <inheritdoc/>
    public Tensor<T> DenoisingStep(Tensor<T> xt, int t, Tensor<T>? condition = null)
    {
        Guard.NotNull(xt, nameof(xt));
        Guard.InRange(t, 0, NumTimesteps - 1, nameof(t));

        // Predict noise
        var predictedNoise = PredictNoise(xt, t, condition);

        // Remove noise using scheduler
        return _scheduler.RemoveNoise(xt, predictedNoise, t, _ops);
    }

    /// <inheritdoc/>
    public Tensor<T> Sample(int[] shape, Tensor<T>? condition = null, int? numInferenceSteps = null)
    {
        Guard.NotNull(shape, nameof(shape));

        if (shape.Length != 4)
        {
            throw new ArgumentException(
                $"Shape must be 4D [batch, channels, height, width], got: [{string.Join(", ", shape)}]",
                nameof(shape));
        }

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        if (channels != _channels || height != _imageSize || width != _imageSize)
        {
            throw new ArgumentException(
                $"Expected shape [*, {_channels}, {_imageSize}, {_imageSize}], got: [{string.Join(", ", shape)}]",
                nameof(shape));
        }

        // Start with random Gaussian noise
        var xt = SampleGaussianNoise(shape);

        // Determine number of steps
        int steps = numInferenceSteps ?? NumTimesteps;

        // Denoising loop
        for (int t = NumTimesteps - 1; t >= 0; t--)
        {
            xt = DenoisingStep(xt, t, condition);

            // Optional: Log progress
            if (t % 100 == 0)
            {
                Console.WriteLine($"Denoising step {NumTimesteps - t}/{NumTimesteps}");
            }
        }

        return xt;
    }

    /// <inheritdoc/>
    public T TrainStep(Tensor<T> images, Tensor<T>? condition = null)
    {
        Guard.NotNull(images, nameof(images));

        var shape = images.Shape;
        if (shape.Length != 4)
        {
            throw new ArgumentException(
                $"Images must be 4D [batch, channels, height, width], got: [{string.Join(", ", shape)}]",
                nameof(images));
        }

        int batch = shape[0];

        // Sample random timesteps for each image in the batch
        var random = new Random();
        var timesteps = new int[batch];
        for (int i = 0; i < batch; i++)
        {
            timesteps[i] = random.Next(NumTimesteps);
        }

        // Sample Gaussian noise
        var noise = SampleGaussianNoise(shape);

        // Add noise to images
        var noisyImages = new Tensor<T>(shape);
        for (int b = 0; b < batch; b++)
        {
            int t = timesteps[b];

            // Extract single image and noise
            var image = ExtractSample(images, b);
            var noiseForImage = ExtractSample(noise, b);

            // Add noise at timestep t
            var noisyImage = _scheduler.AddNoise(image, noiseForImage, t, _ops);

            // Copy back to batch
            CopySampleToBatch(noisyImage, noisyImages, b);
        }

        // Predict noise
        var predictedNoise = _unet.Forward(noisyImages, timesteps, condition, _ops);

        // Compute mean squared error loss
        var loss = ComputeMSELoss(noise, predictedNoise);

        // TODO: Backpropagate and update weights
        // This requires implementing:
        // 1. Gradient computation through U-Net
        // 2. Optimizer (Adam/AdamW)
        // 3. Parameter updates

        return loss;
    }

    private Tensor<T> SampleGaussianNoise(int[] shape)
    {
        var noise = new Tensor<T>(shape);
        var random = new Random();

        for (int i = 0; i < noise.Data.Length; i++)
        {
            // Box-Muller transform for Gaussian samples
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            noise.Data[i] = _ops.FromDouble(z);
        }

        return noise;
    }

    private Tensor<T> ExtractSample(Tensor<T> batch, int index)
    {
        // Extract sample at index from [batch, channels, height, width]
        int channels = batch.Shape[1];
        int height = batch.Shape[2];
        int width = batch.Shape[3];

        var sample = new Tensor<T>(new[] { 1, channels, height, width });

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    sample[0, c, h, w] = batch[index, c, h, w];
                }
            }
        }

        return sample;
    }

    private void CopySampleToBatch(Tensor<T> sample, Tensor<T> batch, int index)
    {
        int channels = sample.Shape[1];
        int height = sample.Shape[2];
        int width = sample.Shape[3];

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    batch[index, c, h, w] = sample[0, c, h, w];
                }
            }
        }
    }

    private T ComputeMSELoss(Tensor<T> target, Tensor<T> prediction)
    {
        // Mean Squared Error: (1/N) * sum((target - prediction)²)
        T sumSquaredError = _ops.Zero;
        int count = target.Data.Length;

        for (int i = 0; i < count; i++)
        {
            var diff = _ops.Subtract(target.Data[i], prediction.Data[i]);
            var squared = _ops.Square(diff);
            sumSquaredError = _ops.Add(sumSquaredError, squared);
        }

        return _ops.Divide(sumSquaredError, _ops.FromDouble(count));
    }

    /// <inheritdoc/>
    public void Save(string path)
    {
        Guard.NotNullOrEmpty(path, nameof(path));

        // TODO: Implement model serialization
        // Save U-Net weights, scheduler configuration, etc.
    }

    /// <inheritdoc/>
    public void Load(string path)
    {
        Guard.NotNullOrEmpty(path, nameof(path));

        // TODO: Implement model deserialization
    }

    /// <inheritdoc/>
    public Tensor<T> Generate(int numSamples)
    {
        var shape = new[] { numSamples, _channels, _imageSize, _imageSize };
        return Sample(shape);
    }
}
```

---

## Testing Strategy

### Unit Tests

```csharp
namespace AiDotNetTests.UnitTests.Diffusion;

using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Mathematics;
using Xunit;

public class NoiseSchedulerTests
{
    [Fact]
    public void LinearSchedule_ComputesCorrectBetas()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var scheduler = new LinearSchedule<double>(1000, 0.0001, 0.02, ops);

        // Act
        var betaStart = scheduler.GetBeta(0);
        var betaEnd = scheduler.GetBeta(999);

        // Assert
        Assert.True(Math.Abs(betaStart - 0.0001) < 0.00001);
        Assert.True(Math.Abs(betaEnd - 0.02) < 0.00001);
    }

    [Fact]
    public void AddNoise_PreservesShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var scheduler = new LinearSchedule<double>(1000, 0.0001, 0.02, ops);

        var x0 = new Tensor<double>(new[] { 2, 3, 32, 32 });
        var noise = new Tensor<double>(new[] { 2, 3, 32, 32 });

        // Act
        var xt = scheduler.AddNoise(x0, noise, 500, ops);

        // Assert
        Assert.Equal(new[] { 2, 3, 32, 32 }, xt.Shape);
    }

    [Fact]
    public void AddNoise_AtT0_ReturnsOriginal()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var scheduler = new LinearSchedule<double>(1000, 0.0001, 0.02, ops);

        var x0 = new Tensor<double>(new[] { 1, 3, 4, 4 });
        for (int i = 0; i < x0.Data.Length; i++)
        {
            x0.Data[i] = i + 1.0; // Fill with sequential values
        }

        var noise = new Tensor<double>(new[] { 1, 3, 4, 4 });
        // noise is all zeros

        // Act
        var xt = scheduler.AddNoise(x0, noise, 0, ops);

        // Assert - Should be very close to x0 (alpha_bar_0 ≈ 1)
        for (int i = 0; i < x0.Data.Length; i++)
        {
            Assert.True(Math.Abs(xt.Data[i] - x0.Data[i]) < 0.01);
        }
    }
}

public class TimeEmbeddingTests
{
    [Fact]
    public void Forward_ReturnsCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var timeEmbed = new TimeEmbedding<double>(256, 1024, 10000, ops);

        var timesteps = new[] { 0, 100, 500, 999 };

        // Act
        var embeddings = timeEmbed.Forward(timesteps, ops);

        // Assert
        Assert.Equal(new[] { 4, 256 }, embeddings.Shape);
    }

    [Fact]
    public void Forward_DifferentTimestepsProduceDifferentEmbeddings()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var timeEmbed = new TimeEmbedding<double>(256, 1024, 10000, ops);

        // Act
        var embed1 = timeEmbed.Forward(new[] { 0 }, ops);
        var embed2 = timeEmbed.Forward(new[] { 500 }, ops);

        // Assert - Embeddings should be different
        bool different = false;
        for (int i = 0; i < 256; i++)
        {
            if (Math.Abs(embed1[0, i] - embed2[0, i]) > 0.01)
            {
                different = true;
                break;
            }
        }
        Assert.True(different);
    }
}

public class DDPMModelTests
{
    [Fact]
    public void Sample_GeneratesCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var scheduler = new LinearSchedule<double>(100, 0.0001, 0.02, ops); // Small T for testing

        // Create mock U-Net (returns zeros for simplicity)
        var unet = new MockUNet<double>(3, 3, 64, 4);

        var model = new DDPMModel<double>(unet, scheduler, 32, 3, ops);

        // Act
        var samples = model.Sample(new[] { 2, 3, 32, 32 });

        // Assert
        Assert.Equal(new[] { 2, 3, 32, 32 }, samples.Shape);
    }

    [Fact]
    public void TrainStep_ComputesLoss()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var scheduler = new LinearSchedule<double>(1000, 0.0001, 0.02, ops);
        var unet = new MockUNet<double>(3, 3, 64, 4);
        var model = new DDPMModel<double>(unet, scheduler, 32, 3, ops);

        var images = new Tensor<double>(new[] { 4, 3, 32, 32 });

        // Act
        var loss = model.TrainStep(images);

        // Assert
        Assert.True(loss >= 0); // Loss should be non-negative
    }
}
```

---

## Training Strategy

### DDPM Training

```csharp
/// <summary>
/// Trains a DDPM model on an image dataset.
/// </summary>
/// <remarks>
/// <b>Training Hyperparameters</b> (from original paper):
///
/// 1. **Optimizer**: Adam with β₁=0.9, β₂=0.999
/// 2. **Learning Rate**: 2 × 10⁻⁴ (constant)
/// 3. **Batch Size**: 128
/// 4. **Training Steps**: 800K steps
/// 5. **EMA**: Exponential Moving Average with decay 0.9999
/// 6. **Image Size**: 32×32 (CIFAR-10) or 256×256 (CelebA-HQ)
///
/// <b>Data Augmentation</b>:
/// - Random horizontal flips
/// - No other augmentation (diffusion provides implicit regularization)
///
/// <b>Training Loop</b>:
/// ```
/// for each batch:
///     1. Sample images x₀ from dataset
///     2. Sample timesteps t uniformly
///     3. Sample noise ε ~ N(0, I)
///     4. Create noisy images x_t
///     5. Predict noise ε_θ(x_t, t)
///     6. Compute loss ||ε - ε_θ||²
///     7. Update model parameters
///     8. Update EMA parameters
/// ```
///
/// <b>Evaluation</b>:
/// - Generate samples every 10K steps
/// - Compute FID (Frechet Inception Distance)
/// - Visual inspection of sample quality
/// </remarks>
public class DDPMTrainer<T>
{
    // TODO: Implement full training loop with:
    // - Data loading and batching
    // - Adam optimizer
    // - EMA for stable generation
    // - Gradient clipping
    // - Checkpointing
    // - FID evaluation
    // - Tensorboard logging
}
```

### Expected Results

| Dataset | Image Size | FID Score | Training Time |
|---------|-----------|-----------|---------------|
| CIFAR-10 | 32×32 | 3.17 | ~5 days on 8 V100 GPUs |
| CelebA-HQ | 256×256 | 5.11 | ~14 days on 8 V100 GPUs |
| ImageNet | 256×256 | 7.72 | ~21 days on 8 V100 GPUs |

---

## Common Pitfalls

### Pitfall 1: Incorrect Noise Scaling
**Problem**: Images don't denoise properly or become artifacts.
**Solution**: Verify alpha_bar computation and noise scaling factors.

### Pitfall 2: Timestep Embedding Issues
**Problem**: Model can't distinguish between timesteps.
**Solution**: Ensure sinusoidal embeddings cover full frequency range.

### Pitfall 3: Numerical Instability
**Problem**: NaN or Inf values during training.
**Solution**: Use gradient clipping, check scheduler values, verify softmax stability.

### Pitfall 4: Too Few Sampling Steps
**Problem**: Generated images are blurry or noisy.
**Solution**: Use at least 250 steps for DDPM, or implement DDIM for faster sampling.

### Pitfall 5: Poor U-Net Architecture
**Problem**: Model can't capture fine details.
**Solution**: Use attention blocks at lower resolutions, increase model capacity.

---

## Performance Benchmarks

### Computational Requirements

| Component | Parameters | Memory | Training Time (1M images) |
|-----------|-----------|--------|---------------------------|
| U-Net (Small) | 35M | ~4 GB | ~3 days (8 GPUs) |
| U-Net (Base) | 100M | ~12 GB | ~7 days (8 GPUs) |
| U-Net (Large) | 400M | ~40 GB | ~14 days (8 GPUs) |

### Generation Speed

| Method | Steps | Time per Image (256×256) |
|--------|-------|--------------------------|
| DDPM | 1000 | ~10 seconds |
| DDIM | 50 | ~0.5 seconds |
| DDIM | 250 | ~2.5 seconds |

---

## Next Steps

1. **Implement DDIM Sampling**: Much faster inference
2. **Add Conditioning**:
   - Class-conditional generation
   - Text-to-image (CLIP guidance)
3. **Latent Diffusion**:
   - VAE encoder/decoder
   - Work in compressed latent space
4. **Advanced Features**:
   - Classifier-free guidance
   - Inpainting and editing
   - Super-resolution

---

## Resources

- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Diffusion Models Tutorial](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
