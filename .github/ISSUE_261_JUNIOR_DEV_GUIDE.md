# Junior Developer Implementation Guide: Issue #261
## Introduce Diffusion Models (Core) – Part 1

### Overview
This guide will walk you through implementing the foundational diffusion model infrastructure for AiDotNet. Diffusion models are a powerful class of generative models that create images by learning to reverse a noise-adding process.

---

## Understanding Diffusion Models

### What Are Diffusion Models?

Think of diffusion models like a "reverse time machine" for noise:

1. **Forward Process (Adding Noise)**: Imagine taking a clear photo and gradually adding more and more static/noise over many steps until the photo becomes pure random noise. This is deterministic and mathematical.

2. **Reverse Process (Removing Noise)**: Now imagine training a model to reverse this process - starting from pure noise and gradually removing it step by step to recover a clear image.

**Real-World Analogy**:
- Forward: Like slowly dissolving a sugar cube in water over 1000 steps
- Reverse: Training an AI to magically reconstruct the sugar cube from the dissolved solution, step by step

### Key Concepts

#### 1. Timesteps (t)
- Diffusion happens over many discrete steps (often 1000)
- t=0 is the original clean image
- t=1000 is pure noise
- Each timestep has a specific noise level

#### 2. Noise Schedule (Beta Schedule)
- Controls how much noise is added at each timestep
- Linear schedule: noise increases uniformly
- Cosine schedule: more gradual at start and end
- Defined by parameters: beta_start (e.g., 0.0001) and beta_end (e.g., 0.02)

#### 3. U-Net Architecture
- The neural network that predicts noise at each timestep
- Takes noisy image + timestep → predicts the noise that was added
- "U-shaped" architecture: encoder → bottleneck → decoder
- Skip connections preserve fine details

#### 4. Denoising Process
```
For timestep t from T down to 1:
  1. Input: noisy image at time t
  2. U-Net predicts: noise that was added
  3. Scheduler computes: slightly less noisy image at time t-1
  4. Repeat until clean image emerges
```

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── IDiffusionModel.cs          # Main diffusion model interface
│   └── IStepScheduler.cs           # Scheduler interface for denoising steps
├── Models/
│   └── Generative/
│       └── Diffusion/
│           ├── DiffusionModelBase.cs    # Base class with shared logic
│           └── DDPMModel.cs             # Concrete DDPM implementation
└── Diffusion/
    └── Schedulers/
        ├── SchedulerConfig.cs      # Configuration for schedulers
        ├── BetaSchedule.cs         # Enum: Linear, Cosine, etc.
        ├── PredictionType.cs       # Enum: Epsilon, V, Sample
        ├── StepSchedulerBase.cs    # Base scheduler implementation
        ├── DDIMScheduler.cs        # Fast deterministic sampler
        └── DDPMScheduler.cs        # Original DDPM sampler
```

### Inheritance Pattern
```
IDiffusionModel<T> (interface in src/Interfaces/)
    ↓ implements IFullModel<T, Tensor<T>, Tensor<T>>
    ↓
DiffusionModelBase<T> (abstract base in src/Models/Generative/Diffusion/)
    ↓
DDPMModel<T> (concrete implementation)


IStepScheduler<T> (interface in src/Interfaces/)
    ↓
StepSchedulerBase<T> (abstract base in src/Diffusion/Schedulers/)
    ↓
DDIMScheduler<T> / DDPMScheduler<T> (concrete implementations)
```

---

## Step-by-Step Implementation

### Step 1: Define Core Enums and Configuration

#### File: `src/Diffusion/Schedulers/BetaSchedule.cs`
```csharp
namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Defines the noise schedule for diffusion models.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> The beta schedule controls how noise is added over time.
///
/// Think of it like turning up static on a TV:
/// - Linear: static increases steadily from quiet to loud
/// - Cosine: static increases smoothly, avoiding sudden jumps
/// - Quadratic: static accelerates as it gets louder
///
/// Most modern models use Cosine because it preserves image quality better.
/// </remarks>
public enum BetaSchedule
{
    /// <summary>
    /// Linear noise schedule - noise increases uniformly.
    /// Used in original DDPM paper. Simple but can be too aggressive.
    /// </summary>
    Linear,

    /// <summary>
    /// Cosine noise schedule - smooth noise increase following cosine curve.
    /// Recommended default. Provides better sample quality. From "Improved DDPM" paper.
    /// </summary>
    Cosine,

    /// <summary>
    /// Quadratic noise schedule - noise increases quadratically.
    /// Faster noise increase, fewer steps needed.
    /// </summary>
    Quadratic
}
```

#### File: `src/Diffusion/Schedulers/PredictionType.cs`
```csharp
namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Defines what the model predicts at each denoising step.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> The prediction type determines what the U-Net is trained to output.
///
/// Think of it like different strategies for cleaning a dirty window:
/// - Epsilon: Predict what dirt is on the window (most common)
/// - Sample: Predict what the clean window looks like directly
/// - VPrediction: Predict a "velocity" toward the clean window (advanced)
///
/// Epsilon (noise prediction) is the standard approach used by Stable Diffusion.
/// </remarks>
public enum PredictionType
{
    /// <summary>
    /// Predict the noise (epsilon) that was added.
    /// This is the standard approach used in DDPM, DDIM, and Stable Diffusion.
    /// </summary>
    Epsilon,

    /// <summary>
    /// Predict the original clean sample directly.
    /// Can be faster but less stable during training.
    /// </summary>
    Sample,

    /// <summary>
    /// Predict the "velocity" in the probability flow ODE.
    /// Advanced technique from "Progressive Distillation" paper.
    /// </summary>
    VPrediction
}
```

#### File: `src/Diffusion/Schedulers/SchedulerConfig.cs`
```csharp
namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Configuration for diffusion schedulers.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This holds all the settings that control how noise is added and removed.
///
/// Key settings:
/// - NumTrainTimesteps: How many steps in the full diffusion process (usually 1000)
/// - BetaStart/BetaEnd: The noise range (typical: 0.0001 to 0.02)
/// - BetaSchedule: How noise increases (Linear vs Cosine)
/// - PredictionType: What the model predicts (usually Epsilon/noise)
///
/// These defaults come from the original DDPM paper and work well for most cases.
/// </remarks>
public class SchedulerConfig<T>
{
    /// <summary>
    /// Number of diffusion steps during training.
    /// Default: 1000 (from DDPM paper)
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> More steps = smoother noise addition but slower generation.
    /// Think of it like the number of frames in a video from clear image to noise.
    /// </remarks>
    public int NumTrainTimesteps { get; set; } = 1000;

    /// <summary>
    /// Starting noise level (minimum beta).
    /// Default: 0.0001 (from DDPM paper)
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The smallest amount of noise added.
    /// Too high = image degrades too quickly.
    /// </remarks>
    public T BetaStart { get; set; } = default(T); // Will use NumOps.FromDouble(0.0001)

    /// <summary>
    /// Ending noise level (maximum beta).
    /// Default: 0.02 (from DDPM paper)
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The maximum amount of noise added.
    /// Too low = not enough noise variety; too high = information lost too quickly.
    /// </remarks>
    public T BetaEnd { get; set; } = default(T); // Will use NumOps.FromDouble(0.02)

    /// <summary>
    /// The noise schedule type.
    /// Default: Cosine (recommended)
    /// </summary>
    public BetaSchedule Schedule { get; set; } = BetaSchedule.Cosine;

    /// <summary>
    /// What the model predicts.
    /// Default: Epsilon (standard for most models)
    /// </summary>
    public PredictionType PredictionType { get; set; } = PredictionType.Epsilon;
}
```

---

### Step 2: Create Scheduler Interface

#### File: `src/Interfaces/IStepScheduler.cs`
```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for diffusion model schedulers that control the denoising process.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A scheduler controls how noise is removed step-by-step.
///
/// Think of it like a recipe for cleaning:
/// - You have a dirty dish (noisy image)
/// - The recipe tells you exactly how much soap, water, and scrubbing at each step
/// - Following the recipe, you gradually get a clean dish (clear image)
///
/// Different schedulers = different cleaning strategies:
/// - DDPM: Thorough but slow (50-1000 steps)
/// - DDIM: Faster with similar quality (20-50 steps)
/// - DPM-Solver: Very fast (10-20 steps)
/// </remarks>
public interface IStepScheduler<T>
{
    /// <summary>
    /// Performs one denoising step.
    /// </summary>
    /// <param name="modelOutput">The noise predicted by the model at this timestep.</param>
    /// <param name="timestep">Current timestep (t) in the denoising process.</param>
    /// <param name="sample">The noisy sample at timestep t.</param>
    /// <returns>The denoised sample at timestep t-1.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the core denoising operation.
    ///
    /// Process:
    /// 1. Model predicts noise in current sample
    /// 2. Step() uses that prediction to compute a less noisy sample
    /// 3. Repeat for each timestep until image is clean
    ///
    /// Example:
    /// - timestep = 500: very noisy image
    /// - modelOutput: predicted noise
    /// - returns: slightly less noisy image at timestep 499
    /// </remarks>
    Tensor<T> Step(Tensor<T> modelOutput, int timestep, Tensor<T> sample);

    /// <summary>
    /// Adds noise to a clean sample for training.
    /// </summary>
    /// <param name="originalSample">The clean image.</param>
    /// <param name="noise">Random noise to add.</param>
    /// <param name="timestep">Which timestep's noise level to use.</param>
    /// <returns>Noisy sample at the specified timestep.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is used during training to create noisy examples.
    ///
    /// Training process:
    /// 1. Start with clean image
    /// 2. AddNoise() corrupts it to a specific timestep
    /// 3. Model learns to predict the noise that was added
    /// 4. Repeat with different images and timesteps
    /// </remarks>
    Tensor<T> AddNoise(Tensor<T> originalSample, Tensor<T> noise, int timestep);

    /// <summary>
    /// Gets the timesteps to use during inference.
    /// </summary>
    /// <param name="numInferenceSteps">How many steps to use (fewer = faster).</param>
    /// <returns>Array of timestep values to iterate through.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This determines the denoising schedule.
    ///
    /// Example: If trained on 1000 steps but you want to generate in 50 steps:
    /// - Returns: [1000, 980, 960, ..., 40, 20, 0]
    /// - You skip most timesteps for speed
    /// - DDIM and DPM-Solver allow this; DDPM doesn't work well with skipping
    /// </remarks>
    int[] GetTimesteps(int numInferenceSteps);

    /// <summary>
    /// Initializes noise for generation.
    /// </summary>
    /// <param name="shape">Shape of the tensor [batch, channels, height, width].</param>
    /// <returns>Random noise tensor.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Creates the starting point for generation.
    ///
    /// Think of it like a blank canvas covered in random static.
    /// From this randomness, the model will sculpt a coherent image.
    /// </remarks>
    Tensor<T> InitNoise(int[] shape);
}
```

---

### Step 3: Implement Base Scheduler

#### File: `src/Diffusion/Schedulers/StepSchedulerBase.cs`
```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Base implementation for diffusion schedulers.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This base class handles the math common to all schedulers.
///
/// Key responsibilities:
/// - Compute noise schedules (betas, alphas)
/// - Provide helper methods for subclasses
/// - Handle numeric operations generically
/// </remarks>
public abstract class StepSchedulerBase<T> : IStepScheduler<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected readonly SchedulerConfig<T> Config;

    // Precomputed noise schedule values
    protected Vector<T> Betas;        // Noise at each timestep
    protected Vector<T> Alphas;       // (1 - beta)
    protected Vector<T> AlphasCumprod; // Cumulative product of alphas

    /// <summary>
    /// Initializes the scheduler with configuration.
    /// </summary>
    /// <param name="config">Scheduler configuration.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up the noise schedule.
    ///
    /// This precomputes values used throughout denoising:
    /// - betas: noise amounts
    /// - alphas: signal preservation (1 - beta)
    /// - alphas_cumprod: cumulative signal decay
    ///
    /// Precomputing saves time during generation.
    /// </remarks>
    protected StepSchedulerBase(SchedulerConfig<T> config)
    {
        Config = config ?? throw new ArgumentNullException(nameof(config));

        // Validate configuration
        if (Config.NumTrainTimesteps <= 0)
            throw new ArgumentException("NumTrainTimesteps must be positive", nameof(config));

        // Initialize noise schedule
        InitializeNoiseSchedule();
    }

    /// <summary>
    /// Initializes the noise schedule based on configuration.
    /// </summary>
    private void InitializeNoiseSchedule()
    {
        int numSteps = Config.NumTrainTimesteps;
        Betas = new Vector<T>(numSteps);

        // Get beta start/end values
        T betaStart = NumOps.Equals(Config.BetaStart, default(T))
            ? NumOps.FromDouble(0.0001)
            : Config.BetaStart;
        T betaEnd = NumOps.Equals(Config.BetaEnd, default(T))
            ? NumOps.FromDouble(0.02)
            : Config.BetaEnd;

        // Compute beta schedule
        switch (Config.Schedule)
        {
            case BetaSchedule.Linear:
                // Linear interpolation from beta_start to beta_end
                for (int i = 0; i < numSteps; i++)
                {
                    double t = (double)i / (numSteps - 1);
                    T beta = NumOps.Add(
                        betaStart,
                        NumOps.Multiply(
                            NumOps.Subtract(betaEnd, betaStart),
                            NumOps.FromDouble(t)
                        )
                    );
                    Betas[i] = beta;
                }
                break;

            case BetaSchedule.Cosine:
                // Cosine schedule from "Improved DDPM" paper
                double s = 0.008; // offset to prevent beta from being too small
                for (int i = 0; i < numSteps; i++)
                {
                    double t = (double)i / numSteps;
                    double alphaBar = Math.Cos((t + s) / (1 + s) * Math.PI / 2);
                    alphaBar = alphaBar * alphaBar;

                    double alphaBarPrev = i > 0
                        ? Math.Cos(((double)(i - 1) / numSteps + s) / (1 + s) * Math.PI / 2)
                        : 1.0;
                    alphaBarPrev = alphaBarPrev * alphaBarPrev;

                    double beta = 1.0 - (alphaBar / alphaBarPrev);
                    beta = Math.Max(0.0, Math.Min(0.999, beta)); // Clamp
                    Betas[i] = NumOps.FromDouble(beta);
                }
                break;

            case BetaSchedule.Quadratic:
                // Quadratic schedule
                T sqrtBetaStart = NumOps.Sqrt(betaStart);
                T sqrtBetaEnd = NumOps.Sqrt(betaEnd);
                for (int i = 0; i < numSteps; i++)
                {
                    double t = (double)i / (numSteps - 1);
                    T sqrtBeta = NumOps.Add(
                        sqrtBetaStart,
                        NumOps.Multiply(
                            NumOps.Subtract(sqrtBetaEnd, sqrtBetaStart),
                            NumOps.FromDouble(t)
                        )
                    );
                    Betas[i] = NumOps.Multiply(sqrtBeta, sqrtBeta);
                }
                break;
        }

        // Compute alphas and cumulative product
        Alphas = new Vector<T>(numSteps);
        AlphasCumprod = new Vector<T>(numSteps);

        T cumprod = NumOps.One;
        for (int i = 0; i < numSteps; i++)
        {
            Alphas[i] = NumOps.Subtract(NumOps.One, Betas[i]);
            cumprod = NumOps.Multiply(cumprod, Alphas[i]);
            AlphasCumprod[i] = cumprod;
        }
    }

    /// <summary>
    /// Performs one denoising step (implemented by subclasses).
    /// </summary>
    public abstract Tensor<T> Step(Tensor<T> modelOutput, int timestep, Tensor<T> sample);

    /// <summary>
    /// Adds noise to a sample for training.
    /// </summary>
    public virtual Tensor<T> AddNoise(Tensor<T> originalSample, Tensor<T> noise, int timestep)
    {
        if (timestep < 0 || timestep >= Config.NumTrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(timestep));

        // Formula: noisy = sqrt(alpha_cumprod) * original + sqrt(1 - alpha_cumprod) * noise
        T alphaCumprod = AlphasCumprod[timestep];
        T sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        T sqrtOneMinusAlphaCumprod = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));

        // Scale original sample
        Tensor<T> scaledOriginal = originalSample.Multiply(sqrtAlphaCumprod);

        // Scale noise
        Tensor<T> scaledNoise = noise.Multiply(sqrtOneMinusAlphaCumprod);

        // Combine
        return scaledOriginal.Add(scaledNoise);
    }

    /// <summary>
    /// Gets timesteps for inference (default: evenly spaced).
    /// </summary>
    public virtual int[] GetTimesteps(int numInferenceSteps)
    {
        if (numInferenceSteps <= 0)
            throw new ArgumentException("numInferenceSteps must be positive", nameof(numInferenceSteps));

        // Evenly spaced timesteps from numTrainTimesteps down to 0
        int[] timesteps = new int[numInferenceSteps];
        int step = Config.NumTrainTimesteps / numInferenceSteps;

        for (int i = 0; i < numInferenceSteps; i++)
        {
            timesteps[i] = Config.NumTrainTimesteps - 1 - (i * step);
        }

        return timesteps;
    }

    /// <summary>
    /// Initializes random noise tensor.
    /// </summary>
    public virtual Tensor<T> InitNoise(int[] shape)
    {
        // Generate standard normal random noise
        Random random = new Random();
        Tensor<T> noise = new Tensor<T>(shape);

        for (int i = 0; i < noise.Length; i++)
        {
            // Box-Muller transform for normal distribution
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            noise[i] = NumOps.FromDouble(z);
        }

        return noise;
    }
}
```

---

### Step 4: Implement DDIM Scheduler

#### File: `src/Diffusion/Schedulers/DDIMScheduler.cs`
```csharp
namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Denoising Diffusion Implicit Models (DDIM) scheduler.
/// Fast deterministic sampling with fewer steps than DDPM.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> DDIM is a faster alternative to DDPM.
///
/// Key differences from DDPM:
/// - Deterministic: same seed → same image (no randomness during sampling)
/// - Fast: 20-50 steps vs 1000 steps for DDPM
/// - Quality: similar to DDPM despite fewer steps
///
/// How it works:
/// - Instead of adding small noise at each step (DDPM)
/// - DDIM takes bigger "jumps" directly toward the clean image
/// - Uses a parameter eta to control stochasticity (eta=0 is fully deterministic)
///
/// Use cases:
/// - Production image generation (need speed)
/// - When reproducibility matters (same seed = same output)
/// - Interactive applications (need <5 second generation)
///
/// From paper: "Denoising Diffusion Implicit Models" (Song et al., 2021)
/// </remarks>
public class DDIMScheduler<T> : StepSchedulerBase<T>
{
    private readonly T _eta;

    /// <summary>
    /// Initializes DDIM scheduler.
    /// </summary>
    /// <param name="config">Scheduler configuration.</param>
    /// <param name="eta">
    /// Stochasticity parameter (0 = deterministic, 1 = stochastic like DDPM).
    /// Default: 0.0 (fully deterministic).
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> Eta controls randomness vs determinism.
    ///
    /// - eta = 0.0: Fully deterministic (recommended)
    ///   * Same starting noise → identical result
    ///   * Faster, cleaner trajectories
    ///   * Best for most use cases
    ///
    /// - eta = 1.0: Stochastic (like DDPM)
    ///   * Adds randomness at each step
    ///   * More variety in outputs
    ///   * Slower convergence
    ///
    /// - eta = 0.5: Hybrid
    ///   * Some randomness for variety
    ///   * Still relatively predictable
    /// </remarks>
    public DDIMScheduler(SchedulerConfig<T> config, T eta = default(T))
        : base(config)
    {
        _eta = NumOps.Equals(eta, default(T)) ? NumOps.Zero : eta;

        // Validate eta is in [0, 1]
        if (NumOps.LessThan(_eta, NumOps.Zero) || NumOps.GreaterThan(_eta, NumOps.One))
            throw new ArgumentException("eta must be in [0, 1]", nameof(eta));
    }

    /// <summary>
    /// Performs one DDIM denoising step.
    /// </summary>
    public override Tensor<T> Step(Tensor<T> modelOutput, int timestep, Tensor<T> sample)
    {
        if (timestep < 0 || timestep >= Config.NumTrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(timestep));

        // Get alpha values
        T alphaCumprod = AlphasCumprod[timestep];
        T alphaCumprodPrev = timestep > 0 ? AlphasCumprod[timestep - 1] : NumOps.One;

        T sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        T sqrtOneMinusAlphaCumprod = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));

        // Predict original sample from noise (x0 prediction)
        Tensor<T> predOriginalSample;

        if (Config.PredictionType == PredictionType.Epsilon)
        {
            // modelOutput is predicted noise
            // Formula: x0 = (sample - sqrt(1-alpha) * noise) / sqrt(alpha)
            Tensor<T> noisePart = modelOutput.Multiply(sqrtOneMinusAlphaCumprod);
            predOriginalSample = sample.Subtract(noisePart).Divide(sqrtAlphaCumprod);
        }
        else if (Config.PredictionType == PredictionType.Sample)
        {
            // modelOutput is predicted x0 directly
            predOriginalSample = modelOutput;
        }
        else
        {
            throw new NotImplementedException($"PredictionType {Config.PredictionType} not implemented");
        }

        // Compute "direction pointing to x_t"
        T sqrtOneMinusAlphaCumprodPrev = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprodPrev));
        T sqrtAlphaCumprodPrev = NumOps.Sqrt(alphaCumprodPrev);

        // DDIM formula: x_{t-1} = sqrt(alpha_{t-1}) * x0 + sqrt(1-alpha_{t-1}) * epsilon
        Tensor<T> predSample = predOriginalSample.Multiply(sqrtAlphaCumprodPrev);
        Tensor<T> noiseTerm = modelOutput.Multiply(sqrtOneMinusAlphaCumprodPrev);

        // Add eta-scaled noise if eta > 0 (stochastic)
        if (NumOps.GreaterThan(_eta, NumOps.Zero))
        {
            // Compute variance
            T variance = NumOps.Multiply(
                _eta,
                NumOps.Sqrt(
                    NumOps.Divide(
                        NumOps.Subtract(NumOps.One, alphaCumprodPrev),
                        NumOps.Subtract(NumOps.One, alphaCumprod)
                    )
                )
            );
            variance = NumOps.Multiply(variance, NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod)));

            // Sample random noise
            Tensor<T> noise = InitNoise(sample.Shape);
            noiseTerm = noiseTerm.Add(noise.Multiply(variance));
        }

        return predSample.Add(noiseTerm);
    }
}
```

---

### Step 5: Create Diffusion Model Interface

#### File: `src/Interfaces/IDiffusionModel.cs`
```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for diffusion generative models.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A diffusion model generates images from noise.
///
/// Think of it like a sculptor:
/// - Start with a block of marble (random noise)
/// - Chip away gradually following a plan (denoising steps)
/// - End with a beautiful statue (generated image)
///
/// The model learns:
/// 1. How to add noise to images (forward process)
/// 2. How to remove noise step by step (reverse process)
/// 3. What patterns to create from noise (trained on dataset)
///
/// Applications:
/// - Text-to-image (Stable Diffusion, DALL-E)
/// - Image editing and inpainting
/// - Super-resolution
/// - Video generation
/// </remarks>
public interface IDiffusionModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Generates images from random noise.
    /// </summary>
    /// <param name="numSamples">Number of images to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps (fewer = faster, lower quality).</param>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    /// <returns>Generated images as tensor.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main generation method.
    ///
    /// Process:
    /// 1. Create random noise tensor(s)
    /// 2. Run denoising loop for numInferenceSteps
    /// 3. Return generated image(s)
    ///
    /// Parameters:
    /// - numSamples: How many images (1 for single, 4 for batch)
    /// - numInferenceSteps: Quality vs speed tradeoff
    ///   * 20 steps: Fast, decent quality
    ///   * 50 steps: Balanced (recommended)
    ///   * 100+ steps: Slow, best quality
    /// - seed: Set for reproducible results
    ///
    /// Example:
    /// var images = model.Generate(numSamples: 4, numInferenceSteps: 50, seed: 42);
    /// // Returns 4 images, each taking ~50 denoising steps
    /// </remarks>
    Tensor<T> Generate(int numSamples, int numInferenceSteps, int? seed = null);

    /// <summary>
    /// Gets the scheduler used for denoising.
    /// </summary>
    IStepScheduler<T> Scheduler { get; }
}
```

---

### Step 6: Implement DDPM Model

#### File: `src/Models/Generative/Diffusion/DDPMModel.cs`
```csharp
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;

namespace AiDotNet.Models.Generative.Diffusion;

/// <summary>
/// Denoising Diffusion Probabilistic Model (DDPM).
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> DDPM is the foundational diffusion model.
///
/// This is the original diffusion model from "Denoising Diffusion Probabilistic Models"
/// (Ho et al., 2020). While newer models like DDIM are faster, DDPM is important to understand.
///
/// How it works:
/// 1. Training: Learn to predict noise added to images at random timesteps
/// 2. Generation: Start with noise, predict and remove noise step by step
/// 3. Result: Clean generated image emerges from noise
///
/// Architecture:
/// - U-Net neural network predicts noise
/// - Scheduler controls denoising trajectory
/// - Timestep conditioning tells model "how noisy" the input is
///
/// Key insight: By learning to denoise at ALL noise levels, the model learns
/// the entire data distribution and can generate novel samples.
/// </remarks>
public class DDPMModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IStepScheduler<T> _scheduler;
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _imageChannels;

    // The U-Net would go here in a full implementation
    // For this guide, we'll use a placeholder
    private readonly INeuralNetwork<T>? _unet;

    /// <summary>
    /// Gets the scheduler used for denoising.
    /// </summary>
    public IStepScheduler<T> Scheduler => _scheduler;

    /// <summary>
    /// Initializes a new DDPM model.
    /// </summary>
    /// <param name="scheduler">The scheduler for controlling denoising steps.</param>
    /// <param name="imageHeight">Height of generated images. Default: 256</param>
    /// <param name="imageWidth">Width of generated images. Default: 256</param>
    /// <param name="imageChannels">Number of channels (3 for RGB, 1 for grayscale). Default: 3</param>
    /// <param name="unet">Optional pre-trained U-Net. If null, must call Train() first.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up the diffusion model.
    ///
    /// The scheduler is REQUIRED because it controls how noise is added/removed.
    /// The U-Net is the neural network that predicts noise.
    ///
    /// Example:
    /// var config = new SchedulerConfig<double> { NumTrainTimesteps = 1000 };
    /// var scheduler = new DDIMScheduler<double>(config);
    /// var model = new DDPMModel<double>(scheduler, imageHeight: 64, imageWidth: 64);
    /// </remarks>
    public DDPMModel(
        IStepScheduler<T> scheduler,
        int imageHeight = 256,
        int imageWidth = 256,
        int imageChannels = 3,
        INeuralNetwork<T>? unet = null)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));

        if (imageHeight <= 0) throw new ArgumentException("Image height must be positive", nameof(imageHeight));
        if (imageWidth <= 0) throw new ArgumentException("Image width must be positive", nameof(imageWidth));
        if (imageChannels <= 0) throw new ArgumentException("Image channels must be positive", nameof(imageChannels));

        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _imageChannels = imageChannels;
        _unet = unet;
    }

    /// <summary>
    /// Generates images from random noise.
    /// </summary>
    public Tensor<T> Generate(int numSamples, int numInferenceSteps, int? seed = null)
    {
        if (_unet == null)
            throw new InvalidOperationException("Model not trained. Call Train() first or provide a trained U-Net.");

        if (numSamples <= 0)
            throw new ArgumentException("numSamples must be positive", nameof(numSamples));
        if (numInferenceSteps <= 0)
            throw new ArgumentException("numInferenceSteps must be positive", nameof(numInferenceSteps));

        // Set random seed if provided
        if (seed.HasValue)
            new Random(seed.Value);

        // Initialize random noise
        int[] shape = new[] { numSamples, _imageChannels, _imageHeight, _imageWidth };
        Tensor<T> sample = _scheduler.InitNoise(shape);

        // Get timesteps for inference
        int[] timesteps = _scheduler.GetTimesteps(numInferenceSteps);

        // Denoising loop
        for (int i = 0; i < timesteps.Length; i++)
        {
            int t = timesteps[i];

            // Create timestep tensor (broadcast to batch)
            Tensor<T> timestepTensor = new Tensor<T>(new[] { numSamples, 1 });
            for (int b = 0; b < numSamples; b++)
                timestepTensor[b, 0] = NumOps.FromDouble(t);

            // Predict noise with U-Net
            // In full implementation: modelOutput = _unet.Forward(sample, timestepTensor);
            // For now, placeholder:
            Tensor<T> modelOutput = sample; // Replace with actual U-Net call

            // Denoise one step
            sample = _scheduler.Step(modelOutput, t, sample);
        }

        return sample;
    }

    // IFullModel<T, Tensor<T>, Tensor<T>> implementation
    public Tensor<T> Predict(Tensor<T> input) => Generate(numSamples: 1, numInferenceSteps: 50);

    public void Train(Tensor<T> inputs, Tensor<T> outputs, int epochs = 100)
    {
        // Training implementation would go here
        // 1. For each epoch
        // 2.   For each batch
        // 3.     Sample random timesteps
        // 4.     Add noise to images using AddNoise()
        // 5.     Predict noise with U-Net
        // 6.     Compute loss (MSE between predicted and actual noise)
        // 7.     Backpropagate and update weights
        throw new NotImplementedException("Training will be implemented in future PR");
    }

    // Other IFullModel methods (serialization, etc.) omitted for brevity
    // Would be implemented in production code
}
```

---

## Unit Testing Strategy

### Test File: `tests/Diffusion/SchedulerTests.cs`

```csharp
using Xunit;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Tests.Diffusion;

public class DDIMSchedulerTests
{
    [Fact]
    public void Constructor_WithValidConfig_Initializes()
    {
        // Arrange
        var config = new SchedulerConfig<double>
        {
            NumTrainTimesteps = 1000,
            BetaStart = 0.0001,
            BetaEnd = 0.02,
            Schedule = BetaSchedule.Linear
        };

        // Act
        var scheduler = new DDIMScheduler<double>(config);

        // Assert
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void AddNoise_ProducesCorrectNoiseLevel()
    {
        // Arrange
        var config = new SchedulerConfig<double> { NumTrainTimesteps = 1000 };
        var scheduler = new DDIMScheduler<double>(config);

        var originalSample = new Tensor<double>(new[] { 1, 3, 4, 4 });
        var noise = new Tensor<double>(new[] { 1, 3, 4, 4 });

        // Fill with known values
        for (int i = 0; i < originalSample.Length; i++)
        {
            originalSample[i] = 1.0;
            noise[i] = 0.0; // No noise
        }

        // Act
        var noisySample = scheduler.AddNoise(originalSample, noise, timestep: 500);

        // Assert - at timestep 500, should be partially noised
        Assert.NotEqual(originalSample, noisySample);
    }

    [Fact]
    public void Step_DeterministicWithEtaZero()
    {
        // Arrange
        var config = new SchedulerConfig<double> { NumTrainTimesteps = 1000 };
        var scheduler = new DDIMScheduler<double>(config, eta: 0.0);

        var modelOutput = new Tensor<double>(new[] { 1, 3, 4, 4 });
        var sample = new Tensor<double>(new[] { 1, 3, 4, 4 });

        // Act - run twice with same inputs
        var result1 = scheduler.Step(modelOutput, timestep: 500, sample);
        var result2 = scheduler.Step(modelOutput, timestep: 500, sample);

        // Assert - should be identical (deterministic)
        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], precision: 10);
        }
    }

    [Theory]
    [InlineData(10)]
    [InlineData(50)]
    [InlineData(100)]
    public void GetTimesteps_ReturnsCorrectCount(int numSteps)
    {
        // Arrange
        var config = new SchedulerConfig<double> { NumTrainTimesteps = 1000 };
        var scheduler = new DDIMScheduler<double>(config);

        // Act
        var timesteps = scheduler.GetTimesteps(numSteps);

        // Assert
        Assert.Equal(numSteps, timesteps.Length);
        Assert.True(timesteps[0] > timesteps[timesteps.Length - 1]); // Descending
    }
}
```

---

## Integration with PredictionModelBuilder

### File: `src/PredictionModelBuilder.cs` (additions)

```csharp
// Add private field
private IStepScheduler<T>? _diffusionScheduler;

// Add configuration method
/// <summary>
/// Configures the scheduler for diffusion models.
/// </summary>
/// <param name="scheduler">The step scheduler to use.</param>
/// <returns>This builder instance for method chaining.</returns>
/// <remarks>
/// <b>For Beginners:</b> The scheduler controls how diffusion models remove noise.
///
/// Different schedulers offer different tradeoffs:
/// - DDIM: Fast and deterministic (20-50 steps)
/// - DDPM: Slower but thorough (100-1000 steps)
/// - DPM-Solver: Very fast (10-20 steps)
///
/// Example:
/// builder.ConfigureDiffusionScheduler(new DDIMScheduler<double>(config));
/// </remarks>
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureDiffusionScheduler(IStepScheduler<T> scheduler)
{
    _diffusionScheduler = scheduler;
    return this;
}

// In Build() method, use the scheduler when creating diffusion models
// If _diffusionScheduler is null, create a default DDIM scheduler
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Not Using INumericOperations<T>
**Wrong:**
```csharp
double beta = 0.0001; // Hardcoded double
```

**Correct:**
```csharp
T beta = NumOps.FromDouble(0.0001);
```

### Pitfall 2: Using default! Operator
**Wrong:**
```csharp
public T BetaStart { get; set; } = default!;
```

**Correct:**
```csharp
public T BetaStart { get; set; } = default(T); // Will be set in constructor
// OR
protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
public T BetaStart { get; set; } = NumOps.Zero;
```

### Pitfall 3: Incorrect Inheritance
**Wrong:**
```csharp
public class DDIMScheduler<T> : IStepScheduler<T> // Direct implementation
```

**Correct:**
```csharp
public class DDIMScheduler<T> : StepSchedulerBase<T> // Inherit from base
```

### Pitfall 4: Missing XML Documentation
**Wrong:**
```csharp
public Tensor<T> Step(Tensor<T> modelOutput, int timestep, Tensor<T> sample)
```

**Correct:**
```csharp
/// <summary>
/// Performs one denoising step.
/// </summary>
/// <param name="modelOutput">The noise predicted by the model.</param>
/// <param name="timestep">Current timestep in denoising process.</param>
/// <param name="sample">The noisy sample to denoise.</param>
/// <returns>Denoised sample at previous timestep.</returns>
/// <remarks>
/// <b>For Beginners:</b> This removes one step of noise from the sample.
/// </remarks>
public Tensor<T> Step(Tensor<T> modelOutput, int timestep, Tensor<T> sample)
```

---

## Testing Checklist

- [ ] SchedulerConfig validates parameters
- [ ] BetaSchedule enums work correctly
- [ ] StepSchedulerBase initializes noise schedules
- [ ] DDIMScheduler produces deterministic results when eta=0
- [ ] DDIMScheduler adds stochasticity when eta>0
- [ ] AddNoise correctly corrupts samples
- [ ] GetTimesteps returns correct count and descending order
- [ ] InitNoise produces standard normal distribution
- [ ] DDPMModel constructor validates parameters
- [ ] Generate() throws if model not trained
- [ ] Integration test: full generation pipeline
- [ ] Test with multiple numeric types (double, float)
- [ ] Code coverage >= 90%

---

## Next Steps

After completing this issue:

1. **Test thoroughly** - Ensure all unit tests pass
2. **Integration test** - Create end-to-end generation test
3. **Documentation** - Add XML docs to all public members
4. **Code review** - Follow project standards
5. **Move to Issue #262** - Implement latent diffusion with text conditioning
6. **Move to Issue #263** - Add more schedulers (PNDM, DPM-Solver++)

---

## Resources

### Papers
- **DDPM**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **DDIM**: "Denoising Diffusion Implicit Models" (Song et al., 2021)
- **Improved DDPM**: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)

### Key Concepts
- **Forward process**: q(x_t | x_{t-1})
- **Reverse process**: p_θ(x_{t-1} | x_t)
- **Training objective**: Predict noise ε from noisy x_t
- **Sampling**: Iteratively denoise from x_T to x_0

### Formulas
- **Noise schedule**: β_t defines noise at timestep t
- **Alpha**: α_t = 1 - β_t
- **Alpha cumprod**: ᾱ_t = ∏(α_i) for i=1 to t
- **Noisy sample**: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

---

## Questions?

If you get stuck:
1. Review the papers linked above
2. Check existing layer implementations (AttentionLayer, ConvolutionalLayer)
3. Look at PredictionModelBuilder for integration patterns
4. Ask in PR comments for clarification

Good luck! This is foundational work that will enable all future diffusion features.
