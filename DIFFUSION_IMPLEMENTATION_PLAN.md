# AiDotNet Diffusion Models - Comprehensive Implementation Plan

## Executive Summary

This document outlines a complete implementation plan for diffusion models in AiDotNet that will exceed industry standards with **full feature parity with HuggingFace diffusers**. Priority focus is on **video diffusion** (SVD, AnimateDiff), with support for audio and 3D modalities.

### Key Requirements (From Discussion)
- **Primary Focus**: Multi-modal generation (Video first, then Audio, then 3D)
- **Architecture**: Interfaces extending IFullModel + Base classes + Concrete implementations
- **Weight Loading**: Support both HuggingFace Hub download AND local .pt/.safetensors/.onnx files
- **Builder Pattern**: Use existing `ConfigureModel()` method on AiModelBuilder
- **Facade Pattern**: Users interact via AiModelBuilder/AiModelResult only
- **AutoML Integration**: Full support for automated diffusion model selection/tuning
- **Agents Integration**: Full support for AI agents to create and configure diffusion models
- **Engine**: Extend existing IEngine with diffusion-specific operations
- **Breaking Changes**: OK - we can refactor existing APIs for better design

---

## Current Issues to Fix

### 1. LatentDiffusionModel Problems
```csharp
// CURRENT (BAD) - Hardcoded, no flexibility
public class LatentDiffusionModel<T>
{
    private readonly VAEEncoder<T> _encoder;  // Hardcoded concrete type
    private readonly VAEDecoder<T> _decoder;  // Hardcoded concrete type
    private readonly DiffusionUNet<T> _unet;  // Hardcoded concrete type

    // Duplicates Box-Muller instead of using RandomHelper
    private Tensor<T> SampleNoiseTensor(...) { /* Box-Muller duplicate */ }

    // Duplicates MSE instead of using ILossFunction
    private T ComputeMSELoss(...) { /* MSE duplicate */ }
}
```

### 2. Missing Integrations
- Not using `IEngine<T>` for GPU acceleration
- Not using `GradientTape<T>` properly for autodiff
- Not using existing `RandomHelper.CreateSeededRandom()`
- Not using existing loss functions from `AiDotNet.LossFunctions`
- Not following nullable parameter pattern with defaults
- No AutoML support
- No Agents support

---

## Proposed Architecture

### Interface Hierarchy (Recommended)

```
IFullModel<T, Tensor<T>, Tensor<T>>
    │
    ├── IDiffusionModel<T>  (existing - keep)
    │       │
    │       ├── ILatentDiffusionModel<T>  (NEW)
    │       │       Methods: EncodeToLatent(), DecodeFromLatent()
    │       │
    │       ├── IVideoDiffusionModel<T>  (NEW)
    │       │       Methods: GenerateVideo(), InterpolateFrames()
    │       │
    │       ├── IAudioDiffusionModel<T>  (NEW)
    │       │       Methods: GenerateAudio(), TextToSpeech()
    │       │
    │       └── I3DDiffusionModel<T>  (NEW)
    │               Methods: GeneratePointCloud(), GenerateMesh()
    │
    ├── INoisePredictor<T> : IFullModel<T, Tensor<T>, Tensor<T>>  (NEW)
    │       │
    │       ├── IUNetNoisePredictor<T>
    │       ├── IDiTNoisePredictor<T>  (Transformer-based)
    │       └── IVideoNoisePredictor<T>  (3D convolutions)
    │
    ├── IVAEModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>  (NEW)
    │       Methods: Encode(), Decode(), Sample()
    │
    └── IConditioningModule<T>  (NEW)
            │
            ├── ITextConditioner<T>  (CLIP, T5)
            ├── IImageConditioner<T>  (IP-Adapter)
            └── IControlConditioner<T>  (ControlNet, T2I-Adapter)
```

### Base Classes

```
DiffusionModelBase<T>  (existing - enhance)
    │
    ├── LatentDiffusionModelBase<T>  (NEW)
    │       │
    │       ├── LatentDiffusionModel<T>  (refactor existing)
    │       ├── SDXLModel<T>
    │       └── StableCascadeModel<T>
    │
    ├── VideoDiffusionModelBase<T>  (NEW)
    │       │
    │       ├── StableVideoDiffusion<T>
    │       ├── AnimateDiffModel<T>
    │       └── VideoCrafterModel<T>
    │
    ├── AudioDiffusionModelBase<T>  (NEW)
    │       │
    │       ├── AudioLDM<T>
    │       └── DiffWaveModel<T>
    │
    └── ThreeDDiffusionModelBase<T>  (NEW)
            │
            ├── PointEModel<T>
            ├── ShapEModel<T>
            └── DreamFusionModel<T>

NoisePredictorBase<T>  (NEW)
    │
    ├── UNetNoisePredictor<T>
    ├── DiTNoisePredictor<T>
    └── UViTNoisePredictor<T>

VAEModelBase<T>  (NEW)
    │
    ├── StandardVAE<T>  (SD-style)
    ├── TinyVAE<T>  (fast inference)
    └── TemporalVAE<T>  (for video)
```

---

## Options Pattern (Following Existing Conventions)

### DiffusionModelOptions<T>
```csharp
/// <summary>
/// Options for configuring diffusion models with industry-standard defaults.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> All parameters have sensible defaults based on
/// Stable Diffusion and other production models. You only need to customize
/// what you want to change.
/// </remarks>
public class DiffusionModelOptions<T> // Already exists - needs enhancement
{
    // === Model Architecture ===

    /// <summary>
    /// The noise prediction network. If null, creates a default UNet.
    /// </summary>
    /// <remarks>
    /// Default: UNetNoisePredictor with 320 base channels (Stable Diffusion architecture).
    /// Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", 2022.
    /// </remarks>
    public INoisePredictor<T>? NoisePredictor { get; set; }

    /// <summary>
    /// The VAE for latent space encoding/decoding. If null, creates a default VAE.
    /// </summary>
    /// <remarks>
    /// Default: StandardVAE with 4 latent channels and 0.18215 scale factor.
    /// Reference: Stable Diffusion VAE configuration.
    /// </remarks>
    public IVAEModel<T>? VAE { get; set; }

    /// <summary>
    /// The conditioning module for text/image guidance. If null, no conditioning.
    /// </summary>
    public IConditioningModule<T>? Conditioner { get; set; }

    // === Scheduler ===

    /// <summary>
    /// The noise scheduler. If null, uses DDIM scheduler.
    /// </summary>
    /// <remarks>
    /// Default: DDIMScheduler with 1000 training steps, cosine beta schedule.
    /// DDIM allows fewer inference steps (20-50) while maintaining quality.
    /// </remarks>
    public IStepScheduler<T>? Scheduler { get; set; }

    // === Latent Space ===

    /// <summary>
    /// Number of latent channels. Default: 4 (Stable Diffusion standard).
    /// </summary>
    public int LatentChannels { get; set; } = 4;

    /// <summary>
    /// Scale factor for latent values. Default: 0.18215 (Stable Diffusion standard).
    /// </summary>
    /// <remarks>
    /// This normalizes the latent distribution for stable training.
    /// Value derived from the VAE's learned latent distribution.
    /// </remarks>
    public double LatentScaleFactor { get; set; } = 0.18215;

    // === Training ===

    /// <summary>
    /// Loss function for training. If null, uses MSE loss.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Learning rate for training. Default: 1e-4 (standard for diffusion).
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    // === Inference ===

    /// <summary>
    /// Default number of inference steps. Default: 50.
    /// </summary>
    /// <remarks>
    /// With DDIM/DPM-Solver, 20-50 steps usually suffice.
    /// More steps = higher quality but slower.
    /// </remarks>
    public int DefaultInferenceSteps { get; set; } = 50;

    /// <summary>
    /// Classifier-free guidance scale. Default: 7.5.
    /// </summary>
    /// <remarks>
    /// Higher values = stronger adherence to conditioning.
    /// Typical range: 1.0 (no guidance) to 15.0 (very strong).
    /// Reference: Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022.
    /// </remarks>
    public double GuidanceScale { get; set; } = 7.5;

    // === Engine & Optimization ===

    /// <summary>
    /// Compute engine for acceleration. If null, auto-detects best available.
    /// </summary>
    public IEngine<T>? Engine { get; set; }

    /// <summary>
    /// Whether to enable JIT compilation. Default: true.
    /// </summary>
    public bool EnableJitCompilation { get; set; } = true;

    /// <summary>
    /// Whether to use gradient checkpointing to save memory. Default: false.
    /// </summary>
    public bool UseGradientCheckpointing { get; set; } = false;

    /// <summary>
    /// Whether to use attention slicing to reduce memory. Default: false.
    /// </summary>
    public bool UseAttentionSlicing { get; set; } = false;
}
```

### VideoDiffusionOptions<T>
```csharp
public class VideoDiffusionOptions<T> : DiffusionModelOptions<T>
{
    /// <summary>
    /// Number of frames to generate. Default: 25.
    /// </summary>
    public int NumFrames { get; set; } = 25;

    /// <summary>
    /// Frames per second. Default: 7.
    /// </summary>
    public int FPS { get; set; } = 7;

    /// <summary>
    /// Motion bucket ID for controlling motion amount. Default: 127.
    /// </summary>
    /// <remarks>
    /// Lower values = less motion, higher values = more motion.
    /// Range: 1-255 (SVD default: 127).
    /// </remarks>
    public int MotionBucketId { get; set; } = 127;

    /// <summary>
    /// Noise augmentation level. Default: 0.02.
    /// </summary>
    public double NoiseAugStrength { get; set; } = 0.02;

    /// <summary>
    /// Temporal VAE for video encoding. If null, uses standard VAE per-frame.
    /// </summary>
    public IVAEModel<T>? TemporalVAE { get; set; }
}
```

---

## Complete Model Implementation List

### Tier 1: Core Foundation (Refactoring + New Interfaces)

| Component | Type | Status | Priority | Description |
|-----------|------|--------|----------|-------------|
| INoisePredictor<T> | Interface | NEW | P0 | Base interface for denoising networks |
| IVAEModel<T> | Interface | NEW | P0 | Base interface for VAE |
| IConditioningModule<T> | Interface | NEW | P0 | Base interface for conditioning |
| ILatentDiffusionModel<T> | Interface | NEW | P0 | Latent diffusion specific |
| IVideoDiffusionModel<T> | Interface | NEW | P0 | Video diffusion specific |
| IAudioDiffusionModel<T> | Interface | NEW | P0 | Audio diffusion specific |
| I3DDiffusionModel<T> | Interface | NEW | P0 | 3D diffusion specific |
| NoisePredictorBase<T> | Base Class | NEW | P0 | Common noise predictor functionality |
| VAEModelBase<T> | Base Class | NEW | P0 | Common VAE functionality |
| LatentDiffusionModelBase<T> | Base Class | NEW | P0 | Common latent diffusion functionality |
| VideoDiffusionModelBase<T> | Base Class | NEW | P0 | Common video diffusion functionality |
| DiffusionModelOptions<T> | Options | ENHANCE | P0 | Full options with defaults |

### Tier 2: Noise Predictors (Denoising Networks)

| Model | Status | Architecture | Use Case |
|-------|--------|--------------|----------|
| UNetNoisePredictor<T> | NEW | Standard U-Net | SD, SDXL |
| DiTNoisePredictor<T> | NEW | Diffusion Transformer | SD3, Sora |
| UViTNoisePredictor<T> | NEW | U-ViT Hybrid | PixArt |
| VideoUNetPredictor<T> | NEW | 3D U-Net | SVD |
| TemporalTransformer<T> | NEW | Temporal attention | AnimateDiff |

### Tier 3: VAE Models

| Model | Status | Description |
|-------|--------|-------------|
| StandardVAE<T> | NEW | SD-style VAE (4 channels) |
| SDXLVAE<T> | NEW | SDXL VAE (improved quality) |
| TinyVAE<T> | NEW | Fast inference VAE |
| TemporalVAE<T> | NEW | Video-aware VAE |
| AudioVAE<T> | NEW | Audio spectrogram VAE |

### Tier 4: Image Diffusion Models

| Model | Status | Description |
|-------|--------|-------------|
| LatentDiffusionModel<T> | REFACTOR | Base latent diffusion |
| SDXLModel<T> | NEW | Stable Diffusion XL |
| PixArtModel<T> | NEW | Efficient DiT |
| StableCascadeModel<T> | NEW | Wuerstchen architecture |
| ConsistencyModel<T> | NEW | Single-step generation |
| FlowMatchingModel<T> | NEW | Flow matching |
| EDMModel<T> | NEW | Elucidated diffusion |

### Tier 5: Video Diffusion Models (PRIMARY FOCUS)

| Model | Status | Description |
|-------|--------|-------------|
| StableVideoDiffusion<T> | NEW | Image-to-video (SVD) |
| AnimateDiffModel<T> | NEW | Motion module approach |
| VideoCrafterModel<T> | NEW | Text-to-video |
| ModelScopeModel<T> | NEW | Text-to-video |
| I2VGenXL<T> | NEW | High-res image-to-video |

### Tier 6: Audio Diffusion Models

| Model | Status | Description |
|-------|--------|-------------|
| AudioLDM<T> | NEW | Audio latent diffusion |
| AudioLDM2<T> | NEW | Improved audio generation |
| DiffWaveModel<T> | NEW | Waveform diffusion |
| RiffusionModel<T> | NEW | Spectrogram diffusion |
| MusicGenModel<T> | NEW | Music generation |

### Tier 7: 3D Diffusion Models

| Model | Status | Description |
|-------|--------|-------------|
| PointEModel<T> | NEW | Point cloud diffusion |
| ShapEModel<T> | NEW | 3D mesh generation |
| Zero123Model<T> | NEW | Novel view synthesis |
| DreamFusionModel<T> | NEW | Text-to-3D via SDS |
| MVDreamModel<T> | NEW | Multi-view diffusion |

### Tier 8: Controlled Generation

| Model | Status | Description |
|-------|--------|-------------|
| ControlNetModel<T> | NEW | Spatial conditioning |
| T2IAdapterModel<T> | NEW | Lightweight adapters |
| IPAdapterModel<T> | NEW | Image prompt adapter |
| InstructPix2Pix<T> | NEW | Instruction editing |

### Tier 9: Personalization

| Component | Status | Description |
|-----------|--------|-------------|
| LoRAAdapter<T> | NEW | Low-rank adaptation |
| DreamBoothTrainer<T> | NEW | Subject fine-tuning |
| TextualInversion<T> | NEW | Custom embeddings |

---

## Schedulers (Complete List)

### Existing (Verify/Enhance)
| Scheduler | Status |
|-----------|--------|
| DDIMScheduler | EXISTS |
| PNDMScheduler | EXISTS |
| EulerDiscreteScheduler | EXISTS |
| EulerAncestralScheduler | EXISTS |
| HeunDiscreteScheduler | EXISTS |
| LMSDiscreteScheduler | EXISTS |
| DPMSolverMultistepScheduler | EXISTS |

### New Required
| Scheduler | Priority | Description |
|-----------|----------|-------------|
| UniPCMultistepScheduler | P1 | Universal predictor-corrector |
| DPM2AncestralScheduler | P1 | DPM++ 2M ancestral |
| DEISMultistepScheduler | P2 | Diffusion exponential integrator |
| SASolverScheduler | P2 | Stochastic Adams solver |
| LCMScheduler | P1 | Latent consistency |
| TCDScheduler | P2 | Trajectory consistency |
| FlowMatchScheduler | P1 | For flow matching models |
| EDMScheduler | P1 | Karras scheduler |

---

## Engine Extensions

### New Operations for IEngine<T>

```csharp
public interface IEngine<T>
{
    // Existing operations...

    // === NEW: Diffusion-Specific Operations ===

    /// <summary>
    /// Efficient flash attention implementation.
    /// </summary>
    Tensor<T> FlashAttention(Tensor<T> query, Tensor<T> key, Tensor<T> value,
                              Tensor<T>? mask = null, double scale = 1.0);

    /// <summary>
    /// Group normalization optimized for diffusion.
    /// </summary>
    Tensor<T> GroupNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta,
                         int numGroups, double epsilon = 1e-5);

    /// <summary>
    /// Sinusoidal time embedding computation.
    /// </summary>
    Tensor<T> SinusoidalEmbedding(Tensor<T> timesteps, int embeddingDim);

    /// <summary>
    /// 3D convolution for video models.
    /// </summary>
    Tensor<T> Conv3D(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias = null,
                      int[] stride = null, int[] padding = null);

    /// <summary>
    /// Efficient upsample operation.
    /// </summary>
    Tensor<T> Upsample(Tensor<T> input, int[] scaleFactor, InterpolationMode mode);

    /// <summary>
    /// Cross-attention between two sequences.
    /// </summary>
    Tensor<T> CrossAttention(Tensor<T> query, Tensor<T> context, int numHeads);
}
```

---

## Weight Loading System

### WeightLoader Classes

```csharp
/// <summary>
/// Loads weights from various formats into AiDotNet models.
/// </summary>
public interface IWeightLoader<T>
{
    Task<bool> CanLoadAsync(string source);
    Task LoadWeightsAsync(IFullModel<T, Tensor<T>, Tensor<T>> model, string source);
}

/// <summary>
/// Loads PyTorch .pt and .safetensors files.
/// </summary>
public class PyTorchWeightLoader<T> : IWeightLoader<T>
{
    public Task LoadFromSafeTensorsAsync(IFullModel<T, ...> model, string path);
    public Task LoadFromPickleAsync(IFullModel<T, ...> model, string path);
}

/// <summary>
/// Loads ONNX models.
/// </summary>
public class OnnxWeightLoader<T> : IWeightLoader<T>
{
    public Task LoadFromOnnxAsync(IFullModel<T, ...> model, string path);
}

/// <summary>
/// Downloads and caches models from HuggingFace Hub.
/// </summary>
public class HuggingFaceHub<T>
{
    public Task<string> DownloadModelAsync(string repoId, string filename = null);
    public Task<IFullModel<T, ...>> LoadPipelineAsync(string repoId);
}
```

---

## AutoML Integration

### DiffusionAutoML Support

```csharp
public class DiffusionAutoML<T> : AutoMLModelBase<T>
{
    /// <summary>
    /// Search space for diffusion model hyperparameters.
    /// </summary>
    protected override void DefineSearchSpace()
    {
        // Architecture choices
        AddCategoricalParameter("noise_predictor",
            ["unet", "dit", "uvit"]);

        // Scheduler choices
        AddCategoricalParameter("scheduler",
            ["ddim", "dpm_solver", "euler", "lcm"]);

        // Training parameters
        AddFloatParameter("learning_rate", 1e-6, 1e-3, log: true);
        AddIntParameter("inference_steps", 10, 100);
        AddFloatParameter("guidance_scale", 1.0, 15.0);

        // Architecture parameters
        AddIntParameter("base_channels", 128, 512, step: 64);
        AddIntParameter("num_res_blocks", 1, 4);
    }
}
```

---

## Helper Classes

### DiffusionNoiseHelper<T>
```csharp
public static class DiffusionNoiseHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Sample Gaussian noise using existing RandomHelper.
    /// </summary>
    public static Tensor<T> SampleGaussian(int[] shape, int? seed = null)
    {
        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        var tensor = new Tensor<T>(shape);
        var span = tensor.AsWritableSpan();

        // Use existing Box-Muller from a helper or inline once
        for (int i = 0; i < span.Length; i += 2)
        {
            var (z0, z1) = BoxMullerTransform(rng);
            span[i] = NumOps.FromDouble(z0);
            if (i + 1 < span.Length)
                span[i + 1] = NumOps.FromDouble(z1);
        }

        return tensor;
    }

    private static (double, double) BoxMullerTransform(Random rng)
    {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        while (u1 <= double.Epsilon) u1 = rng.NextDouble();

        double mag = Math.Sqrt(-2.0 * Math.Log(u1));
        return (mag * Math.Cos(2 * Math.PI * u2), mag * Math.Sin(2 * Math.PI * u2));
    }
}
```

---

## Implementation Phases

### Phase 1: Foundation (2-3 weeks)
**Goal: Create infrastructure and refactor existing code**

1. Create new interfaces:
   - INoisePredictor<T>
   - IVAEModel<T>
   - IConditioningModule<T>
   - ILatentDiffusionModel<T>
   - IVideoDiffusionModel<T>
   - IAudioDiffusionModel<T>
   - I3DDiffusionModel<T>

2. Create base classes:
   - NoisePredictorBase<T>
   - VAEModelBase<T>
   - LatentDiffusionModelBase<T>
   - VideoDiffusionModelBase<T>
   - AudioDiffusionModelBase<T>
   - ThreeDDiffusionModelBase<T>

3. Create helpers:
   - DiffusionNoiseHelper<T>
   - DiffusionTensorHelper<T>

4. Refactor LatentDiffusionModel<T> to use interfaces

5. Extend IEngine with diffusion operations

6. Implement weight loaders:
   - PyTorchWeightLoader<T>
   - OnnxWeightLoader<T>
   - HuggingFaceHub<T>

### Phase 2: Video Diffusion (2-3 weeks)
**Goal: Implement video generation models (PRIMARY FOCUS)**

1. Implement TemporalVAE<T>
2. Implement VideoUNetPredictor<T>
3. Implement StableVideoDiffusion<T>
4. Implement AnimateDiffModel<T>
5. Implement VideoCrafterModel<T>
6. Add video-specific schedulers
7. Create video generation tests

### Phase 3: Production Image Models (2 weeks)
**Goal: Feature parity with HuggingFace for image generation**

1. Implement DiTNoisePredictor<T>
2. Implement SDXLModel<T>
3. Implement PixArtModel<T>
4. Implement ConsistencyModel<T>
5. Implement ControlNetModel<T>
6. Implement IPAdapterModel<T>

### Phase 4: Audio Diffusion (2 weeks)
**Goal: Audio generation capabilities**

1. Implement AudioVAE<T>
2. Implement AudioLDM<T>
3. Implement DiffWaveModel<T>
4. Implement RiffusionModel<T>

### Phase 5: 3D Diffusion (2 weeks)
**Goal: 3D generation capabilities**

1. Implement PointEModel<T>
2. Implement ShapEModel<T>
3. Implement Zero123Model<T>
4. Implement DreamFusionModel<T>

### Phase 6: AutoML & Agents (1 week)
**Goal: Full automation support**

1. Implement DiffusionAutoML<T>
2. Add diffusion model support to Agents
3. Create agent prompts for diffusion model creation

### Phase 7: Optimization & Testing (Ongoing)
**Goal: Production readiness**

1. JIT compilation for all models
2. GPU kernel optimization
3. Memory optimization (gradient checkpointing, attention slicing)
4. Comprehensive test suite (>= 90% coverage)
5. Benchmark against PyTorch diffusers

---

## Test Strategy

### Unit Tests
- Test each interface implementation independently
- Test weight loading from all formats
- Test parameter serialization/deserialization
- Test gradient computation via autodiff

### Integration Tests
- Test full generation pipelines
- Test training loops
- Test AutoML model selection
- Test Agent-based model creation

### Reference Tests
- Compare outputs against HuggingFace diffusers
- Verify numerical precision
- Test deterministic generation with seeds

---

## Success Metrics

- [ ] All models follow IFullModel interface chain
- [ ] All options have nullable parameters with documented defaults
- [ ] Zero code duplication for common operations
- [ ] Full IEngine integration for CPU/GPU acceleration
- [ ] Full JIT compilation support
- [ ] Full autodiff support via GradientTape
- [ ] Weight loading from PyTorch + ONNX + HuggingFace Hub
- [ ] AutoML integration complete
- [ ] Agents integration complete
- [ ] >= 90% test coverage
- [ ] Benchmark parity with PyTorch diffusers
- [ ] Video generation working end-to-end
- [ ] Audio generation working end-to-end
- [ ] 3D generation working end-to-end

---

## File Structure

```
src/
├── Interfaces/
│   ├── IDiffusionModel.cs (existing)
│   ├── INoisePredictor.cs (NEW)
│   ├── IVAEModel.cs (NEW)
│   ├── IConditioningModule.cs (NEW)
│   ├── ILatentDiffusionModel.cs (NEW)
│   ├── IVideoDiffusionModel.cs (NEW)
│   ├── IAudioDiffusionModel.cs (NEW)
│   └── I3DDiffusionModel.cs (NEW)
│
├── Diffusion/
│   ├── DiffusionModelBase.cs (existing - enhance)
│   ├── DDPMModel.cs (existing)
│   ├── LatentDiffusionModelBase.cs (NEW)
│   ├── LatentDiffusionModel.cs (refactor)
│   ├── VideoDiffusionModelBase.cs (NEW)
│   ├── AudioDiffusionModelBase.cs (NEW)
│   ├── ThreeDDiffusionModelBase.cs (NEW)
│   │
│   ├── NoisePredictors/
│   │   ├── NoisePredictorBase.cs (NEW)
│   │   ├── UNetNoisePredictor.cs (NEW)
│   │   ├── DiTNoisePredictor.cs (NEW)
│   │   └── VideoUNetPredictor.cs (NEW)
│   │
│   ├── VAE/
│   │   ├── VAEModelBase.cs (NEW)
│   │   ├── StandardVAE.cs (NEW)
│   │   ├── TinyVAE.cs (NEW)
│   │   └── TemporalVAE.cs (NEW)
│   │
│   ├── Conditioning/
│   │   ├── ConditioningModuleBase.cs (NEW)
│   │   ├── CLIPConditioner.cs (NEW)
│   │   ├── T5Conditioner.cs (NEW)
│   │   └── ImageConditioner.cs (NEW)
│   │
│   ├── Schedulers/ (existing - add new ones)
│   │   ├── UniPCMultistepScheduler.cs (NEW)
│   │   ├── LCMScheduler.cs (NEW)
│   │   └── EDMScheduler.cs (NEW)
│   │
│   ├── Models/
│   │   ├── Image/
│   │   │   ├── SDXLModel.cs (NEW)
│   │   │   ├── PixArtModel.cs (NEW)
│   │   │   └── ConsistencyModel.cs (NEW)
│   │   │
│   │   ├── Video/
│   │   │   ├── StableVideoDiffusion.cs (NEW)
│   │   │   ├── AnimateDiffModel.cs (NEW)
│   │   │   └── VideoCrafterModel.cs (NEW)
│   │   │
│   │   ├── Audio/
│   │   │   ├── AudioLDM.cs (NEW)
│   │   │   └── DiffWaveModel.cs (NEW)
│   │   │
│   │   └── ThreeD/
│   │       ├── PointEModel.cs (NEW)
│   │       ├── ShapEModel.cs (NEW)
│   │       └── Zero123Model.cs (NEW)
│   │
│   ├── ControlledGeneration/
│   │   ├── ControlNetModel.cs (NEW)
│   │   ├── T2IAdapterModel.cs (NEW)
│   │   └── IPAdapterModel.cs (NEW)
│   │
│   ├── Personalization/
│   │   ├── LoRAAdapter.cs (NEW)
│   │   ├── DreamBoothTrainer.cs (NEW)
│   │   └── TextualInversion.cs (NEW)
│   │
│   └── WeightLoading/
│       ├── IWeightLoader.cs (NEW)
│       ├── PyTorchWeightLoader.cs (NEW)
│       ├── OnnxWeightLoader.cs (NEW)
│       └── HuggingFaceHub.cs (NEW)
│
├── Helpers/
│   ├── DiffusionNoiseHelper.cs (NEW)
│   └── DiffusionTensorHelper.cs (NEW)
│
└── AutoML/
    └── DiffusionAutoML.cs (NEW)
```

---

## Questions Resolved

1. **Interface Granularity**: Using specialized sub-interfaces (ILatentDiffusionModel, IVideoDiffusionModel, etc.) extending IDiffusionModel
2. **VAE Separation**: VAE is a separate model implementing IVAEModel<T> : IFullModel
3. **Scheduler Integration**: Will enhance to work with Tensor<T> directly
4. **Weight Loading**: Full support for PyTorch (.pt/.safetensors) + ONNX + HuggingFace Hub
5. **Memory Optimization**: Will implement gradient checkpointing, attention slicing as options
6. **Priority Order**: Video diffusion first, then audio, then 3D
7. **Testing Strategy**: Test against HuggingFace diffusers reference implementations
8. **Builder Pattern**: Use existing ConfigureModel() on AiModelBuilder
9. **Engine Extension**: Add diffusion-specific operations to existing IEngine

