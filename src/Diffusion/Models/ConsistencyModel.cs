using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Consistency Model for single-step or few-step image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Consistency Models can generate high-quality images in a single step by learning to map
/// any point on a probability flow ODE trajectory directly to the trajectory's origin
/// (the clean data). This enables extremely fast generation compared to traditional
/// diffusion models that require 20-50+ steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional diffusion models are like walking down a path one step
/// at a time - they need many steps to go from noise to a clear image. Consistency Models
/// are like teleportation - they can jump directly from any point on the path to the
/// destination (clean image).
///
/// Key advantages:
/// - Single-step generation possible (fastest mode)
/// - Progressive refinement: 1-step, 2-step, 4-step, etc.
/// - Quality improves with more steps but plateaus quickly
/// - Same or better quality as DDPM with 1000x fewer steps
///
/// How it works:
/// 1. The model learns that all points on a denoising path should map to the same clean image
/// 2. This "consistency" property allows direct prediction from any noise level
/// 3. The model can self-refine by treating its output as a new starting point
///
/// Use cases:
/// - Real-time image generation
/// - Interactive applications
/// - Mobile/edge deployment
/// - Batch processing at scale
/// </para>
/// <para>
/// Technical details:
/// - Based on probability flow ODEs (deterministic diffusion)
/// - Two training methods: distillation from pretrained diffusion, direct training
/// - Uses boundary condition f(x, eps) = x (identity at minimal noise)
/// - Supports multistep sampling for quality/speed tradeoff
///
/// Reference: Song et al., "Consistency Models", ICML 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Consistency Model
/// var model = new ConsistencyModel&lt;float&gt;();
///
/// // Single-step generation (fastest)
/// var image1 = model.GenerateFromText(
///     prompt: "A beautiful sunset over mountains",
///     numInferenceSteps: 1);
///
/// // Two-step generation (better quality)
/// var image2 = model.GenerateFromText(
///     prompt: "A beautiful sunset over mountains",
///     numInferenceSteps: 2);
///
/// // Four-step generation (highest quality)
/// var image4 = model.GenerateFromText(
///     prompt: "A beautiful sunset over mountains",
///     numInferenceSteps: 4);
///
/// // Generate with progressive refinement
/// var refined = model.GenerateWithProgressiveRefinement(
///     prompt: "Detailed landscape painting",
///     maxSteps: 4,
///     returnIntermediates: true);
/// </code>
/// </example>
public class ConsistencyModel<T> : LatentDiffusionModelBase<T>
{
    /// <summary>
    /// Standard consistency model latent channels.
    /// </summary>
    private const int CM_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard VAE scale factor.
    /// </summary>
    private const int CM_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// The noise predictor (U-Net or DiT).
    /// </summary>
    private readonly UNetNoisePredictor<T> _noisePredictor;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// The conditioning module for text encoding.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// The sigma schedule for the ODE.
    /// </summary>
    private readonly T[] _sigmas;

    /// <summary>
    /// Number of training timesteps.
    /// </summary>
    private readonly int _numTrainSteps;

    /// <summary>
    /// Minimum sigma value (epsilon).
    /// </summary>
    private readonly double _sigmaMin;

    /// <summary>
    /// Maximum sigma value.
    /// </summary>
    private readonly double _sigmaMax;

    /// <summary>
    /// Rho parameter for sigma schedule.
    /// </summary>
    private readonly double _rho;

    /// <summary>
    /// Whether this model was trained via distillation.
    /// </summary>
    private readonly bool _isDistilled;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _noisePredictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => CM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount =>
        _noisePredictor.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the minimum sigma value used by this model.
    /// </summary>
    public double SigmaMin => _sigmaMin;

    /// <summary>
    /// Gets the maximum sigma value used by this model.
    /// </summary>
    public double SigmaMax => _sigmaMax;

    /// <summary>
    /// Gets whether this model was trained via distillation.
    /// </summary>
    public bool IsDistilled => _isDistilled;

    /// <summary>
    /// Initializes a new instance of ConsistencyModel with default parameters.
    /// </summary>
    /// <remarks>
    /// Creates a Consistency Model with standard parameters:
    /// - 18 training timesteps
    /// - Sigma range: 0.002 to 80
    /// - Rho: 7 (for sigma schedule)
    /// </remarks>
    /// <summary>
    /// Initializes a new instance of ConsistencyModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="noisePredictor">Optional custom noise predictor.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner">Optional conditioning module for text encoding.</param>
    /// <param name="numTrainSteps">Number of training timesteps (default: 18).</param>
    /// <param name="sigmaMin">Minimum sigma value (default: 0.002).</param>
    /// <param name="sigmaMax">Maximum sigma value (default: 80.0).</param>
    /// <param name="rho">Rho parameter for sigma schedule (default: 7.0).</param>
    /// <param name="isDistilled">Whether the model was trained via distillation.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ConsistencyModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? noisePredictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int numTrainSteps = 18,
        double sigmaMin = 0.002,
        double sigmaMax = 80.0,
        double rho = 7.0,
        bool isDistilled = false,
        int? seed = null)
        : base(options ?? CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler())
    {
        _numTrainSteps = numTrainSteps;
        _sigmaMin = sigmaMin;
        _sigmaMax = sigmaMax;
        _rho = rho;
        _isDistilled = isDistilled;
        _conditioner = conditioner;

        // Create noise predictor
        _noisePredictor = noisePredictor ?? CreateDefaultNoisePredictor(seed);

        // Create VAE
        _vae = vae ?? CreateDefaultVAE(seed);

        // Precompute sigma schedule
        _sigmas = ComputeSigmaSchedule();
    }

    /// <summary>
    /// Creates the default options for Consistency Model.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 18,
            BetaStart = 0.00085,
            BetaEnd = 0.012,
            BetaSchedule = BetaSchedule.ScaledLinear
        };
    }

    /// <summary>
    /// Creates the default scheduler.
    /// </summary>
    private static INoiseScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default noise predictor.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultNoisePredictor(int? seed)
    {
        return new UNetNoisePredictor<T>(
            inputChannels: CM_LATENT_CHANNELS,
            outputChannels: CM_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            seed: seed);
    }

    /// <summary>
    /// Creates the default VAE.
    /// </summary>
    private StandardVAE<T> CreateDefaultVAE(int? seed)
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: CM_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

    /// <summary>
    /// Computes the sigma schedule for the ODE.
    /// </summary>
    private T[] ComputeSigmaSchedule()
    {
        var sigmas = new T[_numTrainSteps + 1];
        var oneOverRho = 1.0 / _rho;

        for (int i = 0; i <= _numTrainSteps; i++)
        {
            var t = (double)i / _numTrainSteps;
            // sigma(t) = (sigma_min^(1/rho) + t * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho
            var sigmaMinPow = Math.Pow(_sigmaMin, oneOverRho);
            var sigmaMaxPow = Math.Pow(_sigmaMax, oneOverRho);
            var sigma = Math.Pow(sigmaMinPow + t * (sigmaMaxPow - sigmaMinPow), _rho);
            sigmas[i] = NumOps.FromDouble(sigma);
        }

        return sigmas;
    }

    /// <summary>
    /// Applies the consistency function to map from any noise level to clean data.
    /// </summary>
    /// <param name="x">The noisy sample.</param>
    /// <param name="sigma">The current noise level (sigma).</param>
    /// <param name="conditioning">Optional conditioning tensor.</param>
    /// <returns>The predicted clean sample.</returns>
    /// <remarks>
    /// The consistency function f(x, sigma) should satisfy:
    /// - f(x, sigma_min) = x (boundary condition at minimal noise)
    /// - f(x, sigma) = f(x', sigma') for all x, x' on same ODE trajectory
    /// </remarks>
    public virtual Tensor<T> ConsistencyFunction(Tensor<T> x, T sigma, Tensor<T>? conditioning)
    {
        var sigmaDouble = NumOps.ToDouble(sigma);

        // Skip denoising at boundary condition
        if (sigmaDouble <= _sigmaMin * 1.001)
        {
            return x;
        }

        // Scale for c_skip and c_out parameterization
        var sigmaData = 0.5; // Standard data sigma
        var cSkip = sigmaData * sigmaData / (sigmaDouble * sigmaDouble + sigmaData * sigmaData);
        var cOut = sigmaDouble * sigmaData / Math.Sqrt(sigmaDouble * sigmaDouble + sigmaData * sigmaData);
        var cIn = 1.0 / Math.Sqrt(sigmaDouble * sigmaDouble + sigmaData * sigmaData);

        // Scale input
        var scaledInput = ScaleTensor(x, cIn);

        // Get model prediction (predicts the denoised sample or noise)
        var timestep = SigmaToTimestep(sigmaDouble);
        var modelOutput = _noisePredictor.PredictNoise(scaledInput, timestep, conditioning);

        // Apply skip connection and output scaling
        // f(x, sigma) = c_skip * x + c_out * F(c_in * x, sigma)
        var result = new Tensor<T>(x.Shape);
        var xSpan = x.AsSpan();
        var outSpan = modelOutput.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var cSkipT = NumOps.FromDouble(cSkip);
        var cOutT = NumOps.FromDouble(cOut);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Add(
                NumOps.Multiply(cSkipT, xSpan[i]),
                NumOps.Multiply(cOutT, outSpan[i]));
        }

        return result;
    }

    /// <summary>
    /// Generates an image using consistency sampling.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Image width (should be divisible by 8).</param>
    /// <param name="height">Image height (should be divisible by 8).</param>
    /// <param name="numInferenceSteps">Number of sampling steps (1, 2, 4, etc.).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The generated image tensor.</returns>
    /// <remarks>
    /// For consistency models:
    /// - 1 step: Fastest, single forward pass
    /// - 2 steps: Good balance of speed and quality
    /// - 4 steps: Near-optimal quality
    /// - 8+ steps: Diminishing returns
    /// </remarks>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 2,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Clamp steps to valid range for consistency model
        numInferenceSteps = Math.Max(1, Math.Min(numInferenceSteps, _numTrainSteps));

        // Get conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;

        if (_conditioner != null)
        {
            var promptTokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);

            var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
            if (effectiveGuidanceScale > 1.0)
            {
                if (!string.IsNullOrEmpty(negativePrompt))
                {
                    var negTokens = _conditioner.Tokenize(negativePrompt ?? string.Empty);
                    negativeEmbedding = _conditioner.EncodeText(negTokens);
                }
                else
                {
                    negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
                }
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / CM_VAE_SCALE_FACTOR;
        var latentWidth = width / CM_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, CM_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize with noise at maximum sigma
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Scale noise by sigma_max
        latents = ScaleTensor(latents, _sigmaMax);

        // Compute step indices for multistep sampling
        var stepIndices = ComputeStepIndices(numInferenceSteps);

        // Consistency sampling loop
        for (int i = 0; i < stepIndices.Length; i++)
        {
            var stepIdx = stepIndices[i];
            var sigma = _sigmas[stepIdx];

            // Apply consistency function
            Tensor<T> denoised;
            var effectiveGuidance = guidanceScale ?? GuidanceScale;

            if (effectiveGuidance > 1.0 && negativeEmbedding != null && promptEmbedding != null)
            {
                // Classifier-free guidance
                var condDenoised = ConsistencyFunction(latents, sigma, promptEmbedding);
                var uncondDenoised = ConsistencyFunction(latents, sigma, negativeEmbedding);
                denoised = ApplyGuidance(uncondDenoised, condDenoised, effectiveGuidance);
            }
            else
            {
                denoised = ConsistencyFunction(latents, sigma, promptEmbedding);
            }

            // If not the last step, add noise at the next sigma level
            if (i < stepIndices.Length - 1)
            {
                var nextIdx = stepIndices[i + 1];
                var nextSigma = _sigmas[nextIdx];

                // Add noise scaled by next sigma
                var noise = SampleNoiseTensor(latentShape, rng);
                latents = new Tensor<T>(latentShape);
                var denoisedSpan = denoised.AsSpan();
                var noiseSpan = noise.AsSpan();
                var latentsSpan = latents.AsWritableSpan();

                for (int j = 0; j < latentsSpan.Length; j++)
                {
                    latentsSpan[j] = NumOps.Add(denoisedSpan[j], NumOps.Multiply(nextSigma, noiseSpan[j]));
                }
            }
            else
            {
                // Last step: use denoised result directly
                latents = denoised;
            }
        }

        // Decode to image
        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Generates images with progressive refinement, optionally returning intermediates.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="maxSteps">Maximum number of refinement steps.</param>
    /// <param name="returnIntermediates">Whether to return intermediate images.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>List of generated images at each step (if returnIntermediates) or just the final image.</returns>
    public virtual List<Tensor<T>> GenerateWithProgressiveRefinement(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int maxSteps = 4,
        bool returnIntermediates = false,
        double? guidanceScale = null,
        int? seed = null)
    {
        var results = new List<Tensor<T>>();

        for (int steps = 1; steps <= maxSteps; steps *= 2)
        {
            var image = GenerateFromText(
                prompt,
                negativePrompt,
                width,
                height,
                steps,
                guidanceScale,
                seed);

            if (returnIntermediates || steps == maxSteps)
            {
                results.Add(image);
            }
        }

        return results;
    }

    /// <summary>
    /// Computes the step indices for multistep sampling.
    /// </summary>
    private int[] ComputeStepIndices(int numSteps)
    {
        if (numSteps == 1)
        {
            return new[] { _numTrainSteps };
        }

        // Evenly spaced steps from max to min
        var indices = new int[numSteps];
        for (int i = 0; i < numSteps; i++)
        {
            var t = (double)(numSteps - 1 - i) / (numSteps - 1);
            indices[i] = (int)Math.Round(t * _numTrainSteps);
        }

        return indices;
    }

    /// <summary>
    /// Converts sigma to timestep for the noise predictor.
    /// </summary>
    private int SigmaToTimestep(double sigma)
    {
        // Find closest sigma in schedule
        for (int i = 0; i < _sigmas.Length - 1; i++)
        {
            var sigmaI = NumOps.ToDouble(_sigmas[i]);
            var sigmaI1 = NumOps.ToDouble(_sigmas[i + 1]);

            if (sigma >= sigmaI1 && sigma <= sigmaI)
            {
                // Linear interpolation to get timestep
                var t = (sigma - sigmaI1) / (sigmaI - sigmaI1 + 1e-8);
                return (int)(i + t);
            }
        }

        return sigma >= NumOps.ToDouble(_sigmas[0]) ? 0 : _numTrainSteps;
    }

    /// <summary>
    /// Scales a tensor by a scalar value.
    /// </summary>
    private Tensor<T> ScaleTensor(Tensor<T> tensor, double scale)
    {
        var result = new Tensor<T>(tensor.Shape);
        var inputSpan = tensor.AsSpan();
        var resultSpan = result.AsWritableSpan();
        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Multiply(scaleT, inputSpan[i]);
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _noisePredictor.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _noisePredictor.SetParameters(parameters);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new ConsistencyModel<T>(
            numTrainSteps: _numTrainSteps,
            sigmaMin: _sigmaMin,
            sigmaMax: _sigmaMax,
            rho: _rho,
            isDistilled: _isDistilled,
            conditioner: _conditioner,
            seed: RandomGenerator.Next());

        clone.SetParameters(GetParameters());

        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ConsistencyModel",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Consistency Model for single-step or few-step image generation",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("num_train_steps", _numTrainSteps);
        metadata.SetProperty("sigma_min", _sigmaMin);
        metadata.SetProperty("sigma_max", _sigmaMax);
        metadata.SetProperty("rho", _rho);
        metadata.SetProperty("is_distilled", _isDistilled);

        return metadata;
    }
}
