using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion;

/// <summary>
/// DDPM (Denoising Diffusion Probabilistic Models) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DDPM is the foundational diffusion model architecture that introduced the modern
/// approach to diffusion-based generation. It learns to reverse a gradual noising
/// process to generate data from pure noise.
/// </para>
/// <para>
/// <b>For Beginners:</b> DDPM is like learning to restore a damaged photograph.
///
/// Training process:
/// 1. Take a clear photo
/// 2. Add a specific amount of noise (determined by timestep)
/// 3. Train a neural network to predict what noise was added
/// 4. Repeat with different photos and noise levels
///
/// Generation process:
/// 1. Start with pure random noise
/// 2. Ask the trained model "what noise is in this?"
/// 3. Remove the predicted noise to get a slightly clearer image
/// 4. Repeat 1000 times (or use DDIM for faster generation)
/// 5. End up with a new, never-before-seen image
///
/// This implementation provides a minimal but functional DDPM that serves as:
/// - A reference implementation for understanding diffusion
/// - A base for more sophisticated diffusion models
/// - A demonstration of scheduler integration
/// </para>
/// <para>
/// <b>Reference:</b> "Denoising Diffusion Probabilistic Models" by Ho et al., 2020
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DDPM model for image generation
/// var options = new DiffusionModelOptions&lt;float&gt;
/// {
///     Height = 64,
///     Width = 64,
///     Channels = 3,
///     NumTimesteps = 1000
/// };
/// var model = new DDPMModel&lt;float&gt;(options);
///
/// // Generate an image from random noise
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 3, 64, 64 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Denoising Diffusion Probabilistic Models", "https://arxiv.org/abs/2006.11239", Year = 2020, Authors = "Ho et al.")]
public class DDPMModel<T> : DiffusionModelBase<T>
{
    #region Fields

    /// <summary>
    /// The UNet noise predictor per Ho et al. 2020 Section 3.
    /// Uses the standard architecture: residual blocks with GroupNorm → SiLU → Conv3x3,
    /// sinusoidal time embeddings, and self-attention at lower resolutions.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// Optional custom noise prediction function for testing or custom architectures.
    /// When provided, overrides the UNet for noise prediction.
    /// </summary>
    private readonly Func<Tensor<T>, int, Tensor<T>>? _customPredictor;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the DDPM model.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture specification.</param>
    /// <param name="options">Configuration options for the diffusion model. If null, uses default options.</param>
    /// <param name="scheduler">Optional custom scheduler. If null, creates one from options.</param>
    /// <param name="noisePredictor">
    /// Optional custom noise prediction function. If null, uses a placeholder that returns zeros.
    /// In production, this would be a neural network.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create a DDPM model by providing:
    /// <list type="bullet">
    /// <item><description>Options to configure learning rate, timesteps, and noise schedule</description></item>
    /// <item><description>Optionally, a custom scheduler (otherwise one is created from options)</description></item>
    /// <item><description>Optionally, a noise predictor (the neural network that learns patterns)</description></item>
    /// </list>
    /// Without a noise predictor, this is a "skeleton" model useful for:
    /// <list type="bullet">
    /// <item><description>Testing the scheduler integration</description></item>
    /// <item><description>Understanding the diffusion pipeline</description></item>
    /// <item><description>Serving as a template for custom implementations</description></item>
    /// </list></para>
    /// <example>
    /// <code>
    /// // Create a minimal DDPM for testing with defaults
    /// var model = new DDPMModel&lt;double&gt;();
    ///
    /// // Or with custom options
    /// var options = new DiffusionModelOptions&lt;double&gt;
    /// {
    ///     LearningRate = 0.0001,
    ///     TrainTimesteps = 1000,
    ///     DefaultInferenceSteps = 50
    /// };
    /// var model = new DDPMModel&lt;double&gt;(options);
    ///
    /// // Generate samples (note: without a trained noise predictor, results are random)
    /// var samples = model.Generate(new[] { 1, 3, 64, 64 }, numInferenceSteps: 50);
    /// </code>
    /// </example>
    /// </remarks>
    /// <summary>
    /// Creates a DDPM model with the standard UNet architecture from Ho et al. 2020.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture specification.</param>
    /// <param name="options">Configuration options. Default: 1000 timesteps, linear beta schedule.</param>
    /// <param name="scheduler">Optional custom scheduler. Default: DDIM scheduler.</param>
    /// <param name="unet">Optional custom UNet. Default: creates per-paper architecture.</param>
    /// <param name="customPredictor">Optional function override for noise prediction (for testing).</param>
    /// <param name="channels">Number of data channels. Default: 3 (RGB images).</param>
    /// <param name="imageSize">Spatial size of input data. Default: 32 (CIFAR-10 size per paper).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DDPMModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        Func<Tensor<T>, int, Tensor<T>>? customPredictor = null,
        int channels = 3,
        int imageSize = 32,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear,
                DefaultInferenceSteps = 50
            },
            scheduler,
            architecture)
    {
        _customPredictor = customPredictor;

        // Per Ho et al. 2020 Table 1: UNet with channel multipliers [1, 2, 2, 2] for CIFAR-10
        // Self-attention at 16×16 resolution, GroupNorm with 32 groups, SiLU activations
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: architecture,
            inputChannels: channels,
            outputChannels: channels,
            baseChannels: 128,
            channelMultipliers: [1, 2, 2, 2],
            numResBlocks: 2,
            attentionResolutions: [1],  // Attention at 16×16 (after first downsample)
            contextDim: 0,  // No cross-attention (unconditional DDPM)
            numHeads: 4,
            inputHeight: imageSize,
            seed: seed);
    }

    /// <summary>
    /// Initializes a new instance of the DDPM model with a scheduler only.
    /// </summary>
    public DDPMModel(INoiseScheduler<T> scheduler, Func<Tensor<T>, int, Tensor<T>>? customPredictor = null)
        : this(architecture: null, options: null, scheduler: scheduler, unet: null, customPredictor: customPredictor)
    {
    }

    /// <summary>
    /// Initializes a new instance of the DDPM model with a seed for reproducibility.
    /// </summary>
    public DDPMModel(int seed)
        : this(architecture: null, options: null, scheduler: null, unet: null, customPredictor: null, seed: seed)
    {
    }

    #endregion

    #region Core Methods

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Predicts the noise in a noisy sample. In a full implementation, this would
    /// use a trained neural network (typically a UNet). This minimal implementation
    /// returns zeros as a placeholder.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the "magic" happens in a real diffusion model.
    /// The neural network looks at a noisy image and tries to guess what noise was added.
    /// Better predictions = better generated images.
    /// </para>
    /// </remarks>
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        if (noisySample == null)
            throw new ArgumentNullException(nameof(noisySample));

        // Use custom predictor override if provided (for testing)
        if (_customPredictor != null)
        {
            return _customPredictor(noisySample, timestep);
        }

        // Use the UNet noise predictor per Ho et al. 2020
        return _unet.PredictNoise(noisySample, timestep, null);
    }

    /// <summary>
    /// Creates a DDPM model with a custom scheduler configuration.
    /// </summary>
    /// <param name="config">The scheduler configuration.</param>
    /// <param name="unet">Optional custom UNet noise predictor.</param>
    /// <returns>A new DDPM model instance.</returns>
    public static DDPMModel<T> Create(
        SchedulerConfig<T> config,
        UNetNoisePredictor<T>? unet = null)
    {
        var scheduler = new DDIMScheduler<T>(config);
        return new DDPMModel<T>(scheduler: scheduler, unet: unet);
    }

    /// <summary>
    /// Creates a DDPM model with a custom noise predictor function (for testing).
    /// </summary>
    public static DDPMModel<T> Create(
        SchedulerConfig<T> config,
        Func<Tensor<T>, int, Tensor<T>> customPredictor)
    {
        var scheduler = new DDIMScheduler<T>(config);
        return new DDPMModel<T>(scheduler: scheduler, customPredictor: customPredictor);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _unet.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        _unet.SetParameters(parameters);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedUnet = (UNetNoisePredictor<T>)_unet.Clone();
        return new DDPMModel<T>(
            scheduler: Scheduler,
            unet: clonedUnet,
            customPredictor: _customPredictor);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion
}
