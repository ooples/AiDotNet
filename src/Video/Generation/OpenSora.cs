using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Generation;

/// <summary>
/// OpenSora - Open-source Sora-like video generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OpenSora generates videos from text descriptions, similar to how
/// image generation models like DALL-E or Stable Diffusion work but for videos.
///
/// Key capabilities:
/// - Text-to-Video: Generate videos from text descriptions
/// - Image-to-Video: Animate still images
/// - Video continuation: Extend existing videos
/// - Variable length: Generate videos of different durations
/// - Multiple aspect ratios: Support various video dimensions
///
/// Example prompts:
/// - "A cat playing with a ball in a sunny garden"
/// - "Time-lapse of a flower blooming"
/// - "A spaceship flying through an asteroid field"
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Spatiotemporal DiT (Diffusion Transformer) architecture
/// - Variable resolution and duration support
/// - Efficient 3D attention mechanisms
/// - Progressive training strategy
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an OpenSora model for text-to-video generation
/// var openSora = new OpenSora&lt;double&gt;();
///
/// // Or configure with custom generation parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 3);
/// var model = new OpenSora&lt;double&gt;(architecture, numFrames: 16, numInferenceSteps: 50);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Open-Sora: Democratizing Efficient Video Production for All",
    "https://arxiv.org/abs/2412.20404",
    Year = 2024,
    Authors = "Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, Yang You")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class OpenSora<T> : NeuralNetworkBase<T>, AiDotNet.Interfaces.ITrainingObjectiveProvider<T>
{
    private readonly OpenSoraOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numFrames;
    private int _hiddenDim;
    private int _numLayers;
    private int _numInferenceSteps;
    // The timestep of the Train call in progress, read by ForwardForTraining; null outside Train.
    private double? _trainingTime;
    private double _guidanceScale;

    // Patch embedding for spatiotemporal input
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _patchEmbed = new ConvolutionalLayer<T>(1, 1, 1, 0);

    // DiT blocks (transformer layers with proper self-attention)
    // Following the Diffusion Transformer (DiT) architecture
    private List<ConvolutionalLayer<T>> _ditQKV = [];       // QKV projections
    private List<ConvolutionalLayer<T>> _ditAttnProj = [];  // Attention output projections
    private List<ConvolutionalLayer<T>> _ditFFN1 = [];      // FFN expand layers
    private List<ConvolutionalLayer<T>> _ditFFN2 = [];      // FFN contract layers
    private int _numHeads = 16;                              // Number of attention heads
    private int _headDim;                                    // Dimension per head

    // Text encoder projection
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _textProjection = new ConvolutionalLayer<T>(1, 1, 1, 0);

    // Time embedding
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _timeEmbed = new ConvolutionalLayer<T>(1, 1, 1, 0);

    // Final layer
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _finalLayer = new ConvolutionalLayer<T>(1, 1, 1, 0);

    // VAE decoder (latent to pixel)
    private List<ConvolutionalLayer<T>> _vaeDecoder = [];

    // VAE encoder (pixel to latent) - learned convolutional layers for proper image encoding
    private List<ConvolutionalLayer<T>> _vaeEncoder = [];

    // Noise schedule
    private double[] _betas = [];
    private double[] _alphasCumprod = [];

    // LayerNorm's unit gamma and zero beta: derived from the feature shape, never parameters.
    // Rebuilt when the spatial shape changes.
    [Scratch]
    private Tensor<T>? _layerNormGamma;
    [Scratch]
    private Tensor<T>? _layerNormBeta;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the output frame height.
    /// </summary>
    internal int OutputHeight => _height;

    /// <summary>
    /// Gets the output frame width.
    /// </summary>
    internal int OutputWidth => _width;

    /// <summary>
    /// Gets the number of frames to generate.
    /// </summary>
    internal int NumFrames => _numFrames;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// </summary>
    internal double GuidanceScale { get; set; }

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a default OpenSora video generation model with small default dimensions.
    /// </summary>
    public OpenSora()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            inputHeight: 256,
            inputWidth: 256,
            inputDepth: 3,
            outputSize: 256 * 256 * 3)) { }

    /// <summary>
    /// Creates an OpenSora video generation model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFrames">Number of frames to generate (default: 16).</param>
    /// <param name="hiddenDim">Hidden dimension of DiT blocks (default: 1152).</param>
    /// <param name="numLayers">Number of DiT transformer layers (default: 28).</param>
    /// <param name="numInferenceSteps">Number of diffusion inference steps (default: 50).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale (default: 7.5).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OpenSora generates videos from text descriptions, similar to
    /// how DALL-E generates images from text. Key parameters:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>numFrames:</b> How many video frames to generate.</item>
    /// <item><b>hiddenDim:</b> Model capacity - larger values give better quality but slower.</item>
    /// <item><b>numLayers:</b> Number of transformer layers - more layers = deeper reasoning.</item>
    /// <item><b>numInferenceSteps:</b> More steps = higher quality but slower generation.</item>
    /// <item><b>guidanceScale:</b> How closely to follow the text prompt (higher = more faithful).</item>
    /// </list>
    /// </para>
    /// </remarks>
    public OpenSora(
        NeuralNetworkArchitecture<T> architecture,
        int numFrames = 16,
        int hiddenDim = 1152,
        int numLayers = 28,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        OpenSoraOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new OpenSoraOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFrames = numFrames;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numInferenceSteps = numInferenceSteps;
        _guidanceScale = guidanceScale;
        GuidanceScale = guidanceScale;
        _numHeads = 16;
        // Each head owns a contiguous block of _headDim channels, so the heads must tile hiddenDim
        // exactly - the same requirement as PyTorch's nn.MultiheadAttention.
        if (_hiddenDim % _numHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), hiddenDim,
                $"hiddenDim must be divisible by the {_numHeads} attention heads.");
        _headDim = _hiddenDim / _numHeads;

        // Initialize noise schedule before InitializeLayers
        (_betas, _alphasCumprod) = InitializeNoiseSchedule(_numInferenceSteps);

        // Initialize layers using the proper pattern
        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Generates a video from a text prompt.
    /// </summary>
    /// <param name="textEmbedding">Text embedding from encoder [B, 768] or similar.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated video frames.</returns>
    public List<Tensor<T>> GenerateFromText(Tensor<T> textEmbedding, int? seed = null)
    {
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Process text embedding
        var textCondition = ProcessTextEmbedding(textEmbedding);

        // Initialize latent noise
        int latentH = _height / 8;
        int latentW = _width / 8;
        var latents = InitializeLatents([1, 4, latentH, latentW], random);

        // Denoising loop
        for (int step = 0; step < _numInferenceSteps; step++)
        {
            double t = 1.0 - (double)step / _numInferenceSteps;
            var timeEmbed = CreateTimeEmbedding(t);

            // Conditional prediction
            var noisePredCond = PredictNoise(latents, textCondition, timeEmbed);

            // Unconditional prediction
            var noisePredUncond = PredictNoise(latents, null, timeEmbed);

            // Classifier-free guidance
            var noisePred = ApplyGuidance(noisePredUncond, noisePredCond, GuidanceScale);

            // Denoising step
            latents = DenoisingStep(latents, noisePred, step);
        }

        // Decode latents to video frames
        return DecodeToFrames(latents);
    }

    /// <summary>
    /// Generates a video from an image (image-to-video).
    /// </summary>
    public List<Tensor<T>> GenerateFromImage(Tensor<T> image, Tensor<T>? textEmbedding = null, int? seed = null)
    {
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        if (image.Rank == 3) image = AddBatchDimension(image);

        // Encode image to latent
        var imageLatent = EncodeImage(image);

        // Initialize with image-conditioned noise
        var latents = InitializeLatentsFromImage(imageLatent, random);

        // Get text conditioning if provided
        var textCondition = textEmbedding != null ? ProcessTextEmbedding(textEmbedding) : null;

        // Denoising
        for (int step = 0; step < _numInferenceSteps; step++)
        {
            double t = 1.0 - (double)step / _numInferenceSteps;
            var timeEmbed = CreateTimeEmbedding(t);
            var noisePred = PredictNoise(latents, textCondition, timeEmbed);
            latents = DenoisingStep(latents, noisePred, step);
        }

        return DecodeToFrames(latents);
    }

    /// <summary>
    /// Extends an existing video.
    /// </summary>
    public List<Tensor<T>> ExtendVideo(List<Tensor<T>> existingFrames, Tensor<T>? textEmbedding = null, int? seed = null)
    {
        // Use the last frame as conditioning for video extension
        var lastFrame = existingFrames[existingFrames.Count - 1];

        var newFrames = GenerateFromImage(lastFrame, textEmbedding, seed);

        var result = new List<Tensor<T>>(existingFrames);
        result.AddRange(newFrames);
        return result;
    }

    /// <summary>
    /// Generates video with custom duration and aspect ratio.
    /// </summary>
    public List<Tensor<T>> GenerateCustom(
        Tensor<T> textEmbedding,
        int numFrames,
        int height,
        int width,
        int? seed = null)
    {
        // For simplicity, generate at default size and resize
        var frames = GenerateFromText(textEmbedding, seed);

        // Resize frames to target dimensions
        var resized = new List<Tensor<T>>();
        foreach (var frame in frames)
        {
            resized.Add(ResizeFrame(frame, height, width));
        }

        // Adjust frame count
        while (resized.Count < numFrames && resized.Count > 0)
        {
            resized.Add(resized[resized.Count - 1]);
        }

        return resized.Take(numFrames).ToList();
    }

    /// <summary>
    /// Performs a single denoising prediction step on the input latents.
    /// </summary>
    /// <param name="input">Input latent tensor [B, C, H, W], or a single latent [C, H, W].</param>
    /// <returns>Predicted denoised output, in the input's shape.</returns>
    /// <remarks>
    /// A single latent <c>[C, H, W]</c> - the shape this model's default architecture declares - is
    /// denoised as a batch of one and returned without the batch axis, the same promotion the model
    /// already applies to single images elsewhere (see <c>AddBatchDimension</c>). The denoiser indexes
    /// axis 3 of its features, so a rank-3 latent used to throw IndexOutOfRange.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (input.Rank == 3)
        {
            var denoised = PredictCore(AddBatchDimension(input));
            return denoised.Reshape([denoised.Shape[1], denoised.Shape[2], denoised.Shape[3]]);
        }

        // Create default time embedding at t=0.5 (mid-point)
        var timeEmbed = CreateTimeEmbedding(0.5);

        // Predict noise without text conditioning
        var noisePred = PredictNoise(input, null, timeEmbed);

        // Apply single denoising step
        return DenoisingStep(input, noisePred, _numInferenceSteps / 2);
    }

    /// <summary>
    /// Trains the model using the diffusion training objective.
    /// </summary>
    /// <param name="input">Clean input video latents.</param>
    /// <param name="expectedOutput">Target (typically the same as input for diffusion training).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // DDPM epsilon-prediction (Ho et al. 2020, Algorithm 1), the objective Open-Sora trains its DiT on: draw a
        // timestep and noise, form x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps, and fit the network's
        // predicted noise to eps. The regression target is the drawn noise, so expectedOutput is not used.
        // This used to compute the loss and a gradient by hand, never backpropagate it, and then call
        // UpdateParameters with a hard-coded rate on every layer, so the model never trained. It now runs on the
        // tape like every other network, with the base training loop's optimizer and loss.
        var random = RandomHelper.CreateSecureRandom();
        int timestep = random.Next(_numInferenceSteps);
        double alphaCumprod = _alphasCumprod[timestep];
        var noise = InitializeLatents(input._shape, random);
        var noisyInput = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(input, NumOps.FromDouble(Math.Sqrt(alphaCumprod))),
            Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(Math.Sqrt(1 - alphaCumprod))));

        _trainingTime = 1.0 - (double)timestep / _numInferenceSteps;
        SetTrainingMode(true);
        try
        {
            TrainWithTape(noisyInput, noise);
        }
        finally
        {
            _trainingTime = null;
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// During <see cref="Train"/>, the denoiser's forward at the step's timestep: the time embedding is built here,
    /// inside the taped forward, so its layer receives gradients too. Outside training it is the ordinary forward.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _trainingTime is { } time ? PredictNoise(input, null, CreateTimeEmbedding(time)) : base.ForwardForTraining(input);

    /// <summary>
    /// Each step conditions on a freshly drawn timestep, so a compiled plan that froze the first step's time
    /// embedding would train every later step at the wrong noise level. Train on the eager tape.
    /// </summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc/>
    AiDotNet.Enums.TrainingObjectiveKind AiDotNet.Interfaces.ITrainingObjectiveProvider<T>.TrainingObjectiveKind
        => AiDotNet.Enums.TrainingObjectiveKind.DiffusionDenoising;

    /// <summary>The clip being learned is x_0 itself: Train noises its input, not a separate target.</summary>
    Tensor<T> AiDotNet.Interfaces.ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        => input;

    /// <summary>
    /// L_simple over a fixed (timestep, noise) quadrature, through the configured loss: four timesteps spread over
    /// the schedule, each with seeded noise. A single Train call draws one random timestep, and the epsilon-MSE at
    /// one noise level is not comparable with the next step's at another, so this is the quantity whose decrease
    /// means the denoiser learned. It is deterministic and never updates parameters.
    /// </summary>
    T AiDotNet.Interfaces.ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        const int Points = 4;
        int points = Math.Min(Points, _numInferenceSteps);
        double total = 0;
        for (int k = 0; k < points; k++)
        {
            int timestep = Math.Min(_numInferenceSteps - 1, (int)((k + 0.5) * _numInferenceSteps / points));
            double alphaCumprod = _alphasCumprod[timestep];
            var noise = InitializeLatents(target._shape, new Random(20260928 + k));
            var noisy = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(target, NumOps.FromDouble(Math.Sqrt(alphaCumprod))),
                Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(Math.Sqrt(1 - alphaCumprod))));
            var predicted = PredictNoise(noisy, null, CreateTimeEmbedding(1.0 - (double)timestep / _numInferenceSteps));
            total += NumOps.ToDouble(LossFunction.CalculateLoss(predicted.ToVector(), noise.ToVector()));
        }

        return NumOps.FromDouble(total / points);
    }

    #endregion

    #region Private Methods

    private (double[] betas, double[] alphasCumprod) InitializeNoiseSchedule(int steps)
    {
        var betas = new double[steps];
        var alphasCumprod = new double[steps];

        // Linear schedule
        double betaStart = 0.00085;
        double betaEnd = 0.012;

        for (int i = 0; i < steps; i++)
        {
            betas[i] = betaStart + (betaEnd - betaStart) * i / (steps - 1);
        }

        double alphaCumprod = 1.0;
        for (int i = 0; i < steps; i++)
        {
            alphaCumprod *= (1.0 - betas[i]);
            alphasCumprod[i] = alphaCumprod;
        }

        return (betas, alphasCumprod);
    }

    private Tensor<T> InitializeLatents(int[] shape, Random random)
    {
        var latents = new Tensor<T>(shape);
        for (int i = 0; i < latents.Data.Length; i++)
        {
            latents.Data.Span[i] = NumOps.FromDouble(random.NextGaussian());
        }
        return latents;
    }

    private Tensor<T> InitializeLatentsFromImage(Tensor<T> imageLatent, Random random)
    {
        var noise = InitializeLatents(imageLatent._shape, random);

        // Mix image latent with noise (50/50 blend)
        var scaledImg = Engine.TensorMultiplyScalar(imageLatent, NumOps.FromDouble(0.5));
        var scaledN = Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(0.5));
        return Engine.TensorAdd(scaledImg, scaledN);
    }

    private Tensor<T> ProcessTextEmbedding(Tensor<T> textEmbedding)
    {
        if (textEmbedding.Rank == 2)
        {
            int batch = textEmbedding.Shape[0];
            int dim = textEmbedding.Shape[1];
            var reshaped = new Tensor<T>([batch, dim, 1, 1]);
            textEmbedding.Data.Span.CopyTo(reshaped.Data.Span);
            textEmbedding = reshaped;
        }

        return _textProjection.Forward(textEmbedding);
    }

    private Tensor<T> CreateTimeEmbedding(double t)
    {
        var timeInput = new Tensor<T>([1, 1, 1, 1]);
        timeInput[0, 0, 0, 0] = NumOps.FromDouble(t);
        return _timeEmbed.Forward(timeInput);
    }

    private Tensor<T> PredictNoise(Tensor<T> latents, Tensor<T>? textCondition, Tensor<T> timeEmbed)
    {
        // Patch embedding
        var features = _patchEmbed.Forward(latents);

        // Add conditioning
        if (textCondition != null)
        {
            features = AddCondition(features, textCondition);
        }
        features = AddCondition(features, timeEmbed);

        // DiT blocks with multi-head self-attention. Training checkpoints them: only segment boundaries are kept
        // and each segment is recomputed in the backward (Chen et al. 2016), sqrt(N) blocks per segment, the same
        // trade NoisePredictorBase makes. At Open-Sora's sequence length a taped step that kept every block's
        // activations would not fit; the reference implementation trains with gradient checkpointing as well.
        if (_trainingTime is not null)
        {
            var blocks = new Func<Tensor<T>, Tensor<T>>[_numLayers];
            for (int b = 0; b < _numLayers; b++)
            {
                int block = b;
                blocks[b] = x => DiTBlock(block, x);
            }

            int segmentSize = Math.Max(1, (int)Math.Sqrt(_numLayers));
            features = AiDotNet.Tensors.Engines.Autodiff.GradientCheckpointing<T>.Checkpoint(blocks, features, segmentSize);
        }
        else
        {
            for (int i = 0; i < _numLayers; i++)
                features = DiTBlock(i, features);
        }

        // Final prediction
        var noise = _finalLayer.Forward(features);

        // Unpatchify
        return UnpatchifyNoise(noise, latents._shape);
    }

    /// <summary>One pre-norm DiT block: self-attention and a GELU feed-forward, each with a residual.</summary>
    private Tensor<T> DiTBlock(int i, Tensor<T> features)
    {
        var residual = features;

        // Pre-norm (layer normalization)
        var normed = LayerNorm(features);

        // Multi-head self-attention
        var qkv = _ditQKV[i].Forward(normed);
        var attended = DiTMultiHeadAttention(qkv, features._shape);
        attended = _ditAttnProj[i].Forward(attended);

        // First residual connection
        features = AddTensors(features, attended);

        // Pre-norm for FFN
        residual = features;
        normed = LayerNorm(features);

        // FFN with GELU activation
        var ffnOut = _ditFFN1[i].Forward(normed);
        ffnOut = ApplyGELU(ffnOut);
        ffnOut = _ditFFN2[i].Forward(ffnOut);

        // Second residual connection
        features = AddTensors(features, ffnOut);
        return features;
    }

    /// <summary>
    /// Converts patched noise back to full resolution using pixel shuffle and bilinear interpolation.
    /// </summary>
    private Tensor<T> UnpatchifyNoise(Tensor<T> patchedNoise, int[] targetShape)
    {
        if (patchedNoise.Rank != 4 || targetShape.Length != 4 || patchedNoise.Shape[1] < targetShape[1])
            throw new ArgumentException(
                $"OpenSora cannot unpatchify [{string.Join(", ", patchedNoise._shape)}] to [{string.Join(", ", targetShape)}].");
        // The final layer emits more channels than the latent has; the noise is read from the leading ones, as the
        // element loop this replaces did (it indexed channel c < targetShape[1] of the patched output).
        if (patchedNoise.Shape[1] > targetShape[1])
            // TensorSlice copies into a contiguous result and records its backward; a Narrow view is strided, and a
            // batch > 1 then reached code that needs a contiguous span.
            patchedNoise = Engine.TensorSlice(patchedNoise, new[] { 0, 0, 0, 0 },
                new[] { patchedNoise.Shape[0], targetShape[1], patchedNoise.Shape[2], patchedNoise.Shape[3] });
        if (patchedNoise.Shape[2] == targetShape[2] && patchedNoise.Shape[3] == targetShape[3])
            return patchedNoise;
        // Bilinear with half-pixel centres and edge clamping (PyTorch's align_corners=False): the arithmetic the
        // element loop computed, as an engine op the tape can differentiate.
        return Engine.Interpolate(patchedNoise, new[] { targetShape[2], targetShape[3] },
            AiDotNet.Tensors.Engines.InterpolateMode.Bilinear, alignCorners: false);
    }

    private Tensor<T> DenoisingStep(Tensor<T> latents, Tensor<T> noisePred, int step)
    {
        double alphaCumprod = _alphasCumprod[step];
        double alphaCumprodPrev = step > 0 ? _alphasCumprod[step - 1] : 1.0;

        double sqrtAlphaCumprod = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaCumprod = Math.Sqrt(1 - alphaCumprod);
        double sqrtAlphaCumprodPrev = Math.Sqrt(alphaCumprodPrev);
        double sqrtOneMinusAlphaCumprodPrev = Math.Sqrt(1 - alphaCumprodPrev);

        // x0 = (x - sqrtOneMinusAlphaCumprod * noise) / sqrtAlphaCumprod
        var scaledNoisePred = Engine.TensorMultiplyScalar(noisePred, NumOps.FromDouble(sqrtOneMinusAlphaCumprod));
        var x0 = Engine.TensorDivideScalar(
            Engine.TensorSubtract(latents, scaledNoisePred),
            NumOps.FromDouble(sqrtAlphaCumprod));
        // next = sqrtAlphaCumprodPrev * x0 + sqrtOneMinusAlphaCumprodPrev * noise
        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(x0, NumOps.FromDouble(sqrtAlphaCumprodPrev)),
            Engine.TensorMultiplyScalar(noisePred, NumOps.FromDouble(sqrtOneMinusAlphaCumprodPrev)));
    }

    private Tensor<T> ApplyGuidance(Tensor<T> uncond, Tensor<T> cond, double scale)
    {
        // guided = uncond + scale * (cond - uncond)
        var diff = Engine.TensorSubtract(cond, uncond);
        var scaled = Engine.TensorMultiplyScalar(diff, NumOps.FromDouble(scale));
        return Engine.TensorAdd(uncond, scaled);
    }

    private List<Tensor<T>> DecodeToFrames(Tensor<T> latents)
    {
        // Decode through VAE
        // The VAE decoder layers are initialized with specific input dimensions:
        // - Layer 0: expects [latentDim, latentH, latentW], outputs 256 channels
        // - Layer 1: expects [256, latentH*2, latentW*2], outputs 128 channels
        // - Layer 2: expects [128, latentH*4, latentW*4], outputs 64 channels
        // - Layer 3: expects [64, _height, _width], outputs _channels
        //
        // The correct flow is: layer -> upsample (except for final layer)
        var decoded = latents;
        for (int i = 0; i < _vaeDecoder.Count; i++)
        {
            decoded = _vaeDecoder[i].Forward(decoded);
            decoded = ApplySiLU(decoded);

            // Upsample after each layer EXCEPT the final layer
            // Final layer already outputs at target resolution
            if (i < _vaeDecoder.Count - 1)
            {
                decoded = Upsample2x(decoded);
            }
        }
        decoded = ApplySigmoid(decoded);

        // Generate temporally-varying frames
        var frames = new List<Tensor<T>>();
        int batchSize = decoded.Shape[0];
        int channels = decoded.Shape[1];
        int height = decoded.Shape[2];
        int width = decoded.Shape[3];

        for (int f = 0; f < _numFrames; f++)
        {
            // Compute temporal position t ∈ [0, 1]
            double t = (double)f / Math.Max(1, _numFrames - 1);

            // Create frame with temporal modulation
            var frame = new Tensor<T>([batchSize, channels, height, width]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    // Apply temporal frequency modulation per channel
                    double freq = 2.0 * Math.PI * (c + 1) / channels;
                    double temporalMod = 0.1 * Math.Sin(freq * t);

                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            double baseVal = Convert.ToDouble(decoded[b, c, h, w]);

                            // Add spatiotemporal variation: blend base value with temporal modulation
                            // Include spatial position for more varied motion
                            double spatialFactor = (double)(h + w) / (height + width);
                            double motion = temporalMod * (1.0 + 0.5 * Math.Sin(2.0 * Math.PI * spatialFactor));

                            double finalVal = MathHelper.Clamp(baseVal + motion, 0.0, 1.0);
                            frame[b, c, h, w] = NumOps.FromDouble(finalVal);
                        }
                    }
                }
            }

            frames.Add(frame);
        }

        return frames;
    }

    /// <summary>
    /// Encodes an image to latent space using a learned VAE encoder.
    /// Uses strided convolutional layers for spatial downsampling with learned features.
    /// </summary>
    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        // Ensure input has batch dimension
        if (image.Rank == 3) image = AddBatchDimension(image);

        // Check if encoder is initialized (should be after InitializeNetworkLayers)
        if (_vaeEncoder.Count == 0)
        {
            // Fallback to simple downsampling if encoder not initialized
            return FallbackEncodeImage(image);
        }

        // Process through learned VAE encoder layers
        var encoded = image;
        for (int i = 0; i < _vaeEncoder.Count; i++)
        {
            encoded = _vaeEncoder[i].Forward(encoded);

            // Apply ReLU activation between layers (except final layer)
            if (i < _vaeEncoder.Count - 1)
            {
                for (int j = 0; j < encoded.Length; j++)
                {
                    double val = NumOps.ToDouble(encoded.Data.Span[j]);
                    encoded.Data.Span[j] = NumOps.FromDouble(Math.Max(0, val));
                }
            }
        }

        return encoded;
    }

    /// <summary>
    /// Fallback image encoding using simple average pooling when learned encoder is not available.
    /// </summary>
    private Tensor<T> FallbackEncodeImage(Tensor<T> image)
    {
        int latentH = _height / 8;
        int latentW = _width / 8;

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int srcH = image.Shape[2];
        int srcW = image.Shape[3];

        // Create latent with 4 channels (standard VAE latent dimension)
        var latent = new Tensor<T>([batchSize, 4, latentH, latentW]);

        // Simple average pooling fallback
        for (int b = 0; b < batchSize; b++)
        {
            for (int lh = 0; lh < latentH; lh++)
            {
                for (int lw = 0; lw < latentW; lw++)
                {
                    int srcY0 = lh * 8;
                    int srcY1 = Math.Min(srcY0 + 8, srcH);
                    int srcX0 = lw * 8;
                    int srcX1 = Math.Min(srcX0 + 8, srcW);

                    double[] channelSums = new double[channels];
                    int count = 0;

                    for (int y = srcY0; y < srcY1; y++)
                    {
                        for (int x = srcX0; x < srcX1; x++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                channelSums[c] += Convert.ToDouble(image[b, c, y, x]);
                            }
                            count++;
                        }
                    }

                    if (channels >= 3)
                    {
                        double r = channelSums[0] / count;
                        double g = channelSums[1] / count;
                        double blue = channelSums[2] / count;

                        latent[b, 0, lh, lw] = NumOps.FromDouble(0.7 * r + 0.3 * g);
                        latent[b, 1, lh, lw] = NumOps.FromDouble(0.4 * g + 0.6 * blue);
                        latent[b, 2, lh, lw] = NumOps.FromDouble(0.5 * r + 0.5 * blue);
                        latent[b, 3, lh, lw] = NumOps.FromDouble((r + g + blue) / 3.0);
                    }
                    else
                    {
                        double gray = channelSums[0] / count;
                        for (int c = 0; c < 4; c++)
                        {
                            latent[b, c, lh, lw] = NumOps.FromDouble(gray);
                        }
                    }
                }
            }
        }

        return latent;
    }

    private Tensor<T> ResizeFrame(Tensor<T> frame, int targetH, int targetW)
    {
        if (frame.Rank == 3) frame = AddBatchDimension(frame);

        int batchSize = frame.Shape[0];
        int channels = frame.Shape[1];
        int srcH = frame.Shape[2];
        int srcW = frame.Shape[3];

        var resized = new Tensor<T>([batchSize, channels, targetH, targetW]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < targetH; h++)
                    for (int w = 0; w < targetW; w++)
                    {
                        int srcY = Math.Min((int)((double)h * srcH / targetH), srcH - 1);
                        int srcX = Math.Min((int)((double)w * srcW / targetW), srcW - 1);
                        resized[b, c, h, w] = frame[b, c, srcY, srcX];
                    }

        return resized;
    }

    private Tensor<T> AddCondition(Tensor<T> features, Tensor<T> condition)
    {
        // A per-channel conditioning vector ([1 or B, C, 1, 1]) added to every position of [B, C, H, W] features.
        // The previous element loop indexed the condition by flat NCHW position modulo its length, which is not the
        // channel, so the embedding landed on scrambled channels; and it was invisible to the tape.
        if (condition.Rank != 4 || condition.Shape[1] != features.Shape[1] || condition.Shape[2] != 1 || condition.Shape[3] != 1
            || (condition.Shape[0] != 1 && condition.Shape[0] != features.Shape[0]))
            throw new ArgumentException(
                $"OpenSora conditioning must be [1 or batch, {features.Shape[1]}, 1, 1] against features " +
                $"[{string.Join(", ", features._shape)}], got [{string.Join(", ", condition._shape)}].", nameof(condition));
        // TensorAdd broadcasts by the NumPy rule, so [1 or B, C, 1, 1] reaches every position of its channel.
        return Engine.TensorAdd(features, condition);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        Engine.TensorAdd(a, b);

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, height * 2, width * 2]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        T val = input[b, c, h, w];
                        output[b, c, h * 2, w * 2] = val;
                        output[b, c, h * 2, w * 2 + 1] = val;
                        output[b, c, h * 2 + 1, w * 2] = val;
                        output[b, c, h * 2 + 1, w * 2 + 1] = val;
                    }

        return output;
    }

    // Engine.GELU is the tanh approximation this used to evaluate per element in double,
    // 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))), vectorized and without a boxed value per element.
    internal Tensor<T> ApplyGELU(Tensor<T> input) =>
        Engine.GELU(input);

    private Tensor<T> ApplySiLU(Tensor<T> input) =>
        input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            return NumOps.FromDouble(x / (1.0 + Math.Exp(-x)));
        });

    private Tensor<T> ApplySigmoid(Tensor<T> input) =>
        Engine.Sigmoid(input);

    /// <summary>
    /// Applies layer normalization (standardization) across spatial dimensions.
    /// </summary>
    /// <summary>
    /// Normalizes each sample over all of its channels and positions (no learned affine).
    /// </summary>
    /// <remarks>
    /// A per-element indexer loop here allocated an index array and boxed a value on every read and
    /// write: 22 GB of the 81 GB the paper-scale fixture allocated came from this method alone, and
    /// it ran on one thread. <c>Engine.LayerNorm</c> normalizes over the trailing dimensions that
    /// gamma's shape names, so a [C, H, W] gamma of ones and beta of zeros is the same computation -
    /// population variance, epsilon 1e-5 - in one vectorized pass.
    /// </remarks>
    internal Tensor<T> LayerNorm(Tensor<T> input)
    {
        int[] normalizedShape = [input.Shape[1], input.Shape[2], input.Shape[3]];
        if (_layerNormGamma is null || !_layerNormGamma._shape.AsSpan().SequenceEqual(normalizedShape))
        {
            _layerNormGamma = new Tensor<T>(normalizedShape);
            _layerNormGamma.Data.Span.Fill(NumOps.One);
            _layerNormBeta = new Tensor<T>(normalizedShape);
        }

        return Engine.LayerNorm(input, _layerNormGamma, _layerNormBeta!, 1e-5, out _, out _);
    }

    /// <summary>
    /// Local-window multi-head self-attention over the flattened spatial positions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Query i attends to keys j with <c>max(0, i - w/2) &lt;= j &lt; min(seqLen, i + w/2)</c>, where
    /// <c>w = min(seqLen, 64)</c>, and each head owns a contiguous block of <c>_headDim</c> channels.
    /// </para>
    /// <para>
    /// Queries are processed in tiles of <see cref="AttentionQueryTile"/>. A tile's queries can only
    /// reach the keys within half a window of it, so each tile is two batched GEMMs over at most
    /// <c>tile + w</c> keys with an additive mask for the exact band. That keeps the cost linear in the
    /// sequence, as the window intends: a dense score matrix at a 256x256 frame (16,384 positions) would
    /// be ~17 GB per sample. The previous scalar loop computed the same scores one element at a time
    /// through the tensor indexer, and was 28% of the paper-scale fixture's CPU and 18 GB of its
    /// allocation, on a single thread.
    /// </para>
    /// </remarks>
    internal Tensor<T> DiTMultiHeadAttention(Tensor<T> qkv, int[] inputShape)
    {
        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int seqLen = height * width;
        int halfWindow = Math.Min(seqLen, 64) / 2;

        // A window of zero width (seqLen == 1) attends to nothing, so the block contributes nothing.
        if (halfWindow == 0)
            return new Tensor<T>(inputShape);

        int batchHeads = batchSize * _numHeads;
        int headDim = _headDim;

        // qkv is [B, 3C, H, W] = [B, 3, heads, headDim, seq]; permuted to [3, B*heads, seq, headDim],
        // each of Q, K and V is one contiguous block.
        var seqMajor = Engine.TensorPermute(
            Engine.Reshape(qkv, [batchSize, 3, _numHeads, headDim, seqLen]), [1, 0, 2, 4, 3]).Contiguous();
        var all = seqMajor.Data.Span;
        int block = batchHeads * seqLen * headDim;

        var attended = new Tensor<T>([batchHeads, seqLen, headDim]);
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));
        T excluded = NumOps.FromDouble(-1e30);

        for (int tileStart = 0; tileStart < seqLen; tileStart += AttentionQueryTile)
        {
            int tileEnd = Math.Min(seqLen, tileStart + AttentionQueryTile);
            int queries = tileEnd - tileStart;
            int keyStart = Math.Max(0, tileStart - halfWindow);
            int keyEnd = Math.Min(seqLen, tileEnd - 1 + halfWindow);
            int keys = keyEnd - keyStart;

            var query = CopySequenceRows(all.Slice(0, block), batchHeads, seqLen, headDim, tileStart, queries);
            var key = CopySequenceRows(all.Slice(block, block), batchHeads, seqLen, headDim, keyStart, keys);
            var value = CopySequenceRows(all.Slice(2 * block, block), batchHeads, seqLen, headDim, keyStart, keys);

            var scores = Engine.BatchMatMul(query, Engine.TensorPermute(key, [0, 2, 1]).Contiguous());
            Engine.TensorMultiplyScalarInPlace(scores, scale);

            // Zero inside each query's band, a large negative value outside it. Finite rather than
            // -infinity: it underflows exp() to exactly zero without producing inf - inf in the
            // softmax's max subtraction, and fits a float.
            var mask = new Tensor<T>([1, queries, keys]);
            var maskSpan = mask.Data.Span;
            for (int r = 0; r < queries; r++)
            {
                int i = tileStart + r;
                int allowedStart = Math.Max(0, i - halfWindow) - keyStart;
                int allowedEnd = Math.Min(seqLen, i + halfWindow) - keyStart;
                var row = maskSpan.Slice(r * keys, keys);
                row.Slice(0, allowedStart).Fill(excluded);
                row.Slice(allowedEnd).Fill(excluded);
            }

            Engine.TensorBroadcastAddInPlace(scores, mask);
            var output = Engine.BatchMatMul(Engine.Softmax(scores, -1), value);   // [B*heads, queries, headDim]

            var source = output.Data.Span;
            var destination = attended.Data.Span;
            for (int bh = 0; bh < batchHeads; bh++)
            {
                source.Slice(bh * queries * headDim, queries * headDim)
                    .CopyTo(destination.Slice((bh * seqLen + tileStart) * headDim, queries * headDim));
            }
        }

        var headMajor = Engine.TensorPermute(
            Engine.Reshape(attended, [batchSize, _numHeads, seqLen, headDim]), [0, 1, 3, 2]);
        return Engine.Reshape(headMajor.Contiguous(), [batchSize, channels, height, width]);
    }

    /// <summary>Queries per attention tile; with the 64-wide window a tile reads at most 128 keys.</summary>
    private const int AttentionQueryTile = 64;

    /// <summary>
    /// Copies sequence rows [start, start + count) of every batch-head of a contiguous
    /// [batchHeads, seqLen, headDim] block into a new [batchHeads, count, headDim] tensor.
    /// </summary>
    private static Tensor<T> CopySequenceRows(
        ReadOnlySpan<T> source, int batchHeads, int seqLen, int headDim, int start, int count)
    {
        var result = new Tensor<T>([batchHeads, count, headDim]);
        var destination = result.Data.Span;
        for (int bh = 0; bh < batchHeads; bh++)
        {
            source.Slice((bh * seqLen + start) * headDim, count * headDim)
                .CopyTo(destination.Slice(bh * count * headDim, count * headDim));
        }

        return result;
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the neural network layers for OpenSora.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up all the building blocks of the OpenSora model.
    /// OpenSora uses a Diffusion Transformer (DiT) architecture for video generation.
    /// </para>
    /// <para>
    /// The layers include:
    /// <list type="bullet">
    /// <item><b>Patch Embedding:</b> Converts spatiotemporal video patches into embeddings.</item>
    /// <item><b>DiT Blocks:</b> Transformer layers with multi-head self-attention and FFN.</item>
    /// <item><b>Text/Time Projections:</b> Project conditioning signals into the hidden space.</item>
    /// <item><b>VAE Encoder/Decoder:</b> Compress images to latent space and back.</item>
    /// </list>
    /// </para>
    /// <para>
    /// If you provide custom layers in the architecture, those are used instead.
    /// Otherwise, the default OpenSora layers are created automatically.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            // Use default layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultOpenSoraLayers(
                Architecture, _height, _width, _channels, _hiddenDim, _numLayers, _numHeads));

            // Store references to specific layers for direct access
            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection for direct access.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OpenSora has several distinct components that need to be
    /// accessed individually during the forward pass. This method organizes these layers:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Patch Embedding:</b> The first layer that converts video patches into embeddings.</item>
    /// <item><b>DiT Layers:</b> QKV projections, attention output projections, and FFN layers.</item>
    /// <item><b>Text/Time Projections:</b> Conditioning signal projections.</item>
    /// <item><b>VAE Layers:</b> Encoder and decoder for latent space compression.</item>
    /// </list>
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Patch embedding
        if (Layers.Count > idx && Layers[idx] is ConvolutionalLayer<T> patchEmbed)
        {
            _patchEmbed = patchEmbed;
            idx++;
        }

        // DiT blocks: QKV, AttnProj, FFN1, FFN2 for each layer
        _ditQKV.Clear();
        _ditAttnProj.Clear();
        _ditFFN1.Clear();
        _ditFFN2.Clear();

        for (int i = 0; i < _numLayers && idx + 3 < Layers.Count; i++)
        {
            if (Layers[idx] is ConvolutionalLayer<T> qkv)
                _ditQKV.Add(qkv);
            idx++;

            if (Layers[idx] is ConvolutionalLayer<T> attnProj)
                _ditAttnProj.Add(attnProj);
            idx++;

            if (Layers[idx] is ConvolutionalLayer<T> ffn1)
                _ditFFN1.Add(ffn1);
            idx++;

            if (Layers[idx] is ConvolutionalLayer<T> ffn2)
                _ditFFN2.Add(ffn2);
            idx++;
        }

        // Text projection
        if (Layers.Count > idx && Layers[idx] is ConvolutionalLayer<T> textProj)
        {
            _textProjection = textProj;
            idx++;
        }

        // Time embedding
        if (Layers.Count > idx && Layers[idx] is ConvolutionalLayer<T> timeEmb)
        {
            _timeEmbed = timeEmb;
            idx++;
        }

        // Final layer
        if (Layers.Count > idx && Layers[idx] is ConvolutionalLayer<T> finalLyr)
        {
            _finalLayer = finalLyr;
            idx++;
        }

        // VAE decoder (4 layers)
        _vaeDecoder.Clear();
        for (int i = 0; i < 4 && idx < Layers.Count; i++)
        {
            if (Layers[idx] is ConvolutionalLayer<T> decLayer)
                _vaeDecoder.Add(decLayer);
            idx++;
        }

        // VAE encoder (4 layers)
        _vaeEncoder.Clear();
        for (int i = 0; i < 4 && idx < Layers.Count; i++)
        {
            if (Layers[idx] is ConvolutionalLayer<T> encLayer)
                _vaeEncoder.Add(encLayer);
            idx++;
        }
    }

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "OpenSora" },
            { "Description", "Open-source Sora-like Video Generation" },
            { "OutputHeight", _height },
            { "OutputWidth", _width },
            { "NumFrames", _numFrames },
            { "NumLayers", _numLayers },
            { "GuidanceScale", _guidanceScale }
        },
        ModelDataProvider = () => SerializeForMetadata()
    };



    /// <summary>
    /// Restores model configuration from serialized data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you load a saved model, this method reads back all
    /// the configuration values (like dimensions, number of layers, etc.) and rebuilds
    /// the network architecture to match what was saved.
    /// </para>
    /// <para>
    /// This ensures that after loading, the model has exactly the same structure
    /// as when it was saved, including all the learned weights.
    /// </para>
    /// </remarks>


    #endregion
}
