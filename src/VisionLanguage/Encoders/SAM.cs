using AiDotNet.LearningRateSchedulers;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Segment Anything Model (SAM) vision encoder for promptable image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SAM (Kirillov et al., 2023) consists of a ViT image encoder producing image embeddings, a prompt
/// encoder that handles points/boxes/masks, and a lightweight mask decoder. The image encoder uses
/// windowed attention with occasional global attention blocks for efficiency at high resolution.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Segment Anything" (Kirillov et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> SAM (Segment Anything Model) from Meta is a promptable
/// segmentation model — you give it an image and a prompt (a point click, a bounding box,
/// or a rough mask) and it segments the object you indicated. Trained on over 1 billion masks,
/// its ViT encoder uses windowed attention with occasional global attention for efficient
/// high-resolution processing. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SAM model for promptable image segmentation
/// // using ViT encoder with windowed attention for high-resolution images
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 256);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SAM&lt;double&gt;(architecture, "sam.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SAM&lt;double&gt;(architecture, new SAMOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Segment Anything",
    "https://arxiv.org/abs/2304.02643",
    Year = 2023,
    Authors = "Kirillov et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, LearningRate = 8e-4,
                WeightDecay = 0.1, ReferenceBatchSize = 256, WarmupSteps = 250,
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.1,
                Milestones = [60000, 86666],
                Source = "Kirillov et al. 2023, Training recipe: AdamW with beta1 0.9, beta2 0.999, linear warmup for 250 iterations, initial rate 8e-4 after warmup, decreased 10x at 60k and again at 86666 iterations over a 90k-iteration run, batch size 256, weight decay 0.1.")]
[TaskVariantOf(typeof(AiDotNet.ComputerVision.Segmentation.Foundation.SAM<>), "image encoder for vision-language models")]
public partial class SAM<T> : VisionLanguageModelBase<T>, IVisualEncoder<T>
{
    private readonly SAMOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private SamNetwork<T>? _network;
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);


    /// <summary>The network, the same one the promptable segmentation SAM runs.</summary>
    internal SamNetwork<T> Network => _network ?? throw new InvalidOperationException("The native network is not initialized.");

    public SAM(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SAMOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SAMOptions(); Options = _options;
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.MaskDecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    public SAM(
        NeuralNetworkArchitecture<T> architecture,
        SAMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SAMOptions(); Options = _options;
        if (architecture.InputType == InputType.ThreeDimensional && architecture.InputHeight > 0)
        {
            // The encoder square holds the architecture's longer side, rounded up to the patch: an image of any size up
            // to it is zero-padded to the square (SamNetwork.Preprocess, the official Sam.preprocess).
            int longer = Math.Max(architecture.InputHeight, architecture.InputWidth), patch = _options.PatchSize;
            _options = new SAMOptions(_options) { ImageSize = (longer + patch - 1) / patch * patch }; Options = _options;
        }
        _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("SAM declares its paper recipe.");
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.MaskDecoderDim;
        InitializeLayers();
    }

    /// <summary>The image embedding's channels (the mask decoder's width, 256 in the released models).</summary>
    public int EmbeddingDimension => _options.MaskDecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <summary>The configuration the options describe: the encoder's width, depth and heads, its global-attention
    /// blocks at every quarter of the depth (as the released ViT-B/L/H place them), and the decoder.</summary>
    private SamConfiguration Configuration()
    {
        int depth = _options.NumLayers;
        var global = Enumerable.Range(1, 4).Select(k => k * depth / 4 - 1).Where(i => i >= 0).Distinct().ToArray();
        return new SamConfiguration
        {
            ImageSize = _options.ImageSize, PatchSize = _options.PatchSize, EncoderDim = _options.EmbeddingDim, EncoderDepth = depth,
            EncoderHeads = _options.NumHeads, MlpRatio = _options.FfnMultiplier, WindowSize = _options.WindowSize,
            GlobalAttention = global, RelativePositions = _options.UseRelativePositionalEncoding, OutChannels = _options.MaskDecoderDim,
            DecoderDepth = _options.NumMaskDecoderLayers, MultimaskOutputs = _options.NumMultimaskOutputs,
            PixelMean = _options.ImageMean, PixelStd = _options.ImageStd,
        };
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode || _options is null || _network is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("SAM builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new SamNetwork<T>(Engine, layers, Configuration());
        int seed = _options.Seed ?? 42;
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        Layers.AddRange(layers);
    }

    /// <summary>Loads an official SAM checkpoint's state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    private Tensor<T> Batched(Tensor<T> image) => image.Rank == 3
        ? Engine.Reshape(image, new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] })
        : image;

    /// <summary>
    /// The image embedding <c>[MaskDecoderDim, g, g]</c> (or <c>[B, MaskDecoderDim, g, g]</c>) of images in 0..1 at most
    /// the encoder size: normalized by the options' mean and std, zero-padded to a square, and run through SAM's image
    /// encoder.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));
        bool single = image.Rank == 3;
        var embedding = Network.EncodeImage(Network.Preprocess(PreprocessImage(Batched(image))));
        return single ? Engine.Reshape(embedding, new[] { embedding.Shape[1], embedding.Shape[2], embedding.Shape[3] }) : embedding;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => EncodeImage(input);

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EncodeImage(input);

    /// <summary>
    /// One step of SAM's training on an image and its ground-truth mask (<c>H·W</c> values): the paper's interactive
    /// prompt simulation through the full network (<see cref="SamTraining{T}"/>), the encoder trained with the decoder.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var image = Batched(input);
        // The network's preprocessing normalizes with the options' mean and std; images arrive in 0..1.
        var training = new SamTraining<T>(Engine, Network, 20.0, 1.0, 2.0, 0.25);
        var objective = training.Objective(image, expected, _trainingRandom);
        // The objective runs the whole network itself; TrainWithCustomLoss would also run (and discard) a forward pass.
        TrainWithCustomObjective(image, expected, (_, _) => objective(), _optimizer);
        return true;
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>SAM normalizes and pads to a square in its network (<see cref="EncodeImage"/>); the image passes
    /// through unchanged here.</summary>
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => image;

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SAM-Native" : "SAM-ONNX",
            Description = "Segment Anything Model (Kirillov et al., 2023): the image encoder",
            FeatureCount = _options.MaskDecoderDim,
            Complexity = _options.NumLayers,
        };
        m.AdditionalInfo["Architecture"] = "SAM";
        m.AdditionalInfo["MaskDecoderDim"] = _options.MaskDecoderDim.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SAM<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
