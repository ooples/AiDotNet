using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Vision Mamba (Vim) image classification: the official network with a linear classifier on its class token.
/// </summary>
/// <remarks>
/// <para>
/// The image is cut into patches; a class token is inserted in the middle of the patch sequence; each layer runs a
/// bidirectional Mamba mixer (a forward and a backward selective-scan branch, averaged) on the RMS-normalized
/// residual stream; and the class token's final features are classified. The network is shared with the Vision Mamba
/// segmenter and checked against the official code (hustvl/Vim).
/// </para>
/// <para><b>For Beginners:</b> This model classifies images using Mamba instead of attention.
/// It cuts the image into small patches, reads them forwards and backwards with Mamba blocks (fast sequence
/// processing), and outputs class probabilities. Vision Mamba needs O(n) computation vs O(n^2) for Transformers,
/// making it faster for high-resolution images.</para>
/// <para>
/// <b>Reference:</b> Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space
/// Model", ICML 2024.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", "https://arxiv.org/abs/2401.09417", Year = 2024, Authors = "Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-3, Beta1 = 0.9, WeightDecay = 0.05, ReferenceBatchSize = 1024,
                Schedule = LearningRateSchedulerType.CosineAnnealing, ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                MinLearningRate = 0,
                Source = "Zhu et al. 2024, Sec. 4.1: AdamW with a momentum of 0.9, a total batch size of 1024 and a weight decay of 0.05, 300 epochs with a cosine schedule and a 1e-3 initial learning rate, and EMA (whose decay the paper does not state).")]
public partial class VisionMambaModel<T> : ImageClassifierModelLayoutBase<T>
{
    private readonly VisionMambaOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly VimConfiguration _configuration;
    private VimNetwork<T>? _network;

    /// <summary>Draws stochastic-depth decisions during training.</summary>
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    public override bool SupportsTraining => true;

    public override ModelOptions GetOptions() => _options;

    public int ImageHeight => _configuration.ImageHeight;
    public int ImageWidth => _configuration.ImageWidth;
    public int PatchSize => _configuration.PatchSize;
    public int ModelDimension => _configuration.Dim;
    public int NumLayers => _configuration.Depth;
    public int NumClasses => _options.NumClasses;
    public int NumPatches => _configuration.Patches;

    /// <summary>The network, shared with the Vision Mamba segmenter.</summary>
    internal VimNetwork<T> Network => _network ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The linear classifier on the class token (softmax output).</summary>
    internal DenseLayer<T>? Classifier { get; private set; }

    /// <summary>Creates a Vision Mamba classifier.</summary>
    public VisionMambaModel(
        NeuralNetworkArchitecture<T> architecture,
        VisionMambaOptions? options = null,
        ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.ImageClassification))
    {
        _options = options ?? new VisionMambaOptions();
        _options.Validate();
        Options = _options;
        _configuration = new VimConfiguration
        {
            ImageHeight = _options.ImageHeight, ImageWidth = _options.ImageWidth, Channels = _options.Channels,
            PatchSize = _options.PatchSize, Stride = _options.EffectiveStride, Dim = _options.ModelDimension, Depth = _options.NumLayers,
            StateDim = _options.StateDimension, DropPathRate = _options.DropPathRate,
        };
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("Vision Mamba declares its paper recipe.");
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("VisionMambaModel builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new VimNetwork<T>(Engine, layers, _configuration);
        int seed = _options.Seed ?? 42;
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        _network.InitializeLikeReference(random);
        Classifier = new DenseLayer<T>(_options.NumClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
        layers.Add(Classifier);
        // segm_init_weights: N(0, 0.02²) truncated at ±2, zero bias.
        var w = TextToSpeech.CodecBased.TorchParameters.Gaussian<T>(random, _options.NumClasses * _configuration.Dim, 0.02);
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, Classifier, _configuration.Dim, _options.NumClasses,
            w.ToArray().Select(v => NumOps.ToDouble(v)).ToArray(), new double[_options.NumClasses]);
        Layers.AddRange(layers);
    }

    /// <summary>Loads an official VisionMamba state dict (the network, then <c>head.weight</c> and <c>head.bias</c>).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        Network.LoadTorchWeights(read);
        int d = _configuration.Dim, classes = _options.NumClasses;
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, Classifier!, d, classes,
            read("head.weight", new[] { classes, d }), read("head.bias", new[] { classes }));
    }

    /// <summary>The class probabilities of an image <c>[C, H, W]</c> or a batch <c>[B, C, H, W]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        bool single = input.Rank == 3;
        if (input.Rank is not (3 or 4))
            throw new ArgumentException($"Input must be 3D [C,H,W] or 4D [batch,C,H,W], got {input.Rank}D.", nameof(input));
        var x = single ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }) : input;
        var features = Network.ClassFeatures(Network.Forward(x, IsTrainingMode, _trainingRandom));
        var output = Classifier!.Forward(features);
        return single ? Engine.Reshape(output, new[] { output.Shape[1] }) : output;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
        return true;
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "VisionMamba" },
                { "ImageHeight", _configuration.ImageHeight },
                { "ImageWidth", _configuration.ImageWidth },
                { "PatchSize", _configuration.PatchSize },
                { "Stride", _configuration.Stride },
                { "Channels", _configuration.Channels },
                { "ModelDimension", _configuration.Dim },
                { "NumLayers", _configuration.Depth },
                { "NumClasses", _options.NumClasses },
                { "StateDimension", _configuration.StateDim },
                { "NumPatches", _configuration.Patches },
                { "LayerCount", Layers.Count }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }
}
