using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements the EfficientNet architecture with compound scaling.
/// </summary>
/// <remarks>
/// <para>
/// EfficientNet (Tan &amp; Le, 2019) introduced compound scaling, which uniformly scales
/// network width, depth, and resolution using a principled approach. This achieves
/// state-of-the-art accuracy with significantly fewer parameters than previous models.
/// </para>
/// <para>
/// Architecture overview (EfficientNet-B0 baseline):
/// <code>
/// Input (3x224x224)
///   ↓
/// Stem: Conv 3x3, 32, stride 2 → BN → Swish
///   ↓
/// Stage 1: MBConv1 (k3, c16, n1, s1, SE)
///   ↓
/// Stage 2: MBConv6 (k3, c24, n2, s2, SE)
///   ↓
/// Stage 3: MBConv6 (k5, c40, n2, s2, SE)
///   ↓
/// Stage 4: MBConv6 (k3, c80, n3, s2, SE)
///   ↓
/// Stage 5: MBConv6 (k5, c112, n3, s1, SE)
///   ↓
/// Stage 6: MBConv6 (k5, c192, n4, s2, SE)
///   ↓
/// Stage 7: MBConv6 (k3, c320, n1, s1, SE)
///   ↓
/// Head: Conv 1x1, 1280 → BN → Swish → GlobalAvgPool → FC
/// </code>
/// Where k=kernel size, c=output channels, n=num layers, s=stride.
/// </para>
/// <para>
/// <b>For Beginners:</b> EfficientNet achieves excellent accuracy while being very efficient.
///
/// Key innovations:
/// - Compound Scaling: Balances network width, depth, and resolution together
/// - MBConv blocks: Mobile Inverted Bottleneck with Squeeze-and-Excitation
/// - Swish activation: Smooth, self-gated activation function (x * sigmoid(x))
/// - Neural Architecture Search (NAS): The baseline B0 was found via automated search
///
/// The scaling philosophy: increasing only one dimension (width/depth/resolution)
/// quickly saturates accuracy. Compound scaling increases all three proportionally.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "https://arxiv.org/abs/1905.11946", Year = 2019, Authors = "Mingxing Tan, Quoc V. Le")]
[PaperOptimizer(OptimizerKind.RmsProp, Rho = 0.9, Momentum = 0.9, Epsilon = 1e-3, LearningRate = 0.256, ReferenceBatchSize = 4096,
                Schedule = LearningRateSchedulerType.Exponential, DecayRate = 0.9873893,
                ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                WeightDecay = 1e-5,
                Source = "Tan and Le 2019, Sec. 5.2: RMSProp with decay 0.9 and momentum 0.9, weight decay 1e-5, and an initial learning rate of 0.256 decaying by 0.97 every 2.4 epochs. The batch size (4096) and the epsilon (1e-3) are the official TPU code's (tpu/models/official/efficientnet, after MnasNet), which scales the rate linearly with the batch. The decay is applied per epoch at 0.97^(1/2.4), the same decay over every 2.4 epochs without the official staircase.")]
public partial class EfficientNetNetwork<T> : ImageClassifierModelLayoutBase<T>
{
    private readonly EfficientNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly EfficientNetConfiguration _configuration;

    /// <summary>
    /// Gets the EfficientNet variant.
    /// </summary>
    public EfficientNetVariant Variant => _configuration.Variant;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _configuration.NumClasses;

    /// <summary>
    /// Gets the input resolution for this variant.
    /// </summary>
    public int InputResolution => _configuration.GetInputHeight();

    /// <summary>
    /// Initializes a new instance of the <see cref="EfficientNetNetwork{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The EfficientNet-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public EfficientNetNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224, inputWidth: 224, inputDepth: 3,
            outputSize: 1000),
            configuration: EfficientNetConfiguration.CreateB0(1000))
    {
    }

    public EfficientNetNetwork(NeuralNetworkArchitecture<T> architecture,
        EfficientNetConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        EfficientNetOptions? options = null)
        // CategoricalCrossEntropyLoss (the MultiClassClassification default) expects softmax
        // PROBABILITIES. The classification head ends in a SoftmaxActivation DenseLayer (per
        // CreateDefaultEfficientNetLayers, matching Tan & Le 2019 §3 and the codebase classifier
        // convention used by VGG et al.), so loss and head are consistent. The previous head used
        // IdentityActivation (raw logits) against this probability loss, producing a wrong gradient
        // (differentiated as if the logits were probabilities) that diverged training
        // (Training_ShouldReduceLoss saw 0.33 -> 7094).
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new EfficientNetOptions()).MaxGradNorm)
    {
        _options = options;
        Options = _options;
        Guard.NotNull(configuration);
        _configuration = configuration;

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(EfficientNetNetwork<T>));

        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("EfficientNet declares its paper recipe.");
        // The official code trains with TensorFlow 1's tf.train.RMSPropOptimizer, whose mean-square accumulator starts
        // at 1 and whose epsilon sits inside the root. Starting it at 0 divides the first steps by a near-zero root and
        // moves every weight by about 3·lr, which the paper's rate does not survive.
        if (optimizer is null && _optimizer.GetOptions() is RootMeanSquarePropagationOptimizerOptions<T, Tensor<T>, Tensor<T>> rmsProp)
        {
            rmsProp.InitialAccumulatorValue = 1.0;
            rmsProp.EpsilonInsideRoot = true;
        }

        InitializeLayers();
    }

    /// <summary>
    /// Creates an EfficientNet-B0 network (baseline model).
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A configured EfficientNet-B0 network.</returns>
    public static EfficientNetNetwork<T> EfficientNetB0(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B1 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB1(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B1, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B2 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB2(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B2, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B3 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB3(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B3, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B4 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB4(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B4, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B5 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB5(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B5, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B6 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB6(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B6, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates an EfficientNet-B7 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB7(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B7, numClasses, inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates a minimal EfficientNet network optimized for fast test execution.
    /// </summary>
    /// <remarks>
    /// Uses 32x32 input resolution with 1.0 width/depth multipliers,
    /// resulting in significantly fewer layers than standard variants.
    /// Construction time is typically under 50ms, compared to hundreds of ms for B0.
    /// </remarks>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A minimal EfficientNet network for testing.</returns>
    public static EfficientNetNetwork<T> ForTesting(int numClasses = 10, int inputChannels = 3)
    {
        // Test-scale configuration is GENERATED. ModelTestScale synthesizes it from the
        // constructor -- smallest variant, bounded dims, declared defaults left alone --
        // and numClasses is passed through because it is the caller's intent, not a size
        // the generator can infer.
        var config = AiDotNet.Testing.ModelTestScale.CreateBoundedOptions(
                typeof(EfficientNetConfiguration),
                new global::System.Collections.Generic.Dictionary<string, int> { ["numClasses"] = numClasses })
            as EfficientNetConfiguration
            ?? throw new global::System.InvalidOperationException(
                "EfficientNetConfiguration has no generated test-scale bound.");
        var architecture = CreateArchitectureFromConfig(config);
        return new EfficientNetNetwork<T>(architecture, config);
    }

    private static NeuralNetworkArchitecture<T> CreateArchitectureFromConfig(EfficientNetConfiguration config)
    {
        var resolution = config.GetInputHeight();
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: config.InputChannels * resolution * resolution,
            inputHeight: resolution,
            inputWidth: resolution,
            inputDepth: config.InputChannels,
            outputSize: config.NumClasses,
            layers: null
        );
    }

    /// <summary>The convolutional trunk, shared with the detection backbone.</summary>
    internal EfficientNetTrunk<T> Trunk => _trunk ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The fully connected classifier over the pooled head features.</summary>
    internal DenseLayer<T>? Classifier { get; private set; }

    private EfficientNetTrunk<T>? _trunk;

    /// <summary>Draws the stochastic-depth and dropout masks during training.</summary>
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    /// <summary>
    /// Builds the network: the shared EfficientNet trunk (stem, seven MBConv stages and the head convolution), global
    /// average pooling, dropout and the fully connected classifier, initialized as the official code initializes them.
    /// </summary>
    protected sealed override void InitializeLayers()
    {
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                $"{nameof(EfficientNetNetwork<T>)} builds the paper's network from its {nameof(EfficientNetConfiguration)}; custom layers are not supported.");
        var config = _configuration;
        var layers = new List<LayerBase<T>>();
        _trunk = new EfficientNetTrunk<T>(Engine, layers, new EfficientNetTrunkConfiguration(
            config.GetWidthMultiplier(), config.GetDepthMultiplier(), config.InputChannels, config.DropConnectRate));
        int seed = _options.Seed ?? 42;
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        _trunk.InitializeLikeReference(random);
        var outputActivation = Architecture.TaskType == NeuralNetworkTaskType.BinaryClassification
            ? (IActivationFunction<T>)new SigmoidActivation<T>()
            : new SoftmaxActivation<T>();
        Classifier = new DenseLayer<T>(config.NumClasses, outputActivation);
        layers.Add(Classifier);
        // The official dense initializer: U(±1/√out) for the weight, a zero bias.
        int inputs = _trunk.Configuration.HeadChannels;
        double bound = 1.0 / Math.Sqrt(config.NumClasses);
        var w = new double[config.NumClasses * inputs];
        for (int i = 0; i < w.Length; i++) w[i] = (2 * random.NextDouble() - 1) * bound;
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, Classifier, inputs, config.NumClasses, w, new double[config.NumClasses]);
        Layers.AddRange(layers);
    }

    /// <summary>
    /// The official code's weight decay skips batch normalization: its L2 loss sums every trainable variable whose name
    /// does not contain "batch_normalization" (tpu/models/official/efficientnet/main.py).
    /// </summary>
    protected override bool ExcludesFromWeightDecay(ILayer<T> layer) => layer is BatchNormalizationLayer<T>;

    /// <summary>Loads a timm <c>tf_efficientnet</c> state dict (the trunk, then <c>classifier.weight</c> and
    /// <c>classifier.bias</c>).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        Trunk.LoadTorchWeights(read);
        int inputs = Trunk.Configuration.HeadChannels, classes = _configuration.NumClasses;
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, Classifier!, inputs, classes,
            read("classifier.weight", new[] { classes, inputs }), read("classifier.bias", new[] { classes }));
    }

    /// <summary>
    /// The class probabilities of an image <c>[C, H, W]</c> or a batch <c>[B, C, H, W]</c>: the trunk, global average
    /// pooling, dropout (training only) and the classifier.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        bool single = input.Rank == 3;
        var x = single ? PromoteToBatchedTensor(input) : input;
        var head = Trunk.Forward(x, IsTrainingMode, _trainingRandom)[^1];
        var pooled = Engine.ReduceMean(head, new[] { 2, 3 }, false);
        double rate = _configuration.GetDropoutRate();
        if (IsTrainingMode && rate > 0)
            pooled = TextToSpeech.CodecBased.T5Seq2Seq<T>.Dropout(Engine, pooled, rate, _trainingRandom);
        var output = Classifier!.Forward(pooled);
        return single ? Engine.Reshape(output, new[] { output.Shape[1] }) : output;
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <inheritdoc />
    /// <remarks>Every layer is sized and initialized at construction, and <see cref="NeuralNetworkBase{T}.Layers"/> is
    /// not a chain (an excitation layer reads its block's pooled features), so there is nothing to walk.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }
    /// <inheritdoc />
    /// <summary>
    /// Routes inference through <see cref="NeuralNetworkBase{T}.PredictCompiled"/> for
    /// compiled-plan replay; <see cref="Forward"/> remains the eager fallback.
    /// </summary>
    protected override Tensor<T> PredictEager(Tensor<T> input) => Forward(input);

    /// <inheritdoc />
    /// <remarks>
    /// EfficientNet's stem-and-blocks pipeline expects rank-4 <c>[B, C, H, W]</c>
    /// per Tan &amp; Le 2019 §3 (input is a 4D image tensor). When the test
    /// harness or a CNN-style caller hands us a single-sample rank-3
    /// <c>[C, H, W]</c> tensor — and a target shaped to the per-sample output —
    /// we promote both to a leading batch dim so downstream Conv/BN/SE blocks
    /// see the canonical 4D shape and the loss target's rank matches the
    /// promoted prediction. Without this, Conv2D treats <c>C</c> as the
    /// batch axis and produces <c>[1280, NumClasses]</c> instead of
    /// <c>[1, NumClasses]</c>, breaking
    /// <see cref="LossFunctionBase{T}.EnsureTargetMatchesPredicted"/>.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Mirror the SetTrainingMode wrapper that NeuralNetworkBase.Train
        // applies — without this, BatchNorm / Dropout layers stay in
        // inference mode during training and either skip running-stat
        // updates or produce non-stochastic forward output, breaking
        // training semantics.
        SetTrainingMode(true);
        try
        {
            var (processedInput, processedTarget) = EnsureBatchForCnnTraining(input, expectedOutput);
            TrainWithTape(processedInput, processedTarget, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// EfficientNet's stem-and-blocks pipeline expects rank-4 [B, C, H, W]
    /// per Tan &amp; Le 2019 §3 — same constraint as
    /// <see cref="Train"/>'s EnsureBatchForCnnTraining promotion. The
    /// base <see cref="NeuralNetworkBase{T}.GetNamedLayerActivations"/>
    /// walks Layers without promotion, so a single-sample rank-3
    /// [C, H, W] input (the model-family test base's default for vision
    /// models) hit the SE block's broadcast multiply with mismatched
    /// rank-3 shapes — InvertedResidualBlock's SE branch global-pools
    /// to [B, C, 1, 1] and broadcasts back against [B, C, H, W];
    /// without batch promotion the rank-3 input collapsed into
    /// [1280, 2, 2] vs an SE output of [1, 1280, 1] which can't
    /// broadcast (#1224 Cluster F). Promote rank-3 → rank-4 so the
    /// pipeline sees the canonical NCHW shape it was built for.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var widthCoeff = _configuration.GetWidthMultiplier();
        var depthCoeff = _configuration.GetDepthMultiplier();
        var resolution = _configuration.GetInputHeight();

        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "EfficientNetNetwork" },
                { "Variant", _configuration.Variant.ToString() },
                { "WidthCoefficient", widthCoeff },
                { "DepthCoefficient", depthCoeff },
                { "Resolution", resolution },
                { "NumClasses", _configuration.NumClasses },
                { "InputShape", $"{_configuration.InputChannels}x{resolution}x{resolution}" },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc />


    /// <summary>
    /// Deserializes and validates network-specific configuration data.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <exception cref="InvalidDataException">
    /// Thrown when the serialized configuration does not match the current instance's configuration.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method performs validation-only deserialization. The serialized configuration values
    /// are read and compared against the current instance's configuration to ensure compatibility.
    /// </para>
    /// <para>
    /// <b>Design rationale:</b> The network's layer structure is created during construction based
    /// on the configuration. Changing the configuration during deserialization would not recreate
    /// the layers, leading to an inconsistent state. Therefore, deserialization requires that the
    /// target instance was created with a matching configuration.
    /// </para>
    /// <para>
    /// To load a model with a different configuration, create a new network instance with the
    /// desired configuration, then call <see cref="NeuralNetworkBase{T}.Load"/> on that instance.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Gets the layer at the specified index.
    /// </summary>
    public ILayer<T> GetLayer(int index)
    {
        if (index < 0 || index >= Layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {Layers.Count - 1}.");
        }
        return Layers[index];
    }
}
