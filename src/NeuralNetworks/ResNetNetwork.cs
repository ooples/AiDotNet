using AiDotNet.Optimizers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a ResNet (Residual Network) neural network architecture for image classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// ResNet networks are deep convolutional neural networks that introduced skip connections (residual
/// connections) to enable training of very deep networks. They learn residual functions with reference
/// to the layer inputs, rather than learning unreferenced functions directly.
/// </para>
/// <para>
/// <b>For Beginners:</b> ResNet networks revolutionized deep learning by solving the "vanishing gradient"
/// problem that made very deep networks hard to train. Key benefits include:
/// <list type="bullet">
/// <item>Can train networks with 100+ layers (compared to ~20 layers for earlier architectures)</item>
/// <item>Skip connections allow gradients to flow more easily during training</item>
/// <item>Each block learns the "residual" (difference) rather than the complete transformation</item>
/// <item>Winner of ImageNet 2015 competition with top-5 error of 3.57%</item>
/// </list>
/// </para>
/// <para>
/// <b>Architecture Variants:</b>
/// <list type="bullet">
/// <item><b>ResNet18/34:</b> Use BasicBlock (2 conv layers per block)</item>
/// <item><b>ResNet50/101/152:</b> Use BottleneckBlock (1x1-3x3-1x1 conv pattern) for efficiency</item>
/// </list>
/// </para>
/// <para>
/// <b>Typical Usage:</b>
/// <code>
/// // Create ResNet50 for 1000-class classification
/// var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 1000);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224,
///     inputWidth: 224,
///     inputDepth: 3,
///     outputSize: 1000,
///     taskType: NeuralNetworkTaskType.MultiClassClassification);
/// var network = new ResNetNetwork&lt;float&gt;(architecture, config);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Deep Residual Learning for Image Recognition", "https://arxiv.org/abs/1512.03385", Year = 2016, Authors = "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun")]
[PaperOptimizer(OptimizerKind.SgdMomentum, LearningRate = 0.1, Momentum = 0.9, WeightDecay = 1e-4, ReferenceBatchSize = 256,
                Schedule = LearningRateSchedulerType.ReduceOnPlateau,
                Source = "He et al. 2016, Sec. 3.4 (Implementation): SGD, lr 0.1 divided by 10 when the error plateaus, momentum 0.9, weight decay 1e-4.")]
public partial class ResNetNetwork<T> : ImageClassifierModelLayoutBase<T>
{
    private readonly ResNetOptions _options;

    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The ResNet configuration specifying the variant and parameters.
    /// </summary>
    private readonly ResNetConfiguration _configuration;

    /// <summary>
    /// Gets the ResNet variant being used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variant determines how deep the network is (ResNet18, 34, 50, 101, or 152)
    /// and which block type is used (BasicBlock for 18/34, BottleneckBlock for 50/101/152).
    /// </para>
    /// </remarks>
    public ResNetVariant Variant => _configuration.Variant;

    /// <summary>
    /// Gets whether this variant uses bottleneck blocks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ResNet50 and deeper use bottleneck blocks (1x1-3x3-1x1 convolutions)
    /// which are more parameter efficient than the basic blocks used in ResNet18/34.
    /// </para>
    /// </remarks>
    public bool UsesBottleneck => _configuration.UsesBottleneck;

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    public int NumClasses => _configuration.NumClasses;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public ResNetNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224, inputWidth: 224, inputDepth: 3,
            outputSize: 1000),
            configuration: ResNetConfiguration.CreateResNet50(1000))
    {
    }

    /// <summary>
    /// Initializes a new instance of the ResNetNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The ResNet-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <exception cref="InvalidInputTypeException">Thrown when the input type is not three-dimensional.</exception>
    /// <exception cref="ArgumentNullException">Thrown when configuration is null.</exception>
    /// <remarks>
    /// <para>
    /// ResNet networks require three-dimensional input data (channels, height, width).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a ResNet network, you need to provide:
    /// <list type="bullet">
    /// <item>An architecture object that describes the input/output dimensions</item>
    /// <item>A configuration that specifies which ResNet variant to use</item>
    /// <item>Optionally, custom optimizer and loss function (good defaults are provided)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public ResNetNetwork(NeuralNetworkArchitecture<T> architecture,
        ResNetConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ResNetOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new ResNetOptions()).MaxGradNorm)
    {
        _options = options;
        Options = _options;
        Guard.NotNull(configuration);
        _configuration = configuration;

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(ResNetNetwork<T>));

        // Validate that architecture matches configuration
        ValidateArchitectureMatchesConfiguration(architecture, configuration);

        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Validates that the architecture parameters match the configuration.
    /// </summary>
    private static void ValidateArchitectureMatchesConfiguration(
        NeuralNetworkArchitecture<T> architecture,
        ResNetConfiguration configuration)
    {
        if (architecture.OutputSize != configuration.NumClasses)
        {
            throw new ArgumentException(
                $"Architecture output size ({architecture.OutputSize}) does not match " +
                $"configuration NumClasses ({configuration.NumClasses}).",
                nameof(architecture));
        }

        var inputShape = architecture.GetInputShape();
        if (inputShape[0] != configuration.InputChannels ||
            inputShape[1] != configuration.InputHeight ||
            inputShape[2] != configuration.InputWidth)
        {
            throw new ArgumentException(
                $"Architecture input shape [{inputShape[0]}, {inputShape[1]}, {inputShape[2]}] " +
                $"does not match configuration input shape " +
                $"[{configuration.InputChannels}, {configuration.InputHeight}, {configuration.InputWidth}].",
                nameof(architecture));
        }
    }

    /// <summary>
    /// Creates a minimal ResNet network optimized for fast test execution.
    /// </summary>
    /// <remarks>
    /// Uses ResNet18 (smallest variant) with 32x32 input resolution,
    /// resulting in significantly fewer layers than standard variants.
    /// Construction time is typically under 50ms.
    /// </remarks>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A minimal ResNet network for testing.</returns>
    public static ResNetNetwork<T> ForTesting(int numClasses = 10, int inputChannels = 3)
    {
        // Test-scale configuration is GENERATED. ModelTestScale synthesizes it from the
        // constructor -- smallest variant, bounded dims, declared defaults left alone --
        // and numClasses is passed through because it is the caller's intent, not a size
        // the generator can infer.
        var config = AiDotNet.Testing.ModelTestScale.CreateBoundedOptions(
                typeof(ResNetConfiguration),
                new global::System.Collections.Generic.Dictionary<string, int> { ["numClasses"] = numClasses })
            as ResNetConfiguration
            ?? throw new global::System.InvalidOperationException(
                "ResNetConfiguration has no generated test-scale bound.");
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: inputChannels * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: inputChannels,
            outputSize: numClasses,
            layers: null
        );
        return new ResNetNetwork<T>(architecture, config);
    }

    /// <summary>The convolutional trunk, shared with the detection backbone.</summary>
    internal ResNetTrunk<T> Trunk => _trunk ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The fully connected classifier over the pooled C5 features, when the configuration includes one.</summary>
    internal DenseLayer<T>? Classifier { get; private set; }

    private ResNetTrunk<T>? _trunk;

    /// <summary>
    /// Builds the network: the shared ResNet trunk (stem, then the four stages conv2_x to conv5_x), global average
    /// pooling, and the fully connected classifier, initialized as torchvision initializes them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds the ResNet network layer by layer:
    /// <list type="number">
    /// <item>Initial 7x7 convolution with stride 2, BatchNorm and ReLU</item>
    /// <item>Max pooling 3x3 with stride 2</item>
    /// <item>Four stages of residual blocks (conv2_x through conv5_x)</item>
    /// <item>Global average pooling</item>
    /// <item>Fully connected classification layer</item>
    /// </list>
    /// </para>
    /// </remarks>
    protected sealed override void InitializeLayers()
    {
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                $"{nameof(ResNetNetwork<T>)} builds the paper's network from its {nameof(ResNetConfiguration)}; custom layers are not supported.");
        var config = _configuration;
        var layers = new List<LayerBase<T>>();
        _trunk = new ResNetTrunk<T>(Engine, layers, config.Trunk);
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42);
        _trunk.InitializeLikeReference(random);
        if (config.IncludeClassifier)
        {
            var outputActivation = Architecture.TaskType == NeuralNetworkTaskType.BinaryClassification
                ? (IActivationFunction<T>)new ActivationFunctions.SigmoidActivation<T>()
                : new ActivationFunctions.SoftmaxActivation<T>();
            Classifier = new DenseLayer<T>(config.NumClasses, outputActivation);
            layers.Add(Classifier);
            // nn.Linear's default: U(±1/√in) for the weight and the bias.
            int inputs = config.Trunk.OutputChannels[3];
            double bound = 1.0 / Math.Sqrt(inputs);
            var w = new double[config.NumClasses * inputs];
            var b = new double[config.NumClasses];
            for (int i = 0; i < w.Length; i++) w[i] = (2 * random.NextDouble() - 1) * bound;
            for (int i = 0; i < b.Length; i++) b[i] = (2 * random.NextDouble() - 1) * bound;
            TextToSpeech.CodecBased.TorchParameters.Linear(Engine, Classifier, inputs, config.NumClasses, w, b);
        }
        Layers.AddRange(layers);
    }

    /// <summary>Loads a torchvision ResNet state dict (the trunk, then <c>fc.weight</c> and <c>fc.bias</c>).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        Trunk.LoadTorchWeights(read);
        if (Classifier is null) return;
        int inputs = _configuration.Trunk.OutputChannels[3], classes = _configuration.NumClasses;
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, Classifier, inputs, classes,
            read("fc.weight", new[] { classes, inputs }), read("fc.bias", new[] { classes }));
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    /// <inheritdoc />
    /// <remarks>Every layer is sized and initialized at construction (each convolution knows its input channels), and
    /// <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain: a projection shortcut reads its block's input, not the
    /// layer before it. Walking the list as a chain would give those convolutions the wrong input depth.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);
    /// <summary>
    /// Performs a forward pass through the ResNet network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process (shape: [channels, height, width] for a single example, or [batch, channels, height, width] for a batch).</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <exception cref="TensorShapeMismatchException">Thrown when the input shape doesn't match expected shape.</exception>
    /// <remarks>
    /// <para>
    /// The forward pass sequentially processes the input through each layer of the network:
    /// initial conv, residual blocks, global pooling, and classification layer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network makes predictions. You give it an image
    /// (as a tensor), and it processes it through all the ResNet layers to produce a prediction.
    /// The output contains probabilities for each class.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var expectedShape = Architecture.GetInputShape();

        // Validate input shape - accept both 3D [C,H,W] and 4D [B,C,H,W]
        bool addedBatch = false;
        Tensor<T> processedInput;

        if (input.Rank == 3)
        {
            // 3D input: validate against expected shape directly
            TensorValidator.ValidateShape(
                input,
                expectedShape,
                nameof(ResNetNetwork<T>),
                "forward pass");
            addedBatch = true;
            processedInput = PromoteToBatchedTensor(input);
        }
        else if (input.Rank == 4)
        {
            // 4D input: validate the non-batch dimensions [C,H,W] match expected shape
            var actualNonBatch = new int[] { input.Shape[1], input.Shape[2], input.Shape[3] };
            if (!actualNonBatch.SequenceEqual(expectedShape))
            {
                throw new TensorShapeMismatchException(
                    $"Shape mismatch in ResNetNetwork during forward pass: Expected non-batch dimensions [{string.Join(", ", expectedShape)}], but got [{string.Join(", ", actualNonBatch)}].");
            }
            processedInput = input;
        }
        else
        {
            throw new TensorShapeMismatchException(
                $"Shape mismatch in ResNetNetwork during forward pass: Expected 3D [C,H,W] or 4D [B,C,H,W] input, but got rank {input.Rank}.");
        }

        // The C5 map, globally average-pooled to [B, C], then the classifier.
        var c5 = Trunk.Forward(processedInput)[3];
        var output = Engine.ReduceMean(c5, new[] { 2, 3 }, false);
        if (Classifier is not null) output = Classifier.Forward(output);

        // Remove batch dimension if we added it
        if (addedBatch && output.Rank == 2 && output.Shape[0] == 1)
        {
            output = output.Reshape([output.Shape[1]]);
        }

        return output;
    }


    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Makes a prediction using the ResNet network for the given input.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor containing class probabilities.</returns>
    /// <summary>
    /// Routes inference through <see cref="NeuralNetworkBase{T}.PredictCompiled"/> so the
    /// forward pass gets traced and replayed by <c>CompiledModelHost</c> after warmup —
    /// matching PyTorch's <c>torch.compile</c> default. The eager forward is
    /// <see cref="Forward"/>.
    /// </summary>
    protected override Tensor<T> PredictEager(Tensor<T> input) => Forward(input);

    // Hotfix: bypass the compiled-replay path in NeuralNetworkBase.Predict.
    // Forward has shape-conditional control flow (rank-3 → rank-4 batch
    // promotion and the trailing Reshape that strips the synthetic batch
    // dim) that the tracer does not capture correctly — routing through
    // PredictCompiled returned an intermediate feature-map shape instead of
    // the final logits, and also cached the first input so Predict(x2)
    // returned Predict(x1)'s output. Tracked upstream at
    // ooples/AiDotNet.Tensors#228; remove this override once that lands.
    //
    // Also force eval mode here (matches PyTorch model.eval()). The base
    // class defaults IsTrainingMode=true at construction, which would let
    // BatchNorm use batch stats instead of the (zero-init) running stats
    // and produce non-deterministic Predict output across calls.
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        SetTrainingMode(false);
        // #1622 verify-then-trust compiled gate; no-op unless acceleration is engaged.
        return Accelerate(input, () => Forward(input));
    }

    /// <inheritdoc/>
    /// <remarks>A single <c>[C, H, W]</c> image is batched for training; its target is batched with it when it is a label vector.</remarks>
    protected override (Tensor<T> Input, Tensor<T>? Target) PrepareTrainingPair(Tensor<T> input, Tensor<T>? target, long draw)
    {
        if (target is null) return (input.Rank == 3 ? PromoteToBatchedTensor(input) : input, null);
        var (batchedInput, batchedTarget) = EnsureBatchForCnnTraining(input, target);
        return (batchedInput, batchedTarget);
    }
    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    /// <returns>The total parameter count.</returns>
    /// <remarks>
    /// <para>
    /// ResNet networks have fewer parameters than VGG despite being deeper:
    /// <list type="bullet">
    /// <item>ResNet18: ~11.7 million parameters</item>
    /// <item>ResNet34: ~21.8 million parameters</item>
    /// <item>ResNet50: ~25.6 million parameters</item>
    /// <item>ResNet101: ~44.5 million parameters</item>
    /// <item>ResNet152: ~60.2 million parameters</item>
    /// </list>
    /// </para>
    /// </remarks>
    public new long GetParameterCount()
    {
        return base.GetParameterCount();
    }

    /// <summary>
    /// Retrieves metadata about the ResNet network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "ResNet" },
                { "Variant", _configuration.Variant.ToString() },
                { "NumClasses", _configuration.NumClasses },
                { "UsesBottleneck", _configuration.UsesBottleneck },
                { "BlockCounts", _configuration.BlockCounts },
                { "Expansion", _configuration.Expansion },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Layers.Count > 0 ? Layers[Layers.Count - 1].GetOutputShape() : Array.Empty<int>() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "NumConvLayers", _configuration.NumConvLayers },
                { "NumWeightLayers", _configuration.NumWeightLayers },
                { "ZeroInitResidual", _configuration.ZeroInitResidual }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes ResNet network-specific data to a binary writer.
    /// </summary>


    /// <summary>
    /// Deserializes ResNet network-specific data from a binary reader.
    /// </summary>

}
