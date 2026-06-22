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
/// Implements the DenseNet (Densely Connected Convolutional Network) architecture.
/// </summary>
/// <remarks>
/// <para>
/// DenseNet (Huang et al., 2017) connects each layer to every other layer in a
/// feed-forward fashion. This creates strong gradient flow and feature reuse,
/// enabling very deep networks with fewer parameters.
/// </para>
/// <para>
/// Architecture overview:
/// <code>
/// Input (3x224x224)
///   ↓
/// Stem: Conv 7x7, stride 2 → BN → ReLU → MaxPool 3x3, stride 2
///   ↓
/// Dense Block 1 (6 layers, k=32) → Transition 1
///   ↓
/// Dense Block 2 (12 layers, k=32) → Transition 2
///   ↓
/// Dense Block 3 (24 layers, k=32) → Transition 3
///   ↓
/// Dense Block 4 (16 layers, k=32)
///   ↓
/// BN → ReLU → Global Average Pool → FC (num_classes)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> DenseNet is designed to maximize information flow
/// through the network by connecting each layer directly to all subsequent layers.
///
/// Key innovations:
/// - Dense Connectivity: Each layer receives features from ALL previous layers
/// - Feature Reuse: Reduces redundant feature learning, fewer parameters
/// - Strong Gradient Flow: Direct connections help train very deep networks
/// - Compact Models: Can achieve similar accuracy with fewer parameters than ResNet
///
/// The "growth rate" (k) determines how many new feature maps each layer adds.
/// Typical values are 12, 24, or 32. Higher values increase capacity but also cost.
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
[ResearchPaper("Densely Connected Convolutional Networks", "https://arxiv.org/abs/1608.06993", Year = 2017, Authors = "Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger")]
public class DenseNetNetwork<T> : NeuralNetworkBase<T>
{
    private readonly DenseNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly DenseNetConfiguration _configuration;

    /// <summary>
    /// Gets the DenseNet variant.
    /// </summary>
    public DenseNetVariant Variant => _configuration.Variant;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _configuration.NumClasses;

    /// <summary>
    /// Gets the growth rate (k).
    /// </summary>
    public int GrowthRate => _configuration.GrowthRate;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DenseNetNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224, inputWidth: 224, inputDepth: 3,
            outputSize: 1000),
            configuration: DenseNetConfiguration.CreateDenseNet121(1000))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseNetNetwork{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The DenseNet-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam with lr=1e-4 and AMSGrad enabled via <c>new AdamOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(this, new AdamOptimizerOptions&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt; { InitialLearningRate = 1e-4, UseAMSGrad = true })</c>).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
    public DenseNetNetwork(
        NeuralNetworkArchitecture<T> architecture,
        DenseNetConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        DenseNetOptions? options = null)
        : base(architecture, lossFunction ?? GetDenseNetDefaultLoss(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new DenseNetOptions();
        Options = _options;
        Guard.NotNull(configuration);
        _configuration = configuration;

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(DenseNetNetwork<T>));

        // Issue #1393: switched default optimizer from vanilla Adam(lr=1e-3)
        // to AMSGrad-mode Adam(lr=1e-4) to stop the optimizer from drifting
        // past the converged point on small / memorization fixtures.
        //
        // Two compounding root causes in the prior default:
        //   1. lr=1e-3 was aggressive for DenseNet's dense-connectivity
        //      gradient paths — every parameter sees many backward routes,
        //      so the effective per-step update is amplified vs a plain
        //      sequential CNN. Lowering to 1e-4 (conventional CV-Adam base
        //      per Kingma & Ba 2014 §2.1) shrinks per-step magnitude.
        //   2. Vanilla Adam's per-parameter v̂ denominator can decay
        //      alongside gradients near convergence so the effective step
        //      size doesn't shrink in lockstep, leading to post-convergence
        //      drift. AMSGrad (Reddi, Kale, Kumar 2018) maintains a running
        //      v̂_max, guaranteeing the denominator can only grow —
        //      eliminating the drift.
        //
        // We tested issue's Option A (paper-faithful SGD + momentum 0.9 +
        // lr=0.1 per Huang 2017 §3) and plain Adam(1e-4); both still
        // overshot the single-sample memorization fixture
        // (Adam: 0.208→0.225; SGD: 0.177→0.225 short→long). AMSGrad-mode
        // Adam(1e-4) is the same configuration NeuralNetworkBase's
        // GetOrCreateBaseOptimizer hands out as the framework-wide tape
        // default, just made explicit on DenseNet so the public Train()
        // path benefits from the AMSGrad stability fix without going
        // through the tape-only path. Callers passing an explicit
        // optimizer are unaffected.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-4,
                UseAMSGrad = true
            });
        _lossFunction = lossFunction ?? GetDenseNetDefaultLoss(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Returns the appropriate loss function for DenseNet. DenseNet outputs raw logits
    /// (no softmax activation), so classification tasks use CrossEntropyWithLogitsLoss
    /// which applies LogSoftmax internally per Huang et al. 2017.
    /// </summary>
    private static ILossFunction<T> GetDenseNetDefaultLoss(NeuralNetworkTaskType taskType)
    {
        // Match each task type to the corresponding *WithLogits loss so the DenseNet
        // classification head's raw-logits output flows into a numerically stable loss
        // that applies the activation internally:
        //   * BinaryClassification — outputSize is typically 1, so softmax([x]) collapses
        //     to a constant 1 (zero loss). Use BinaryCrossEntropyWithLogitsLoss instead,
        //     which is the sigmoid+BCE fused form.
        //   * MultiLabelClassification — labels are independent (not mutually exclusive),
        //     so softmax cross-entropy is the wrong objective. BCE-with-logits per output
        //     is the standard multi-label loss.
        //   * MultiClassClassification — the only task type where softmax cross-entropy
        //     is correct (mutually exclusive classes).
        return taskType switch
        {
            NeuralNetworkTaskType.BinaryClassification     => new BinaryCrossEntropyWithLogitsLoss<T>(),
            NeuralNetworkTaskType.MultiClassClassification => new CrossEntropyWithLogitsLoss<T>(),
            NeuralNetworkTaskType.MultiLabelClassification => new BinaryCrossEntropyWithLogitsLoss<T>(),
            _ => NeuralNetworkHelper<T>.GetDefaultLossFunction(taskType),
        };
    }

    /// <summary>
    /// Creates a DenseNet-121 network.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A configured DenseNet-121 network.</returns>
    public static DenseNetNetwork<T> DenseNet121(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new DenseNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates a DenseNet-169 network.
    /// </summary>
    public static DenseNetNetwork<T> DenseNet169(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new DenseNetConfiguration(DenseNetVariant.DenseNet169, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new DenseNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates a DenseNet-201 network.
    /// </summary>
    public static DenseNetNetwork<T> DenseNet201(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new DenseNetConfiguration(DenseNetVariant.DenseNet201, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new DenseNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates a DenseNet-264 network.
    /// </summary>
    public static DenseNetNetwork<T> DenseNet264(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new DenseNetConfiguration(DenseNetVariant.DenseNet264, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new DenseNetNetwork<T>(architecture, config);
    }

    /// <summary>
    /// Creates a minimal DenseNet network optimized for fast test execution.
    /// </summary>
    /// <remarks>
    /// Uses [2, 2, 2, 2] block configuration with small growth rate (8) and 32x32 input,
    /// resulting in significantly fewer layers than standard variants.
    /// Construction time is typically under 50ms, compared to ~500ms for DenseNet-121.
    /// </remarks>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A minimal DenseNet network for testing.</returns>
    public static DenseNetNetwork<T> ForTesting(int numClasses = 10, int inputChannels = 3)
    {
        var config = DenseNetConfiguration.CreateForTesting(numClasses);
        var architecture = CreateArchitectureFromConfig(config);
        return new DenseNetNetwork<T>(architecture, config);
    }

    private static NeuralNetworkArchitecture<T> CreateArchitectureFromConfig(DenseNetConfiguration config)
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: config.InputChannels * config.InputHeight * config.InputWidth,
            inputHeight: config.InputHeight,
            inputWidth: config.InputWidth,
            inputDepth: config.InputChannels,
            outputSize: config.NumClasses,
            layers: null
        );
    }

    /// <inheritdoc />
    protected sealed override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use DenseNet-specific layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultDenseNetLayers(Architecture, _configuration));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The output class logits.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // DenseNet's conv / BatchNorm stack operates on batched images
        // [B, C, H, W]; the per-channel BatchNorm broadcasts its [1, C, 1] params
        // against the channel axis at dim 1, and the conv/dense-block layers
        // preserve the input rank. A single image supplied as rank-3 [C, H, W]
        // therefore propagates as rank-3 with channels at dim 0, so the BatchNorm
        // broadcast misaligns (e.g. [64, 32, 32] vs [1, 64, 1]). Add a leading
        // batch dim so the whole stack stays 4-D with channels at dim 1, matching
        // the standard image-model input contract.
        if (input.Rank == 3)
        {
            input = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        }

        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Add the batch dim if caller passed unbatched [C, H, W]. The
        // convolutional / BatchNorm stack consistently treats axis 0 as
        // the batch axis (channel index lives at axis 1), so a rank-3
        // input would be misread as [B=channelsExpanded, H, W] after
        // the first conv promoted channels — that's the
        // "[64, 32, 32] vs [1, 64, 1]" broadcast mismatch the
        // DenseNetNetwork_Predict_ProducesOutput test caught.
        bool addedBatch = false;
        if (input.Rank == 3)
        {
            input = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
            addedBatch = true;
        }

        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        // Strip the added unit-batch on the way out so the caller's
        // single-sample input contract is preserved.
        if (addedBatch && output.Rank >= 1 && output.Shape[0] == 1)
        {
            var squeezed = new int[output.Rank - 1];
            for (int d = 0; d < squeezed.Length; d++) squeezed[d] = output.Shape[d + 1];
            output = Engine.Reshape(output, squeezed);
        }
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        // #1622 verify-then-trust compiled gate; no-op unless acceleration is engaged.
        return Accelerate(input, () => Forward(input));
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainWithTape(input, expectedOutput, _optimizer);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "DenseNetNetwork" },
                { "Variant", _configuration.Variant.ToString() },
                { "GrowthRate", _configuration.GrowthRate },
                { "NumClasses", _configuration.NumClasses },
                { "InputShape", $"{_configuration.InputChannels}x{_configuration.InputHeight}x{_configuration.InputWidth}" },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_configuration.Variant);
        writer.Write(_configuration.InputChannels);
        writer.Write(_configuration.InputHeight);
        writer.Write(_configuration.InputWidth);
        writer.Write(_configuration.NumClasses);
        writer.Write(_configuration.GrowthRate);
        writer.Write(_configuration.CompressionFactor);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var variant = (DenseNetVariant)reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var inputHeight = reader.ReadInt32();
        var inputWidth = reader.ReadInt32();
        var numClasses = reader.ReadInt32();
        var growthRate = reader.ReadInt32();
        var compressionFactor = reader.ReadDouble();

        if (variant != _configuration.Variant ||
            inputChannels != _configuration.InputChannels ||
            inputHeight != _configuration.InputHeight ||
            inputWidth != _configuration.InputWidth ||
            numClasses != _configuration.NumClasses ||
            growthRate != _configuration.GrowthRate ||
            Math.Abs(compressionFactor - _configuration.CompressionFactor) > 0.001)
        {
            throw new InvalidDataException("Serialized DenseNet configuration does not match current configuration.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var config = new DenseNetConfiguration(
            _configuration.Variant,
            _configuration.NumClasses,
            _configuration.InputHeight,
            _configuration.InputWidth,
            _configuration.InputChannels,
            _configuration.GrowthRate,
            _configuration.CompressionFactor,
            _configuration.CustomBlockLayers);

        return new DenseNetNetwork<T>(Architecture, config, _optimizer, _lossFunction);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        return DeepCopy();
    }

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
