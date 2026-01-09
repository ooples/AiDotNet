using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
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
public class EfficientNetNetwork<T> : NeuralNetworkBase<T>
{
    private readonly ILossFunction<T> _lossFunction;
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
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
    public EfficientNetNetwork(
        NeuralNetworkArchitecture<T> architecture,
        EfficientNetConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(EfficientNetNetwork<T>));

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

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
        var config = EfficientNetConfiguration.CreateForTesting(numClasses);
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
            // Use EfficientNet-specific layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultEfficientNetLayers(Architecture, _configuration));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The output class logits.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;


        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Performs backward propagation through the network.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
        return gradient;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }

        _optimizer.UpdateParameters(Layers);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var widthCoeff = _configuration.GetWidthMultiplier();
        var depthCoeff = _configuration.GetDepthMultiplier();
        var resolution = _configuration.GetInputHeight();

        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConvolutionalNeuralNetwork,
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
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_configuration.Variant);
        writer.Write(_configuration.InputChannels);
        writer.Write(_configuration.NumClasses);
    }

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
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read serialized configuration values
        var variant = (EfficientNetVariant)reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var numClasses = reader.ReadInt32();

        // Validate configuration matches - layer structure depends on these values
        // and cannot be changed after construction
        if (variant != _configuration.Variant ||
            inputChannels != _configuration.InputChannels ||
            numClasses != _configuration.NumClasses)
        {
            throw new InvalidDataException(
                $"Serialized EfficientNet configuration (Variant={variant}, InputChannels={inputChannels}, " +
                $"NumClasses={numClasses}) does not match current configuration " +
                $"(Variant={_configuration.Variant}, InputChannels={_configuration.InputChannels}, " +
                $"NumClasses={_configuration.NumClasses}). Create a new network with matching configuration to load this model.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var config = new EfficientNetConfiguration(
            _configuration.Variant,
            _configuration.NumClasses,
            _configuration.InputChannels);

        return new EfficientNetNetwork<T>(Architecture, config, _optimizer, _lossFunction);
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
