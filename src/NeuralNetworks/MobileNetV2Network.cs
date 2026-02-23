using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements the MobileNetV2 architecture for efficient mobile inference.
/// </summary>
/// <remarks>
/// <para>
/// MobileNetV2 (Sandler et al., 2018) introduced the inverted residual structure with
/// linear bottlenecks, making it highly efficient for mobile and embedded vision applications.
/// </para>
/// <para>
/// Architecture overview:
/// <code>
/// Input (3x224x224)
///   ↓
/// Conv 3x3, 32, stride 2 → BN → ReLU6
///   ↓
/// InvertedResidual (t=1, c=16, n=1, s=1)
///   ↓
/// InvertedResidual (t=6, c=24, n=2, s=2)
///   ↓
/// InvertedResidual (t=6, c=32, n=3, s=2)
///   ↓
/// InvertedResidual (t=6, c=64, n=4, s=2)
///   ↓
/// InvertedResidual (t=6, c=96, n=3, s=1)
///   ↓
/// InvertedResidual (t=6, c=160, n=3, s=2)
///   ↓
/// InvertedResidual (t=6, c=320, n=1, s=1)
///   ↓
/// Conv 1x1, 1280 → BN → ReLU6
///   ↓
/// Global Average Pool (1x1)
///   ↓
/// FC (num_classes)
/// </code>
/// Where t=expansion, c=output channels, n=repeat count, s=stride.
/// </para>
/// <para>
/// <b>For Beginners:</b> MobileNetV2 is designed to be efficient on mobile devices.
///
/// Key innovations:
/// - Inverted Residuals: Expand → Depthwise Conv → Project (opposite of traditional bottlenecks)
/// - Linear Bottlenecks: No activation after the projection layer (preserves information)
/// - ReLU6: Activation capped at 6 for better quantization on mobile devices
/// - Depthwise Separable Convolutions: Much fewer parameters than standard convolutions
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MobileNetV2Network<T> : NeuralNetworkBase<T>
{
    private readonly MobileNetV2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly MobileNetV2Configuration _configuration;

    /// <summary>
    /// Gets the width multiplier used by this network.
    /// </summary>
    public MobileNetV2WidthMultiplier WidthMultiplier => _configuration.WidthMultiplier;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _configuration.NumClasses;

    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV2Network{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The MobileNetV2-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
    public MobileNetV2Network(
        NeuralNetworkArchitecture<T> architecture,
        MobileNetV2Configuration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        MobileNetV2Options? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MobileNetV2Options();
        Options = _options;
        Guard.NotNull(configuration);
        _configuration = configuration;

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(MobileNetV2Network<T>));

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 1.0.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A configured MobileNetV2 network.</returns>
    public static MobileNetV2Network<T> MobileNetV2_100(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha100, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV2Network<T>(architecture, config);
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 0.35 (smallest).
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_035(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha035, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV2Network<T>(architecture, config);
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 0.5.
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_050(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha050, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV2Network<T>(architecture, config);
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 0.75.
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_075(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha075, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV2Network<T>(architecture, config);
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 1.3.
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_130(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha130, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV2Network<T>(architecture, config);
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 1.4 (largest).
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_140(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha140, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV2Network<T>(architecture, config);
    }

    private static NeuralNetworkArchitecture<T> CreateArchitectureFromConfig(MobileNetV2Configuration config)
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
            // Use MobileNetV2-specific layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultMobileNetV2Layers(Architecture, _configuration));
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
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConvolutionalNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "MobileNetV2Network" },
                { "WidthMultiplier", _configuration.Alpha },
                { "NumClasses", _configuration.NumClasses },
                { "InputShape", $"{_configuration.InputChannels}x{_configuration.InputHeight}x{_configuration.InputWidth}" },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_configuration.WidthMultiplier);
        writer.Write(_configuration.InputChannels);
        writer.Write(_configuration.InputHeight);
        writer.Write(_configuration.InputWidth);
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
        var widthMultiplier = (MobileNetV2WidthMultiplier)reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var inputHeight = reader.ReadInt32();
        var inputWidth = reader.ReadInt32();
        var numClasses = reader.ReadInt32();

        // Validate configuration matches - layer structure depends on these values
        // and cannot be changed after construction
        if (widthMultiplier != _configuration.WidthMultiplier ||
            inputChannels != _configuration.InputChannels ||
            inputHeight != _configuration.InputHeight ||
            inputWidth != _configuration.InputWidth ||
            numClasses != _configuration.NumClasses)
        {
            throw new InvalidDataException(
                $"Serialized MobileNetV2 configuration (WidthMultiplier={widthMultiplier}, InputChannels={inputChannels}, " +
                $"InputHeight={inputHeight}, InputWidth={inputWidth}, NumClasses={numClasses}) does not match current configuration " +
                $"(WidthMultiplier={_configuration.WidthMultiplier}, InputChannels={_configuration.InputChannels}, " +
                $"InputHeight={_configuration.InputHeight}, InputWidth={_configuration.InputWidth}, " +
                $"NumClasses={_configuration.NumClasses}). Create a new network with matching configuration to load this model.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var config = new MobileNetV2Configuration(
            _configuration.WidthMultiplier,
            _configuration.NumClasses,
            _configuration.InputHeight,
            _configuration.InputWidth,
            _configuration.InputChannels);

        return new MobileNetV2Network<T>(Architecture, config, _optimizer, _lossFunction);
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
