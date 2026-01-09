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
public class DenseNetNetwork<T> : NeuralNetworkBase<T>
{
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
    /// Initializes a new instance of the <see cref="DenseNetNetwork{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The DenseNet-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
    public DenseNetNetwork(
        NeuralNetworkArchitecture<T> architecture,
        DenseNetConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(DenseNetNetwork<T>));

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
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
                { "NetworkType", "DenseNetNetwork" },
                { "Variant", _configuration.Variant.ToString() },
                { "GrowthRate", _configuration.GrowthRate },
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
            _configuration.CompressionFactor);

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
