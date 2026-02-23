using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements the MobileNetV3 architecture for efficient mobile inference.
/// </summary>
/// <remarks>
/// <para>
/// MobileNetV3 (Howard et al., 2019) builds on MobileNetV2 with three key improvements:
/// 1. Hard-Swish activation: x * min(max(0, x+3), 6) / 6 - computationally efficient
/// 2. Squeeze-and-Excitation blocks: Adaptive channel weighting
/// 3. Efficient network head: Reduced computational cost in final layers
/// </para>
/// <para>
/// <b>For Beginners:</b> MobileNetV3 is the latest in the MobileNet family, optimized for
/// both accuracy and latency on mobile devices.
///
/// Key innovations over V2:
/// - Hard-Swish: A faster activation function that works better with quantization
/// - SE blocks: Helps the network learn which channels are most important
/// - Network search: The architecture was found using neural architecture search (NAS)
/// - Two variants: "Large" for higher accuracy, "Small" for extreme efficiency
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MobileNetV3Network<T> : NeuralNetworkBase<T>
{
    private readonly MobileNetV3Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly MobileNetV3Configuration _configuration;

    /// <summary>
    /// Gets the MobileNetV3 variant used by this network.
    /// </summary>
    public MobileNetV3Variant Variant => _configuration.Variant;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _configuration.NumClasses;

    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV3Network{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The MobileNetV3-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
    public MobileNetV3Network(
        NeuralNetworkArchitecture<T> architecture,
        MobileNetV3Configuration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        MobileNetV3Options? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MobileNetV3Options();
        Options = _options;
        Guard.NotNull(configuration);
        _configuration = configuration;

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(MobileNetV3Network<T>));

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a MobileNetV3-Large network.
    /// </summary>
    public static MobileNetV3Network<T> MobileNetV3Large(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV3Configuration(MobileNetV3Variant.Large, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV3Network<T>(architecture, config);
    }

    /// <summary>
    /// Creates a MobileNetV3-Small network.
    /// </summary>
    public static MobileNetV3Network<T> MobileNetV3Small(int numClasses = 1000, int inputChannels = 3)
    {
        var config = new MobileNetV3Configuration(MobileNetV3Variant.Small, numClasses, inputChannels: inputChannels);
        var architecture = CreateArchitectureFromConfig(config);
        return new MobileNetV3Network<T>(architecture, config);
    }

    private static NeuralNetworkArchitecture<T> CreateArchitectureFromConfig(MobileNetV3Configuration config)
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
            // Use MobileNetV3-specific layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultMobileNetV3Layers(Architecture, _configuration));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
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
                { "NetworkType", "MobileNetV3Network" },
                { "Variant", _configuration.Variant.ToString() },
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
        writer.Write((int)_configuration.Variant);
        writer.Write((int)_configuration.WidthMultiplier);
        writer.Write(_configuration.InputChannels);
        writer.Write(_configuration.InputHeight);
        writer.Write(_configuration.InputWidth);
        writer.Write(_configuration.NumClasses);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var variant = (MobileNetV3Variant)reader.ReadInt32();
        var widthMultiplier = (MobileNetV3WidthMultiplier)reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var inputHeight = reader.ReadInt32();
        var inputWidth = reader.ReadInt32();
        var numClasses = reader.ReadInt32();

        if (variant != _configuration.Variant ||
            widthMultiplier != _configuration.WidthMultiplier ||
            inputChannels != _configuration.InputChannels ||
            inputHeight != _configuration.InputHeight ||
            inputWidth != _configuration.InputWidth ||
            numClasses != _configuration.NumClasses)
        {
            throw new InvalidDataException("Serialized MobileNetV3 configuration does not match current configuration.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var config = new MobileNetV3Configuration(
            _configuration.Variant,
            _configuration.NumClasses,
            _configuration.WidthMultiplier,
            _configuration.InputHeight,
            _configuration.InputWidth,
            _configuration.InputChannels);

        return new MobileNetV3Network<T>(Architecture, config, _optimizer, _lossFunction);
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
