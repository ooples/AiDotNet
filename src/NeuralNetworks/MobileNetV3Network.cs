using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Specifies the MobileNetV3 architecture variant.
/// </summary>
public enum MobileNetV3Variant
{
    /// <summary>
    /// MobileNetV3-Small: Optimized for low resource use cases (CPU, ~2.5M params).
    /// </summary>
    Small,

    /// <summary>
    /// MobileNetV3-Large: Standard version with higher accuracy (~5.4M params).
    /// </summary>
    Large
}

/// <summary>
/// Specifies the MobileNetV3 width multiplier.
/// </summary>
public enum MobileNetV3WidthMultiplier
{
    /// <summary>
    /// Width multiplier of 0.75.
    /// </summary>
    Alpha075,

    /// <summary>
    /// Width multiplier of 1.0 (standard model).
    /// </summary>
    Alpha100
}

/// <summary>
/// Configuration for a MobileNetV3 network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MobileNetV3Configuration<T>
{
    /// <summary>
    /// Gets or sets the MobileNetV3 variant (Small or Large).
    /// </summary>
    public MobileNetV3Variant Variant { get; set; } = MobileNetV3Variant.Large;

    /// <summary>
    /// Gets or sets the width multiplier.
    /// </summary>
    public MobileNetV3WidthMultiplier WidthMultiplier { get; set; } = MobileNetV3WidthMultiplier.Alpha100;

    /// <summary>
    /// Gets or sets the number of input channels.
    /// </summary>
    public int InputChannels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the input image height.
    /// </summary>
    public int InputHeight { get; set; } = 224;

    /// <summary>
    /// Gets or sets the input image width.
    /// </summary>
    public int InputWidth { get; set; } = 224;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    public int NumClasses { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the loss function.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer.
    /// </summary>
    public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; set; }

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets the numeric width multiplier value.
    /// </summary>
    public double Alpha => WidthMultiplier switch
    {
        MobileNetV3WidthMultiplier.Alpha075 => 0.75,
        MobileNetV3WidthMultiplier.Alpha100 => 1.0,
        _ => 1.0
    };
}

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
    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly MobileNetV3Configuration<T> _config;

    /// <summary>
    /// Gets the MobileNetV3 variant used by this network.
    /// </summary>
    public MobileNetV3Variant Variant => _config.Variant;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV3Network{T}"/> class.
    /// </summary>
    /// <param name="config">The MobileNetV3 configuration.</param>
    public MobileNetV3Network(MobileNetV3Configuration<T> config)
        : base(CreateArchitecture(config),
               config.LossFunction ?? new CrossEntropyLoss<T>(),
               config.MaxGradNorm)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _lossFunction = config.LossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = config.Optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a MobileNetV3-Large network.
    /// </summary>
    public static MobileNetV3Network<T> MobileNetV3Large(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV3Network<T>(new MobileNetV3Configuration<T>
        {
            Variant = MobileNetV3Variant.Large,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates a MobileNetV3-Small network.
    /// </summary>
    public static MobileNetV3Network<T> MobileNetV3Small(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV3Network<T>(new MobileNetV3Configuration<T>
        {
            Variant = MobileNetV3Variant.Small,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(MobileNetV3Configuration<T> config)
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

    /// <summary>
    /// Scales channel count by the width multiplier.
    /// </summary>
    private int MakeScaledChannels(int channels)
    {
        int scaled = (int)Math.Round(channels * _config.Alpha);
        return Math.Max(8, (scaled + 4) / 8 * 8);
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (_config.Variant == MobileNetV3Variant.Large)
        {
            InitializeLargeLayers();
        }
        else
        {
            InitializeSmallLayers();
        }
    }

    /// <summary>
    /// Block configuration: (expansion, out_channels, kernel, stride, use_se, use_hs)
    /// use_hs: use HardSwish (true) or ReLU (false)
    /// </summary>
    private record BlockConfig(int Expansion, int OutChannels, int Kernel, int Stride, bool UseSE, bool UseHS);

    private void InitializeLargeLayers()
    {
        var layers = new List<ILayer<T>>();
        int currentHeight = _config.InputHeight;
        int currentWidth = _config.InputWidth;

        // Initial conv: 3x3, stride 2, HardSwish
        int firstConvChannels = MakeScaledChannels(16);
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: _config.InputChannels,
            outputDepth: firstConvChannels,
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 1,
            activation: new IdentityActivation<T>()));

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        layers.Add(new BatchNormalizationLayer<T>(firstConvChannels));
        layers.Add(new ActivationLayer<T>([firstConvChannels, currentHeight, currentWidth],
            activationFunction: new HardSwishActivation<T>()));

        int currentChannels = firstConvChannels;

        // MobileNetV3-Large block configuration
        // (expansion, out_channels, kernel, stride, use_se, use_hardswish)
        var blockConfigs = new BlockConfig[]
        {
            new(1, 16, 3, 1, false, false),   // RE
            new(4, 24, 3, 2, false, false),   // RE
            new(3, 24, 3, 1, false, false),   // RE
            new(3, 40, 5, 2, true, false),    // RE, SE
            new(3, 40, 5, 1, true, false),    // RE, SE
            new(3, 40, 5, 1, true, false),    // RE, SE
            new(6, 80, 3, 2, false, true),    // HS
            new(2, 80, 3, 1, false, true),    // HS  (2.5 rounded to 2)
            new(2, 80, 3, 1, false, true),    // HS  (2.3 rounded to 2)
            new(2, 80, 3, 1, false, true),    // HS  (2.3 rounded to 2)
            new(6, 112, 3, 1, true, true),    // HS, SE
            new(6, 112, 3, 1, true, true),    // HS, SE
            new(6, 160, 5, 2, true, true),    // HS, SE
            new(6, 160, 5, 1, true, true),    // HS, SE
            new(6, 160, 5, 1, true, true)     // HS, SE
        };

        // Add inverted residual blocks
        foreach (var config in blockConfigs)
        {
            int scaledOutChannels = MakeScaledChannels(config.OutChannels);
            var activation = config.UseHS
                ? (IActivationFunction<T>)new HardSwishActivation<T>()
                : (IActivationFunction<T>)new ReLU6Activation<T>();

            layers.Add(new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: scaledOutChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: config.Expansion,
                stride: config.Stride,
                useSE: config.UseSE,
                seRatio: 4,
                activation: activation));

            currentHeight = (currentHeight + config.Stride - 1) / config.Stride;
            currentWidth = (currentWidth + config.Stride - 1) / config.Stride;
            currentChannels = scaledOutChannels;
        }

        // Last stage: 1x1 conv to 960, global avg pool, 1x1 conv to 1280
        int lastConvChannels = MakeScaledChannels(960);
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: lastConvChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>()));

        layers.Add(new BatchNormalizationLayer<T>(lastConvChannels));
        layers.Add(new ActivationLayer<T>([lastConvChannels, currentHeight, currentWidth],
            activationFunction: new HardSwishActivation<T>()));

        // Global average pooling
        layers.Add(new AdaptiveAvgPoolingLayer<T>(lastConvChannels, currentHeight, currentWidth, 1, 1));

        // Classifier head: 1x1 conv to 1280 (no BN), HardSwish, then output
        int headChannels = 1280;
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: lastConvChannels,
            outputDepth: headChannels,
            kernelSize: 1,
            inputHeight: 1,
            inputWidth: 1,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>()));

        layers.Add(new ActivationLayer<T>([headChannels, 1, 1],
            activationFunction: new HardSwishActivation<T>()));

        // Flatten
        layers.Add(new FlattenLayer<T>([headChannels, 1, 1]));

        // Final classifier
        layers.Add(new DenseLayer<T>(headChannels, _config.NumClasses,
            activationFunction: new IdentityActivation<T>()));

        Layers.AddRange(layers);
    }

    private void InitializeSmallLayers()
    {
        var layers = new List<ILayer<T>>();
        int currentHeight = _config.InputHeight;
        int currentWidth = _config.InputWidth;

        // Initial conv: 3x3, stride 2, HardSwish
        int firstConvChannels = MakeScaledChannels(16);
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: _config.InputChannels,
            outputDepth: firstConvChannels,
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 1,
            activation: new IdentityActivation<T>()));

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        layers.Add(new BatchNormalizationLayer<T>(firstConvChannels));
        layers.Add(new ActivationLayer<T>([firstConvChannels, currentHeight, currentWidth],
            activationFunction: new HardSwishActivation<T>()));

        int currentChannels = firstConvChannels;

        // MobileNetV3-Small block configuration
        var blockConfigs = new BlockConfig[]
        {
            new(1, 16, 3, 2, true, false),    // SE, RE
            new(4, 24, 3, 2, false, false),   // RE  (4.5 rounded to 4)
            new(3, 24, 3, 1, false, false),   // RE  (3.67 rounded to 4->3)
            new(4, 40, 5, 2, true, true),     // SE, HS
            new(6, 40, 5, 1, true, true),     // SE, HS
            new(6, 40, 5, 1, true, true),     // SE, HS
            new(3, 48, 5, 1, true, true),     // SE, HS
            new(3, 48, 5, 1, true, true),     // SE, HS
            new(6, 96, 5, 2, true, true),     // SE, HS
            new(6, 96, 5, 1, true, true),     // SE, HS
            new(6, 96, 5, 1, true, true)      // SE, HS
        };

        // Add inverted residual blocks
        foreach (var config in blockConfigs)
        {
            int scaledOutChannels = MakeScaledChannels(config.OutChannels);
            var activation = config.UseHS
                ? (IActivationFunction<T>)new HardSwishActivation<T>()
                : (IActivationFunction<T>)new ReLU6Activation<T>();

            layers.Add(new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: scaledOutChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: config.Expansion,
                stride: config.Stride,
                useSE: config.UseSE,
                seRatio: 4,
                activation: activation));

            currentHeight = (currentHeight + config.Stride - 1) / config.Stride;
            currentWidth = (currentWidth + config.Stride - 1) / config.Stride;
            currentChannels = scaledOutChannels;
        }

        // Last stage: 1x1 conv to 576
        int lastConvChannels = MakeScaledChannels(576);
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: lastConvChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>()));

        layers.Add(new BatchNormalizationLayer<T>(lastConvChannels));
        layers.Add(new ActivationLayer<T>([lastConvChannels, currentHeight, currentWidth],
            activationFunction: new HardSwishActivation<T>()));

        // Global average pooling
        layers.Add(new AdaptiveAvgPoolingLayer<T>(lastConvChannels, currentHeight, currentWidth, 1, 1));

        // Classifier head: 1x1 conv to 1024 (no BN), HardSwish
        int headChannels = 1024;
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: lastConvChannels,
            outputDepth: headChannels,
            kernelSize: 1,
            inputHeight: 1,
            inputWidth: 1,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>()));

        layers.Add(new ActivationLayer<T>([headChannels, 1, 1],
            activationFunction: new HardSwishActivation<T>()));

        // Flatten
        layers.Add(new FlattenLayer<T>([headChannels, 1, 1]));

        // Final classifier
        layers.Add(new DenseLayer<T>(headChannels, _config.NumClasses,
            activationFunction: new IdentityActivation<T>()));

        Layers.AddRange(layers);
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
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
                { "Variant", _config.Variant.ToString() },
                { "WidthMultiplier", _config.Alpha },
                { "NumClasses", _config.NumClasses },
                { "InputShape", $"{_config.InputChannels}x{_config.InputHeight}x{_config.InputWidth}" },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_config.Variant);
        writer.Write((int)_config.WidthMultiplier);
        writer.Write(_config.InputChannels);
        writer.Write(_config.InputHeight);
        writer.Write(_config.InputWidth);
        writer.Write(_config.NumClasses);
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

        if (variant != _config.Variant ||
            widthMultiplier != _config.WidthMultiplier ||
            inputChannels != _config.InputChannels ||
            inputHeight != _config.InputHeight ||
            inputWidth != _config.InputWidth ||
            numClasses != _config.NumClasses)
        {
            throw new InvalidDataException("Serialized MobileNetV3 configuration does not match current configuration.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MobileNetV3Network<T>(new MobileNetV3Configuration<T>
        {
            Variant = _config.Variant,
            WidthMultiplier = _config.WidthMultiplier,
            InputChannels = _config.InputChannels,
            InputHeight = _config.InputHeight,
            InputWidth = _config.InputWidth,
            NumClasses = _config.NumClasses,
            MaxGradNorm = _config.MaxGradNorm,
            LossFunction = _lossFunction,
            Optimizer = _optimizer
        });
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
