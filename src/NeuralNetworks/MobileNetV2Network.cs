using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Specifies the MobileNetV2 width multiplier.
/// </summary>
/// <remarks>
/// The width multiplier scales the number of channels in each layer.
/// Lower values produce smaller, faster models with some accuracy trade-off.
/// </remarks>
public enum MobileNetV2WidthMultiplier
{
    /// <summary>
    /// Width multiplier of 0.35 (smallest model, fastest inference).
    /// </summary>
    Alpha035,

    /// <summary>
    /// Width multiplier of 0.5.
    /// </summary>
    Alpha050,

    /// <summary>
    /// Width multiplier of 0.75.
    /// </summary>
    Alpha075,

    /// <summary>
    /// Width multiplier of 1.0 (standard model).
    /// </summary>
    Alpha100,

    /// <summary>
    /// Width multiplier of 1.4 (larger model, higher accuracy).
    /// </summary>
    Alpha140
}

/// <summary>
/// Configuration for a MobileNetV2 network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MobileNetV2Configuration<T>
{
    /// <summary>
    /// Gets or sets the width multiplier that scales the number of channels.
    /// </summary>
    public MobileNetV2WidthMultiplier WidthMultiplier { get; set; } = MobileNetV2WidthMultiplier.Alpha100;

    /// <summary>
    /// Gets or sets the number of input channels (e.g., 3 for RGB, 1 for grayscale).
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
    /// Gets or sets the loss function. Defaults to CrossEntropyLoss for classification.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer. Defaults to Adam.
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
        MobileNetV2WidthMultiplier.Alpha035 => 0.35,
        MobileNetV2WidthMultiplier.Alpha050 => 0.5,
        MobileNetV2WidthMultiplier.Alpha075 => 0.75,
        MobileNetV2WidthMultiplier.Alpha100 => 1.0,
        MobileNetV2WidthMultiplier.Alpha140 => 1.4,
        _ => 1.0
    };
}

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
    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly MobileNetV2Configuration<T> _config;

    /// <summary>
    /// Gets the width multiplier used by this network.
    /// </summary>
    public MobileNetV2WidthMultiplier WidthMultiplier => _config.WidthMultiplier;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV2Network{T}"/> class.
    /// </summary>
    /// <param name="config">The MobileNetV2 configuration.</param>
    public MobileNetV2Network(MobileNetV2Configuration<T> config)
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
    /// Initializes a new MobileNetV2 with width multiplier 1.0.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A configured MobileNetV2 network.</returns>
    public static MobileNetV2Network<T> MobileNetV2_100(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV2Network<T>(new MobileNetV2Configuration<T>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha100,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 0.35 (smallest).
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_035(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV2Network<T>(new MobileNetV2Configuration<T>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha035,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 0.5.
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_050(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV2Network<T>(new MobileNetV2Configuration<T>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha050,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 0.75.
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_075(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV2Network<T>(new MobileNetV2Configuration<T>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha075,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Initializes a new MobileNetV2 with width multiplier 1.4 (largest).
    /// </summary>
    public static MobileNetV2Network<T> MobileNetV2_140(int numClasses = 1000, int inputChannels = 3)
    {
        return new MobileNetV2Network<T>(new MobileNetV2Configuration<T>
        {
            WidthMultiplier = MobileNetV2WidthMultiplier.Alpha140,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(MobileNetV2Configuration<T> config)
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
        // Ensure channels is divisible by 8 (important for efficient computation)
        int scaled = (int)Math.Round(channels * _config.Alpha);
        return Math.Max(8, (scaled + 4) / 8 * 8); // Round to nearest 8, minimum 8
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        var layers = new List<ILayer<T>>();
        int currentHeight = _config.InputHeight;
        int currentWidth = _config.InputWidth;

        // Initial convolution: 3x3, stride 2
        int firstConvChannels = MakeScaledChannels(32);
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
            activationFunction: new ReLU6Activation<T>()));

        int currentChannels = firstConvChannels;

        // MobileNetV2 inverted residual block configuration:
        // (expansion, output_channels, num_blocks, stride)
        var blockConfigs = new (int expansion, int outChannels, int numBlocks, int stride)[]
        {
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)
        };

        // Add inverted residual blocks
        foreach (var (expansion, outChannels, numBlocks, stride) in blockConfigs)
        {
            int scaledOutChannels = MakeScaledChannels(outChannels);

            // First block in each stage may have stride > 1
            layers.Add(new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: scaledOutChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: expansion,
                stride: stride,
                useSE: false, // MobileNetV2 doesn't use SE
                activation: new ReLU6Activation<T>()));

            // Update dimensions after first block
            currentHeight = (currentHeight + stride - 1) / stride;
            currentWidth = (currentWidth + stride - 1) / stride;
            currentChannels = scaledOutChannels;

            // Remaining blocks in the stage (stride=1)
            for (int i = 1; i < numBlocks; i++)
            {
                layers.Add(new InvertedResidualBlock<T>(
                    inChannels: currentChannels,
                    outChannels: currentChannels, // Same channels
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    expansionRatio: expansion,
                    stride: 1,
                    useSE: false,
                    activation: new ReLU6Activation<T>()));
            }
        }

        // Final 1x1 convolution
        int finalConvChannels = _config.WidthMultiplier == MobileNetV2WidthMultiplier.Alpha140 ? 1792 : 1280;
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: finalConvChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>()));

        layers.Add(new BatchNormalizationLayer<T>(finalConvChannels));
        layers.Add(new ActivationLayer<T>([finalConvChannels, currentHeight, currentWidth],
            activationFunction: new ReLU6Activation<T>()));

        // Global average pooling
        layers.Add(new AdaptiveAvgPoolingLayer<T>(finalConvChannels, currentHeight, currentWidth, 1, 1));

        // Flatten
        layers.Add(new FlattenLayer<T>([finalConvChannels, 1, 1]));

        // Classification head
        layers.Add(new DenseLayer<T>(finalConvChannels, _config.NumClasses,
            activationFunction: new IdentityActivation<T>()));

        Layers.AddRange(layers);
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The output class logits.</returns>
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
        writer.Write((int)_config.WidthMultiplier);
        writer.Write(_config.InputChannels);
        writer.Write(_config.InputHeight);
        writer.Write(_config.InputWidth);
        writer.Write(_config.NumClasses);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var widthMultiplier = (MobileNetV2WidthMultiplier)reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var inputHeight = reader.ReadInt32();
        var inputWidth = reader.ReadInt32();
        var numClasses = reader.ReadInt32();

        if (widthMultiplier != _config.WidthMultiplier ||
            inputChannels != _config.InputChannels ||
            inputHeight != _config.InputHeight ||
            inputWidth != _config.InputWidth ||
            numClasses != _config.NumClasses)
        {
            throw new InvalidDataException("Serialized MobileNetV2 configuration does not match current configuration.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MobileNetV2Network<T>(new MobileNetV2Configuration<T>
        {
            WidthMultiplier = _config.WidthMultiplier,
            InputChannels = _config.InputChannels,
            InputHeight = _config.InputHeight,
            InputWidth = _config.InputWidth,
            NumClasses = _config.NumClasses,
            MaxGradNorm = _config.MaxGradNorm
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        return CreateNewInstance();
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
