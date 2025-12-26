using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Specifies the EfficientNet model variant (B0-B7).
/// </summary>
/// <remarks>
/// Each variant uses different compound scaling coefficients for width, depth, and resolution.
/// Higher variants have more parameters and accuracy but require more compute.
/// </remarks>
public enum EfficientNetVariant
{
    /// <summary>
    /// EfficientNet-B0: Baseline model (5.3M parameters, 224x224 input).
    /// </summary>
    B0,

    /// <summary>
    /// EfficientNet-B1: Scaled model (7.8M parameters, 240x240 input).
    /// </summary>
    B1,

    /// <summary>
    /// EfficientNet-B2: Scaled model (9.2M parameters, 260x260 input).
    /// </summary>
    B2,

    /// <summary>
    /// EfficientNet-B3: Scaled model (12M parameters, 300x300 input).
    /// </summary>
    B3,

    /// <summary>
    /// EfficientNet-B4: Scaled model (19M parameters, 380x380 input).
    /// </summary>
    B4,

    /// <summary>
    /// EfficientNet-B5: Scaled model (30M parameters, 456x456 input).
    /// </summary>
    B5,

    /// <summary>
    /// EfficientNet-B6: Scaled model (43M parameters, 528x528 input).
    /// </summary>
    B6,

    /// <summary>
    /// EfficientNet-B7: Largest model (66M parameters, 600x600 input).
    /// </summary>
    B7
}

/// <summary>
/// Configuration for an EfficientNet network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EfficientNetConfiguration<T>
{
    /// <summary>
    /// Gets or sets the EfficientNet variant (B0-B7).
    /// </summary>
    public EfficientNetVariant Variant { get; set; } = EfficientNetVariant.B0;

    /// <summary>
    /// Gets or sets the number of input channels (e.g., 3 for RGB, 1 for grayscale).
    /// </summary>
    public int InputChannels { get; set; } = 3;

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
    /// Gets the compound scaling coefficients for the selected variant.
    /// </summary>
    /// <returns>A tuple of (widthCoefficient, depthCoefficient, resolution).</returns>
    public (double Width, double Depth, int Resolution) GetScalingCoefficients()
    {
        return Variant switch
        {
            EfficientNetVariant.B0 => (1.0, 1.0, 224),
            EfficientNetVariant.B1 => (1.0, 1.1, 240),
            EfficientNetVariant.B2 => (1.1, 1.2, 260),
            EfficientNetVariant.B3 => (1.2, 1.4, 300),
            EfficientNetVariant.B4 => (1.4, 1.8, 380),
            EfficientNetVariant.B5 => (1.6, 2.2, 456),
            EfficientNetVariant.B6 => (1.8, 2.6, 528),
            EfficientNetVariant.B7 => (2.0, 3.1, 600),
            _ => (1.0, 1.0, 224)
        };
    }

    /// <summary>
    /// Gets the input resolution for this variant.
    /// </summary>
    public int InputResolution => GetScalingCoefficients().Resolution;
}

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
    private readonly EfficientNetConfiguration<T> _config;

    /// <summary>
    /// Gets the EfficientNet variant.
    /// </summary>
    public EfficientNetVariant Variant => _config.Variant;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>
    /// Gets the input resolution for this variant.
    /// </summary>
    public int InputResolution => _config.InputResolution;

    /// <summary>
    /// Initializes a new instance of the <see cref="EfficientNetNetwork{T}"/> class.
    /// </summary>
    /// <param name="config">The EfficientNet configuration.</param>
    public EfficientNetNetwork(EfficientNetConfiguration<T> config)
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
    /// Creates an EfficientNet-B0 network (baseline model).
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A configured EfficientNet-B0 network.</returns>
    public static EfficientNetNetwork<T> EfficientNetB0(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B0,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B1 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB1(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B1,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B2 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB2(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B2,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B3 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB3(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B3,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B4 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB4(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B4,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B5 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB5(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B5,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B6 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB6(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B6,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates an EfficientNet-B7 network.
    /// </summary>
    public static EfficientNetNetwork<T> EfficientNetB7(int numClasses = 1000, int inputChannels = 3)
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = EfficientNetVariant.B7,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(EfficientNetConfiguration<T> config)
    {
        var resolution = config.InputResolution;
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

    /// <summary>
    /// Scales channel count by the width coefficient.
    /// </summary>
    private int MakeScaledChannels(int channels, double widthCoefficient)
    {
        // Ensure channels is divisible by 8 (important for efficient computation)
        int scaled = (int)Math.Round(channels * widthCoefficient);
        return Math.Max(8, (scaled + 4) / 8 * 8); // Round to nearest 8, minimum 8
    }

    /// <summary>
    /// Scales layer repeat count by the depth coefficient.
    /// </summary>
    private int MakeScaledDepth(int numLayers, double depthCoefficient)
    {
        return (int)Math.Ceiling(numLayers * depthCoefficient);
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        var layers = new List<ILayer<T>>();
        var (widthCoeff, depthCoeff, resolution) = _config.GetScalingCoefficients();

        int currentHeight = resolution;
        int currentWidth = resolution;

        // Stem: 3x3 conv, stride 2
        int stemChannels = MakeScaledChannels(32, widthCoeff);
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: _config.InputChannels,
            outputDepth: stemChannels,
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 1,
            activation: new IdentityActivation<T>()));

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        layers.Add(new BatchNormalizationLayer<T>(stemChannels));
        layers.Add(new ActivationLayer<T>([stemChannels, currentHeight, currentWidth],
            activationFunction: new SwishActivation<T>()));

        int currentChannels = stemChannels;

        // EfficientNet-B0 block configuration:
        // (expansion, output_channels, num_layers, stride, kernel_size)
        // Note: kernel_size is stored but we currently use 3x3 for all blocks
        var blockConfigs = new (int expansion, int outChannels, int numLayers, int stride, int kernelSize)[]
        {
            (1, 16, 1, 1, 3),   // Stage 1: MBConv1
            (6, 24, 2, 2, 3),   // Stage 2: MBConv6
            (6, 40, 2, 2, 5),   // Stage 3: MBConv6 (5x5 in original, using 3x3)
            (6, 80, 3, 2, 3),   // Stage 4: MBConv6
            (6, 112, 3, 1, 5),  // Stage 5: MBConv6 (5x5 in original, using 3x3)
            (6, 192, 4, 2, 5),  // Stage 6: MBConv6 (5x5 in original, using 3x3)
            (6, 320, 1, 1, 3)   // Stage 7: MBConv6
        };

        // Add MBConv blocks with SE and Swish activation
        foreach (var (expansion, outChannels, numLayers, stride, kernelSize) in blockConfigs)
        {
            int scaledOutChannels = MakeScaledChannels(outChannels, widthCoeff);
            int scaledNumLayers = MakeScaledDepth(numLayers, depthCoeff);

            // First block in each stage may have stride > 1
            layers.Add(new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: scaledOutChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: expansion,
                stride: stride,
                useSE: true, // EfficientNet uses SE blocks
                seRatio: 4,
                activation: new SwishActivation<T>()));

            // Update dimensions after first block
            currentHeight = (currentHeight + stride - 1) / stride;
            currentWidth = (currentWidth + stride - 1) / stride;
            currentChannels = scaledOutChannels;

            // Remaining blocks in the stage (stride=1)
            for (int i = 1; i < scaledNumLayers; i++)
            {
                layers.Add(new InvertedResidualBlock<T>(
                    inChannels: currentChannels,
                    outChannels: currentChannels, // Same channels
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    expansionRatio: expansion,
                    stride: 1,
                    useSE: true,
                    seRatio: 4,
                    activation: new SwishActivation<T>()));
            }
        }

        // Head: 1x1 conv
        int headChannels = MakeScaledChannels(1280, widthCoeff);
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: headChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>()));

        layers.Add(new BatchNormalizationLayer<T>(headChannels));
        layers.Add(new ActivationLayer<T>([headChannels, currentHeight, currentWidth],
            activationFunction: new SwishActivation<T>()));

        // Global average pooling
        layers.Add(new AdaptiveAvgPoolingLayer<T>(headChannels, currentHeight, currentWidth, 1, 1));

        // Flatten
        layers.Add(new FlattenLayer<T>([headChannels, 1, 1]));

        // Classification head
        layers.Add(new DenseLayer<T>(headChannels, _config.NumClasses,
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
        var (widthCoeff, depthCoeff, resolution) = _config.GetScalingCoefficients();
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConvolutionalNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "EfficientNetNetwork" },
                { "Variant", _config.Variant.ToString() },
                { "WidthCoefficient", widthCoeff },
                { "DepthCoefficient", depthCoeff },
                { "Resolution", resolution },
                { "NumClasses", _config.NumClasses },
                { "InputShape", $"{_config.InputChannels}x{resolution}x{resolution}" },
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
        writer.Write(_config.InputChannels);
        writer.Write(_config.NumClasses);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var variant = (EfficientNetVariant)reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var numClasses = reader.ReadInt32();

        if (variant != _config.Variant ||
            inputChannels != _config.InputChannels ||
            numClasses != _config.NumClasses)
        {
            throw new InvalidDataException("Serialized EfficientNet configuration does not match current configuration.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new EfficientNetNetwork<T>(new EfficientNetConfiguration<T>
        {
            Variant = _config.Variant,
            InputChannels = _config.InputChannels,
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
