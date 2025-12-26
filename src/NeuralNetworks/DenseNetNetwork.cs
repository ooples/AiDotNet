using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Specifies the DenseNet model variant.
/// </summary>
/// <remarks>
/// Each variant has different numbers of layers per dense block.
/// Higher variants have more layers and better accuracy but require more compute.
/// </remarks>
public enum DenseNetVariant
{
    /// <summary>
    /// DenseNet-121: [6, 12, 24, 16] layers per block (8M parameters).
    /// </summary>
    DenseNet121,

    /// <summary>
    /// DenseNet-169: [6, 12, 32, 32] layers per block (14M parameters).
    /// </summary>
    DenseNet169,

    /// <summary>
    /// DenseNet-201: [6, 12, 48, 32] layers per block (20M parameters).
    /// </summary>
    DenseNet201,

    /// <summary>
    /// DenseNet-264: [6, 12, 64, 48] layers per block (33M parameters).
    /// </summary>
    DenseNet264
}

/// <summary>
/// Configuration for a DenseNet network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DenseNetConfiguration<T>
{
    /// <summary>
    /// Gets or sets the DenseNet variant.
    /// </summary>
    public DenseNetVariant Variant { get; set; } = DenseNetVariant.DenseNet121;

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
    /// Gets or sets the growth rate (k in the paper). Default is 32.
    /// </summary>
    public int GrowthRate { get; set; } = 32;

    /// <summary>
    /// Gets or sets the compression factor for transition layers. Default is 0.5.
    /// </summary>
    public double CompressionFactor { get; set; } = 0.5;

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
    /// Gets the number of layers per dense block for this variant.
    /// </summary>
    public int[] GetBlockLayers()
    {
        return Variant switch
        {
            DenseNetVariant.DenseNet121 => [6, 12, 24, 16],
            DenseNetVariant.DenseNet169 => [6, 12, 32, 32],
            DenseNetVariant.DenseNet201 => [6, 12, 48, 32],
            DenseNetVariant.DenseNet264 => [6, 12, 64, 48],
            _ => [6, 12, 24, 16]
        };
    }
}

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
    private readonly DenseNetConfiguration<T> _config;

    /// <summary>
    /// Gets the DenseNet variant.
    /// </summary>
    public DenseNetVariant Variant => _config.Variant;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>
    /// Gets the growth rate (k).
    /// </summary>
    public int GrowthRate => _config.GrowthRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseNetNetwork{T}"/> class.
    /// </summary>
    /// <param name="config">The DenseNet configuration.</param>
    public DenseNetNetwork(DenseNetConfiguration<T> config)
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
    /// Creates a DenseNet-121 network.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <returns>A configured DenseNet-121 network.</returns>
    public static DenseNetNetwork<T> DenseNet121(int numClasses = 1000, int inputChannels = 3)
    {
        return new DenseNetNetwork<T>(new DenseNetConfiguration<T>
        {
            Variant = DenseNetVariant.DenseNet121,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates a DenseNet-169 network.
    /// </summary>
    public static DenseNetNetwork<T> DenseNet169(int numClasses = 1000, int inputChannels = 3)
    {
        return new DenseNetNetwork<T>(new DenseNetConfiguration<T>
        {
            Variant = DenseNetVariant.DenseNet169,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates a DenseNet-201 network.
    /// </summary>
    public static DenseNetNetwork<T> DenseNet201(int numClasses = 1000, int inputChannels = 3)
    {
        return new DenseNetNetwork<T>(new DenseNetConfiguration<T>
        {
            Variant = DenseNetVariant.DenseNet201,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    /// <summary>
    /// Creates a DenseNet-264 network.
    /// </summary>
    public static DenseNetNetwork<T> DenseNet264(int numClasses = 1000, int inputChannels = 3)
    {
        return new DenseNetNetwork<T>(new DenseNetConfiguration<T>
        {
            Variant = DenseNetVariant.DenseNet264,
            NumClasses = numClasses,
            InputChannels = inputChannels
        });
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(DenseNetConfiguration<T> config)
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
    protected override void InitializeLayers()
    {
        var layers = new List<ILayer<T>>();
        int currentHeight = _config.InputHeight;
        int currentWidth = _config.InputWidth;
        var blockLayers = _config.GetBlockLayers();

        // Stem: 7x7 conv, stride 2, padding 3
        int stemChannels = 64; // 2 * growth rate is typical, but 64 is standard for DenseNet
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: _config.InputChannels,
            outputDepth: stemChannels,
            kernelSize: 7,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 3,
            activation: new IdentityActivation<T>()));

        currentHeight = (currentHeight + 2 * 3 - 7) / 2 + 1; // 112 for 224 input
        currentWidth = (currentWidth + 2 * 3 - 7) / 2 + 1;

        layers.Add(new BatchNormalizationLayer<T>(stemChannels));
        layers.Add(new ActivationLayer<T>([stemChannels, currentHeight, currentWidth],
            activationFunction: new ReLUActivation<T>()));

        // MaxPool 3x3, stride 2, padding 1
        layers.Add(new MaxPoolingLayer<T>(
            inputShape: [stemChannels, currentHeight, currentWidth],
            poolSize: 3,
            strides: 2));

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1; // 56 for 112
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        int currentChannels = stemChannels;

        // Dense blocks and transitions
        for (int i = 0; i < blockLayers.Length; i++)
        {
            int numLayersInBlock = blockLayers[i];

            // Add Dense Block
            var denseBlock = new DenseBlock<T>(
                inputChannels: currentChannels,
                numLayers: numLayersInBlock,
                growthRate: _config.GrowthRate,
                inputHeight: currentHeight,
                inputWidth: currentWidth);

            layers.Add(denseBlock);
            currentChannels = denseBlock.OutputChannels;

            // Add Transition (except after the last block)
            if (i < blockLayers.Length - 1)
            {
                var transition = new TransitionLayer<T>(
                    inputChannels: currentChannels,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    compressionFactor: _config.CompressionFactor);

                layers.Add(transition);
                currentChannels = transition.OutputChannels;
                currentHeight /= 2;
                currentWidth /= 2;
            }
        }

        // Final BN and ReLU
        layers.Add(new BatchNormalizationLayer<T>(currentChannels));
        layers.Add(new ActivationLayer<T>([currentChannels, currentHeight, currentWidth],
            activationFunction: new ReLUActivation<T>()));

        // Global average pooling
        layers.Add(new AdaptiveAvgPoolingLayer<T>(currentChannels, currentHeight, currentWidth, 1, 1));

        // Flatten
        layers.Add(new FlattenLayer<T>([currentChannels, 1, 1]));

        // Classification head
        layers.Add(new DenseLayer<T>(currentChannels, _config.NumClasses,
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
                { "NetworkType", "DenseNetNetwork" },
                { "Variant", _config.Variant.ToString() },
                { "GrowthRate", _config.GrowthRate },
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
        writer.Write(_config.InputChannels);
        writer.Write(_config.InputHeight);
        writer.Write(_config.InputWidth);
        writer.Write(_config.NumClasses);
        writer.Write(_config.GrowthRate);
        writer.Write(_config.CompressionFactor);
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

        if (variant != _config.Variant ||
            inputChannels != _config.InputChannels ||
            inputHeight != _config.InputHeight ||
            inputWidth != _config.InputWidth ||
            numClasses != _config.NumClasses ||
            growthRate != _config.GrowthRate ||
            Math.Abs(compressionFactor - _config.CompressionFactor) > 0.001)
        {
            throw new InvalidDataException("Serialized DenseNet configuration does not match current configuration.");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DenseNetNetwork<T>(new DenseNetConfiguration<T>
        {
            Variant = _config.Variant,
            InputChannels = _config.InputChannels,
            InputHeight = _config.InputHeight,
            InputWidth = _config.InputWidth,
            NumClasses = _config.NumClasses,
            GrowthRate = _config.GrowthRate,
            CompressionFactor = _config.CompressionFactor,
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
