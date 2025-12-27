using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
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
public class ResNetNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

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
    /// Initializes a new instance of the ResNetNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The ResNet-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
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
    public ResNetNetwork(
        NeuralNetworkArchitecture<T> architecture,
        ResNetConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(ResNetNetwork<T>));

        // Validate that architecture matches configuration
        ValidateArchitectureMatchesConfiguration(architecture, configuration);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

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
    /// Initializes the layers of the ResNet network based on the configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// the standard ResNet layers based on the configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method builds the ResNet network layer by layer:
    /// <list type="number">
    /// <item>Initial 7x7 convolution with stride 2</item>
    /// <item>Max pooling 3x3 with stride 2</item>
    /// <item>Four stages of residual blocks (conv2_x through conv5_x)</item>
    /// <item>Global average pooling</item>
    /// <item>Fully connected classification layer</item>
    /// </list>
    /// </para>
    /// </remarks>
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
            // Use ResNet-specific layer configuration
            Layers.AddRange(CreateResNetLayers());
        }
    }

    /// <summary>
    /// Creates the ResNet layers based on the configuration.
    /// </summary>
    private IEnumerable<ILayer<T>> CreateResNetLayers()
    {
        var layers = new List<ILayer<T>>();
        var config = _configuration;

        int currentHeight = config.InputHeight;
        int currentWidth = config.InputWidth;
        int currentChannels = config.InputChannels;

        // Stage 0: Initial convolution (conv1)
        // 7x7 conv, 64, stride 2
        layers.Add(new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            outputDepth: 64,
            kernelSize: 7,
            stride: 2,
            padding: 3,
            activation: new ActivationFunctions.IdentityActivation<T>()));

        currentHeight /= 2;
        currentWidth /= 2;
        currentChannels = 64;

        // Batch normalization after conv1
        layers.Add(new BatchNormalizationLayer<T>(currentChannels));

        // ReLU activation
        layers.Add(new ActivationLayer<T>(
            inputShape: [currentChannels, currentHeight, currentWidth],
            activationFunction: new ActivationFunctions.ReLUActivation<T>()));

        // Max pooling: 3x3 pool, stride 2
        layers.Add(new MaxPoolingLayer<T>(
            inputShape: [currentChannels, currentHeight, currentWidth],
            poolSize: 3,
            strides: 2));

        currentHeight = (currentHeight - 1) / 2 + 1; // Ceiling division for pool output
        currentWidth = (currentWidth - 1) / 2 + 1;

        // Get block configuration for this variant
        int[] blockCounts = config.BlockCounts;
        int[] baseChannels = config.BaseChannels;
        int expansion = config.Expansion;

        // Stage 1-4: Residual blocks
        int inChannels = 64;
        for (int stageIdx = 0; stageIdx < 4; stageIdx++)
        {
            int stageBaseChannels = baseChannels[stageIdx];
            int numBlocks = blockCounts[stageIdx];
            int stride = stageIdx == 0 ? 1 : 2; // First stage has stride 1, others have stride 2

            // First block of stage may downsample
            if (config.UsesBottleneck)
            {
                layers.Add(new BottleneckBlock<T>(
                    inChannels: inChannels,
                    baseChannels: stageBaseChannels,
                    stride: stride,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    zeroInitResidual: config.ZeroInitResidual));
            }
            else
            {
                layers.Add(new BasicBlock<T>(
                    inChannels: inChannels,
                    outChannels: stageBaseChannels * expansion,
                    stride: stride,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    zeroInitResidual: config.ZeroInitResidual));
            }

            // Update dimensions
            if (stride == 2)
            {
                currentHeight /= 2;
                currentWidth /= 2;
            }
            inChannels = stageBaseChannels * expansion;

            // Remaining blocks in stage (stride 1, no channel change)
            for (int blockIdx = 1; blockIdx < numBlocks; blockIdx++)
            {
                if (config.UsesBottleneck)
                {
                    layers.Add(new BottleneckBlock<T>(
                        inChannels: inChannels,
                        baseChannels: stageBaseChannels,
                        stride: 1,
                        inputHeight: currentHeight,
                        inputWidth: currentWidth,
                        zeroInitResidual: config.ZeroInitResidual));
                }
                else
                {
                    layers.Add(new BasicBlock<T>(
                        inChannels: inChannels,
                        outChannels: stageBaseChannels * expansion,
                        stride: 1,
                        inputHeight: currentHeight,
                        inputWidth: currentWidth,
                        zeroInitResidual: config.ZeroInitResidual));
                }
            }
        }

        // Global average pooling
        int finalChannels = baseChannels[3] * expansion; // 512 for BasicBlock, 2048 for Bottleneck
        layers.Add(AdaptiveAvgPoolingLayer<T>.GlobalPool(
            inputChannels: finalChannels,
            inputHeight: currentHeight,
            inputWidth: currentWidth));

        // Flatten for FC layer
        layers.Add(new FlattenLayer<T>(
            inputShape: [finalChannels, 1, 1]));

        // Fully connected classifier
        if (config.IncludeClassifier)
        {
            var outputActivation = Architecture.TaskType == NeuralNetworkTaskType.BinaryClassification
                ? (IActivationFunction<T>)new ActivationFunctions.SigmoidActivation<T>()
                : new ActivationFunctions.SoftmaxActivation<T>();

            layers.Add(new DenseLayer<T>(
                inputSize: finalChannels,
                outputSize: config.NumClasses,
                activationFunction: outputActivation));
        }

        return layers;
    }

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
            processedInput = AddBatchDimension(input);
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

        Tensor<T> output = processedInput;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        // Remove batch dimension if we added it
        if (addedBatch && output.Rank == 2 && output.Shape[0] == 1)
        {
            output = output.Reshape([output.Shape[1]]);
        }

        return output;
    }

    /// <summary>
    /// Adds a batch dimension to a single input tensor.
    /// </summary>
    private static Tensor<T> AddBatchDimension(Tensor<T> input)
    {
        int[] inputShape = input.Shape;
        int[] resultShape = new int[inputShape.Length + 1];
        resultShape[0] = 1;
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 1] = inputShape[i];
        }
        return input.Reshape(resultShape);
    }

    /// <summary>
    /// Performs a backward pass through the network to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
    /// <returns>The gradient of the loss with respect to the network's input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            outputGradient = Layers[i].Backward(outputGradient);
        }
        return outputGradient;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
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

    /// <summary>
    /// Makes a prediction using the ResNet network for the given input.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor containing class probabilities.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Trains the ResNet network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor (one-hot encoded class labels).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var prediction = Predict(input);

        // Calculate loss
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        // Calculate output gradient
        var outputGradient = CalculateOutputGradient(prediction, expectedOutput);
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        // Backpropagation
        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }

        // Update parameters using the optimizer
        ApplyParameterUpdates();
    }

    /// <summary>
    /// Applies parameter updates using the optimizer.
    /// </summary>
    private void ApplyParameterUpdates()
    {
        _optimizer.UpdateParameters(Layers);
    }

    /// <summary>
    /// Calculates the gradient of the loss with respect to the network's output.
    /// </summary>
    private Vector<T> CalculateOutputGradient(Tensor<T> prediction, Tensor<T> expectedOutput)
    {
        return _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
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
    public new int GetParameterCount()
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
            ModelType = ModelType.ConvolutionalNeuralNetwork,
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
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes ResNet network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_configuration.Variant);
        writer.Write(_configuration.NumClasses);
        writer.Write(_configuration.InputHeight);
        writer.Write(_configuration.InputWidth);
        writer.Write(_configuration.InputChannels);
        writer.Write(_configuration.IncludeClassifier);
        writer.Write(_configuration.ZeroInitResidual);
        writer.Write(_configuration.UseAutodiff);
    }

    /// <summary>
    /// Deserializes ResNet network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var variant = (ResNetVariant)reader.ReadInt32();
        var numClasses = reader.ReadInt32();
        var inputHeight = reader.ReadInt32();
        var inputWidth = reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var includeClassifier = reader.ReadBoolean();
        var zeroInitResidual = reader.ReadBoolean();
        _ = reader.ReadBoolean(); // useAutodiff

        // Validate loaded configuration matches current
        if (variant != _configuration.Variant)
        {
            throw new InvalidOperationException(
                $"Serialized ResNet variant ({variant}) does not match current configuration ({_configuration.Variant}).");
        }

        if (numClasses != _configuration.NumClasses)
        {
            throw new InvalidOperationException(
                $"Serialized number of classes ({numClasses}) does not match current configuration ({_configuration.NumClasses}).");
        }

        if (inputHeight != _configuration.InputHeight || inputWidth != _configuration.InputWidth)
        {
            throw new InvalidOperationException(
                $"Serialized input dimensions ({inputHeight}x{inputWidth}) do not match current configuration ({_configuration.InputHeight}x{_configuration.InputWidth}).");
        }

        if (inputChannels != _configuration.InputChannels)
        {
            throw new InvalidOperationException(
                $"Serialized input channels ({inputChannels}) does not match current configuration ({_configuration.InputChannels}).");
        }

        if (includeClassifier != _configuration.IncludeClassifier)
        {
            throw new InvalidOperationException(
                $"Serialized includeClassifier ({includeClassifier}) does not match current configuration ({_configuration.IncludeClassifier}).");
        }

        if (zeroInitResidual != _configuration.ZeroInitResidual)
        {
            throw new InvalidOperationException(
                $"Serialized zeroInitResidual ({zeroInitResidual}) does not match current configuration ({_configuration.ZeroInitResidual}).");
        }
    }

    /// <summary>
    /// Creates a new instance of the ResNet network model.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ResNetNetwork<T>(
            Architecture,
            _configuration,
            null,
            _lossFunction,
            NumOps.ToDouble(MaxGradNorm));
    }
}
