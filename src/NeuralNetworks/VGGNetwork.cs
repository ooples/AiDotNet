using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a VGG (Visual Geometry Group) neural network architecture for image classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// VGG networks are deep convolutional neural networks developed by the Visual Geometry Group at
/// Oxford University. They are characterized by their use of small (3x3) convolution filters stacked
/// in increasing depth, which allows them to learn complex hierarchical features.
/// </para>
/// <para>
/// <b>For Beginners:</b> VGG networks are one of the foundational architectures in deep learning for
/// image recognition. Despite being developed in 2014, they remain popular because:
/// <list type="bullet">
/// <item>They're simple to understand - just stacked convolutions and pooling</item>
/// <item>They serve as excellent baselines for comparing new architectures</item>
/// <item>They're great for transfer learning (using a pre-trained network as a starting point)</item>
/// <item>The features they learn are highly transferable to other visual tasks</item>
/// </list>
/// </para>
/// <para>
/// <b>Architecture:</b> VGG networks consist of:
/// <list type="bullet">
/// <item>Multiple blocks of 3x3 convolutional layers with ReLU activation</item>
/// <item>Max pooling (2x2, stride 2) after each block to reduce spatial dimensions</item>
/// <item>Optional batch normalization after each convolution (in _BN variants)</item>
/// <item>Three fully connected layers (4096 -> 4096 -> num_classes)</item>
/// <item>Dropout regularization in the fully connected layers</item>
/// </list>
/// </para>
/// <para>
/// <b>Typical Usage:</b>
/// <code>
/// // Create VGG16 with batch normalization for 10-class classification
/// var config = new VGGConfiguration(VGGVariant.VGG16_BN, numClasses: 10);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224,
///     inputWidth: 224,
///     inputDepth: 3,
///     outputSize: 10,
///     taskType: NeuralNetworkTaskType.MultiClassClassification);
/// var network = new VGGNetwork&lt;float&gt;(architecture, config);
/// </code>
/// </para>
/// </remarks>
public class VGGNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The VGG configuration specifying the variant and parameters.
    /// </summary>
    private readonly VGGConfiguration _configuration;

    /// <summary>
    /// Gets the VGG variant being used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variant determines how deep the network is (VGG11, 13, 16, or 19)
    /// and whether batch normalization is used (variants ending in _BN).
    /// </para>
    /// </remarks>
    public VGGVariant Variant => _configuration.Variant;

    /// <summary>
    /// Gets whether this network uses batch normalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Batch normalization is a technique that normalizes layer inputs,
    /// making training faster and more stable. VGG variants ending in "_BN" use this.
    /// </para>
    /// </remarks>
    public bool UsesBatchNormalization => _configuration.UseBatchNormalization;

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    public int NumClasses => _configuration.NumClasses;

    /// <summary>
    /// Initializes a new instance of the VGGNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="configuration">The VGG-specific configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Optional loss function (default: based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping (default: 1.0).</param>
    /// <exception cref="InvalidInputTypeException">Thrown when the input type is not three-dimensional.</exception>
    /// <exception cref="ArgumentNullException">Thrown when configuration is null.</exception>
    /// <remarks>
    /// <para>
    /// VGG networks require three-dimensional input data (channels, height, width).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a VGG network, you need to provide:
    /// <list type="bullet">
    /// <item>An architecture object that describes the input/output dimensions</item>
    /// <item>A configuration that specifies which VGG variant to use</item>
    /// <item>Optionally, custom optimizer and loss function (good defaults are provided)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public VGGNetwork(
        NeuralNetworkArchitecture<T> architecture,
        VGGConfiguration configuration,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

        ArchitectureValidator.ValidateInputType(
            architecture,
            InputType.ThreeDimensional,
            nameof(VGGNetwork<T>));

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
        VGGConfiguration configuration)
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
    /// Initializes the layers of the VGG network based on the configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// the standard VGG layers based on the configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method builds the VGG network layer by layer. If you've
    /// provided custom layers, it uses those. Otherwise, it creates the standard VGG
    /// architecture with the appropriate number of convolutional blocks for your chosen variant.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use VGG-specific layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultVGGLayers(Architecture, _configuration));
        }
    }

    /// <summary>
    /// Performs a forward pass through the VGG network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process (shape: [batch, channels, height, width]).</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <exception cref="TensorShapeMismatchException">Thrown when the input shape doesn't match expected shape.</exception>
    /// <remarks>
    /// <para>
    /// The forward pass sequentially processes the input through each layer of the network:
    /// convolution blocks, pooling, fully connected layers, and produces class probabilities.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network makes predictions. You give it an image
    /// (as a tensor), and it processes it through all the VGG layers to produce a prediction.
    /// The output contains probabilities for each class.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        TensorValidator.ValidateShape(
            input,
            Architecture.GetInputShape(),
            nameof(VGGNetwork<T>),
            "forward pass");

        // Add batch dimension if input is 3D (single image)
        bool addedBatch = false;
        Tensor<T> processedInput = input;
        if (input.Rank == 3)
        {
            addedBatch = true;
            processedInput = AddBatchDimension(input);
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
    /// <param name="input">The input tensor with shape [channels, height, width].</param>
    /// <returns>A tensor with shape [1, channels, height, width].</returns>
    private static Tensor<T> AddBatchDimension(Tensor<T> input)
    {
        int[] inputShape = input.Shape;
        int[] resultShape = new int[inputShape.Length + 1];

        // Add batch dimension of size 1
        resultShape[0] = 1;

        // Copy remaining dimensions
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 1] = inputShape[i];
        }

        // Use Reshape which should handle the memory layout correctly
        return input.Reshape(resultShape);
    }

    /// <summary>
    /// Performs a backward pass through the network to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
    /// <returns>The gradient of the loss with respect to the network's input.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass propagates gradients through each layer in reverse order,
    /// computing the gradients needed for parameter updates during training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network learns from its mistakes. After making
    /// a prediction, we calculate how wrong it was, and this method propagates that error
    /// backward through all the layers so each layer knows how to adjust its weights.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After calculating how to improve each layer's weights,
    /// this method actually applies those improvements to make the network better.
    /// </para>
    /// </remarks>
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
    /// Makes a prediction using the VGG network for the given input.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor containing class probabilities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use after training. Give it an
    /// image, and it returns probabilities for each class. The class with the highest
    /// probability is the network's prediction.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Trains the VGG network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor (one-hot encoded class labels).</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration: forward pass, loss calculation,
    /// backward pass, and parameter update.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network learns. You show it an image and tell
    /// it what class the image belongs to. The network makes a guess, compares it to the
    /// correct answer, and adjusts its weights to do better next time.
    /// </para>
    /// </remarks>
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
        var gradients = new List<Tensor<T>>();
        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, currentGradient);
        }

        // Update parameters
        UpdateParameters(gradients);
    }

    /// <summary>
    /// Updates the parameters of the network based on the calculated gradients.
    /// </summary>
    private void UpdateParameters(List<Tensor<T>> gradients)
    {
        ClipGradients(gradients);
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
    /// VGG networks are known for having a large number of parameters, primarily in the
    /// fully connected layers. For example:
    /// <list type="bullet">
    /// <item>VGG16: ~138 million parameters</item>
    /// <item>VGG19: ~144 million parameters</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Parameters are the learnable weights in the network. More
    /// parameters means more capacity to learn complex patterns, but also requires more
    /// memory and training data. VGG networks have many parameters because of their
    /// large fully connected layers.
    /// </para>
    /// </remarks>
    public new int GetParameterCount()
    {
        return base.GetParameterCount();
    }

    /// <summary>
    /// Retrieves metadata about the VGG network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns a summary of the network's structure and configuration,
    /// including the VGG variant, input/output shapes, and layer information.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConvolutionalNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "VGG" },
                { "Variant", _configuration.Variant.ToString() },
                { "NumClasses", _configuration.NumClasses },
                { "UseBatchNormalization", _configuration.UseBatchNormalization },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Layers[Layers.Count - 1].GetOutputShape() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "NumConvLayers", _configuration.NumConvLayers },
                { "NumWeightLayers", _configuration.NumWeightLayers },
                { "DropoutRate", _configuration.DropoutRate }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes VGG network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write VGG configuration
        writer.Write((int)_configuration.Variant);
        writer.Write(_configuration.NumClasses);
        writer.Write(_configuration.InputHeight);
        writer.Write(_configuration.InputWidth);
        writer.Write(_configuration.InputChannels);
        writer.Write(_configuration.DropoutRate);
        writer.Write(_configuration.IncludeClassifier);
        writer.Write(_configuration.UseAutodiff);
    }

    /// <summary>
    /// Deserializes VGG network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read VGG configuration - note: configuration is set in constructor
        // This is for validation/compatibility checking
        var variant = (VGGVariant)reader.ReadInt32();
        var numClasses = reader.ReadInt32();
        var inputHeight = reader.ReadInt32();
        var inputWidth = reader.ReadInt32();
        var inputChannels = reader.ReadInt32();
        var dropoutRate = reader.ReadDouble();
        var includeClassifier = reader.ReadBoolean();
        var useAutodiff = reader.ReadBoolean();

        // Validate loaded configuration matches current
        if (variant != _configuration.Variant)
        {
            throw new InvalidOperationException(
                $"Serialized VGG variant ({variant}) does not match current configuration ({_configuration.Variant}).");
        }
    }

    /// <summary>
    /// Creates a new instance of the VGG network model.
    /// </summary>
    /// <returns>A new instance of the VGG network with the same configuration.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VGGNetwork<T>(
            Architecture,
            _configuration,
            _optimizer,
            _lossFunction,
            Convert.ToDouble(MaxGradNorm)
        );
    }
}
