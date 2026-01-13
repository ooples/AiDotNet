namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Feed-Forward Neural Network (FFNN) for processing data in a forward path.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Feed-Forward Neural Network is the simplest type of artificial neural network, where connections
/// between nodes do not form a cycle. Information moves in only one direction -- forward -- from the input
/// nodes, through the hidden nodes (if any), and to the output nodes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a Feed-Forward Neural Network like a series of information-processing
/// stages arranged in a line. Data flows only forward through these stages, never backward. Each stage
/// (or layer) processes the information and passes it to the next stage. This simple structure makes
/// FFNNs great for many common tasks like classification (deciding which category something belongs to)
/// or regression (predicting a numerical value).
/// </para>
/// </remarks>
public class FeedForwardNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This function measures how well the network is performing and guides the learning process.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as the scorekeeper for the network. It tells the network
    /// how far off its predictions are from the correct answers.
    /// </para>
    /// </remarks>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer determines how the network's internal values are adjusted based on the calculated error.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like the network's learning strategy. It decides how to adjust
    /// the network's settings to improve its performance over time.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Initializes a new instance of the FeedForwardNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, an appropriate loss function is selected based on the task type.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
    /// <remarks>
    /// <para>
    /// Feed-Forward Neural Networks can work with various input dimensions and are typically used for
    /// classification and regression tasks with structured data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a Feed-Forward Neural Network, you need to provide a blueprint (architecture)
    /// that defines the structure of your network. This constructor sets up the network based on that blueprint.
    /// It also prepares the learning strategy (optimizer) and the way to measure mistakes (loss function).
    /// If you don't specify these, it chooses reasonable defaults based on the type of task you're trying to solve.
    /// </para>
    /// </remarks>
    public FeedForwardNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0) : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Select appropriate loss function based on task type if not provided
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default feed-forward layers if none are specified.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the different processing stages (layers) of your
    /// neural network. If you've specified custom layers in your architecture, it will use those.
    /// If not, it will create a standard set of layers commonly used for feed-forward networks,
    /// with the right number of neurons at each stage based on your architecture settings.
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
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultFeedForwardLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the feed-forward neural network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network to generate a prediction.
    /// Unlike the vector-based Predict method, this takes a tensor directly as input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method makes predictions using the trained neural network.
    /// It accepts input data in tensor format (multi-dimensional arrays), processes it through
    /// all the network's layers, and returns the prediction as a tensor. This is useful when
    /// working with data that naturally has multiple dimensions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Ensure the network is in inference mode
        IsTrainingMode = false;

        // Validate input shape
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(FeedForwardNeuralNetwork<T>), "prediction");

        // Just perform a forward pass
        var predictions = Forward(input);

        IsTrainingMode = true;

        return predictions;
    }

    /// <summary>
    /// Performs a forward pass through the network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <exception cref="TensorShapeMismatchException">Thrown when the input shape doesn't match the expected input shape.</exception>
    /// <remarks>
    /// <para>
    /// The forward pass sequentially processes the input through each layer of the network.
    /// This is the core operation for making predictions with the neural network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes your input data and passes it through each layer
    /// of the neural network in sequence. Think of it like an assembly line where each station (layer)
    /// processes the data and passes it to the next station. The final output contains the network's prediction.
    /// This is the engine that powers the prediction process.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(FeedForwardNeuralNetwork<T>), "forward pass");

        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // CPU path: each layer processes input and may download results
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Performs a backward pass through the network to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
    /// <returns>The gradient of the loss with respect to the network's input.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass is used during training to update the network's parameters.
    /// It propagates the gradient backward through each layer, starting from the output layer.
    /// This process is known as "backpropagation" and is essential for training neural networks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> While the forward pass makes predictions, the backward pass is how
    /// the network learns from its mistakes. After making a prediction, we calculate how wrong
    /// the prediction was (the error). This method takes that error and works backward through
    /// the network, calculating how each part contributed to the mistake. This information is
    /// then used to adjust the network's internal settings to make better predictions next time.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer based on their parameter count.
    /// It's typically called during training after calculating parameter updates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After the backward pass calculates how to improve the network,
    /// this method actually applies those improvements. It takes a list of updated settings
    /// (parameters) and distributes them to each layer in the network. This method is
    /// called repeatedly during training to gradually improve the network's accuracy.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Trains the feed-forward neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration, including forward pass, loss calculation,
    /// backward pass, and parameter update.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network learns. You show it some input data and
    /// tell it what the correct output should be. The network makes a guess, compares it to
    /// the correct answer, and then adjusts its internal settings to do better next time.
    /// This process is repeated many times with different examples to train the network.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;

        // Forward pass to get prediction
        var prediction = Forward(input);

        // Calculate primary loss
        var primaryLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Calculate auxiliary losses from layers that support them
        T auxiliaryLoss = NumOps.Zero;
        foreach (var auxLayer in Layers.OfType<IAuxiliaryLossLayer<T>>().Where(l => l.UseAuxiliaryLoss))
        {
            var layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
            var weightedAuxLoss = NumOps.Multiply(layerAuxLoss, auxLayer.AuxiliaryLossWeight);
            auxiliaryLoss = NumOps.Add(auxiliaryLoss, weightedAuxLoss);
        }

        // Combine primary and auxiliary losses
        LastLoss = NumOps.Add(primaryLoss, auxiliaryLoss);

        // Calculate output gradient (derivative of loss with respect to network output)
        // Note: Auxiliary loss gradients are handled within each layer's backward pass
        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = Tensor<T>.FromVector(outputGradient);

        // Backpropagation
        var inputGradient = Backward(outputGradientTensor);

        // Update parameters using the optimizer
        _optimizer.UpdateParameters(Layers);

        IsTrainingMode = false;
    }

    /// <summary>
    /// Retrieves metadata about the feed-forward neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the network's structure and configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like getting a summary of the network's blueprint. It tells you
    /// how many layers it has, what types of layers they are, and other important details about how
    /// the network is set up. This can be useful for documentation or debugging purposes.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.FeedForwardNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes feed-forward neural network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific parameters and state of the feed-forward neural network to a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like saving the network's current state to a file. It records all
    /// the important information about the network so you can reload it later exactly as it is now.
    /// This is useful when you want to save a trained model for later use.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize optimizer and loss function interfaces
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Deserialize and restore optimizer
        var optimizer = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
        if (optimizer != null)
        {
            _optimizer = optimizer;
        }

        // Deserialize and restore loss function
        var lossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        if (lossFunction != null)
        {
            _lossFunction = lossFunction;
        }
    }

    /// <summary>
    /// Creates a new instance of the FeedForwardNeuralNetwork with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new FeedForwardNeuralNetwork instance with the same architecture, optimizer, and loss function as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the FeedForwardNeuralNetwork with the same architecture, optimizer, and loss function
    /// as the current instance. This is useful for model cloning, ensemble methods, or cross-validation scenarios where
    /// multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method creates a fresh copy of the neural network's blueprint.
    /// 
    /// When you need multiple versions of the same type of neural network with identical settings:
    /// - This method creates a new, empty network with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new network has the same structure but no trained data
    /// - This is useful for techniques that need multiple models, like ensemble methods
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new FeedForwardNeuralNetwork<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            Convert.ToDouble(MaxGradNorm));
    }

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Feed-forward neural networks support training through backpropagation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This indicates that the network can learn from data.
    /// Feed-forward networks are designed to be trained, so this property returns true.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;
}
