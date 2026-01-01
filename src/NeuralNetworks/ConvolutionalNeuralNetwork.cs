global using AiDotNet.Validation;
using AiDotNet.Exceptions;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Convolutional Neural Network (CNN) that processes multi-dimensional data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Convolutional Neural Network is specialized for processing data with a grid-like structure,
/// such as images. It uses convolutional layers to automatically detect important features
/// without manual feature extraction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a CNN as an image recognition system that works similarly to how 
/// your eyes and brain process visual information. Just as your brain automatically notices 
/// patterns like edges, shapes, and textures without conscious effort, a CNN automatically 
/// learns to detect these features in images. This makes CNNs excellent for tasks like 
/// recognizing objects in photos, detecting faces, or reading handwritten text.
/// </para>
/// </remarks>
public class ConvolutionalNeuralNetwork<T> : NeuralNetworkBase<T>
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
    /// Initializes a new instance of the ConvolutionalNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <remarks>
    /// <para>
    /// CNNs are typically used with three-dimensional input data (height, width, and channels for images),
    /// but this implementation accepts any rank and lets layers adapt the dimensions as needed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a CNN, you need to provide a blueprint (architecture) 
    /// that defines how your network will be structured. This implementation supports inputs of
    /// different ranks (1D, 2D, 3D, or batched 4D+), and the layers will handle reshaping and
    /// dimension adaptation internally.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0) : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default CNN layers if none are specified.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the different processing stages (layers) of your 
    /// neural network. If you've specified custom layers in your architecture, it will use those. 
    /// If not, it will create a standard set of layers commonly used in image recognition tasks. 
    /// Think of this as either following your custom recipe or using a tried-and-tested recipe 
    /// if you haven't provided one.
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
            // Pass the architecture's output size to ensure the output layer has correct dimensions
            Layers.AddRange(LayerHelper<T>.CreateDefaultCNNLayers(Architecture, outputSize: Architecture.OutputSize));
        }
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
    /// <b>For Beginners:</b> This method takes your input data (already in the right 3D format) 
    /// and passes it through each layer of the neural network in sequence. Think of it like an 
    /// assembly line where each station (layer) processes the data and passes it to the next station. 
    /// The final output contains the network's prediction. This is the engine that powers the 
    /// prediction process.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var expectedShape = Architecture.GetInputShape();
        Tensor<T> processedInput;
        bool hasBatchDims = false;
        int[] batchDimSizes = [];
        int batchSize = 1;

        // Handle any-rank tensors by supporting batch dimensions
        if (input.Rank == expectedShape.Length && ShapesEqual(input.Shape, expectedShape))
        {
            // Exact match - no batch dimensions
            processedInput = input;
        }
        else if (input.Rank > expectedShape.Length)
        {
            // Input has extra dimensions (batch dimensions)
            int extraDims = input.Rank - expectedShape.Length;
            var actualSpatial = input.Shape.Skip(extraDims).ToArray();

            if (!ShapesEqual(actualSpatial, expectedShape))
            {
                throw new TensorShapeMismatchException(
                    $"Shape mismatch in ConvolutionalNeuralNetwork during forward pass: " +
                    $"Expected spatial dimensions [{string.Join(", ", expectedShape)}], " +
                    $"but got [{string.Join(", ", actualSpatial)}].");
            }

            hasBatchDims = true;
            batchDimSizes = input.Shape.Take(extraDims).ToArray();
            for (int i = 0; i < extraDims; i++)
            {
                batchSize *= batchDimSizes[i];
            }

            processedInput = input;
        }
        else
        {
            // Input has fewer dimensions than expected
            throw new TensorShapeMismatchException(
                $"Shape mismatch in ConvolutionalNeuralNetwork during forward pass: " +
                $"Expected shape [{string.Join(", ", expectedShape)}], but got [{string.Join(", ", input.Shape)}].");
        }

        // If we have batch dimensions, process each batch element
        if (hasBatchDims && batchSize > 1)
        {
            int elementsPerBatch = expectedShape.Aggregate(1, (a, b) => a * b);
            var outputs = new List<Tensor<T>>();

            for (int b = 0; b < batchSize; b++)
            {
                // Extract single element from batch
                var elementData = new T[elementsPerBatch];
                Array.Copy(processedInput.Data.ToArray(), b * elementsPerBatch, elementData, 0, elementsPerBatch);
                var element = new Tensor<T>(expectedShape, new Vector<T>(elementData));

                // Process through all layers
                Tensor<T> output = element;
                foreach (var layer in Layers)
                {
                    output = layer.Forward(output);
                }
                outputs.Add(output);
            }

            // Combine outputs back into batched tensor
            if (outputs.Count > 0)
            {
                var outputShape = new int[batchDimSizes.Length + outputs[0].Rank];
                Array.Copy(batchDimSizes, 0, outputShape, 0, batchDimSizes.Length);
                Array.Copy(outputs[0].Shape, 0, outputShape, batchDimSizes.Length, outputs[0].Rank);

                var combinedData = new T[batchSize * outputs[0].Length];
                for (int b = 0; b < batchSize; b++)
                {
                    Array.Copy(outputs[b].Data.ToArray(), 0, combinedData, b * outputs[0].Length, outputs[0].Length);
                }

                return new Tensor<T>(outputShape, new Vector<T>(combinedData));
            }
        }

        // Single element or batch size 1 - process directly through layers
        Tensor<T> result = processedInput;

        // For batch size 1, extract the single element
        if (hasBatchDims && batchSize == 1)
        {
            int elementsPerBatch = expectedShape.Aggregate(1, (a, b) => a * b);
            var elementData = new T[elementsPerBatch];
            Array.Copy(processedInput.Data.ToArray(), 0, elementData, 0, elementsPerBatch);
            result = new Tensor<T>(expectedShape, new Vector<T>(elementData));
        }

        foreach (var layer in Layers)
        {
            result = layer.Forward(result);
        }

        // Restore batch dimensions if needed
        if (hasBatchDims)
        {
            var outputShape = new int[batchDimSizes.Length + result.Rank];
            Array.Copy(batchDimSizes, 0, outputShape, 0, batchDimSizes.Length);
            Array.Copy(result.Shape, 0, outputShape, batchDimSizes.Length, result.Rank);
            return new Tensor<T>(outputShape, new Vector<T>(result.Data.ToArray()));
        }

        return result;
    }

    /// <summary>
    /// Checks if two shapes are equal.
    /// </summary>
    private static bool ShapesEqual(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length) return false;
        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i]) return false;
        }
        return true;
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
    /// Think of it like learning from feedback - you make a guess, see how close you were, and 
    /// adjust your thinking for next time.
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
    /// This method distributes the parameters to each layer based on their parameter count.
    /// It's typically called during training after calculating parameter updates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After the backward pass calculates how to improve the network, 
    /// this method actually applies those improvements. It takes a list of updated settings 
    /// (parameters) and distributes them to each layer in the network. Think of it like 
    /// fine-tuning each part of a machine based on performance feedback. This method is 
    /// called repeatedly during training to gradually improve the network's accuracy.
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
    /// Makes a prediction using the convolutional neural network for the given input.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network to generate a prediction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like asking the network to recognize an image. You give it 
    /// the image data, and it processes it through all its layers to give you its best guess 
    /// about what's in the image.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Trains the convolutional neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration, including forward pass, loss calculation,
    /// backward pass, and parameter update.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network learns. You show it an image (input) and 
    /// tell it what should be in that image (expected output). The network makes a guess, 
    /// compares it to the correct answer, and then adjusts its internal settings to do better next time.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var prediction = Predict(input);

        // Calculate loss
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Store the last loss value
        LastLoss = loss;

        // Calculate output gradient
        var outputGradient = CalculateOutputGradient(prediction, expectedOutput);

        // Convert output gradient back to a tensor
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
    /// <param name="gradients">A list of tensors containing the gradients for each layer.</param>
    /// <remarks>
    /// <para>
    /// This method applies gradient clipping to prevent exploding gradients and then uses the optimizer
    /// to update the parameters of each layer in the network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the network actually learns. After calculating how wrong
    /// the network was (the gradients), this method carefully adjusts the network's internal settings.
    /// It first makes sure the adjustments aren't too big (gradient clipping), then uses a smart
    /// adjustment strategy (the optimizer) to make the network a little bit better at its job.
    /// </para>
    /// </remarks>
    private void UpdateParameters(List<Tensor<T>> gradients)
    {
        // Apply gradient clipping
        ClipGradients(gradients);

        // Use the optimizer to update parameters
        _optimizer.UpdateParameters(Layers);
    }

    /// <summary>
    /// Calculates the gradient of the loss with respect to the network's output.
    /// </summary>
    /// <param name="prediction">The predicted output tensor from the network.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <returns>A vector representing the gradient of the loss with respect to the network's output.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the loss function to compute the derivative of the loss with respect to the
    /// network's output. It flattens the tensors to vectors before calculation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network figures out how wrong its guess was. It compares
    /// the network's prediction to the correct answer and calculates how much it needs to change to
    /// get closer to the right answer. This "how much to change" is called the gradient, and it's
    /// used to guide the learning process.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateOutputGradient(Tensor<T> prediction, Tensor<T> expectedOutput)
    {
        return _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For CNNs, the parameter count includes weights and biases from convolutional layers, pooling layers,
    /// and fully connected layers. The computation typically includes:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Convolutional layer parameters: (kernel_height × kernel_width × input_channels × output_channels) + output_channels (biases)</description></item>
    /// <item><description>Fully connected layer parameters: (input_size × output_size) + output_size (biases)</description></item>
    /// <item><description>Pooling layers typically have no trainable parameters</description></item>
    /// </list>
    /// <para>
    /// <b>For Beginners:</b> CNNs usually have parameters in their convolutional filters (which detect features like
    /// edges and patterns) and in their fully connected layers (which make the final classification). The total number
    /// depends on the filter sizes, number of filters, and size of the fully connected layers. Larger kernels and more
    /// filters mean more parameters and thus more computational requirements.
    /// </para>
    /// </remarks>
    public new int GetParameterCount()
    {
        return base.GetParameterCount();
    }

    /// <summary>
    /// Retrieves metadata about the convolutional neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the network's structure and configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like getting a summary of the network's blueprint. It tells you
    /// how many layers it has, what types of layers they are, and other important details about how
    /// the network is set up.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConvolutionalNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Layers[Layers.Count - 1].GetOutputShape() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes convolutional neural network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific parameters and state of the convolutional neural network to a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like saving the network's current state to a file. It records all 
    /// the important information about the network so you can reload it later exactly as it is now.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
    }

    /// <summary>
    /// Deserializes convolutional neural network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific parameters and state of the convolutional neural network from a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like loading a saved network state from a file. It rebuilds the 
    /// network exactly as it was when you saved it, including all its learned information.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
    }

    /// <summary>
    /// Creates a new instance of the convolutional neural network model.
    /// </summary>
    /// <returns>A new instance of the convolutional neural network model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the convolutional neural network model with the same 
    /// configuration as the current instance. It is used internally during serialization/deserialization 
    /// processes to create a fresh instance that can be populated with the serialized data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method creates a copy of the network structure without copying 
    /// the learned data. Think of it like making a blank copy of the original network's blueprint - 
    /// it has the same structure, same learning strategy, and same error measurement, but none of 
    /// the knowledge that the original network has gained through training. This is primarily 
    /// used when saving or loading models, creating an empty framework that can later be filled 
    /// with the saved knowledge from the original network.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ConvolutionalNeuralNetwork<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            Convert.ToDouble(MaxGradNorm)
        );
    }
}
