global using AiDotNet.Validation;

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
    /// Initializes a new instance of the ConvolutionalNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <exception cref="InvalidInputTypeException">Thrown when the input type is not three-dimensional.</exception>
    /// <remarks>
    /// <para>
    /// CNNs require three-dimensional input data (typically height, width, and channels for images).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a CNN, you need to provide a blueprint (architecture) 
    /// that defines how your network will be structured. This constructor checks that your input 
    /// data has three dimensions - for images, these dimensions typically represent height, width, 
    /// and color channels (like RGB). If you try to use data with the wrong dimensions, the 
    /// constructor will raise an error to let you know.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        ArchitectureValidator.ValidateInputType(architecture, InputType.ThreeDimensional, nameof(ConvolutionalNeuralNetwork<T>));
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCNNLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the neural network based on the provided input vector.
    /// </summary>
    /// <param name="input">The input vector containing the data to process.</param>
    /// <returns>A vector containing the network's prediction.</returns>
    /// <exception cref="VectorLengthMismatchException">Thrown when the input vector length doesn't match the expected input dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method converts the input vector to a tensor with the appropriate shape,
    /// processes it through the network, and returns the result as a vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use to get predictions from your CNN. 
    /// You provide your input data (like an image) as a flat list of numbers (a vector), and this 
    /// method reshapes it into the proper 3D format, runs it through the neural network, and gives 
    /// you back the prediction results (also as a flat list). The method checks that your input 
    /// has the correct number of values before processing.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        // Convert the input Vector to a Tensor with the correct shape
        var inputShape = Architecture.GetInputShape();
    
        // Validate that the input vector length matches the expected shape
        VectorValidator.ValidateLengthForShape(input, inputShape, nameof(ConvolutionalNeuralNetwork<T>), "Predict");

        var inputTensor = new Tensor<T>(inputShape, input);

        // Perform forward pass
        var output = Forward(inputTensor);

        // Flatten the output Tensor to a Vector
        return new Vector<T>([.. output]);
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
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(), 
            nameof(ConvolutionalNeuralNetwork<T>), "forward pass");

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
    /// Saves the neural network's state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the network state to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the number of layers and the state of each layer.
    /// It allows you to save a trained model for later use without retraining.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training a neural network can take a lot of time and computing power. 
    /// This method allows you to save your trained network to a file so you can use it later 
    /// without having to train it again. It's like saving your progress in a video game - you 
    /// don't want to start from the beginning every time you play. The method saves all the 
    /// network's learned knowledge and structure.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        SerializationValidator.ValidateWriter(writer, nameof(ConvolutionalNeuralNetwork<T>));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            string? layerTypeName = layer.GetType().FullName;
            SerializationValidator.ValidateLayerTypeName(layerTypeName);
            writer.Write(layerTypeName!);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Loads the neural network's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the network state from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when there's an issue with layer type information.</exception>
    /// <exception cref="SerializationException">Thrown when there's an error instantiating or deserializing a layer.</exception>
    /// <remarks>
    /// <para>
    /// This method loads the number of layers and reconstructs each layer from the saved state.
    /// It allows you to load a previously trained model without retraining.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is like opening a saved file in a program. It reads a file 
    /// containing your previously trained neural network and rebuilds it exactly as it was when you 
    /// saved it. This is extremely useful because training neural networks can take hours or even days. 
    /// With this method, you can save a trained network and reload it whenever you need to use it, 
    /// without having to train it again from scratch. Think of it like loading a saved game instead 
    /// of starting a new one.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        SerializationValidator.ValidateReader(reader, nameof(ConvolutionalNeuralNetwork<T>));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            SerializationValidator.ValidateLayerTypeName(layerTypeName);

            Type? layerType = Type.GetType(layerTypeName);
            SerializationValidator.ValidateLayerTypeExists(layerTypeName, layerType, nameof(ConvolutionalNeuralNetwork<T>));

            try
            {
                ILayer<T> layer = (ILayer<T>)Activator.CreateInstance(layerType!)!;
                layer.Deserialize(reader);
                Layers.Add(layer);
            }
            catch (Exception ex) when (ex is not SerializationException)
            {
                throw new SerializationException(
                    $"Failed to instantiate or deserialize layer of type {layerTypeName}",
                    nameof(ConvolutionalNeuralNetwork<T>),
                    "Deserialize",
                    ex);
            }
        }
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    /// <returns>The total number of parameters across all layers.</returns>
    /// <remarks>
    /// <para>
    /// This count includes weights and biases from all layers in the network.
    /// The parameter count is useful for understanding the complexity of the model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Neural networks learn by adjusting thousands or even millions of internal 
    /// values called "parameters." This method tells you how many parameters your network has in total. 
    /// Think of parameters like the individual knobs and dials that the network can adjust to improve 
    /// its predictions. More parameters generally mean the network can learn more complex patterns, 
    /// but also requires more data to train effectively and more memory to store. For example, a simple 
    /// network might have a few thousand parameters, while advanced image recognition networks can have 
    /// millions or billions of parameters.
    /// </para>
    /// </remarks>
    public override int GetParameterCount()
    {
        return Layers.Sum(layer => layer.ParameterCount);
    }
}