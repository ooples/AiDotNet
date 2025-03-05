global using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Convolutional Neural Network (CNN) that processes multi-dimensional data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// A Convolutional Neural Network is specialized for processing data with a grid-like structure,
/// such as images. It uses convolutional layers to automatically detect important features
/// without manual feature extraction.
/// </remarks>
public class ConvolutionalNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Initializes a new instance of the ConvolutionalNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <exception cref="InvalidInputTypeException">Thrown when the input type is not three-dimensional.</exception>
    /// <remarks>
    /// CNNs require three-dimensional input data (typically height, width, and channels for images).
    /// </remarks>
    public ConvolutionalNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        ArchitectureValidator.ValidateInputType(architecture, InputType.ThreeDimensional, nameof(ConvolutionalNeuralNetwork<T>));
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// This method either uses custom layers provided in the architecture or creates
    /// default CNN layers if none are specified.
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
    /// This method converts the input vector to a tensor with the appropriate shape,
    /// processes it through the network, and returns the result as a vector.
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
    /// The forward pass sequentially processes the input through each layer of the network.
    /// This is the core operation for making predictions with the neural network.
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
    /// The backward pass is used during training to update the network's parameters.
    /// It propagates the gradient backward through each layer, starting from the output layer.
    /// This process is known as "backpropagation" and is essential for training neural networks.
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
    /// This method distributes the parameters to each layer based on their parameter count.
    /// It's typically called during training after calculating parameter updates.
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
    /// This method saves the number of layers and the state of each layer.
    /// It allows you to save a trained model for later use without retraining.
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
    /// <remarks>
    /// This method loads the number of layers and reconstructs each layer from the saved state.
    /// It allows you to load a previously trained model without retraining.
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
    /// This count includes weights and biases from all layers in the network.
    /// The parameter count is useful for understanding the complexity of the model.
    /// </remarks>
    public override int GetParameterCount()
    {
        return Layers.Sum(layer => layer.ParameterCount);
    }
}