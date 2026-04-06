using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a reconstruction layer that uses multiple fully connected layers to transform inputs into outputs.
/// </summary>
/// <remarks>
/// <para>
/// The ReconstructionLayer is a composite layer that consists of three fully connected layers in sequence.
/// It is typically used in autoencoders or similar architectures to reconstruct the original input from a 
/// compressed representation. The layer provides a deeper transformation path through multiple hidden layers,
/// allowing it to learn more complex reconstruction functions than a single layer could.
/// </para>
/// <para><b>For Beginners:</b> This layer works like a mini-network within your neural network.
/// 
/// Think of the ReconstructionLayer as a specialized team of artists:
/// - The first artist (first fully connected layer) makes a rough sketch
/// - The second artist (second fully connected layer) adds details to the sketch
/// - The third artist (third fully connected layer) finalizes the artwork
/// 
/// In an autoencoder network (a common use for this layer):
/// - Earlier layers compress your data into a compact form (like squeezing information)
/// - This reconstruction layer carefully expands that compact form back to the original size
/// - It learns how to restore information that was "squeezed out" during compression
/// 
/// For example, if you're building an image autoencoder, this layer would help transform
/// the compressed representation back into an image that looks like the original.
/// 
/// By using three layers instead of just one, this layer can learn more sophisticated
/// patterns for reconstructing the data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, 4, 4, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public class ReconstructionLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The first fully connected layer in the reconstruction sequence.
    /// </summary>
    /// <remarks>
    /// This layer processes the input and transforms it to the first hidden dimension.
    /// It applies the hidden activation function to its output.
    /// </remarks>
    private int _hidden1Dim;
    private int _hidden2Dim;
    private readonly FullyConnectedLayer<T> _fc1;

    /// <summary>
    /// The second fully connected layer in the reconstruction sequence.
    /// </summary>
    /// <remarks>
    /// This layer takes the output from the first layer and transforms it to the second hidden dimension.
    /// It also applies the hidden activation function to its output.
    /// </remarks>
    private readonly FullyConnectedLayer<T> _fc2;

    /// <summary>
    /// The third fully connected layer in the reconstruction sequence.
    /// </summary>
    /// <remarks>
    /// This layer takes the output from the second layer and transforms it to the final output dimension.
    /// It applies the output activation function, which is often sigmoid for reconstruction tasks.
    /// </remarks>
    private readonly FullyConnectedLayer<T> _fc3;

    /// <summary>
    /// Flag indicating whether vector activation functions are used instead of scalar activation functions.
    /// </summary>
    /// <remarks>
    /// When true, the layer uses vector activation functions that operate on entire vectors at once.
    /// When false, the layer uses scalar activation functions that operate on individual elements.
    /// </remarks>
    private bool _useVectorActivation;

    /// <summary>
    /// Gets the total number of trainable parameters in the reconstruction layer.
    /// </summary>
    /// <value>
    /// The sum of parameter counts from all three fully connected layers.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the total number of trainable parameters (weights and biases) across all three
    /// fully connected layers that make up this reconstruction layer. This is useful for monitoring the
    /// complexity of the layer or for parameter initialization strategies.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you how many numbers the layer can adjust during training.
    /// 
    /// Each parameter is a number that the neural network learns:
    /// - More parameters mean the layer can learn more complex patterns
    /// - More parameters also require more training data and time
    /// - This layer has parameters in all three of its internal layers
    /// 
    /// Think of parameters like knobs that the network can turn to get better results.
    /// This property tells you the total number of knobs available to this layer.
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        _fc1.ParameterCount + _fc2.ParameterCount + _fc3.ParameterCount;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for ReconstructionLayer, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the ReconstructionLayer has trainable parameters that can be optimized
    /// during the training process using backpropagation. The actual parameters are contained within the
    /// three fully connected layers that make up this layer.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// During training, this layer will learn how to best reconstruct outputs from inputs,
    /// adapting its internal parameters to the specific patterns in your data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Hidden1Dim"] = _fc1.GetOutputShape()[0].ToString();
        metadata["Hidden2Dim"] = _fc2.GetOutputShape()[0].ToString();
        metadata["UseVectorActivation"] = _useVectorActivation.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <remarks>
    /// Returns true since the internal FullyConnectedLayers support GPU execution.
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReconstructionLayer{T}"/> class with scalar activation functions.
    /// </summary>
    /// <param name="inputDimension">The size of the input to the layer.</param>
    /// <param name="hidden1Dimension">The size of the first hidden layer.</param>
    /// <param name="hidden2Dimension">The size of the second hidden layer.</param>
    /// <param name="outputDimension">The size of the output from the layer.</param>
    /// <param name="hiddenActivation">The activation function to apply to hidden layers. Defaults to ReLU if not specified.</param>
    /// <param name="outputActivation">The activation function to apply to the output layer. Defaults to Sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReconstructionLayer with the specified dimensions and scalar activation functions.
    /// It initializes three fully connected layers in sequence, with the output of each layer feeding into the input of the next.
    /// The hidden layers use the specified hidden activation function (or ReLU by default), and the output layer uses the
    /// specified output activation function (or Sigmoid by default).
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new reconstruction layer for your neural network using simple activation functions.
    /// 
    /// When you create this layer, you specify:
    /// - inputDimension: How many features come into the layer
    /// - hidden1Dimension: How many neurons are in the first internal layer
    /// - hidden2Dimension: How many neurons are in the second internal layer
    /// - outputDimension: How many features you want in the output
    /// - hiddenActivation: How to transform values in the hidden layers (defaults to ReLU)
    /// - outputActivation: How to transform the final output (defaults to Sigmoid)
    /// 
    /// The hidden dimensions control the "processing power" of the layer:
    /// - Larger hidden dimensions can learn more complex patterns but require more memory
    /// - In autoencoders, these dimensions are often larger than the input/output to expand the compressed representation
    /// 
    /// The activation functions shape how information flows through the layer:
    /// - ReLU is good for hidden layers as it helps with gradient flow
    /// - Sigmoid is good for outputs that should be between 0 and 1 (like pixel values in images)
    /// </para>
    /// </remarks>
    public ReconstructionLayer(
        int inputDimension,
        int hidden1Dimension,
        int hidden2Dimension,
        int outputDimension,
        IActivationFunction<T>? hiddenActivation = null,
        IActivationFunction<T>? outputActivation = null)
        : base([inputDimension], [outputDimension])
    {
        _useVectorActivation = false;
        _hidden1Dim = hidden1Dimension;
        _hidden2Dim = hidden2Dimension;
        hiddenActivation ??= new ReLUActivation<T>();
        outputActivation ??= new SigmoidActivation<T>();

        _fc1 = new FullyConnectedLayer<T>(inputDimension, hidden1Dimension, hiddenActivation);
        _fc2 = new FullyConnectedLayer<T>(hidden1Dimension, hidden2Dimension, hiddenActivation);
        _fc3 = new FullyConnectedLayer<T>(hidden2Dimension, outputDimension, outputActivation);

        RegisterSubLayer(_fc1);
        RegisterSubLayer(_fc2);
        RegisterSubLayer(_fc3);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ReconstructionLayer{T}"/> class with vector activation functions.
    /// </summary>
    /// <param name="inputDimension">The size of the input to the layer.</param>
    /// <param name="hidden1Dimension">The size of the first hidden layer.</param>
    /// <param name="hidden2Dimension">The size of the second hidden layer.</param>
    /// <param name="outputDimension">The size of the output from the layer.</param>
    /// <param name="hiddenVectorActivation">The vector activation function to apply to hidden layers. Defaults to ReLU if not specified.</param>
    /// <param name="outputVectorActivation">The vector activation function to apply to the output layer. Defaults to Sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReconstructionLayer with the specified dimensions and vector activation functions.
    /// It initializes three fully connected layers in sequence, with the output of each layer feeding into the input of the next.
    /// The hidden layers use the specified hidden vector activation function (or ReLU by default), and the output layer uses the
    /// specified output vector activation function (or Sigmoid by default). Vector activation functions operate on entire vectors
    /// at once, allowing for interactions between different elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new reconstruction layer for your neural network using advanced activation functions.
    /// 
    /// When you create this layer, you specify the same dimensions as the scalar version, but with vector activation functions:
    /// - Vector activations process all outputs together as a group, rather than individually
    /// - This can capture relationships between different output elements
    /// - It's useful for outputs that need to maintain certain properties across all values
    /// 
    /// For example, in an image generation task:
    /// - A vector activation might ensure proper contrast across the entire image
    /// - It could maintain relationships between neighboring pixels
    /// 
    /// This version of the constructor is more advanced and used less frequently than the scalar version,
    /// but it can be powerful for specific types of reconstruction tasks.
    /// </para>
    /// </remarks>
    public ReconstructionLayer(
        int inputDimension,
        int hidden1Dimension,
        int hidden2Dimension,
        int outputDimension,
        IVectorActivationFunction<T>? hiddenVectorActivation = null,
        IVectorActivationFunction<T>? outputVectorActivation = null)
        : base([inputDimension], [outputDimension])
    {
        _useVectorActivation = true;
        _hidden1Dim = hidden1Dimension;
        _hidden2Dim = hidden2Dimension;
        hiddenVectorActivation ??= new ReLUActivation<T>();
        outputVectorActivation ??= new SigmoidActivation<T>();

        _fc1 = new FullyConnectedLayer<T>(inputDimension, hidden1Dimension, hiddenVectorActivation);
        _fc2 = new FullyConnectedLayer<T>(hidden1Dimension, hidden2Dimension, hiddenVectorActivation);
        _fc3 = new FullyConnectedLayer<T>(hidden2Dimension, outputDimension, outputVectorActivation);

        RegisterSubLayer(_fc1);
        RegisterSubLayer(_fc2);
        RegisterSubLayer(_fc3);
    }

    /// <summary>
    /// Performs the forward pass of the reconstruction layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after reconstruction processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the reconstruction layer. It sequentially passes the input through
    /// the three fully connected layers, with each layer's output becoming the input to the next layer. The final
    /// output represents the reconstructed data.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the reconstruction layer.
    /// 
    /// During the forward pass:
    /// 1. Your input data goes through the first fully connected layer
    /// 2. The output from the first layer goes through the second layer
    /// 3. The output from the second layer goes through the third layer
    /// 4. The output from the third layer is the final reconstruction
    /// 
    /// This step-by-step transformation allows the layer to gradually reconstruct complex patterns.
    /// Each layer in the sequence adds detail to the reconstruction, similar to how an artist might
    /// start with a rough sketch and gradually add more detail.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var x = _fc1.Forward(input);
        x = _fc2.Forward(x);
        return _fc3.Forward(x);
    }

    /// <summary>
    /// Performs GPU-accelerated forward pass by chaining through sublayers.
    /// </summary>
    /// <param name="inputs">Input GPU tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        // Chain through the three fully connected layers
        var x = _fc1.ForwardGpu(inputs[0]);
        var x2 = _fc2.ForwardGpu(x);
        x.Dispose(); // Dispose intermediate tensor

        var output = _fc3.ForwardGpu(x2);
        x2.Dispose(); // Dispose intermediate tensor

        return output;
    }


    /// <summary>
    /// Updates the parameters of the reconstruction layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of all three fully connected layers based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. Each of the three internal layers updates its own weights and biases
    /// 2. The learning rate controls how big each update step is
    /// 3. All three layers are updated in a single call to this method
    /// 
    /// Think of it like each of the three artists adjusting their technique based on feedback:
    /// - "My lines were too thick, I'll make them thinner next time"
    /// - "I missed some details, I'll pay more attention to them"
    /// - "My colors were off, I'll mix them differently"
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _fc1.UpdateParameters(learningRate);
        _fc2.UpdateParameters(learningRate);
        _fc3.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Serializes the reconstruction layer to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the state of the reconstruction layer to a binary writer. It writes the
    /// vector activation flag and then serializes each of the three fully connected layers in sequence.
    /// Hidden dimensions are implicitly captured by the FC layers' own serialization (each stores its
    /// input/output sizes). GetMetadata() separately exports Hidden1Dimension and Hidden2Dimension
    /// for the deserialization constructor registered in DeserializationHelper.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the layer's state so it can be loaded later.
    /// 
    /// When serializing:
    /// - First, it saves whether vector activation is used or not
    /// - Then, it asks each of the three internal layers to save their states
    /// - The result is a complete snapshot of the layer that can be restored
    /// 
    /// This is useful for:
    /// - Saving a trained model to disk
    /// - Pausing training and continuing later
    /// - Sharing a trained model with others
    /// 
    /// Think of it like taking a detailed photograph of the layer's current state
    /// that can be used to recreate it exactly.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_useVectorActivation);
        writer.Write(_hidden1Dim);
        writer.Write(_hidden2Dim);
        _fc1.Serialize(writer);
        _fc2.Serialize(writer);
        _fc3.Serialize(writer);
    }

    /// <summary>
    /// Deserializes the reconstruction layer from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the reconstruction layer from a binary reader. It reads the
    /// vector activation flag and then deserializes each of the three fully connected layers in sequence.
    /// This is useful for loading the layer's state from disk or receiving it over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved layer state.
    /// 
    /// When deserializing:
    /// - First, it loads whether vector activation is used or not
    /// - Then, it asks each of the three internal layers to load their states
    /// - The result is a complete restoration of a previously saved layer
    /// 
    /// This is useful for:
    /// - Loading a trained model from disk
    /// - Continuing training from where you left off
    /// - Using a model that someone else trained
    /// 
    /// Think of it like reconstructing the exact state of the layer from a detailed blueprint.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        _useVectorActivation = reader.ReadBoolean();
        _hidden1Dim = reader.ReadInt32();
        _hidden2Dim = reader.ReadInt32();
        _fc1.Deserialize(reader);
        _fc2.Deserialize(reader);
        _fc3.Deserialize(reader);
    }

    /// <summary>
    /// Gets all trainable parameters of the reconstruction layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters from all three fully connected layers.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the reconstruction layer as a single vector.
    /// It collects the parameters from each of the three fully connected layers in sequence and concatenates
    /// them into a single vector. This is useful for optimization algorithms that operate on all parameters
    /// at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the reconstruction layer.
    /// 
    /// The parameters:
    /// - Are the weights and biases from all three internal layers
    /// - Control how the layer processes information
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The parameters from the first layer come first in the vector, followed by the second layer's parameters,
    /// and finally the third layer's parameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for production-grade parameter collection
        return Vector<T>.Concatenate(
            _fc1.GetParameters(),
            _fc2.GetParameters(),
            _fc3.GetParameters());
    }

    /// <summary>
    /// Sets the trainable parameters of the reconstruction layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for all three fully connected layers.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters of the reconstruction layer from a single vector.
    /// It extracts the appropriate portions of the vector for each of the three fully connected layers
    /// and sets their parameters accordingly. This is useful for loading saved model weights or for
    /// implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the weights and biases in the reconstruction layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The method divides this vector into three parts, one for each internal layer
    /// - Each internal layer gets its own specific section of parameters
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        return Vector<T>.Concatenate(
            _fc1.GetParameterGradients(),
            _fc2.GetParameterGradients(),
            _fc3.GetParameterGradients());
    }


    public override void ClearGradients()
    {
        _fc1.ClearGradients();
        _fc2.ClearGradients();
        _fc3.ClearGradients();
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Get parameter counts for each sublayer
        int fc1ParamCount = _fc1.ParameterCount;
        int fc2ParamCount = _fc2.ParameterCount;
        int fc3ParamCount = _fc3.ParameterCount;
        int totalParams = fc1ParamCount + fc2ParamCount + fc3ParamCount;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        // Use Vector.Slice for production-grade parameter distribution
        int offset = 0;
        _fc1.SetParameters(parameters.Slice(offset, fc1ParamCount));
        offset += fc1ParamCount;

        _fc2.SetParameters(parameters.Slice(offset, fc2ParamCount));
        offset += fc2ParamCount;

        _fc3.SetParameters(parameters.Slice(offset, fc3ParamCount));
    }

    /// <summary>
    /// Resets the internal state of the reconstruction layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the reconstruction layer by resetting the state of each
    /// of the three fully connected layers. This is useful when starting to process a new sequence or
    /// batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Each of the three internal layers clears its own temporary state
    /// - The layer forgets any information from previous batches
    /// - The learned parameters (weights and biases) are not reset
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like wiping a chalkboard clean before drawing something new,
    /// but keeping all the chalk and erasers (the parameters) you've collected.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset state in all sublayers
        _fc1.ResetState();
        _fc2.ResetState();
        _fc3.ResetState();
    }

}
