namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Extreme Learning Machine (ELM), a type of feedforward neural network with a unique training approach.
/// </summary>
/// <remarks>
/// <para>
/// An Extreme Learning Machine is a special type of single-hidden-layer feedforward neural network that uses a
/// non-iterative training approach. Unlike traditional neural networks that use backpropagation to adjust all weights,
/// ELMs randomly assign the weights between the input and hidden layer and only train the weights between the hidden
/// and output layer. This is done analytically using a pseudo-inverse operation rather than through iterative
/// optimization, resulting in extremely fast training times while maintaining good generalization performance.
/// </para>
/// <para><b>For Beginners:</b> An Extreme Learning Machine is like a neural network on fast-forward.
/// 
/// Think of it like building a bridge:
/// - Traditional neural networks carefully adjust every piece of the bridge (slow but thorough)
/// - ELMs randomly set up most of the bridge, then only carefully adjust the final section
/// - This approach is much faster but can still create a surprisingly strong bridge
/// 
/// The "extreme" part refers to its extremely fast training time. While traditional networks
/// might take hours or days to train, ELMs can often be trained in seconds or minutes, making
/// them useful for applications where training speed is critical.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ExtremeLearningMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the hidden layer (number of neurons).
    /// </summary>
    /// <value>An integer representing the number of neurons in the hidden layer.</value>
    /// <remarks>
    /// <para>
    /// The hidden layer size determines the dimensionality of the feature space that the ELM projects the input data into.
    /// A larger hidden layer can capture more complex patterns but may lead to overfitting with small datasets.
    /// This is a key hyperparameter that significantly affects the ELM's performance and capacity.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many "pattern detectors" the network has.
    /// 
    /// Think of HiddenLayerSize as:
    /// - The number of different patterns the network can recognize
    /// - Like having a team of people each looking for specific features
    /// - More neurons (larger size) means more patterns can be detected
    /// - But too many neurons might make the network "memorize" rather than "learn"
    /// 
    /// For example, a hidden layer size of 100 means the network has 100 different
    /// pattern detectors that work together to analyze the input data.
    /// </para>
    /// </remarks>
    private readonly int _hiddenLayerSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExtremeLearningMachine{T}"/> class with the specified architecture and hidden layer size.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerSize">The number of neurons in the hidden layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Extreme Learning Machine with the specified architecture and hidden layer size.
    /// The hidden layer size is a key parameter that determines the capacity and learning ability of the ELM.
    /// After setting this parameter, the constructor initializes the network layers.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new ELM with a specific number of pattern detectors.
    /// 
    /// When creating a new ELM:
    /// - The architecture defines the overall structure (input and output sizes)
    /// - The hiddenLayerSize determines how many pattern detectors the network will have
    /// - The constructor sets up the initial structure, but doesn't train the network yet
    /// 
    /// Think of it like assembling a team of a specific size to look for patterns,
    /// where each team member will be randomly assigned what to look for.
    /// </para>
    /// </remarks>
    public ExtremeLearningMachine(NeuralNetworkArchitecture<T> architecture, int hiddenLayerSize) 
        : base(architecture)
    {
        _hiddenLayerSize = hiddenLayerSize;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Extreme Learning Machine based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Extreme Learning Machine. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default layers are created based on the architecture's specifications and
    /// the specified hidden layer size. A typical ELM consists of an input layer, a hidden layer with random weights,
    /// and an output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This builds the structure of the neural network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers suitable for an ELM:
    ///   1. A random input-to-hidden layer with fixed weights
    ///   2. A non-linear activation function (typically sigmoid or tanh)
    ///   3. A trainable hidden-to-output layer
    /// 
    /// The most important part is that only the final layer (hidden-to-output)
    /// will be trained - the other layers will keep their random weights.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultELMLayers(Architecture, _hiddenLayerSize));
        }
    }

    /// <summary>
    /// Makes a prediction using the current state of the Extreme Learning Machine.
    /// </summary>
    /// <param name="input">The input vector to make a prediction for.</param>
    /// <returns>The predicted output vector after passing through the network.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction by passing the input vector through each layer of the Extreme Learning Machine
    /// in sequence. Each layer processes the output of the previous layer, transforming the data until it reaches
    /// the final output layer. The result is a vector representing the network's prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This uses the network to make a prediction based on input data.
    /// 
    /// The prediction process works like this:
    /// - The input data enters the first layer (with random, fixed weights)
    /// - It's transformed into a new representation by the hidden layer
    /// - The activation function adds non-linearity to this representation
    /// - The output layer (the only trained part) converts this into the final prediction
    /// 
    /// This process is very fast because there's no complex computation involved -
    /// just a few simple matrix multiplications and activation functions.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Trains the Extreme Learning Machine using the provided input and target output data.
    /// </summary>
    /// <param name="X">The matrix of input vectors, with each row representing one input sample.</param>
    /// <param name="Y">The matrix of target output vectors, with each row representing one target sample.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Extreme Learning Machine using the provided input and target output data. Unlike
    /// traditional neural networks that use iterative optimization, ELMs use a direct, analytical solution.
    /// The method first passes the input data through the randomly initialized hidden layer to get the hidden layer
    /// output (H). Then it calculates the optimal output weights using the Moore-Penrose pseudo-inverse:
    /// Output Weights = (H^T * H)^(-1) * H^T * Y. This single-step approach allows ELMs to train extremely quickly.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the network to make accurate predictions in a single step.
    /// 
    /// The training process works like this:
    /// - First, pass all input data through the random hidden layer
    /// - This transforms the data into a new representation
    /// - Then, use math (pseudo-inverse) to find the best output weights
    /// - These weights connect the hidden layer to the correct outputs
    /// 
    /// This is why ELMs are so fast - instead of thousands of iterations adjusting weights
    /// little by little (like in traditional networks), ELMs solve for the optimal weights
    /// in a single mathematical operation.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> X, Matrix<T> Y)
    {
        // Forward pass through random hidden layer
        var H = X;
        for (int i = 0; i < 2; i++)  // First two layers: Dense and Activation
        {
            H = Layers[i].Forward(Tensor<T>.FromMatrix(H)).ToMatrix();
        }

        // Calculate output weights using pseudo-inverse
        var HTranspose = H.Transpose();
        var HHTranspose = HTranspose.Multiply(H);
        var HHTransposeInverse = HHTranspose.Inverse();
        var outputWeights = HHTransposeInverse.Multiply(HTranspose).Multiply(Y);

        // Set the calculated weights to the output layer
        ((DenseLayer<T>)Layers[2]).SetWeights(outputWeights);
    }

    /// <summary>
    /// Updates the parameters of all layers in the Extreme Learning Machine.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <exception cref="NotImplementedException">
    /// Always thrown because ELM does not support traditional parameter updates.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method is not implemented for Extreme Learning Machines because they do not use traditional parameter updates.
    /// In an ELM, the input-to-hidden weights are randomly generated and remain fixed, while the hidden-to-output weights
    /// are calculated analytically in a single step rather than through iterative updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method always throws an error because ELMs don't train like regular neural networks.
    /// 
    /// Extreme Learning Machines are different from standard neural networks:
    /// - They don't use backpropagation or gradient descent
    /// - Most of their weights stay fixed (unchangeable) after random initialization
    /// - The output weights are calculated in one step, not iteratively updated
    /// 
    /// If you try to update parameters like in a regular neural network,
    /// you'll get an error because this isn't how ELMs work.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // ELM doesn't update parameters in the traditional sense
        throw new NotImplementedException("ELM does not support traditional parameter updates.");
    }

    /// <summary>
    /// Serializes the Extreme Learning Machine to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Extreme Learning Machine to a binary stream. It writes the number of layers,
    /// followed by the type name and serialized state of each layer. This allows the ELM to be saved to disk
    /// and later restored with all its parameters intact, including both the random input-to-hidden weights
    /// and the analytically calculated hidden-to-output weights.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the network to a file so you can use it later.
    /// 
    /// When saving the Extreme Learning Machine:
    /// - It records how many layers the network has
    /// - For each layer, it saves:
    ///   - What type of layer it is
    ///   - All the weights and settings for that layer
    /// 
    /// This is like taking a snapshot of the entire network - including both the random
    /// hidden layer weights and the calculated output weights. You can later load this
    /// snapshot to use the trained network without having to train it again.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Deserializes the Extreme Learning Machine from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer type information is invalid or instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Extreme Learning Machine from a binary stream. It reads the number of layers,
    /// followed by the type name and serialized state of each layer. This allows a previously saved ELM to be
    /// restored from disk with all its parameters intact, including both the random input-to-hidden weights
    /// and the analytically calculated hidden-to-output weights.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved network from a file.
    /// 
    /// When loading the Extreme Learning Machine:
    /// - First, it reads how many layers the network had
    /// - Then, for each layer, it:
    ///   - Reads what type of layer it was
    ///   - Creates a new layer of that type
    ///   - Loads all the weights and settings for that layer
    ///   - Adds the layer to the network
    /// 
    /// This lets you use a previously trained network without having to train it again.
    /// It's like restoring the complete snapshot of your network, bringing back both
    /// the random hidden layer and the trained output weights.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }
}