namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Hopfield Network, a recurrent neural network designed for pattern storage and retrieval.
/// </summary>
/// <remarks>
/// <para>
/// A Hopfield Network is a type of recurrent artificial neural network that serves as a content-addressable memory system.
/// It can store patterns and retrieve them based on partial or noisy inputs. The network consists of a single layer
/// of fully connected neurons, with each neuron connected to all others except itself.
/// Hopfield networks are particularly useful for pattern recognition, image restoration, and optimization problems.
/// </para>
/// <para><b>For Beginners:</b> A Hopfield Network is like a special memory system that can store and recall patterns.
/// 
/// Think of it like a photo album with a magical property:
/// - You can store a collection of complete photos in the album
/// - Later, if you show the album a damaged or partial photo, it can recall the complete original version
/// 
/// For example:
/// - You might store clear images of the digits 0-9
/// - If you later show the network a smudged or partially erased "7", it can recall the clean version
/// 
/// Hopfield networks work differently from most neural networks:
/// - They don't have separate input and output layers
/// - All neurons are connected to each other (but not to themselves)
/// - They use a special learning rule based on correlations between pattern elements
/// 
/// These networks are useful for tasks like:
/// - Image reconstruction
/// - Pattern recognition
/// - Noise filtering
/// - Solving certain optimization problems
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HopfieldNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the weight matrix that stores the connection strengths between neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix represents the synaptic connection strengths between neurons in the Hopfield network.
    /// The value at position (i,j) represents the strength of the connection from neuron j to neuron i.
    /// In a standard Hopfield network, the weight matrix is symmetric with zeros on the diagonal.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a table that stores how strongly each neuron is connected to every other neuron.
    /// 
    /// Imagine each neuron as a person in a social network:
    /// - The weight between two neurons represents how strongly they influence each other
    /// - Positive weights mean they encourage each other's activation
    /// - Negative weights mean they discourage each other's activation
    /// - The diagonal is zero because neurons don't connect to themselves
    /// 
    /// These connection strengths are what allow the network to store and recall patterns.
    /// </para>
    /// </remarks>
    private Matrix<T> _weights;

    /// <summary>
    /// Gets or sets the size of the network, which is the number of neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The size determines how many neurons are in the network, which equals both the number of rows
    /// and columns in the weight matrix. It also defines the dimensionality of the patterns that
    /// can be stored and retrieved.
    /// </para>
    /// <para><b>For Beginners:</b> This is simply how many neurons are in the network.
    /// 
    /// The size determines:
    /// - How large the patterns can be that you store
    /// - For example, if you're storing images that are 10x10 pixels (100 pixels total),
    ///   your network size would be 100
    /// 
    /// Each neuron represents one element of the pattern (like one pixel in an image).
    /// </para>
    /// </remarks>
    private int _size;

    /// <summary>
    /// Gets the activation function used to determine the state of each neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the weighted sum of inputs to each neuron to determine
    /// its next state. In a classical Hopfield network, this is typically a sign function that returns
    /// +1 for positive inputs and -1 for negative inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This is the rule that decides whether each neuron should be "on" or "off".
    /// 
    /// In a Hopfield network:
    /// - The activation function is usually very simple
    /// - For positive inputs, it outputs +1 (neuron is "on")
    /// - For negative inputs, it outputs -1 (neuron is "off")
    /// 
    /// This binary nature (on/off) is what allows the network to store and recall distinct patterns.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T> _activationFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="HopfieldNetwork{T}"/> class with the specified architecture and size.
    /// </summary>
    /// <param name="architecture">The neural network architecture providing base configuration.</param>
    /// <param name="size">The size of the network, determining the number of neurons.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a Hopfield network with the specified size, initializing the weight matrix
    /// and setting up a sign activation function. The input and output sizes are both set to the specified size,
    /// as Hopfield networks have a single layer that serves as both input and output.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new Hopfield network with a specific number of neurons.
    /// 
    /// When creating a Hopfield network:
    /// - You specify how many neurons you need (the size parameter)
    /// - This determines how large a pattern you can store (e.g., how many pixels in an image)
    /// - The network automatically sets up a weight matrix initialized to zeros
    /// - It uses a special "sign" activation function that outputs either +1 or -1
    /// 
    /// For example, if you're creating a network to store 8x8 pixel images,
    /// you would set the size to 64 (8x8=64).
    /// </para>
    /// </remarks>
    public HopfieldNetwork(NeuralNetworkArchitecture<T> architecture, int size) : base(new NeuralNetworkArchitecture<T>(
        architecture.InputType,
        taskType: architecture.TaskType,
        complexity: architecture.Complexity,
        inputSize: size,
        outputSize: size))
    {
        _size = size;
        _weights = new Matrix<T>(size, size);
        _activationFunction = new SignActivation<T>();

        InitializeWeights();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the weight matrix with zeros.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weight matrix of the Hopfield network. All weights are set to zero,
    /// including the diagonal elements which represent self-connections (these remain zero during training
    /// as neurons don't connect to themselves in a standard Hopfield network).
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial connection strengths between neurons, starting with no connections.
    /// 
    /// Before training, all connection weights are set to zero, meaning:
    /// - No neuron influences any other neuron yet
    /// - The weight matrix is like a blank slate
    /// - Training will establish the actual connection strengths
    /// 
    /// The diagonal of the matrix (where i=j) is also set to zero and stays zero,
    /// which means neurons don't influence themselves directly.
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                if (i != j)
                {
                    _weights[i, j] = NumOps.Zero;
                }
                else
                {
                    _weights[i, j] = NumOps.Zero;
                }
            }
        }
    }

    /// <summary>
    /// Trains the Hopfield network on a set of patterns.
    /// </summary>
    /// <param name="patterns">A list of patterns to store in the network.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Hopfield network using the Hebbian learning rule, which adjusts the weights
    /// based on the correlation between pattern elements. For each pattern, the method updates the weight
    /// matrix by adding the product of each pair of elements. After processing all patterns, the weights
    /// are normalized by dividing by the number of patterns to improve recall performance.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the network to remember a set of patterns.
    /// 
    /// During training:
    /// - For each pattern you want to store (like an image)
    /// - The network looks at each pair of elements in the pattern
    /// - If two elements are both active (+1) or both inactive (-1), it strengthens their connection
    /// - If one is active and one is inactive, it weakens their connection
    /// 
    /// This follows a principle similar to "neurons that fire together, wire together."
    /// 
    /// After training on all patterns, the connections are adjusted (normalized) to ensure
    /// better recall. This process is different from training in most neural networks because:
    /// - It happens in one pass, not through repeated iterations
    /// - It doesn't use backpropagation or gradients
    /// - It has limited capacity (can only store approximately 0.14 × network size patterns reliably)
    /// </para>
    /// </remarks>
    public void Train(List<Vector<T>> patterns)
    {
        foreach (var pattern in patterns)
        {
            for (int i = 0; i < _size; i++)
            {
                for (int j = 0; j < _size; j++)
                {
                    if (i != j)
                    {
                        _weights[i, j] = NumOps.Add(_weights[i, j], NumOps.Multiply(pattern[i], pattern[j]));
                    }
                }
            }
        }

        // Normalize weights
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                _weights[i, j] = NumOps.Divide(_weights[i, j], NumOps.FromDouble(patterns.Count));
            }
        }
    }

    /// <summary>
    /// Performs pattern recall to retrieve a complete pattern from a partial or noisy input.
    /// </summary>
    /// <param name="input">The input pattern to use as a starting point for recall.</param>
    /// <param name="maxIterations">The maximum number of iterations to perform during recall. Default is 100.</param>
    /// <returns>The recalled pattern after the network reaches stability or the maximum number of iterations.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the asynchronous update rule for Hopfield networks. It starts with the provided
    /// input pattern and iteratively updates each neuron based on the weighted sum of inputs from all other neurons.
    /// The process continues until either the pattern stabilizes (no more changes occur) or the maximum number
    /// of iterations is reached. The stable pattern represents the stored memory closest to the input pattern.
    /// </para>
    /// <para><b>For Beginners:</b> This is like showing the network a partial or damaged pattern and asking it to recall the complete version.
    /// 
    /// The recall process works like this:
    /// 1. Start with your input pattern (which might be incomplete or noisy)
    /// 2. For each element in the pattern:
    ///    - Calculate how it's influenced by all other elements using the connection weights
    ///    - Update its state to either "on" (+1) or "off" (-1) based on those influences
    /// 3. Repeat this process until:
    ///    - The pattern stops changing (it's stable), OR
    ///    - You reach the maximum number of allowed iterations
    /// 
    /// For example, if you trained the network on images of letters and show it a smudged "A",
    /// this process would gradually clean up the image until it resembles a complete "A".
    /// 
    /// This recall process doesn't always find the exact pattern that was stored - it finds
    /// the closest stable pattern according to the network's energy function.
    /// </para>
    /// </remarks>
    public Vector<T> Recall(Vector<T> input, int maxIterations = 100)
    {
        var current = input;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var next = new Vector<T>(_size);
            for (int i = 0; i < _size; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < _size; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_weights[i, j], current[j]));
                }
                next[i] = _activationFunction.Activate(sum);
            }

            if (current.Equals(next))
            {
                break;
            }
            current = next;
        }

        return current;
    }

    /// <summary>
    /// Performs a forward pass through the network to generate a prediction from an input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the recalled pattern.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the predict functionality required by the NeuralNetworkBase class.
    /// For Hopfield networks, prediction is equivalent to pattern recall, so this method simply
    /// calls the Recall method with the provided input.
    /// </para>
    /// <para><b>For Beginners:</b> This method is the network's way of making a prediction from an input.
    /// 
    /// In most neural networks, "predict" means taking an input and producing a new output.
    /// In a Hopfield network:
    /// - "Predict" actually means "recall the closest stored pattern"
    /// - It uses the same recall process described above
    /// - The input might be incomplete or noisy
    /// - The output is the complete pattern that best matches the input
    /// 
    /// This method is required because the HopfieldNetwork class extends the base neural network class,
    /// which requires a Predict method. It simply redirects to the Recall method.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        return Recall(input);
    }

    /// <summary>
    /// Not implemented for Hopfield networks, as they don't use gradient-based parameter updates.
    /// </summary>
    /// <param name="parameters">A vector containing parameters to update.</param>
    /// <exception cref="NotImplementedException">Always thrown, as this method is not applicable to Hopfield networks.</exception>
    /// <remarks>
    /// <para>
    /// This method is required by the NeuralNetworkBase class but is not implemented for Hopfield networks.
    /// Hopfield networks use a different training approach (Hebbian learning) that directly sets the weights
    /// rather than using gradient-based updates as in most neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method is not used in Hopfield networks.
    /// 
    /// Most neural networks learn by:
    /// - Making small adjustments to weights based on error gradients
    /// - Updating parameters gradually over many training iterations
    /// 
    /// Hopfield networks are different:
    /// - They learn using the Hebbian rule ("neurons that fire together, wire together")
    /// - Training happens in one pass through the Train method
    /// - They don't use gradient-based updates at all
    /// 
    /// This method exists only because the base neural network class requires it,
    /// but it will throw an error if called.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // Hopfield networks typically don't use gradient-based updates
        throw new NotImplementedException("Hopfield networks do not support gradient-based parameter updates.");
    }

    /// <summary>
    /// Serializes the Hopfield network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized network to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the Hopfield network's state to a binary format that can be stored and
    /// later loaded. It writes the size of the network and then serializes the weight matrix
    /// by writing each element as a double value.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained Hopfield network to a file.
    /// 
    /// After training a network to remember patterns, you'll want to save it to use it later
    /// without training again. This method:
    /// 
    /// 1. Saves the size of the network (how many neurons it has)
    /// 2. Saves all the connection weights between neurons
    /// 
    /// The weights are what allow the network to remember patterns, so saving them
    /// preserves all the network's "knowledge."
    /// 
    /// This is like taking a snapshot of the network's brain that you can reload later.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(_size);
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes the Hopfield network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized network from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <remarks>
    /// <para>
    /// This method loads a previously serialized Hopfield network from a binary format. It reads the size
    /// of the network and then deserializes the weight matrix by reading each element as a double value
    /// and converting it to the appropriate numeric type T.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved Hopfield network from a file.
    /// 
    /// When you want to use a network that was trained earlier, this method:
    /// 
    /// 1. Reads how big the network is (how many neurons)
    /// 2. Creates a new, empty weight matrix of that size
    /// 3. Loads all the connection weights that were saved
    /// 
    /// After loading, the network will remember all the patterns it was trained on before,
    /// without needing to train it again.
    /// 
    /// This is like restoring the network's "brain" from a snapshot.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        _size = reader.ReadInt32();
        _weights = new Matrix<T>(_size, _size);
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Initializes the layers of the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is required by the NeuralNetworkBase class but is empty for Hopfield networks,
    /// as they don't use separate layers like feedforward neural networks. Instead, a Hopfield network
    /// consists of a single fully-connected layer of neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This method is empty because Hopfield networks don't have layers.
    /// 
    /// Most neural networks have distinct layers like:
    /// - Input layer
    /// - Hidden layers
    /// - Output layer
    /// 
    /// But a Hopfield network:
    /// - Has just a single layer of neurons
    /// - Each neuron connects to all others
    /// - The same neurons act as both input and output
    /// 
    /// This method exists only because the base neural network class requires it,
    /// but it doesn't need to do anything for a Hopfield network.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // Hopfield networks don't use layers in the same way as feedforward networks
    }
}