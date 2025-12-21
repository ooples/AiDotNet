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
    public HopfieldNetwork(NeuralNetworkArchitecture<T> architecture, int size, ILossFunction<T>? lossFunction = null) : base(new NeuralNetworkArchitecture<T>(
        architecture.InputType,
        taskType: architecture.TaskType,
        complexity: architecture.Complexity,
        inputSize: size,
        outputSize: size), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
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
    /// - It has limited capacity (can only store approximately 0.14 * network size patterns reliably)
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
        // Number of off-diagonal entries in a symmetric NxN matrix
        int expectedLength = (_size * (_size - 1)) / 2;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Parameter vector length mismatch. Expected {expectedLength} parameters but got {parameters.Length}.", nameof(parameters));
        }

        int paramIndex = 0;
        // Fill upper triangle and mirror to lower triangle; keep diagonal zero
        for (int i = 0; i < _size; i++)
        {
            _weights[i, i] = NumOps.Zero;
            for (int j = i + 1; j < _size; j++)
            {
                var w = parameters[paramIndex++];
                _weights[i, j] = w;
                _weights[j, i] = w;
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

    /// <summary>
    /// Converts an input tensor to a vector, performs pattern recall, and converts back to a tensor.
    /// </summary>
    /// <param name="input">The input tensor containing the pattern to recall.</param>
    /// <returns>A tensor containing the recalled pattern.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the prediction functionality for the Hopfield network. It converts the input tensor
    /// to a vector, performs the recall operation to retrieve the stored pattern most similar to the input,
    /// and then converts the result back to a tensor. This allows the Hopfield network to be used within the
    /// broader neural network framework while maintaining its unique recall-based approach.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the Hopfield network makes predictions.
    /// 
    /// When you provide an input pattern (possibly noisy or incomplete):
    /// 1. The method first converts it to the right format for the network
    /// 2. It then runs the recall process to find the closest stored pattern
    /// 3. Finally, it converts the result back to the expected output format
    /// 
    /// This allows the Hopfield network to be used like other neural networks
    /// where you can simply call Predict() to get a result, even though
    /// the underlying mechanism is quite different.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Convert input tensor to vector
        Vector<T> inputVector = input.ToVector();

        // Ensure the input vector has the correct size
        if (inputVector.Length != _size)
        {
            throw new ArgumentException($"Input vector length ({inputVector.Length}) does not match network size ({_size})");
        }

        // Perform pattern recall
        Vector<T> recalledPattern = Recall(inputVector);

        // Convert the recalled pattern back to a tensor with the same shape as the input
        Tensor<T> result = Tensor<T>.FromVector(recalledPattern);

        // If the input has a non-trivial shape (not just a flat vector),
        // reshape the result to match the input shape
        if (input.Shape.Length > 1 || (input.Shape.Length == 1 && input.Shape[0] == recalledPattern.Length))
        {
            result = result.Reshape(input.Shape);
        }

        return result;
    }

    /// <summary>
    /// Trains the Hopfield network using the provided input patterns.
    /// </summary>
    /// <param name="input">A tensor containing the patterns to store.</param>
    /// <param name="expectedOutput">This parameter is ignored in Hopfield networks.</param>
    /// <remarks>
    /// <para>
    /// This method adapts the standard neural network training interface to the Hopfield network.
    /// It extracts patterns from the input tensor and calls the Hebbian learning-based Train method.
    /// The expectedOutput parameter is ignored since Hopfield networks are autoassociative and
    /// use the same patterns for both input and output.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows the Hopfield network to learn patterns.
    /// 
    /// Unlike traditional neural networks that learn mappings from inputs to outputs:
    /// - Hopfield networks learn to associate patterns with themselves
    /// - The input tensor contains the patterns to be stored
    /// - The expectedOutput parameter is ignored (not needed)
    /// 
    /// This method:
    /// 1. Extracts individual patterns from the input tensor
    /// 2. Converts them to the right format
    /// 3. Calls the core training method that implements Hebbian learning
    /// 
    /// After training, the network will be able to recognize and complete these patterns.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Extract patterns from the input tensor
        // Each row of the input tensor is treated as a separate pattern
        List<Vector<T>> patterns = [];

        for (int i = 0; i < input.Shape[0]; i++)
        {
            // Extract the i-th pattern from the input tensor
            var pattern = new Vector<T>(_size);
            for (int j = 0; j < _size; j++)
            {
                pattern[j] = input[i, j];
            }

            patterns.Add(pattern);
        }

        // Train the network using Hebbian learning
        Train(patterns);

        // Calculate the average energy of all patterns as a measure of loss
        // Lower energy means more stable patterns (better storage)
        T totalEnergy = NumOps.Zero;
        foreach (var pattern in patterns)
        {
            // Calculate energy for this pattern
            T patternEnergy = CalculateEnergy(pattern);

            // Add to total
            totalEnergy = NumOps.Add(totalEnergy, patternEnergy);
        }

        // Calculate average energy (if there are patterns)
        if (patterns.Count > 0)
        {
            LastLoss = NumOps.Divide(totalEnergy, NumOps.FromDouble(patterns.Count));
        }
        else
        {
            LastLoss = NumOps.Zero;
        }
    }

    /// <summary>
    /// Gets metadata about the Hopfield network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the Hopfield network, including its model type, size,
    /// and serialized weights. This information is useful for model management and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides a summary of the Hopfield network.
    /// 
    /// The metadata includes:
    /// - The type of model (Hopfield Network)
    /// - The size of the network (number of neurons)
    /// - The current weight matrix (connection strengths)
    /// - Serialized data that can be used to save and reload the network
    /// 
    /// This information is useful when:
    /// - Managing multiple models
    /// - Saving the network for later use
    /// - Analyzing the network's properties
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.HopfieldNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Size", _size },
                { "WeightMatrixShape", $"{_weights.Rows}x{_weights.Columns}" }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Hopfield network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the Hopfield network's specific data to a binary stream. It includes
    /// the network size and the weight matrix. This data is needed to reconstruct the Hopfield network
    /// when deserializing.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the Hopfield network to a file.
    /// 
    /// It's like taking a snapshot of the network's current state, including:
    /// - The size of the network (how many neurons it has)
    /// - All the connection weights between neurons
    /// 
    /// This allows you to save a trained network and reload it later,
    /// without having to train it again from scratch.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write network size
        writer.Write(_size);

        // Write weight matrix
        writer.Write(_weights.Rows);
        writer.Write(_weights.Columns);

        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes Hopfield network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the Hopfield network's specific data from a binary stream. It retrieves
    /// the network size and the weight matrix. After reading this data, the Hopfield network's state
    /// is fully restored to what it was when saved.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved Hopfield network.
    /// 
    /// It's like restoring the network from a snapshot, retrieving:
    /// - The size of the network
    /// - All the connection weights that were learned during training
    /// 
    /// This allows you to use a previously trained network without
    /// having to train it again on the same patterns.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read network size
        _size = reader.ReadInt32();

        // Read weight matrix dimensions
        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();

        // Initialize weight matrix
        _weights = new Matrix<T>(rows, columns);

        // Read weight values
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Calculates the energy of the current state of the Hopfield network.
    /// </summary>
    /// <param name="state">The state vector to calculate energy for.</param>
    /// <returns>The energy value of the given state.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the energy function of the Hopfield network for a given state.
    /// The energy function is defined as E = -0.5 * sum(sum(w_ij * s_i * s_j)) - sum(theta_i * s_i),
    /// where w_ij are the weights, s_i and s_j are the states of neurons i and j, and theta_i is the
    /// bias for neuron i (typically 0 in standard Hopfield networks). Lower energy values correspond
    /// to more stable states, and the network naturally evolves toward states with minimum energy.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how "stable" a pattern is in the network.
    /// 
    /// Think of energy like a ball rolling on a landscape:
    /// - Lower energy = valleys where the ball comes to rest (stable patterns)
    /// - Higher energy = hills that the ball rolls away from (unstable patterns)
    /// 
    /// The Hopfield network naturally moves toward lower energy states.
    /// When you use the Recall method, the network is essentially rolling downhill
    /// to the nearest valley (stable pattern) from your starting point (input pattern).
    /// 
    /// The stored patterns correspond to the deepest valleys in this energy landscape.
    /// </para>
    /// </remarks>
    public T CalculateEnergy(Vector<T> state)
    {
        if (state.Length != _size)
        {
            throw new ArgumentException($"State vector length ({state.Length}) does not match network size ({_size})");
        }

        // Calculate energy: E = -0.5 * sum(sum(w_ij * s_i * s_j))
        // Lower energy = more stable state
        T energy = NumOps.Zero;

        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                energy = NumOps.Subtract(
                    energy,
                    NumOps.Multiply(
                        NumOps.Multiply(_weights[i, j], state[i]),
                        state[j]
                    )
                );
            }
        }

        // Multiply by 0.5 (we've double-counted each pair)
        energy = NumOps.Multiply(energy, NumOps.FromDouble(0.5));

        return energy;
    }

    /// <summary>
    /// Gets the maximum number of patterns that can be reliably stored in the network.
    /// </summary>
    /// <returns>The estimated capacity of the network.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the theoretical capacity of the Hopfield network based on its size.
    /// The general rule of thumb is that a Hopfield network can reliably store approximately
    /// N/(4*log(N)) patterns, where N is the number of neurons. This is a theoretical upper bound,
    /// and in practice, the capacity might be lower depending on the patterns' similarities.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many different patterns you can reliably store in the network.
    /// 
    /// Hopfield networks have limited memory capacity:
    /// - Too many patterns will cause interference
    /// - This leads to incorrect recall and "spurious patterns"
    /// - The capacity depends on the number of neurons
    /// 
    /// The rule of thumb is that a network with N neurons can store approximately
    /// N/(4*log(N)) patterns. For example, a network with 100 neurons can reliably
    /// store about 5 patterns, not 100 patterns as you might expect.
    /// 
    /// This limited capacity is important to keep in mind when designing applications.
    /// </para>
    /// </remarks>
    public int GetNetworkCapacity()
    {
        // Theoretical capacity of a Hopfield network is approximately N/(4*ln(N))
        double capacity = _size / (4.0 * Math.Log(_size));
        return (int)Math.Floor(capacity);
    }

    /// <summary>
    /// Creates a new instance of the Hopfield Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new Hopfield Network instance with the same architecture and size.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the Hopfield Network with the same architecture and size
    /// as the current instance. It's used in scenarios where a fresh copy of the model is needed
    /// while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the network with the same setup.
    /// 
    /// Think of it like creating a blank version of the network:
    /// - The new network has the same size (number of neurons)
    /// - It has the same architecture (configuration)
    /// - But it starts with no stored patterns - it's a fresh network
    /// - The weight matrix is initialized to zeros
    /// 
    /// This is useful when you want to:
    /// - Start with a clean network with the same structure
    /// - Train it on different patterns
    /// - Compare results between different training approaches
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new HopfieldNetwork<T>(this.Architecture, _size);
    }
}
