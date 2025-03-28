namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Boltzmann Machine, a type of stochastic recurrent neural network.
/// </summary>
/// <remarks>
/// <para>
/// A Boltzmann Machine is a type of neural network that consists of binary units (neurons) with undirected
/// connections between them. Each unit can be either "on" (1) or "off" (0), and the units update their states
/// based on a probability function that depends on the states of other units and the weights of connections.
/// Boltzmann Machines are used for learning probability distributions from input data.
/// </para>
/// <para><b>For Beginners:</b> A Boltzmann Machine is like a network of connected switches that can be either on or off.
/// 
/// Think of it as a group of light bulbs connected by wires:
/// - Each bulb can be on (1) or off (0)
/// - The bulbs are connected to each other with wires of different strengths (weights)
/// - Whether a bulb turns on or off depends on the state of other bulbs and the strength of the connections
/// - The machine "learns" by adjusting the strength of connections to match patterns in data
/// 
/// For example, if you show the network many pictures of handwritten digits, it will eventually learn
/// the common patterns that make up each digit, allowing it to recognize or generate similar digits.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BoltzmannMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the bias values for each unit in the Boltzmann Machine.
    /// </summary>
    /// <value>A vector of bias values, one for each unit.</value>
    /// <remarks>
    /// <para>
    /// Biases represent the individual tendency of each unit to be active (1) or inactive (0), regardless
    /// of inputs from other units. A higher bias makes a unit more likely to be active.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like the default settings for each light bulb in our network.
    /// 
    /// Think of a bias as how much each bulb "wants" to be on or off by default:
    /// - A high positive bias means the bulb prefers to be on
    /// - A negative bias means the bulb prefers to be off
    /// - Zero bias means the bulb has no preference
    /// 
    /// These default tendencies get adjusted along with the connection weights as the network learns.
    /// </para>
    /// </remarks>
    private Vector<T> _biases { get; set; }

    /// <summary>
    /// Gets or sets the weight matrix representing connection strengths between units.
    /// </summary>
    /// <value>A matrix of weights, where each element [i,j] represents the connection strength between units i and j.</value>
    /// <remarks>
    /// <para>
    /// Weights determine how strongly each unit influences other units. A positive weight means that when one unit
    /// is active, it encourages the connected unit to also be active. A negative weight means that when one unit
    /// is active, it discourages the connected unit from being active.
    /// </para>
    /// <para><b>For Beginners:</b> Weights are like the strengths of the wires connecting our light bulbs.
    /// 
    /// Think of weights as how strongly one bulb affects another:
    /// - A positive weight means "if I'm on, you should be on too"
    /// - A negative weight means "if I'm on, you should be off"
    /// - A zero weight means "I don't affect you"
    /// 
    /// The Boltzmann Machine learns by adjusting these connection strengths until the network
    /// naturally produces patterns similar to the training data.
    /// </para>
    /// </remarks>
    private Matrix<T> _weights { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="BoltzmannMachine{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input type is not one-dimensional, when the input size is invalid, or when custom layers are provided.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Boltzmann Machine with the specified architecture. It validates that the architecture
    /// is compatible with a Boltzmann Machine, which requires one-dimensional input and does not support custom layers.
    /// It initializes the biases to zero and the weights to small random values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Boltzmann Machine ready to learn patterns.
    /// 
    /// When setting up a Boltzmann Machine:
    /// - The input must be a simple list of values (one-dimensional)
    /// - The number of units (light bulbs) is determined by the input size
    /// - All biases start at zero (no preference)
    /// - The weights start as small random values
    /// - Unlike other neural networks, Boltzmann Machines don't use separate layers
    /// 
    /// If any of these requirements aren't met, the constructor will show an error message.
    /// </para>
    /// </remarks>
    public BoltzmannMachine(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        if (architecture.InputType != InputType.OneDimensional)
        {
            throw new ArgumentException("Boltzmann Machine requires one-dimensional input.");
        }

        int size = architecture.CalculatedInputSize;
        if (size <= 0)
        {
            throw new ArgumentException("Invalid input size for Boltzmann Machine.");
        }

        _biases = new Vector<T>(size);
        _weights = new Matrix<T>(size, size);

        // Check if custom layers are provided (which is not typical for Boltzmann Machines)
        if (architecture.Layers != null && architecture.Layers.Count > 0)
        {
            throw new ArgumentException("Boltzmann Machine does not support custom layers.");
        }

        InitializeParameters(size);
    }

    /// <summary>
    /// Initializes the layers of the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is overridden from the base class but is not used in Boltzmann Machines, which do not use
    /// layers in the same way as feedforward networks. Instead, the weights and biases are initialized directly.
    /// </para>
    /// <para><b>For Beginners:</b> This method is here because Boltzmann Machines are a special type of neural network.
    /// 
    /// Unlike most neural networks that have layers:
    /// - Boltzmann Machines have a single group of connected units
    /// - There are no separate input, hidden, or output layers
    /// - All units are connected to each other
    /// 
    /// This method is left empty because we initialize the network differently.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // Boltzmann Machine doesn't use layers in the same way as feedforward networks
        // Instead, we'll initialize the weights and biases directly
    }

    /// <summary>
    /// Initializes the weights and biases of the Boltzmann Machine.
    /// </summary>
    /// <param name="size">The number of units in the Boltzmann Machine.</param>
    /// <remarks>
    /// <para>
    /// This method initializes the biases to zero and the weights to small random values between -0.05 and 0.05.
    /// The weights represent the connection strengths between units, and the biases represent the individual
    /// tendencies of units to be active or inactive.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets the starting values for our network.
    /// 
    /// During initialization:
    /// - All biases are set to zero (no preference to be on or off)
    /// - All weights are set to small random values between -0.05 and 0.05
    /// - Small random values are used so the network can learn gradually
    /// - Starting with all zeros would make learning difficult
    /// 
    /// These initial values are just starting points that will be adjusted during training.
    /// </para>
    /// </remarks>
    private void InitializeParameters(int size)
    {
        // Initialize biases to zero and weights to small random values
        for (int i = 0; i < size; i++)
        {
            _biases[i] = NumOps.Zero;
            for (int j = 0; j < size; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1 - 0.05);
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the current state of the Boltzmann Machine.
    /// </summary>
    /// <param name="input">The input vector to make a prediction for.</param>
    /// <returns>The predicted output vector after sampling from the Boltzmann Machine.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction by sampling from the Boltzmann Machine. In Boltzmann Machines,
    /// prediction is done through stochastic sampling, where each unit's state is updated based on a
    /// probability function that depends on the states of other units and the connection weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the network to make a prediction based on input data.
    /// 
    /// The prediction process works like this:
    /// - The input data is used as the starting state
    /// - For each unit, we calculate how likely it is to be on or off
    /// - We decide whether to turn each unit on or off based on these probabilities
    /// - The final state of all units is the prediction
    /// 
    /// This is different from most neural networks because there's randomness involved - 
    /// the same input might give slightly different predictions each time!
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        // In a Boltzmann Machine, prediction is typically done through sampling
        return Sample(input);
    }

    /// <summary>
    /// Generates a sample from the Boltzmann Machine based on the current state.
    /// </summary>
    /// <param name="state">The current state of the Boltzmann Machine units.</param>
    /// <returns>A new state vector after updating each unit based on probabilistic sampling.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a new sample by updating each unit's state based on a probability function
    /// that depends on the states of other units and the connection weights. The probability is calculated
    /// using the sigmoid activation function applied to the weighted sum of inputs plus the bias.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like rolling dice to decide each unit's new state.
    /// 
    /// For each unit (light bulb) in the network:
    /// - We calculate how likely it should be on, based on:
    ///   - Its bias (default tendency)
    ///   - The states of other units and how strongly they're connected
    /// - We convert this into a probability between 0 and 1 using a special function (sigmoid)
    /// - We "roll a dice" (generate a random number)
    /// - If the random number is less than the probability, the unit turns on; otherwise, it turns off
    /// 
    /// This randomness is important because it helps the network explore different possible patterns
    /// instead of always producing the same output.
    /// </para>
    /// </remarks>
    private Vector<T> Sample(Vector<T> state)
    {
        Vector<T> newState = new Vector<T>(state.Length);
        for (int i = 0; i < state.Length; i++)
        {
            T activation = _biases[i];
            for (int j = 0; j < state.Length; j++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(_weights[i, j], state[j]));
            }

            T probability = new SigmoidActivation<T>().Activate(activation);
            newState[i] = NumOps.FromDouble(Random.NextDouble() < Convert.ToDouble(probability) ? 1 : 0);
        }

        return newState;
    }

    /// <summary>
    /// Trains the Boltzmann Machine on the provided data using contrastive divergence.
    /// </summary>
    /// <param name="data">The training data vector.</param>
    /// <param name="epochs">The number of training epochs to perform.</param>
    /// <param name="learningRate">The learning rate for weight updates.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Boltzmann Machine using a simplified version of contrastive divergence.
    /// For each epoch, it performs the following steps:
    /// 1. Use the data as the visible state and sample the hidden state
    /// 2. Compute the positive phase gradient using the outer product of visible and hidden states
    /// 3. Sample a new visible state from the hidden state
    /// 4. Sample a new hidden state from the reconstructed visible state
    /// 5. Compute the negative phase gradient using the outer product of reconstructed states
    /// 6. Update weights and biases based on the difference between positive and negative gradients
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the network to recognize patterns in data.
    /// 
    /// The training process works like this:
    /// - We show the network a training example (like a handwritten digit)
    /// - The network tries to "dream up" a similar pattern
    /// - We compare what the network dreamed with our training example
    /// - We adjust the connection weights to make the network's dreams more like our examples
    /// - We repeat this process many times (epochs)
    /// 
    /// The learning rate controls how quickly the network adjusts - too fast and it might learn
    /// the wrong patterns, too slow and it might take too long to learn anything useful.
    /// </para>
    /// </remarks>
    public void Train(Vector<T> data, int epochs, T learningRate)
    {
        int size = Architecture.CalculatedInputSize;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Vector<T> visibleProbs = data;
            Vector<T> hiddenProbs = Sample(visibleProbs);
            Matrix<T> posGradient = OuterProduct(visibleProbs, hiddenProbs);

            Vector<T> visibleReconstruction = Sample(hiddenProbs);
            Vector<T> hiddenReconstruction = Sample(visibleReconstruction);
            Matrix<T> negGradient = OuterProduct(visibleReconstruction, hiddenReconstruction);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    T weightUpdate = NumOps.Multiply(learningRate, NumOps.Subtract(posGradient[i, j], negGradient[i, j]));
                    _weights[i, j] = NumOps.Add(_weights[i, j], weightUpdate);
                }

                T biasUpdate = NumOps.Multiply(learningRate, NumOps.Subtract(visibleProbs[i], visibleReconstruction[i]));
                _biases[i] = NumOps.Add(_biases[i], biasUpdate);
            }
        }
    }

    /// <summary>
    /// Computes the outer product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>A matrix representing the outer product of the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// The outer product of two vectors is a matrix where each element [i,j] is the product of
    /// the i-th element of the first vector and the j-th element of the second vector.
    /// This is used in the training process to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies every value in one list with every value in another list.
    /// 
    /// The outer product creates a table of all possible multiplications between two lists:
    /// - For vector [a, b, c] and vector [x, y, z]
    /// - The result is a table (matrix):
    ///   [a*x, a*y, a*z]
    ///   [b*x, b*y, b*z]
    ///   [c*x, c*y, c*z]
    /// 
    /// In the Boltzmann Machine, this is used to calculate how to adjust the connection
    /// weights between units based on the training data.
    /// </para>
    /// </remarks>
    private Matrix<T> OuterProduct(Vector<T> v1, Vector<T> v2)
    {
        Matrix<T> result = new Matrix<T>(v1.Length, v2.Length);
        for (int i = 0; i < v1.Length; i++)
        {
            for (int j = 0; j < v2.Length; j++)
            {
                result[i, j] = NumOps.Multiply(v1[i], v2[j]);
            }
        }

        return result;
    }

    /// <summary>
    /// Updates the parameters of the Boltzmann Machine.
    /// </summary>
    /// <param name="parameters">A vector of parameters to update the Boltzmann Machine with.</param>
    /// <exception cref="NotImplementedException">
    /// Always thrown because this method is not typically used in Boltzmann Machines.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method is not typically used in Boltzmann Machines, which are usually trained using
    /// contrastive divergence or other specialized methods rather than traditional gradient descent
    /// with explicit parameter updates. It throws a NotImplementedException to indicate this.
    /// </para>
    /// <para><b>For Beginners:</b> This method is not used in Boltzmann Machines.
    /// 
    /// Unlike other neural networks, Boltzmann Machines:
    /// - Use a special training method (contrastive divergence)
    /// - Don't update their parameters all at once with a pre-calculated list
    /// - Instead make small adjustments based on comparing the input data with generated samples
    /// 
    /// If you try to use this method, you'll get an error message saying it's not implemented.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // This method is not typically used in Boltzmann Machines
        throw new NotImplementedException("UpdateParameters is not implemented for Boltzmann Machines.");
    }

    /// <summary>
    /// Serializes the Boltzmann Machine to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Boltzmann Machine to a binary stream. It writes the size of the
    /// machine, followed by all bias values and then all weight values. This allows the Boltzmann Machine
    /// to be saved to disk and later restored.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the network to a file.
    /// 
    /// When saving the Boltzmann Machine:
    /// - First, we save how many units the network has
    /// - Then we save all the bias values (one per unit)
    /// - Finally, we save all the weight values (one per connection)
    /// 
    /// This is like taking a snapshot of the network so you can reload it later
    /// without having to train it all over again.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(_biases.Length);
        for (int i = 0; i < _biases.Length; i++)
        {
            writer.Write(Convert.ToDouble(_biases[i]));
        }

        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes the Boltzmann Machine from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Boltzmann Machine from a binary stream. It reads the size of the
    /// machine, followed by all bias values and then all weight values. This allows a previously saved
    /// Boltzmann Machine to be restored from disk.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved network from a file.
    /// 
    /// When loading the Boltzmann Machine:
    /// - First, we read how many units the network had
    /// - Then we read all the bias values
    /// - Finally, we read all the weight values
    /// 
    /// This is like restoring a previous snapshot of your network, bringing back
    /// all the patterns it had learned before.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int size = reader.ReadInt32();
        _biases = new Vector<T>(size);
        _weights = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            _biases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}