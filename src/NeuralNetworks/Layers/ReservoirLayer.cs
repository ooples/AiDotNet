namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a reservoir layer used in Echo State Networks (ESNs) for processing sequential data with fixed random weights.
/// </summary>
/// <remarks>
/// <para>
/// The ReservoirLayer implements the core component of an Echo State Network, a type of recurrent neural network
/// where the internal connections (reservoir weights) are randomly initialized and remain fixed during training.
/// This layer maintains a high-dimensional reservoir state that is updated based on the current input and the
/// previous state. The key characteristic of an ESN is that only the output layer is trained, while the reservoir
/// itself remains unchanged.
/// </para>
/// <para><b>For Beginners:</b> This layer works like a complex echo chamber for your data.
/// 
/// Think of the ReservoirLayer as a special room that creates rich echoes:
/// - When you speak a word into this room (input data), it creates complex echoes (reservoir state)
/// - These echoes depend both on what you just said and on the echoes of previous words
/// - The room's shape and materials (reservoir weights) determine how echoes form and persist
/// - Unlike other neural networks, the room's properties are fixed and don't change during training
/// 
/// For example, when processing a sentence word by word:
/// - Each word causes a unique pattern of echoes in the reservoir
/// - These echoes contain information about both the current word and previous words
/// - The patterns are rich enough that a simple output layer can be trained to extract useful information
/// 
/// This approach is powerful because:
/// - The random, fixed reservoir creates complex transformations of the input data
/// - Only the output layer needs to be trained, making learning faster and simpler
/// - It works especially well for time series prediction and certain sequence processing tasks
/// 
/// Echo State Networks are particularly effective when you need to model complex dynamical systems
/// with a simpler training process than traditional recurrent neural networks.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ReservoirLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The size of the input vector at each time step.
    /// </summary>
    /// <remarks>
    /// This value determines how many features are in each input vector to the reservoir.
    /// It is set during initialization and cannot be changed afterward.
    /// </remarks>
    private readonly int _inputSize;
    
    /// <summary>
    /// The size of the reservoir, determining the dimensionality of the reservoir state.
    /// </summary>
    /// <remarks>
    /// This value determines how many neurons are in the reservoir. Larger reservoirs can model
    /// more complex dynamics but require more computation. It is set during initialization and
    /// cannot be changed afterward.
    /// </remarks>
    private readonly int _reservoirSize;
    
    /// <summary>
    /// The probability of connection between any two neurons in the reservoir.
    /// </summary>
    /// <remarks>
    /// This value controls the sparsity of the reservoir weight matrix. Lower values create a more
    /// sparse network, which is typically more computationally efficient and can help prevent
    /// overfitting. It is set during initialization and cannot be changed afterward.
    /// </remarks>
    private readonly double _connectionProbability;
    
    /// <summary>
    /// The spectral radius of the reservoir weight matrix, affecting the memory of the network.
    /// </summary>
    /// <remarks>
    /// This value determines the maximum absolute eigenvalue of the reservoir weight matrix.
    /// It affects how long information persists in the reservoir state. Values close to 1.0
    /// lead to longer memory, while smaller values cause the influence of inputs to fade faster.
    /// It is set during initialization and cannot be changed afterward.
    /// </remarks>
    private readonly double _spectralRadius;
    
    /// <summary>
    /// The scaling factor applied to input before it enters the reservoir.
    /// </summary>
    /// <remarks>
    /// This value controls how strongly the input affects the reservoir state. Higher values
    /// give more importance to the current input, while lower values prioritize the reservoir's
    /// internal dynamics. It is set during initialization and cannot be changed afterward.
    /// </remarks>
    private readonly double _inputScaling;
    
    /// <summary>
    /// The leaking rate determining how quickly the reservoir state updates.
    /// </summary>
    /// <remarks>
    /// This value controls how much the reservoir state is updated at each time step. Values
    /// close to 1.0 cause rapid updates, while smaller values create a smoother evolution of
    /// the reservoir state. It is set during initialization and cannot be changed afterward.
    /// </remarks>
    private readonly double _leakingRate;

    /// <summary>
    /// The weight matrix representing connections between neurons in the reservoir.
    /// </summary>
    /// <remarks>
    /// This matrix holds the fixed random weights of connections between reservoir neurons.
    /// It is initialized randomly based on the connection probability and then scaled to
    /// achieve the desired spectral radius. These weights remain fixed during training.
    /// </remarks>
    private Matrix<T> _reservoirWeights;
    
    /// <summary>
    /// The current state of the reservoir, representing the activation of all neurons.
    /// </summary>
    /// <remarks>
    /// This vector holds the current activation state of all neurons in the reservoir.
    /// It is updated during each forward pass based on the input and the previous state.
    /// The reservoir state is the output of this layer and contains the features that
    /// will be used by subsequent layers for prediction or classification.
    /// </remarks>
    private Vector<T> _reservoirState;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> for ReservoirLayer, indicating that the layer cannot be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the ReservoirLayer does not support traditional training through backpropagation.
    /// In Echo State Networks, the reservoir weights are randomly initialized and remain fixed. Only the output layer
    /// (typically implemented as a separate layer after the reservoir) is trained using the reservoir states as input.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer's internal values don't change during training.
    /// 
    /// A value of false means:
    /// - The random connections inside the reservoir stay fixed
    /// - Error signals don't flow backward through this layer during training
    /// - No gradients are calculated for the reservoir weights
    /// 
    /// This is a key feature of Echo State Networks - the reservoir itself doesn't learn!
    /// Instead, only a readout layer (typically a simple linear layer placed after the reservoir)
    /// is trained to interpret the reservoir states. This makes training much faster and often
    /// more stable than training traditional recurrent neural networks.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReservoirLayer{T}"/> class with specified dimensions and properties.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer at each time step.</param>
    /// <param name="reservoirSize">The size of the reservoir (number of neurons).</param>
    /// <param name="connectionProbability">The probability of connection between any two neurons in the reservoir. Defaults to 0.1.</param>
    /// <param name="spectralRadius">The spectral radius of the reservoir weight matrix, affecting the memory of the network. Defaults to 0.9.</param>
    /// <param name="inputScaling">The scaling factor applied to input before it enters the reservoir. Defaults to 1.0.</param>
    /// <param name="leakingRate">The leaking rate determining how quickly the reservoir state updates. Defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReservoirLayer with the specified dimensions and properties. The reservoir
    /// weights are initialized randomly based on the connection probability, and then scaled to achieve the desired
    /// spectral radius. The reservoir state is initialized to zero. These parameters control the dynamics of the
    /// reservoir and should be tuned based on the specific task.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new reservoir layer for your Echo State Network.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer at each time step
    /// - reservoirSize: How many neurons are in the reservoir (more neurons = more complex patterns)
    /// - connectionProbability: How dense the connections between neurons are (default: 10% chance of connection)
    /// - spectralRadius: How long information persists in the reservoir (default: 0.9, close to 1.0 = longer memory)
    /// - inputScaling: How strongly the input affects the reservoir (default: 1.0)
    /// - leakingRate: How quickly the reservoir state changes (default: 1.0, smaller = smoother changes)
    /// 
    /// These parameters control the "personality" of your reservoir:
    /// - A larger reservoir can capture more complex patterns but needs more computation
    /// - A higher connection probability makes a denser network, which might be more expressive but less efficient
    /// - A spectral radius close to 1.0 gives the network longer memory
    /// - Higher input scaling makes the network more responsive to new inputs
    /// - Lower leaking rates create smoother changes in the reservoir state
    /// 
    /// Tuning these parameters is more of an art than a science, and often requires experimentation
    /// for best results on a specific task.
    /// </para>
    /// </remarks>
    public ReservoirLayer(
        int inputSize, 
        int reservoirSize, 
        double connectionProbability = 0.1, 
        double spectralRadius = 0.9, 
        double inputScaling = 1.0, 
        double leakingRate = 1.0)
        : base([inputSize], [reservoirSize])
    {
        _inputSize = inputSize;
        _reservoirSize = reservoirSize;
        _connectionProbability = connectionProbability;
        _spectralRadius = spectralRadius;
        _inputScaling = inputScaling;
        _leakingRate = leakingRate;

        _reservoirWeights = new Matrix<T>(_reservoirSize, _reservoirSize);
        _reservoirState = new Vector<T>(_reservoirSize);

        InitializeReservoir();
    }

    /// <summary>
    /// Performs the forward pass of the reservoir layer.
    /// </summary>
    /// <param name="input">The input tensor to process, with shape [1, inputSize].</param>
    /// <returns>The output tensor containing the updated reservoir state, with shape [1, reservoirSize].</returns>
    /// <exception cref="ArgumentException">Thrown when the input tensor has incorrect shape.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the reservoir layer. It updates the reservoir state based on
    /// the current input and the previous state. The update follows the Echo State Network dynamics: the input
    /// is scaled, multiplied by the input weights, and added to the product of the reservoir weights and the
    /// previous state. The result is passed through an activation function and combined with the previous state
    /// according to the leaking rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your input data through the reservoir.
    /// 
    /// During the forward pass:
    /// 1. The layer scales your input by the input scaling factor
    /// 2. It combines this scaled input with the previous reservoir state through the reservoir weights
    /// 3. It applies an activation function (usually tanh) to introduce non-linearity
    /// 4. It updates the reservoir state by blending the old state with the new one based on the leaking rate
    /// 
    /// The formula is approximately:
    /// new_state = (1-leakingRate) * old_state + leakingRate * activation(input_scaling * input + reservoir_weights * old_state)
    /// 
    /// This process creates a rich representation of your input sequence in the high-dimensional reservoir state.
    /// The reservoir state is both the output of this layer and serves as memory for processing the next input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[0] != 1)
            throw new ArgumentException("Input must be a 2D tensor with shape [1, inputSize]");

        Vector<T> inputVector = input.ToVector();
        Vector<T> scaledInput = inputVector.Multiply(NumOps.FromDouble(_inputScaling));

        Vector<T> reservoirInput = _reservoirWeights.Multiply(_reservoirState).Add(scaledInput);
        Vector<T> newState = ApplyActivation(reservoirInput);

        _reservoirState = _reservoirState.Multiply(NumOps.FromDouble(1 - _leakingRate))
            .Add(newState.Multiply(NumOps.FromDouble(_leakingRate)));

        return Tensor<T>.FromVector(_reservoirState);
    }

    /// <summary>
    /// Performs the backward pass of the reservoir layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>This method does not return; it throws an exception.</returns>
    /// <exception cref="InvalidOperationException">Always thrown because backward pass is not supported for ReservoirLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method is not supported because Echo State Networks do not train the reservoir through backpropagation.
    /// In ESNs, only the output layer (typically a separate layer after the reservoir) is trained, while the
    /// reservoir weights remain fixed. Therefore, there is no need to compute gradients with respect to the
    /// reservoir parameters or inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method throws an error because reservoir layers don't do backward passes.
    ///
    /// In a standard neural network, the backward pass:
    /// - Calculates how to adjust weights to reduce error
    /// - Propagates error signals backward through the network
    ///
    /// But in Echo State Networks:
    /// - The reservoir weights are fixed and never change
    /// - There's no need to calculate gradients or propagate errors backward
    /// - Only the output layer (after the reservoir) is trained
    ///
    /// If you try to call this method, you'll get an error. Instead, you should:
    /// 1. Collect reservoir states for your entire dataset
    /// 2. Train a simple readout layer (like a linear regression) on these states
    /// 3. Use the trained readout layer to make predictions
    ///
    /// This is what makes Echo State Networks faster and simpler to train than traditional RNNs.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // In ESN, we don't backpropagate through the reservoir
        throw new InvalidOperationException("Backward pass is not supported for ReservoirLayer in Echo State Networks as reservoir weights are typically fixed.");
    }

    /// <summary>
    /// Updates the parameters of the reservoir layer.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Always thrown because parameter updates are not supported for ReservoirLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method is not supported because Echo State Networks do not update the reservoir weights during training.
    /// In ESNs, only the output layer (typically a separate layer after the reservoir) is trained, while the
    /// reservoir weights remain fixed as initially set. Therefore, there is no need to update the reservoir parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method throws an error because reservoir layers don't update their weights.
    ///
    /// In a standard neural network, this method would:
    /// - Update the weights based on the gradients calculated during backward pass
    /// - Adjust the network to better fit the training data
    ///
    /// But in Echo State Networks:
    /// - The reservoir weights are fixed and never change
    /// - No updates are applied to the weights after initialization
    /// - Only the output layer (after the reservoir) is trained
    ///
    /// If you try to call this method, you'll get an error. This is normal and expected
    /// because the core principle of Echo State Networks is that the reservoir itself
    /// remains unchanged during training.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // In ESN, we don't update the reservoir weights
        throw new InvalidOperationException("Parameter update is not supported for ReservoirLayer in Echo State Networks as reservoir weights are typically fixed.");
    }

    /// <summary>
    /// Gets the current state of the reservoir.
    /// </summary>
    /// <returns>A vector representing the current activation of all neurons in the reservoir.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current reservoir state, which represents the activation of all neurons in the
    /// reservoir after processing the input sequence up to the current time step. This state contains the features
    /// that are typically used by a readout layer to make predictions in an Echo State Network.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you access the current "echo pattern" in the reservoir.
    /// 
    /// The reservoir state:
    /// - Represents the collective activation of all neurons in the reservoir
    /// - Contains information about both the current input and the history of past inputs
    /// - Is what makes Echo State Networks powerful for sequence processing
    /// 
    /// You might use this method to:
    /// - Collect reservoir states for different inputs to train a readout layer
    /// - Visualize or analyze the dynamics of the reservoir
    /// - Debug how your network responds to different inputs
    /// 
    /// Think of it like taking a snapshot of all the complex echoes in the room at a specific moment.
    /// These echoes contain rich information that can be decoded by a trained readout layer.
    /// </para>
    /// </remarks>
    public Vector<T> GetState()
    {
        return _reservoirState;
    }

    /// <summary>
    /// Resets the internal state of the reservoir layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the reservoir state to all zeros. This is useful when starting to process a new
    /// sequence or when you want to clear the memory of the network. In Echo State Networks, the reservoir
    /// state serves as memory that accumulates information about the input sequence, so resetting it effectively
    /// erases this memory.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the reservoir's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All neuron activations in the reservoir are set to zero
    /// - The layer forgets any information from previous inputs
    /// - The next input will be processed without any influence from the past
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence of data
    /// - Preventing information from one sequence affecting another
    /// - Testing how the network performs with a clean slate
    /// 
    /// Think of it like silencing all the echoes in the room before you speak a new word.
    /// This ensures that what you hear is only the echo of the current input, not a mix
    /// with previous echoes.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        for (int i = 0; i < _reservoirState.Length; i++)
        {
            _reservoirState[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Gets all parameters of the reservoir layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all reservoir weights, which remain fixed during training.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all reservoir weights as a single vector. In Echo State Networks, these weights
    /// are randomly initialized and remain fixed during training, so this method is primarily useful for
    /// inspection or manual modification of the weights, rather than for training purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you access the fixed random weights of the reservoir.
    /// 
    /// Even though the reservoir weights don't change during training, this method provides access to them for:
    /// - Inspecting the weight values
    /// - Saving the weights for later use
    /// - Manually modifying the weights if needed
    /// - Research or experimental purposes
    /// 
    /// Remember that in Echo State Networks:
    /// - These weights are set randomly during initialization
    /// - They are scaled to achieve the desired spectral radius
    /// - They remain fixed throughout the network's lifetime
    /// - Only the weights in a separate readout layer are trained
    /// 
    /// This method returns all the weights as a single long list (vector).
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // In Echo State Networks, the reservoir weights are typically not trained
        // But we still provide access to them for inspection or manual modification
        int totalParams = _reservoirWeights.Rows * _reservoirWeights.Columns;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
        for (int i = 0; i < _reservoirWeights.Rows; i++)
        {
            for (int j = 0; j < _reservoirWeights.Columns; j++)
            {
                parameters[index++] = _reservoirWeights[i, j];
            }
        }
    
        return parameters;
    }

    /// <summary>
    /// Initializes the reservoir weights and state with proper scaling.
    /// </summary>
    /// <remarks>
    /// This private method initializes the reservoir weights randomly based on the connection probability,
    /// then scales them to achieve the desired spectral radius. The spectral radius is a key hyperparameter
    /// that affects the memory and stability of the reservoir. The method also initializes the reservoir
    /// state to zeros. This initialization is performed once during the construction of the layer.
    /// </remarks>
    private void InitializeReservoir()
    {
        // Initialize reservoir weights
        for (int i = 0; i < _reservoirWeights.Rows; i++)
        {
            for (int j = 0; j < _reservoirWeights.Columns; j++)
            {
                if (Random.NextDouble() < _connectionProbability)
                {
                    _reservoirWeights[i, j] = NumOps.FromDouble(Random.NextDouble() - 0.5);
                }
                else
                {
                    _reservoirWeights[i, j] = NumOps.Zero;
                }
            }
        }

        // Scale the reservoir weights to achieve the desired spectral radius
        T maxEigenvalue = ComputeMaxEigenvalue(_reservoirWeights);
        T scaleFactor = NumOps.FromDouble(_spectralRadius / Convert.ToDouble(maxEigenvalue));
        _reservoirWeights = _reservoirWeights.Multiply(scaleFactor);

        // Initialize reservoir state to zeros
        for (int i = 0; i < _reservoirState.Length; i++)
        {
            _reservoirState[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Computes the maximum eigenvalue (spectral radius) of a matrix using power iteration.
    /// </summary>
    /// <param name="matrix">The matrix for which to compute the maximum eigenvalue.</param>
    /// <returns>The maximum absolute eigenvalue of the matrix.</returns>
    /// <remarks>
    /// This private method implements the power iteration algorithm to compute the maximum eigenvalue
    /// (spectral radius) of the reservoir weight matrix. This value is important for scaling the
    /// weights to achieve the desired spectral radius, which affects the memory and stability of
    /// the reservoir. The method uses improvements like random initialization and Rayleigh quotient
    /// for better convergence.
    /// </remarks>
    private T ComputeMaxEigenvalue(Matrix<T> matrix)
    {
        // Power iteration method with improvements for better convergence
        int maxIterations = 1000;
        double tolerance = 1e-10;
    
        // Start with a random vector instead of all ones
        Vector<T> v = Vector<T>.CreateRandom(matrix.Rows);
        for (int i = 0; i < v.Length; i++)
        {
            v[i] = NumOps.FromDouble(Random.NextDouble() - 0.5);
        }
    
        // Normalize the initial vector
        T initialNorm = v.Norm();
        if (!NumOps.Equals(initialNorm, NumOps.Zero))
        {
            v = v.Divide(initialNorm);
        }
    
        T prevEigenvalue = NumOps.Zero;
        T currentEigenvalue;
    
        for (int i = 0; i < maxIterations; i++)
        {
            // Apply matrix to vector
            Vector<T> Av = matrix.Multiply(v);
        
            // Calculate Rayleigh quotient for better eigenvalue approximation
            T rayleighQuotient = v.DotProduct(Av);
        
            // Normalize the vector
            T norm = Av.Norm();
            if (NumOps.Equals(norm, NumOps.Zero))
            {
                // If we get a zero vector, the matrix might be nilpotent
                return NumOps.Zero;
            }
        
            v = Av.Divide(norm);
            currentEigenvalue = rayleighQuotient;
        
            // Check for convergence
            T diff = NumOps.Abs(NumOps.Subtract(currentEigenvalue, prevEigenvalue));
            if (Convert.ToDouble(diff) < tolerance && i > 5)
            {
                return NumOps.Abs(currentEigenvalue);
            }
        
            prevEigenvalue = currentEigenvalue;
        }
    
        // Return absolute value to ensure positive spectral radius
        return NumOps.Abs(prevEigenvalue);
    }
}