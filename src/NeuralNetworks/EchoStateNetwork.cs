namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Echo State Network (ESN), a type of recurrent neural network with a sparsely connected hidden layer called a reservoir.
/// </summary>
/// <remarks>
/// <para>
/// An Echo State Network is a unique type of recurrent neural network where the connections between neurons in
/// the hidden layer (called the reservoir) are randomly generated and remain fixed during training. Only the
/// output connections from the reservoir to the output layer are trained. The reservoir acts as a dynamic
/// memory that transforms inputs into high-dimensional representations, enabling the network to process
/// temporal patterns effectively. The key characteristic of ESNs is the "echo state property" which ensures
/// that the effect of initial conditions gradually fades away.
/// </para>
/// <para><b>For Beginners:</b> An Echo State Network is like a pool of water that creates ripples from your input.
/// 
/// Think of it this way:
/// - You drop a stone into a pool of water (your input)
/// - The stone creates ripples that bounce around and interact in complex ways (the reservoir)
/// - Someone watches the pattern of ripples and learns to predict what comes next (the output layer)
/// - Only the person watching and predicting is trained - the water itself doesn't change how it ripples
/// 
/// This approach is particularly good for processing sequences, like speech or time series data,
/// because the ripples in the reservoir naturally capture patterns over time without needing
/// complex training procedures.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class EchoStateNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the reservoir (number of neurons in the hidden layer).
    /// </summary>
    /// <value>An integer representing the number of reservoir neurons.</value>
    /// <remarks>
    /// <para>
    /// The reservoir size determines the dimensionality of the internal state space. A larger reservoir can
    /// capture more complex dynamics but requires more computational resources. The optimal size depends
    /// on the complexity of the task and the available data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the size of the pool of water.
    /// 
    /// Think of ReservoirSize as:
    /// - How big your pool of water is
    /// - A larger reservoir (pool) can create more complex ripple patterns
    /// - This allows the network to remember and process more complex sequences
    /// - But a larger reservoir also needs more computing power
    /// 
    /// For example, a reservoir size of 100 means the network has 100 interconnected neurons
    /// that collectively form the network's dynamic memory.
    /// </para>
    /// </remarks>
    private readonly int _reservoirSize;

    /// <summary>
    /// Gets the spectral radius that controls the dynamics of the reservoir.
    /// </summary>
    /// <value>A double between 0 and 1 representing the spectral radius.</value>
    /// <remarks>
    /// <para>
    /// The spectral radius is the largest absolute eigenvalue of the reservoir weight matrix. It controls the
    /// long-term behavior of the reservoir dynamics. A value less than 1.0 ensures the echo state property,
    /// which means that the effect of initial conditions and inputs will gradually fade away over time.
    /// Values closer to 1.0 allow the network to remember inputs for longer periods but may lead to instability.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how long ripples last in the pool.
    /// 
    /// Think of SpectralRadius as:
    /// - How quickly or slowly ripples fade away in your pool
    /// - Values closer to 1.0 make ripples last longer (better long-term memory)
    /// - Values closer to 0.0 make ripples fade quickly (better for rapidly changing patterns)
    /// - It's typically set between 0.7 and 0.99
    /// 
    /// This parameter helps balance between remembering past inputs long enough to be useful
    /// while still being responsive to new inputs.
    /// </para>
    /// </remarks>
    private readonly double _spectralRadius;

    /// <summary>
    /// Gets the sparsity level of connections in the reservoir.
    /// </summary>
    /// <value>A double between 0 and 1 representing the connection sparsity.</value>
    /// <remarks>
    /// <para>
    /// The sparsity parameter determines what fraction of the possible connections between reservoir neurons
    /// are actually present. A value of 0.1 means that only about 10% of all possible connections exist.
    /// Sparse connectivity is a key feature of ESNs, making them computationally efficient and helping to
    /// create rich, diverse dynamics within the reservoir.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how interconnected the pool is.
    /// 
    /// Think of Sparsity as:
    /// - How many invisible barriers or channels exist in your pool
    /// - A higher sparsity value (like 0.9) means very few connections between areas
    /// - A lower value (like 0.1) means many connections between different areas
    /// - Lower sparsity creates richer dynamics but uses more computing power
    /// 
    /// Most Echo State Networks use sparse connections (values around 0.1 or 0.2)
    /// to create complex dynamics while keeping computation manageable.
    /// </para>
    /// </remarks>
    private readonly double _sparsity;

    /// <summary>
    /// Gets or sets the current state of the reservoir.
    /// </summary>
    /// <value>A vector representing the activation values of all neurons in the reservoir.</value>
    /// <remarks>
    /// <para>
    /// The reservoir state represents the current activation values of all neurons in the reservoir.
    /// This state is updated with each new input and carries the network's memory of past inputs.
    /// The reservoir state is what gives ESNs their ability to process sequential data effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This is the current pattern of ripples in the pool.
    /// 
    /// The ReservoirState:
    /// - Represents the current activity of all neurons in the reservoir
    /// - Changes with each new input, but also preserves traces of past inputs
    /// - Acts as the network's "memory" of what it has seen before
    /// - Is what allows the network to process sequences and time-dependent patterns
    /// 
    /// This dynamic memory is central to how the Echo State Network works -
    /// it's what allows the network to "remember" past inputs when processing new ones.
    /// </para>
    /// </remarks>
    private Vector<T> _reservoirState;

    /// <summary>
    /// Gets or sets the vector activation function applied to the input-to-reservoir connections.
    /// </summary>
    /// <value>The vector activation function for input-to-reservoir transformations, or null if using scalar activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the transformation from the input layer to the reservoir.
    /// It processes entire vectors at once rather than individual elements. This can allow for more
    /// complex transformations and is optional in the Echo State Network.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how input signals are transformed before entering the pool.
    /// 
    /// Think of this activation function as:
    /// - A special filter that processes your input before it creates ripples
    /// - It can transform the input in complex ways, working on the entire input at once
    /// - This is an advanced option that allows for more sophisticated input processing
    /// - Most simple ESNs don't need this and use element-wise (scalar) activation instead
    /// 
    /// If this is null, the network will use the scalar activation function instead.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _reservoirInputVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function applied to the reservoir-to-output connections.
    /// </summary>
    /// <value>The vector activation function for reservoir-to-output transformations, or null if using scalar activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the transformation from the reservoir to the output layer.
    /// It processes entire vectors at once rather than individual elements. This can allow for more
    /// complex transformations and is optional in the Echo State Network.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how reservoir signals are transformed before producing output.
    /// 
    /// Think of this activation function as:
    /// - A filter that processes the reservoir state before generating predictions
    /// - It works on the entire reservoir state at once, allowing complex transformations
    /// - This is an advanced option for sophisticated ESN configurations
    /// - Most simple ESNs don't need this and use element-wise (scalar) activation instead
    /// 
    /// If this is null, the network will use the scalar activation function instead.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _reservoirOutputVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function applied within the reservoir.
    /// </summary>
    /// <value>The vector activation function for internal reservoir dynamics, or null if using scalar activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the internal processing within the reservoir itself.
    /// It processes entire vectors at once rather than individual elements. This can allow for more
    /// complex transformations and is optional in the Echo State Network.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how signals propagate within the pool itself.
    /// 
    /// Think of this activation function as:
    /// - Rules for how ripples interact with each other inside the reservoir
    /// - It works on the entire state at once, allowing complex interactions
    /// - This is an advanced option for sophisticated ESN configurations
    /// - Most simple ESNs don't need this and use element-wise (scalar) activation instead
    /// 
    /// If this is null, the network will use the scalar activation function instead.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _reservoirVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function applied to the output layer.
    /// </summary>
    /// <value>The vector activation function for the output layer, or null if using scalar activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the output layer of the network. It processes entire vectors
    /// at once rather than individual elements. This can allow for more complex transformations and is
    /// optional in the Echo State Network.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the final transformation of your predictions.
    /// 
    /// Think of this activation function as:
    /// - A final filter that shapes the network's predictions
    /// - It works on the entire output at once, allowing complex transformations
    /// - This is an advanced option for sophisticated ESN configurations
    /// - Most simple ESNs don't need this and use element-wise (scalar) activation instead
    /// 
    /// If this is null, the network will use the scalar activation function instead.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _outputVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function applied to individual elements in the input-to-reservoir connections.
    /// </summary>
    /// <value>The scalar activation function for input-to-reservoir transformations, or null if using vector activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to each individual element in the transformation from the input layer
    /// to the reservoir. Common choices include hyperbolic tangent (tanh) or sigmoid functions that introduce
    /// non-linearity into the network.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how each input value affects the pool.
    /// 
    /// Think of this activation function as:
    /// - A rule for how strongly each input value creates ripples
    /// - It works on each value separately (unlike vector activation)
    /// - Common choices limit values to certain ranges (like -1 to 1)
    /// - This non-linearity is crucial for the network to learn complex patterns
    /// 
    /// For example, a tanh activation squeezes values between -1 and 1, which keeps
    /// the reservoir dynamics stable and prevents values from growing too large.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _reservoirInputScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function applied to individual elements in the reservoir-to-output connections.
    /// </summary>
    /// <value>The scalar activation function for reservoir-to-output transformations, or null if using vector activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to each individual element in the transformation from the reservoir
    /// to the output layer. Common choices include hyperbolic tangent (tanh), sigmoid, or linear functions
    /// depending on the type of output required.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how each reservoir value affects the output.
    /// 
    /// Think of this activation function as:
    /// - A rule for how each reservoir neuron's state contributes to the prediction
    /// - It works on each value separately (unlike vector activation)
    /// - The choice depends on what kind of output you need
    /// - For instance, linear activation for regression, sigmoid for binary classification
    /// 
    /// This function shapes how the network translates the complex reservoir state
    /// into useful predictions.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _reservoirOutputScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function applied to individual elements within the reservoir.
    /// </summary>
    /// <value>The scalar activation function for internal reservoir dynamics, or null if using vector activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to each individual element in the internal processing within the reservoir.
    /// Common choices include hyperbolic tangent (tanh) or sigmoid functions that help maintain the stability
    /// of the reservoir dynamics.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how each neuron in the pool responds to signals.
    /// 
    /// Think of this activation function as:
    /// - A rule for how each individual neuron in the reservoir responds to input
    /// - It works on each neuron separately (unlike vector activation)
    /// - Usually non-linear functions like tanh that keep values within bounds
    /// - This helps create the rich, complex dynamics in the reservoir
    /// 
    /// Typically, a tanh function is used here to ensure the reservoir dynamics remain
    /// stable and the echo state property is maintained.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _reservoirScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function applied to individual elements in the output layer.
    /// </summary>
    /// <value>The scalar activation function for the output layer, or null if using vector activation.</value>
    /// <remarks>
    /// <para>
    /// This activation function is applied to each individual element in the output layer. The choice depends
    /// on the task: linear for regression, sigmoid for binary classification, softmax for multi-class classification, etc.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the form of your final predictions.
    /// 
    /// Think of this activation function as:
    /// - A rule for shaping the final output values of the network
    /// - It works on each output value separately (unlike vector activation)
    /// - The choice depends on what you're trying to predict:
    ///   - Linear for continuous values (like temperature prediction)
    ///   - Sigmoid for yes/no predictions (between 0 and 1)
    ///   - Tanh for values between -1 and 1
    /// 
    /// This final activation ensures the network's output is in the proper form for your specific problem.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _outputScalarActivation { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="EchoStateNetwork{T}"/> class with vector activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="reservoirSize">The number of neurons in the reservoir.</param>
    /// <param name="spectralRadius">The spectral radius that controls the dynamics of the reservoir. Default is 0.9.</param>
    /// <param name="sparsity">The sparsity level of connections in the reservoir. Default is 0.1.</param>
    /// <param name="reservoirInputVectorActivation">The vector activation function for input-to-reservoir connections.</param>
    /// <param name="reservoirOutputVectorActivation">The vector activation function for reservoir-to-output connections.</param>
    /// <param name="reservoirVectorActivation">The vector activation function for internal reservoir dynamics.</param>
    /// <param name="outputVectorActivation">The vector activation function for the output layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes an Echo State Network with vector activation functions, which process entire
    /// vectors at once rather than individual elements. This allows for more complex transformations and is an
    /// advanced configuration option for the ESN.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up an Echo State Network with advanced vector-based processing.
    /// 
    /// When creating an ESN with this constructor:
    /// - You're choosing to use vector activation functions that process entire groups of values at once
    /// - This is a more advanced configuration that allows for more complex transformations
    /// - These vector activations can capture relationships between different elements in vectors
    /// - Most simple applications use the scalar constructor instead
    /// 
    /// Think of this as setting up a pool with sophisticated rules for how groups of
    /// ripples interact, rather than simple rules for individual ripples.
    /// </para>
    /// </remarks>
    public EchoStateNetwork(NeuralNetworkArchitecture<T> architecture, int reservoirSize, double spectralRadius = 0.9, double sparsity = 0.1, 
        IVectorActivationFunction<T>? reservoirInputVectorActivation = null, IVectorActivationFunction<T>? reservoirOutputVectorActivation = null, 
        IVectorActivationFunction<T>? reservoirVectorActivation = null, IVectorActivationFunction<T>? outputVectorActivation = null) 
        : base(architecture)
    {
        _reservoirSize = reservoirSize;
        _spectralRadius = spectralRadius;
        _sparsity = sparsity;
        _reservoirState = new Vector<T>(_reservoirSize);
        _reservoirInputVectorActivation = reservoirInputVectorActivation;
        _reservoirOutputVectorActivation = reservoirOutputVectorActivation;
        _reservoirVectorActivation = reservoirVectorActivation;
        _outputVectorActivation = outputVectorActivation;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EchoStateNetwork{T}"/> class with scalar activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="reservoirSize">The number of neurons in the reservoir.</param>
    /// <param name="spectralRadius">The spectral radius that controls the dynamics of the reservoir. Default is 0.9.</param>
    /// <param name="sparsity">The sparsity level of connections in the reservoir. Default is 0.1.</param>
    /// <param name="reservoirInputScalarActivation">The scalar activation function for input-to-reservoir connections.</param>
    /// <param name="reservoirOutputScalarActivation">The scalar activation function for reservoir-to-output connections.</param>
    /// <param name="reservoirScalarActivation">The scalar activation function for internal reservoir dynamics.</param>
    /// <param name="outputScalarActivation">The scalar activation function for the output layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes an Echo State Network with scalar activation functions, which process individual
    /// elements one at a time. This is the more common configuration for ESNs and is simpler than using vector
    /// activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a standard Echo State Network with element-by-element processing.
    /// 
    /// When creating an ESN with this constructor:
    /// - You're using scalar activation functions that process each value individually
    /// - This is the more common and straightforward way to configure an ESN
    /// - Typical choices include tanh functions that keep values between -1 and 1
    /// - This approach is sufficient for most applications
    /// 
    /// Think of this as setting up a pool with simple rules for how individual
    /// water molecules behave, which collectively create complex ripple patterns.
    /// </para>
    /// </remarks>
    public EchoStateNetwork(NeuralNetworkArchitecture<T> architecture, int reservoirSize, double spectralRadius = 0.9, double sparsity = 0.1, 
        IActivationFunction<T>? reservoirInputScalarActivation = null, IActivationFunction<T>? reservoirOutputScalarActivation = null, 
        IActivationFunction<T>? reservoirScalarActivation = null, IActivationFunction<T>? outputScalarActivation = null) 
        : base(architecture)
    {
        _reservoirSize = reservoirSize;
        _spectralRadius = spectralRadius;
        _sparsity = sparsity;
        _reservoirState = new Vector<T>(_reservoirSize);
        _reservoirInputScalarActivation = reservoirInputScalarActivation;
        _reservoirOutputScalarActivation = reservoirOutputScalarActivation;
        _reservoirScalarActivation = reservoirScalarActivation;
        _outputScalarActivation = outputScalarActivation;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Echo State Network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Echo State Network. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default layers are created based on the architecture's specifications and
    /// the ESN's parameters. A typical ESN consists of an input layer, a reservoir layer, and an output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This builds the structure of the Echo State Network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers suitable for an ESN:
    ///   1. An input layer that receives external data
    ///   2. A reservoir layer with random, fixed connections
    ///   3. An output layer that learns to interpret the reservoir state
    /// 
    /// The method creates these layers with the appropriate sizes and connections
    /// based on the parameters you specified when creating the network.
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
            int inputSize = Architecture.GetInputShape()[0];
            int outputSize = Architecture.OutputSize;
        
            Layers.AddRange(LayerHelper<T>.CreateDefaultESNLayers(
                inputSize: inputSize,
                outputSize: outputSize,
                reservoirSize: _reservoirSize,
                spectralRadius: _spectralRadius,
                sparsity: _sparsity
            ));
        }
    }

    /// <summary>
    /// Validates that the custom layers form a valid Echo State Network structure.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the layer configuration does not meet the requirements for an Echo State Network.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method checks that the provided layers form a valid Echo State Network structure. An ESN must have
    /// at least 3 layers: an input layer, a reservoir layer, and an output layer. The reservoir layer must be
    /// a ReservoirLayer and cannot be the first or last layer in the network. This ensures that the network
    /// has the proper structure to function as an Echo State Network.
    /// </para>
    /// <para><b>For Beginners:</b> This makes sure your network has the right structure to work as an ESN.
    /// 
    /// The validation checks:
    /// - That you have at least 3 layers (input, reservoir, output)
    /// - That one layer is a special ReservoirLayer
    /// - That the ReservoirLayer isn't the first or last layer
    /// - That various other structural requirements are met
    /// 
    /// This is like making sure all the necessary parts of your water pool are present
    /// and properly arranged before filling it with water.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
        {
            throw new InvalidOperationException("Echo State Network must have at least 3 layers: input, reservoir, and output.");
        }

        // ESN-specific validation
        bool hasInputLayer = false;
        bool hasReservoirLayer = false;
        bool hasOutputLayer = false;

        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];

            if (layer is ReservoirLayer<T>)
            {
                if (hasReservoirLayer)
                {
                    throw new InvalidOperationException("Echo State Network should have only one Reservoir Layer.");
                }
                hasReservoirLayer = true;
            }
            else if (layer is DenseLayer<T>)
            {
                if (i == 0)
                {
                    hasInputLayer = true;
                }
                else if (!hasOutputLayer)
                {
                    hasOutputLayer = true;
                }
            }
        }

        if (!hasInputLayer)
        {
            throw new InvalidOperationException("Echo State Network must start with an input layer (DenseLayer).");
        }

        if (!hasReservoirLayer)
        {
            throw new InvalidOperationException("Echo State Network must contain a Reservoir Layer.");
        }

        if (!hasOutputLayer)
        {
            throw new InvalidOperationException("Echo State Network must contain an output layer (DenseLayer).");
        }

        // Ensure the reservoir layer is not the first or last layer
        int reservoirIndex = layers.FindIndex(l => l is ReservoirLayer<T>);
        if (reservoirIndex == 0 || reservoirIndex == layers.Count - 1)
        {
            throw new InvalidOperationException("The Reservoir Layer cannot be the first or last layer in the network.");
        }
    }

    /// <summary>
    /// Makes a prediction using the current state of the Echo State Network.
    /// </summary>
    /// <param name="input">The input vector to make a prediction for.</param>
    /// <returns>The predicted output vector after passing through the network.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input through the Echo State Network to produce a prediction. The input first
    /// passes through the input layer, then the reservoir layer (which updates the reservoir state), and finally
    /// the output layer. The reservoir state is maintained between predictions, allowing the network to exhibit
    /// memory of past inputs, which is essential for processing sequential data.
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through the network to make a prediction.
    /// 
    /// The prediction process works like this:
    /// - The input enters the network through the input layer
    /// - The reservoir layer combines this input with its current state to create a new state
    /// - This update creates complex "ripples" in the reservoir that depend on both current and past inputs
    /// - The output layer reads these ripples and produces a prediction
    /// - The new reservoir state is preserved for the next prediction
    /// 
    /// This preservation of state between predictions is what gives the ESN its ability to
    /// process sequences and remember patterns over time.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            if (i == 1) // Reservoir layer
            {
                var reservoirLayer = (ReservoirLayer<T>)Layers[i];
                _reservoirState = reservoirLayer.Forward(Tensor<T>.FromVector(current), Tensor<T>.FromVector(_reservoirState)).ToVector();
                current = _reservoirState;
            }
            else
            {
                current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
            }
        }

        return current;
    }

    /// <summary>
    /// Trains the Echo State Network using the provided input and target output data.
    /// </summary>
    /// <param name="X">The matrix of input vectors, with each row representing one input sample.</param>
    /// <param name="Y">The matrix of target output vectors, with each row representing one target sample.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Echo State Network using the provided input and target output data. Unlike traditional
    /// neural networks, ESNs only train the output layer weights while keeping the reservoir weights fixed. This is
    /// done using ridge regression, which efficiently computes the optimal output weights to map the reservoir states
    /// to the target outputs. This approach is much faster than backpropagation and is a key advantage of ESNs.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the network to make predictions from the reservoir patterns.
    /// 
    /// The training process works like this:
    /// - For each input sample, run it through the network to get a reservoir state
    /// - Collect all these reservoir states
    /// - Use a mathematical technique called ridge regression to find the best weights
    ///   for the output layer to turn reservoir states into correct predictions
    /// - Only the output layer weights are trained - the reservoir itself remains fixed
    /// 
    /// This approach is much faster than traditional neural network training and is one
    /// of the main advantages of Echo State Networks. It's like training someone to
    /// interpret ripple patterns without changing how the water ripples.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> X, Matrix<T> Y)
    {
        // Collect reservoir states
        var states = new List<Vector<T>>();
        for (int i = 0; i < X.Rows; i++)
        {
            var input = X.GetRow(i);
            Predict(input); // This updates the ReservoirState
            states.Add(_reservoirState);
        }

        // Concatenate states into a matrix
        var stateMatrix = new Matrix<T>(states.Count, _reservoirSize);
        for (int i = 0; i < states.Count; i++)
        {
            stateMatrix.SetRow(i, states[i]);
        }

        // Calculate output weights using ridge regression
        var regularization = NumOps.FromDouble(1e-8); // Small regularization term
        var stateTranspose = stateMatrix.Transpose();
        var outputWeights = stateTranspose.Multiply(stateMatrix)
            .Add(Matrix<T>.CreateIdentity(_reservoirSize).Multiply(regularization))
            .Inverse()
            .Multiply(stateTranspose)
            .Multiply(Y);

        // Set the calculated weights to the output layer
        ((DenseLayer<T>)Layers[3]).UpdateParameters(outputWeights.Flatten());
    }

    /// <summary>
    /// Updates the parameters of all layers in the Echo State Network.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <exception cref="NotImplementedException">
    /// Always thrown because ESN does not support traditional parameter updates.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method is not implemented for Echo State Networks because they do not use traditional parameter updates.
    /// In an ESN, only the output layer weights are trained, and this is done using ridge regression rather than
    /// gradient-based optimization. The reservoir weights remain fixed after initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method always throws an error because ESNs don't train like regular neural networks.
    /// 
    /// Echo State Networks are different from standard neural networks:
    /// - They don't use backpropagation or gradient descent
    /// - Their reservoir weights stay fixed (unchangeable) after initialization
    /// - Only the output layer weights are trained, using ridge regression
    /// 
    /// If you try to update parameters like in a regular neural network,
    /// you'll get an error because this isn't how ESNs work.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // ESN doesn't update parameters in the traditional sense
        throw new NotImplementedException("ESN does not support traditional parameter updates.");
    }

    /// <summary>
    /// Serializes the Echo State Network to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Echo State Network to a binary stream. It writes the number of layers,
    /// followed by the type name and serialized state of each layer. This allows the ESN to be saved to disk
    /// and later restored with its trained parameters intact.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the network to a file so you can use it later.
    /// 
    /// When saving the Echo State Network:
    /// - It records how many layers the network has
    /// - For each layer, it saves:
    ///   - What type of layer it is
    ///   - All the weights and settings for that layer
    /// 
    /// This is like taking a snapshot of the entire network - including both the reservoir
    /// (which doesn't change during training) and the output weights (which do change).
    /// You can later load this snapshot to use the trained network without having to
    /// train it again.
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
    /// Deserializes the Echo State Network from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer type information is invalid or instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Echo State Network from a binary stream. It reads the number of layers,
    /// followed by the type name and serialized state of each layer. This allows a previously saved ESN to be
    /// restored from disk with all its trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved network from a file.
    /// 
    /// When loading the Echo State Network:
    /// - First, it reads how many layers the network had
    /// - Then, for each layer, it:
    ///   - Reads what type of layer it was
    ///   - Creates a new layer of that type
    ///   - Loads all the weights and settings for that layer
    ///   - Adds the layer to the network
    /// 
    /// This lets you use a previously trained network without having to train it again.
    /// It's like restoring a complete snapshot of your network, bringing back
    /// both the reservoir and the trained output weights.
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