using AiDotNet.NeuralNetworks.Options;

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
    private readonly EchoStateNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    private int _reservoirSize;

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
    private double _spectralRadius;

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
    private double _sparsity;

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
    /// The weight matrix for input-to-reservoir connections.
    /// </summary>
    private Matrix<T> _inputWeights;

    /// <summary>
    /// The weight matrix for reservoir-to-reservoir connections.
    /// </summary>
    private Matrix<T> _reservoirWeights;

    /// <summary>
    /// The weight matrix for reservoir-to-output connections.
    /// </summary>
    private Matrix<T> _outputWeights;

    /// <summary>
    /// The bias vector for the reservoir.
    /// </summary>
    private Vector<T> _reservoirBias;

    /// <summary>
    /// The bias vector for the output layer.
    /// </summary>
    private Vector<T> _outputBias;

    /// <summary>
    /// The current state of the reservoir.
    /// </summary>
    private Vector<T> _currentState;

    /// <summary>
    /// Indicates whether the network is being trained.
    /// </summary>
    private bool _isTraining = false;

    /// <summary>
    /// Leaking rate for controlling the update speed of reservoir neurons.
    /// Value between 0 and 1, default is 1.0 (no leaking).
    /// </summary>
    private T _leakingRate;

    /// <summary>
    /// Regularization parameter for ridge regression during training.
    /// </summary>
    private T _regularization;

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    private Random _random = RandomHelper.CreateSecureRandom();

    /// <summary>
    /// Input dimension size.
    /// </summary>
    private int _inputSize;

    /// <summary>
    /// Output dimension size.
    /// </summary>
    private int _outputSize;

    /// <summary>
    /// Collected states during training for regression.
    /// </summary>
    private List<Vector<T>> _collectedStates;

    /// <summary>
    /// Collected targets during training for regression.
    /// </summary>
    private List<Vector<T>> _collectedTargets;

    /// <summary>
    /// Warmup period for discarding initial transient reservoir states during training.
    /// </summary>
    private int _warmupPeriod;

    private ILossFunction<T> _lossFunction;

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
    public EchoStateNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int reservoirSize,
        double spectralRadius = 0.9,
        double sparsity = 0.1,
        double leakingRate = 1.0,
        double regularization = 1e-4,
        int warmupPeriod = 10,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? reservoirInputVectorActivation = null,
        IVectorActivationFunction<T>? reservoirOutputVectorActivation = null,
        IVectorActivationFunction<T>? reservoirVectorActivation = null,
        IVectorActivationFunction<T>? outputVectorActivation = null,
        EchoStateNetworkOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new EchoStateNetworkOptions();
        Options = _options;
        _reservoirSize = reservoirSize;
        _spectralRadius = spectralRadius;
        _sparsity = sparsity;
        _inputSize = architecture.InputSize;
        _outputSize = architecture.OutputSize;
        _leakingRate = NumOps.FromDouble(leakingRate);
        _regularization = NumOps.FromDouble(regularization);
        _warmupPeriod = warmupPeriod;
        _isTraining = false;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // Initialize the reservoir state and other vectors/matrices
        _reservoirState = new Vector<T>(_reservoirSize);
        _inputWeights = new Matrix<T>(_inputSize, _reservoirSize);
        _reservoirWeights = new Matrix<T>(_reservoirSize, _reservoirSize);
        _outputWeights = new Matrix<T>(_reservoirSize, _outputSize);
        _reservoirBias = new Vector<T>(_reservoirSize);
        _outputBias = new Vector<T>(_outputSize);
        _currentState = new Vector<T>(_inputSize);

        // Initialize activation functions
        _reservoirInputVectorActivation = reservoirInputVectorActivation;
        _reservoirOutputVectorActivation = reservoirOutputVectorActivation;
        _reservoirVectorActivation = reservoirVectorActivation;
        _outputVectorActivation = outputVectorActivation;

        // Initialize collections for training
        _collectedStates = new List<Vector<T>>();
        _collectedTargets = new List<Vector<T>>();

        // Initialize weights with random values
        InitializeWeights();

        // Initialize layers
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
    public EchoStateNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int reservoirSize,
        double spectralRadius = 0.9,
        double sparsity = 0.1,
        double leakingRate = 1.0,
        double regularization = 1e-4,
        int warmupPeriod = 10,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? reservoirInputScalarActivation = null,
        IActivationFunction<T>? reservoirOutputScalarActivation = null,
        IActivationFunction<T>? reservoirScalarActivation = null,
        IActivationFunction<T>? outputScalarActivation = null,
        EchoStateNetworkOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new EchoStateNetworkOptions();
        Options = _options;
        _reservoirSize = reservoirSize;
        _spectralRadius = spectralRadius;
        _sparsity = sparsity;
        _inputSize = architecture.InputSize;
        _outputSize = architecture.OutputSize;
        _leakingRate = NumOps.FromDouble(leakingRate);
        _regularization = NumOps.FromDouble(regularization);
        _warmupPeriod = warmupPeriod;
        _isTraining = false;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // Initialize the reservoir state and other vectors/matrices
        _reservoirState = new Vector<T>(_reservoirSize);
        _inputWeights = new Matrix<T>(_inputSize, _reservoirSize);
        _reservoirWeights = new Matrix<T>(_reservoirSize, _reservoirSize);
        _outputWeights = new Matrix<T>(_reservoirSize, _outputSize);
        _reservoirBias = new Vector<T>(_reservoirSize);
        _outputBias = new Vector<T>(_outputSize);
        _currentState = new Vector<T>(_inputSize);

        // Initialize activation functions
        _reservoirInputScalarActivation = reservoirInputScalarActivation;
        _reservoirOutputScalarActivation = reservoirOutputScalarActivation;
        _reservoirScalarActivation = reservoirScalarActivation;
        _outputScalarActivation = outputScalarActivation;

        // Initialize collections for training
        _collectedStates = new List<Vector<T>>();
        _collectedTargets = new List<Vector<T>>();

        // Initialize weights with random values
        InitializeWeights();

        // Initialize layers
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the weights and reservoir state.
    /// </summary>
    private void InitializeWeights()
    {
        // Initialize weights with small random values
        _inputWeights = new Matrix<T>(_inputSize, _reservoirSize);
        _reservoirWeights = new Matrix<T>(_reservoirSize, _reservoirSize);
        _outputWeights = new Matrix<T>(_reservoirSize, _outputSize);
        _reservoirBias = new Vector<T>(_reservoirSize);
        _outputBias = new Vector<T>(_outputSize);
        _currentState = new Vector<T>(_reservoirSize); // Start with zero state
        _leakingRate = NumOps.FromDouble(1.0); // Default to no leaking
        _regularization = NumOps.FromDouble(1e-4); // Default regularization
        _warmupPeriod = 10; // Default warmup period

        // Initialize input weights and reservoir bias
        for (int i = 0; i < _inputSize; i++)
        {
            for (int j = 0; j < _reservoirSize; j++)
            {
                _inputWeights[i, j] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * 0.1);
            }
        }

        // Initialize reservoir weights with sparse connections based on sparsity
        for (int i = 0; i < _reservoirSize; i++)
        {
            for (int j = 0; j < _reservoirSize; j++)
            {
                if (_random.NextDouble() < _sparsity)
                {
                    _reservoirWeights[i, j] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * 0.1);
                }
                else
                {
                    _reservoirWeights[i, j] = NumOps.Zero;
                }
            }

            _reservoirBias[i] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * 0.1);
        }

        // Scale reservoir weights to achieve desired spectral radius
        _reservoirWeights = ScaleToSpectralRadius(_reservoirWeights, _spectralRadius);

        // Initialize output weights and bias to zero
        // (These will be learned during training)
        for (int i = 0; i < _reservoirSize; i++)
        {
            for (int j = 0; j < _outputSize; j++)
            {
                _outputWeights[i, j] = NumOps.Zero;
            }
        }

        for (int i = 0; i < _outputSize; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }

        // Initialize collection lists for training
        _collectedStates = new List<Vector<T>>();
        _collectedTargets = new List<Vector<T>>();
    }

    /// <summary>
    /// Scales a matrix to achieve the desired spectral radius.
    /// </summary>
    /// <param name="matrix">The matrix to scale.</param>
    /// <param name="targetRadius">The target spectral radius.</param>
    /// <returns>The scaled matrix.</returns>
    private Matrix<T> ScaleToSpectralRadius(Matrix<T> matrix, double targetRadius)
    {
        // Calculate the current spectral radius using the power method
        double currentRadius = CalculateSpectralRadius(matrix);

        // Scale the matrix to achieve the target radius
        double scaleFactor = targetRadius / currentRadius;

        // Create a new scaled matrix
        Matrix<T> scaledMatrix = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                scaledMatrix[i, j] = NumOps.Multiply(matrix[i, j], NumOps.FromDouble(scaleFactor));
            }
        }

        return scaledMatrix;
    }

    /// <summary>
    /// Calculates the spectral radius of a matrix using the power method.
    /// </summary>
    /// <param name="matrix">The matrix to calculate the spectral radius for.</param>
    /// <returns>The spectral radius.</returns>
    private double CalculateSpectralRadius(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> x = new Vector<T>(n);

        // Initialize with a random vector
        for (int i = 0; i < n; i++)
        {
            x[i] = NumOps.FromDouble(_random.NextDouble());
        }

        // Normalize
        x = NormalizeVector(x);

        // Iterate using power method (typically 100 iterations is sufficient)
        for (int iter = 0; iter < 100; iter++)
        {
            Vector<T> y = matrix.Multiply(x);
            x = NormalizeVector(y);
        }

        // Calculate Rayleigh quotient
        Vector<T> Ax = matrix.Multiply(x);
        T rayleighQuotient = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            rayleighQuotient = NumOps.Add(rayleighQuotient, NumOps.Multiply(Ax[i], x[i]));
        }

        return Math.Abs(Convert.ToDouble(rayleighQuotient));
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// </summary>
    /// <param name="vector">The vector to normalize.</param>
    /// <returns>The normalized vector.</returns>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        // Compute L2 norm using vectorized dot product: sqrt(sum(vector^2))
        T normSquared = Engine.DotProduct(vector, vector);
        T norm = NumOps.Sqrt(normSquared);

        // Avoid division by zero
        if (MathHelper.AlmostEqual(norm, NumOps.Zero))
        {
            return new Vector<T>(vector.Length); // Return zero vector
        }

        // Vectorized normalization: vector / norm
        return (Vector<T>)Engine.Divide(vector, norm);
    }

    /// <summary>
    /// Updates the reservoir state based on the input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    private void UpdateReservoirState(Vector<T> input)
    {
        // Vectorized input contribution: transpose(input_weights) * input
        var inputWeightsTransposed = Engine.MatrixTranspose(_inputWeights);
        Vector<T> inputContribution = Engine.MatrixVectorMultiply(inputWeightsTransposed, input);

        // Vectorized reservoir contribution: transpose(reservoir_weights) * current_state
        var reservoirWeightsTransposed = Engine.MatrixTranspose(_reservoirWeights);
        Vector<T> reservoirContribution = Engine.MatrixVectorMultiply(reservoirWeightsTransposed, _currentState);

        // Vectorized sum: input_contribution + reservoir_contribution + bias
        Vector<T> preActivation = Engine.Add(Engine.Add(inputContribution, reservoirContribution), _reservoirBias);

        // Apply activation function
        Vector<T> activated;
        if (_reservoirScalarActivation != null)
        {
            // Scalar activation must be applied element-wise
            activated = new Vector<T>(_reservoirSize);
            for (int i = 0; i < _reservoirSize; i++)
            {
                activated[i] = _reservoirScalarActivation.Activate(preActivation[i]);
            }
        }
        else if (_reservoirVectorActivation != null)
        {
            // Use vectorized activation
            activated = _reservoirVectorActivation.Activate(preActivation);
        }
        else
        {
            // Default to vectorized tanh using Engine
            activated = Engine.Tanh(preActivation);
        }

        // Apply leaking rate (vectorized)
        Vector<T> newState;
        if (MathHelper.AlmostEqual(_leakingRate, NumOps.One))
        {
            // No leaking
            newState = activated;
        }
        else
        {
            // Vectorized leaky integration: (1-a)*previous_state + a*new_state
            T oneMinusAlpha = NumOps.Subtract(NumOps.One, _leakingRate);
            var previousScaled = Engine.Multiply(_currentState, oneMinusAlpha);
            var activatedScaled = Engine.Multiply(activated, _leakingRate);
            newState = Engine.Add(previousScaled, activatedScaled);
        }

        // Update the current state
        _currentState = newState;
    }

    /// <summary>
    /// Computes the output based on the current reservoir state.
    /// </summary>
    /// <returns>The output vector.</returns>
    private Vector<T> ComputeOutput()
    {
        // Vectorized output: transpose(output_weights) * reservoir_state + output_bias
        var outputWeightsTransposed = Engine.MatrixTranspose(_outputWeights);
        Vector<T> linearOutput = Engine.MatrixVectorMultiply(outputWeightsTransposed, _currentState);
        Vector<T> preActivation = Engine.Add(linearOutput, _outputBias);

        // Apply output activation if specified
        Vector<T> output;
        if (_outputScalarActivation != null)
        {
            // Scalar activation must be applied element-wise
            output = new Vector<T>(_outputSize);
            for (int i = 0; i < _outputSize; i++)
            {
                output[i] = _outputScalarActivation.Activate(preActivation[i]);
            }
        }
        else if (_outputVectorActivation != null)
        {
            // Use vectorized activation
            output = _outputVectorActivation.Activate(preActivation);
        }
        else
        {
            // No activation, linear output
            output = preActivation;
        }

        return output;
    }

    /// <summary>
    /// Resets the reservoir state to zeros.
    /// </summary>
    public void ResetReservoirState()
    {
        // Vectorized reset using Engine tensor fill
        var stateTensor = new Tensor<T>(_currentState.ToArray(), [_reservoirSize]);
        Engine.TensorFill(stateTensor, NumOps.Zero);
        var zeroArray = stateTensor.ToArray();
        for (int i = 0; i < _reservoirSize; i++)
        {
            _currentState[i] = zeroArray[i];
        }
    }

    /// <summary>
    /// Sets the leaking rate for the reservoir.
    /// </summary>
    /// <param name="leakingRate">The leaking rate (between 0 and 1).</param>
    public void SetLeakingRate(double leakingRate)
    {
        if (leakingRate < 0 || leakingRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(leakingRate), "Leaking rate must be between 0 and 1.");
        }

        _leakingRate = NumOps.FromDouble(leakingRate);
    }

    /// <summary>
    /// Sets the regularization parameter for ridge regression.
    /// </summary>
    /// <param name="regularization">The regularization parameter.</param>
    public void SetRegularization(double regularization)
    {
        if (regularization < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(regularization), "Regularization must be non-negative.");
        }

        _regularization = NumOps.FromDouble(regularization);
    }

    /// <summary>
    /// Sets the warmup period for discarding initial transient reservoir states.
    /// </summary>
    /// <param name="warmupPeriod">The warmup period.</param>
    public void SetWarmupPeriod(int warmupPeriod)
    {
        if (warmupPeriod < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(warmupPeriod), "Warmup period must be non-negative.");
        }

        _warmupPeriod = warmupPeriod;
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
    /// Updates the output layer parameters (weights and biases) of the Echo State Network.
    /// </summary>
    /// <param name="parameters">A vector containing the output weights and biases to update.</param>
    /// <exception cref="ArgumentException">Thrown when parameter vector length doesn't match expected size.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the ESN's output layer parameters from a flat parameter vector. The parameter vector
    /// must have a length equal to (reservoirSize × outputSize) + outputSize. Note that this only updates the
    /// output layer - the reservoir weights remain fixed. While ESNs typically train using ridge regression
    /// (see the Train method), this method allows for direct parameter updates for external optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the output layer weights directly.
    ///
    /// Echo State Networks are different from standard neural networks:
    /// - Their reservoir weights stay fixed (unchangeable) after initialization
    /// - Only the output layer weights are trainable
    /// - They typically use ridge regression for training (not gradient descent)
    ///
    /// This method allows you to:
    /// - Directly set the output layer weights and biases
    /// - Integrate with external optimization algorithms
    /// - Transfer parameters from other sources
    ///
    /// <b>Note:</b> The reservoir weights are NOT affected by this method and remain fixed.
    /// For typical ESN training, use the Train method with ridge regression instead.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int outputWeightCount = _reservoirSize * _outputSize;
        int expectedLength = outputWeightCount + _outputSize;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Parameter vector length mismatch. Expected {expectedLength} parameters but got {parameters.Length}.", nameof(parameters));
        }

        int paramIndex = 0;

        for (int i = 0; i < _reservoirSize; i++)
        {
            for (int j = 0; j < _outputSize; j++)
            {
                _outputWeights[i, j] = parameters[paramIndex++];
            }
        }

        for (int i = 0; i < _outputSize; i++)
        {
            _outputBias[i] = parameters[paramIndex++];
        }
    }

    /// <summary>
    /// Makes a prediction using the Echo State Network.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through the Echo State Network to make a prediction.
    /// It first flattens the input to a vector, then updates the reservoir state based on
    /// this input, and finally computes the output based on the updated reservoir state.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the ESN processes new information and makes predictions.
    /// 
    /// The prediction process works like this:
    /// 1. The input is prepared and flattened to a vector
    /// 2. The reservoir state is updated based on the input
    /// 3. The output is computed from the current reservoir state
    /// 
    /// The key difference from traditional neural networks is that the ESN's internal connections
    /// (the reservoir) aren't trained - only the output connections are adjusted during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Extract input as vector
        Vector<T> inputVector = input.ToVector();

        // Check if input size matches expected size
        if (inputVector.Length != _inputSize)
        {
            throw new ArgumentException($"Input vector length ({inputVector.Length}) does not match expected input size ({_inputSize}).");
        }

        // Update reservoir state
        UpdateReservoirState(inputVector);

        // Compute output
        Vector<T> outputVector = ComputeOutput();

        // Create and return output tensor
        return new Tensor<T>(new[] { 1, _outputSize }, outputVector);
    }

    /// <summary>
    /// Processes a sequence of inputs through the Echo State Network.
    /// </summary>
    /// <param name="inputSequence">The sequence of input tensors.</param>
    /// <param name="resetState">Whether to reset the reservoir state before processing.</param>
    /// <returns>The sequence of output tensors.</returns>
    /// <remarks>
    /// <para>
    /// This method processes a sequence of inputs through the Echo State Network, maintaining the
    /// reservoir state between time steps. This is particularly useful for time series prediction
    /// and sequence processing tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This processes a sequence of inputs one after another.
    /// 
    /// When processing a sequence:
    /// 1. The reservoir state can be reset (optional) to start fresh
    /// 2. Each input in the sequence is processed in order
    /// 3. The state of the reservoir carries information between steps
    /// 4. A sequence of outputs is produced corresponding to each input
    /// 
    /// This maintains the "memory" of the network across the sequence, making ESNs
    /// particularly good for time series and sequential data.
    /// </para>
    /// </remarks>
    public List<Tensor<T>> PredictSequence(List<Tensor<T>> inputSequence, bool resetState = true)
    {
        if (resetState)
        {
            ResetReservoirState();
        }

        List<Tensor<T>> outputs = new List<Tensor<T>>();

        foreach (var input in inputSequence)
        {
            outputs.Add(Predict(input));
        }

        return outputs;
    }

    /// <summary>
    /// Trains the Echo State Network on a single batch of data.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Echo State Network on a single batch of data. For ESNs, training
    /// is different from traditional neural networks. Instead of using backpropagation to update
    /// all weights, only the output weights are trained, typically using ridge regression.
    /// During the training phase, the method collects reservoir states and corresponding target
    /// outputs to be used in the regression.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the ESN learns from examples.
    /// 
    /// The training process works like this:
    /// 1. If this is the first training call, start collecting reservoir states and targets
    /// 2. Update the reservoir state based on the input
    /// 3. Collect the current reservoir state and the expected output
    /// 4. When training is complete, solve for the optimal output weights using ridge regression
    /// 
    /// Unlike traditional neural networks where all weights are adjusted gradually,
    /// ESNs learn by mathematically solving for the optimal output weights in one step.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Mark that we're training
        if (!_isTraining)
        {
            _isTraining = true;
            _collectedStates.Clear();
            _collectedTargets.Clear();
        }

        // Flatten input and output
        Vector<T> inputVector = input.ToVector();
        Vector<T> targetVector = expectedOutput.ToVector();

        // Check input and output sizes
        if (inputVector.Length != _inputSize)
        {
            throw new ArgumentException($"Input vector length ({inputVector.Length}) does not match expected input size ({_inputSize}).");
        }

        if (targetVector.Length != _outputSize)
        {
            throw new ArgumentException($"Target vector length ({targetVector.Length}) does not match expected output size ({_outputSize}).");
        }

        // Update reservoir state
        UpdateReservoirState(inputVector);

        // Calculate current prediction
        Vector<T> prediction = ComputeOutput();

        // Calculate and store loss
        LastLoss = _lossFunction.CalculateLoss(prediction, targetVector);

        // Collect state and target (skip if we're still in warmup period)
        if (_collectedStates.Count >= _warmupPeriod)
        {
            _collectedStates.Add(_currentState.Clone());
            _collectedTargets.Add(targetVector.Clone());
        }
    }

    /// <summary>
    /// Finalizes training by computing the optimal output weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method finalizes the training of the Echo State Network by computing the optimal
    /// output weights using ridge regression. It solves the equation (X^T X + ?I)^(-1) X^T Y,
    /// where X is the matrix of collected reservoir states, Y is the matrix of target outputs,
    /// and ? is the regularization parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This completes the training by solving for the best output weights.
    /// 
    /// After collecting all training examples:
    /// 1. We create matrices from all collected states and targets
    /// 2. We solve a mathematical equation (ridge regression) to find the weights
    /// 3. These weights will minimize the error between predictions and targets
    /// 4. The regularization parameter helps prevent overfitting
    /// 
    /// This one-shot learning approach is more efficient than the iterative
    /// approach used in traditional neural networks.
    /// </para>
    /// </remarks>
    public void FinalizeTraining()
    {
        if (!_isTraining || _collectedStates.Count == 0)
        {
            throw new InvalidOperationException("No training data collected. Call Train first.");
        }

        // Prepare matrices for ridge regression
        // X: Matrix of reservoir states
        // Y: Matrix of target outputs
        int numSamples = _collectedStates.Count;

        Matrix<T> X = new Matrix<T>(numSamples, _reservoirSize);
        Matrix<T> Y = new Matrix<T>(numSamples, _outputSize);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < _reservoirSize; j++)
            {
                X[i, j] = _collectedStates[i][j];
            }

            for (int j = 0; j < _outputSize; j++)
            {
                Y[i, j] = _collectedTargets[i][j];
            }
        }

        // Perform ridge regression: (X^T X + ?I)^(-1) X^T Y
        // Step 1: Compute X^T X
        Matrix<T> XtX = X.Transpose().Multiply(X);

        // Step 2: Add regularization (X^T X + ?I)
        Matrix<T> regularized = XtX.Clone();
        for (int i = 0; i < _reservoirSize; i++)
        {
            regularized[i, i] = NumOps.Add(regularized[i, i], _regularization);
        }

        // Step 3: Compute (X^T X + ?I)^(-1)
        Matrix<T> inverse = ComputeInverse(regularized);

        // Step 4: Compute X^T Y
        Matrix<T> XtY = X.Transpose().Multiply(Y);

        // Step 5: Compute (X^T X + ?I)^(-1) X^T Y
        Matrix<T> weights = inverse.Multiply(XtY);

        // Update output weights
        for (int i = 0; i < _reservoirSize; i++)
        {
            for (int j = 0; j < _outputSize; j++)
            {
                _outputWeights[i, j] = weights[i, j];
            }
        }

        // Compute bias terms (mean of target - mean of prediction)
        // For each output dimension, we need to compute:
        // bias = mean(targets) - mean(weights * states)
        for (int j = 0; j < _outputSize; j++)
        {
            T targetSum = NumOps.Zero;
            for (int i = 0; i < numSamples; i++)
            {
                targetSum = NumOps.Add(targetSum, Y[i, j]);
            }
            T targetMean = NumOps.Divide(targetSum, NumOps.FromDouble(numSamples));

            // For each sample, compute the output without bias
            T outputSum = NumOps.Zero;
            for (int i = 0; i < numSamples; i++)
            {
                T output = NumOps.Zero;
                for (int k = 0; k < _reservoirSize; k++)
                {
                    output = NumOps.Add(output, NumOps.Multiply(_outputWeights[k, j], X[i, k]));
                }
                outputSum = NumOps.Add(outputSum, output);
            }
            T outputMean = NumOps.Divide(outputSum, NumOps.FromDouble(numSamples));

            // Bias is target mean - output mean
            _outputBias[j] = NumOps.Subtract(targetMean, outputMean);
        }

        // Reset training state
        _isTraining = false;
        _collectedStates.Clear();
        _collectedTargets.Clear();
    }

    /// <summary>
    /// Computes the inverse of a matrix using Gaussian elimination.
    /// </summary>
    /// <param name="matrix">The matrix to invert.</param>
    /// <returns>The inverse of the matrix.</returns>
    private Matrix<T> ComputeInverse(Matrix<T> matrix)
    {
        // For simplicity, we'll assume the matrix is invertible and not ill-conditioned
        // A more robust implementation would use SVD or other techniques

        int n = matrix.Rows;
        if (n != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square.");
        }

        // Create augmented matrix [A|I]
        Matrix<T> augmented = new Matrix<T>(n, 2 * n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
            }

            // Identity matrix on the right
            augmented[i, i + n] = NumOps.One;
        }

        // Gaussian elimination
        for (int i = 0; i < n; i++)
        {
            // Find pivot
            T pivot = augmented[i, i];
            int pivotRow = i;

            // Find the row with the largest absolute value in this column
            for (int j = i + 1; j < n; j++)
            {
                if (Math.Abs(Convert.ToDouble(augmented[j, i])) > Math.Abs(Convert.ToDouble(pivot)))
                {
                    pivot = augmented[j, i];
                    pivotRow = j;
                }
            }

            // Swap rows if needed
            if (pivotRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    T temp = augmented[i, j];
                    augmented[i, j] = augmented[pivotRow, j];
                    augmented[pivotRow, j] = temp;
                }
            }

            // Scale the pivot row
            for (int j = 0; j < 2 * n; j++)
            {
                augmented[i, j] = NumOps.Divide(augmented[i, j], pivot);
            }

            // Eliminate other rows
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                {
                    T factor = augmented[j, i];
                    for (int k = 0; k < 2 * n; k++)
                    {
                        augmented[j, k] = NumOps.Subtract(
                            augmented[j, k],
                            NumOps.Multiply(factor, augmented[i, k])
                        );
                    }
                }
            }
        }

        // Extract the inverse
        Matrix<T> inverse = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, j + n];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Gets metadata about the Echo State Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the Echo State Network, including its model type,
    /// reservoir size, spectral radius, sparsity, and other configuration parameters.
    /// This information is useful for model management and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your ESN's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (Echo State Network)
    /// - Details about reservoir size and connectivity
    /// - Information about activation functions
    /// - Serialized data that can be used to save and reload the model
    /// 
    /// This information is useful for tracking different model configurations
    /// and for saving/loading models for later use.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.EchoStateNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ReservoirSize", _reservoirSize },
                { "SpectralRadius", _spectralRadius },
                { "Sparsity", _sparsity },
                { "InputSize", _inputSize },
                { "OutputSize", _outputSize },
                { "LeakingRate", Convert.ToDouble(_leakingRate) },
                { "Regularization", Convert.ToDouble(_regularization) },
                { "WarmupPeriod", _warmupPeriod }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Echo State Network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific parameters and state of the Echo State Network to a binary stream.
    /// It includes the reservoir size, spectral radius, sparsity, weight matrices, activation functions,
    /// and other configuration parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the special configuration and current state of your ESN.
    /// 
    /// It's like taking a snapshot of the network that includes:
    /// - Its structural configuration (reservoir size, connectivity, etc.)
    /// - The weight matrices that determine how signals flow
    /// - The activation functions that process signals
    /// - The current state of the reservoir
    /// 
    /// This allows you to save the network and reload it later exactly as it was.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write basic configuration
        writer.Write(_reservoirSize);
        writer.Write(_spectralRadius);
        writer.Write(_sparsity);
        writer.Write(_inputSize);
        writer.Write(_outputSize);
        writer.Write(Convert.ToDouble(_leakingRate));
        writer.Write(Convert.ToDouble(_regularization));
        writer.Write(_warmupPeriod);

        // Write activation function information
        // Write scalar activation function flags
        writer.Write(_reservoirInputScalarActivation != null);
        writer.Write(_reservoirOutputScalarActivation != null);
        writer.Write(_reservoirScalarActivation != null);
        writer.Write(_outputScalarActivation != null);

        // Write vector activation function flags
        writer.Write(_reservoirInputVectorActivation != null);
        writer.Write(_reservoirOutputVectorActivation != null);
        writer.Write(_reservoirVectorActivation != null);
        writer.Write(_outputVectorActivation != null);

        // Serialize activation functions if present
        if (_reservoirInputScalarActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _reservoirInputScalarActivation);
        }

        if (_reservoirOutputScalarActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _reservoirOutputScalarActivation);
        }

        if (_reservoirScalarActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _reservoirScalarActivation);
        }

        if (_outputScalarActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _outputScalarActivation);
        }

        if (_reservoirInputVectorActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _reservoirInputVectorActivation);
        }

        if (_reservoirOutputVectorActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _reservoirOutputVectorActivation);
        }

        if (_reservoirVectorActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _reservoirVectorActivation);
        }

        if (_outputVectorActivation != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _outputVectorActivation);
        }

        // Write weight matrices and bias vectors
        SerializeMatrix(writer, _inputWeights);
        SerializeMatrix(writer, _reservoirWeights);
        SerializeMatrix(writer, _outputWeights);
        SerializeVector(writer, _reservoirBias);
        SerializeVector(writer, _outputBias);

        // Write current state
        SerializeVector(writer, _currentState);
    }

    /// <summary>
    /// Serializes a matrix to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="matrix">The matrix to serialize.</param>
    private void SerializeMatrix(BinaryWriter writer, Matrix<T> matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                writer.Write(Convert.ToDouble(matrix[i, j]));
            }
        }
    }

    /// <summary>
    /// Serializes a vector to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="vector">The vector to serialize.</param>
    private void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            writer.Write(Convert.ToDouble(vector[i]));
        }
    }

    /// <summary>
    /// Deserializes Echo State Network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific parameters and state of the Echo State Network from a binary stream.
    /// It reconstructs the reservoir size, spectral radius, sparsity, weight matrices, activation functions,
    /// and other configuration parameters from the serialized data.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds the ESN from saved data.
    /// 
    /// It's like restoring the network from a snapshot, including:
    /// - Its structural configuration (reservoir size, connectivity, etc.)
    /// - The weight matrices that determine how signals flow
    /// - The activation functions that process signals
    /// - The state of the reservoir at the time it was saved
    /// 
    /// This allows you to continue using the network exactly where you left off.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read basic configuration
        _reservoirSize = reader.ReadInt32();
        _spectralRadius = reader.ReadDouble();
        _sparsity = reader.ReadDouble();
        _inputSize = reader.ReadInt32();
        _outputSize = reader.ReadInt32();
        _leakingRate = NumOps.FromDouble(reader.ReadDouble());
        _regularization = NumOps.FromDouble(reader.ReadDouble());
        _warmupPeriod = reader.ReadInt32();

        // Read activation function flags
        bool hasReservoirInputScalarActivation = reader.ReadBoolean();
        bool hasReservoirOutputScalarActivation = reader.ReadBoolean();
        bool hasReservoirScalarActivation = reader.ReadBoolean();
        bool hasOutputScalarActivation = reader.ReadBoolean();

        bool hasReservoirInputVectorActivation = reader.ReadBoolean();
        bool hasReservoirOutputVectorActivation = reader.ReadBoolean();
        bool hasReservoirVectorActivation = reader.ReadBoolean();
        bool hasOutputVectorActivation = reader.ReadBoolean();

        // Deserialize activation functions if present
        if (hasReservoirInputScalarActivation)
        {
            _reservoirInputScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        }

        if (hasReservoirOutputScalarActivation)
        {
            _reservoirOutputScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        }

        if (hasReservoirScalarActivation)
        {
            _reservoirScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        }

        if (hasOutputScalarActivation)
        {
            _outputScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        }

        if (hasReservoirInputVectorActivation)
        {
            _reservoirInputVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        }

        if (hasReservoirOutputVectorActivation)
        {
            _reservoirOutputVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        }

        if (hasReservoirVectorActivation)
        {
            _reservoirVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        }

        if (hasOutputVectorActivation)
        {
            _outputVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        }

        // Read weight matrices and bias vectors
        _inputWeights = DeserializeMatrix(reader);
        _reservoirWeights = DeserializeMatrix(reader);
        _outputWeights = DeserializeMatrix(reader);
        _reservoirBias = DeserializeVector(reader);
        _outputBias = DeserializeVector(reader);

        // Read current state
        _currentState = DeserializeVector(reader);

        // Initialize training collections
        _collectedStates = new List<Vector<T>>();
        _collectedTargets = new List<Vector<T>>();
        _isTraining = false;
    }

    /// <summary>
    /// Deserializes a matrix from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <returns>The deserialized matrix.</returns>
    private Matrix<T> DeserializeMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();

        Matrix<T> matrix = new Matrix<T>(rows, columns);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        return matrix;
    }

    /// <summary>
    /// Deserializes a vector from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <returns>The deserialized vector.</returns>
    private Vector<T> DeserializeVector(BinaryReader reader)
    {
        int length = reader.ReadInt32();

        Vector<T> vector = new Vector<T>(length);

        for (int i = 0; i < length; i++)
        {
            vector[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return vector;
    }

    /// <summary>
    /// Creates a new instance of the EchoStateNetwork with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new EchoStateNetwork instance with the same architecture and configuration as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the EchoStateNetwork with the same architecture, reservoir size,
    /// spectral radius, sparsity, and activation functions as the current instance. This is useful for model cloning,
    /// ensemble methods, or cross-validation scenarios where multiple instances of the same model with identical
    /// configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the ESN's blueprint.
    /// 
    /// When you need multiple versions of the same type of ESN with identical settings:
    /// - This method creates a new, empty ESN with the same configuration
    /// - It's like making a copy of your pool design before building it
    /// - The new ESN has the same structure but no trained data
    /// - This is useful for techniques that need multiple models, like ensemble methods
    /// 
    /// For example, when training on different data streams,
    /// you'd want each ESN to have the same architecture and reservoir properties.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_reservoirInputVectorActivation != null || _reservoirOutputVectorActivation != null ||
            _reservoirVectorActivation != null || _outputVectorActivation != null)
        {
            // If using vector activations
            return new EchoStateNetwork<T>(
                Architecture,
                _reservoirSize,
                _spectralRadius,
                _sparsity,
                Convert.ToDouble(_leakingRate),
                Convert.ToDouble(_regularization),
                _warmupPeriod,
                _lossFunction,
                _reservoirInputVectorActivation,
                _reservoirOutputVectorActivation,
                _reservoirVectorActivation,
                _outputVectorActivation);
        }
        else
        {
            // If using scalar activations
            return new EchoStateNetwork<T>(
                Architecture,
                _reservoirSize,
                _spectralRadius,
                _sparsity,
                Convert.ToDouble(_leakingRate),
                Convert.ToDouble(_regularization),
                _warmupPeriod,
                _lossFunction,
                _reservoirInputScalarActivation,
                _reservoirOutputScalarActivation,
                _reservoirScalarActivation,
                _outputScalarActivation);
        }
    }
}
