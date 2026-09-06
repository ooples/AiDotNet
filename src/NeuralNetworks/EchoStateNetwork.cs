using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
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
/// <example>
/// <code>
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 100, 1 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new EchoStateNetwork&lt;float&gt;())
///     .Build(trainX, trainY);
/// var output = result.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("The 'echo state' approach to analysing and training recurrent neural networks", "https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf", Year = 2001, Authors = "Herbert Jaeger")]
public partial class EchoStateNetwork<T> : SequenceModelLayoutBase<T>
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
    private T _spectralRadius;

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
    private T _sparsity;

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
    [AiDotNet.Attributes.Buffer]
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
    [AiDotNet.Attributes.FrozenParameter]
    private Matrix<T> _inputWeights;

    /// <summary>
    /// The weight matrix for reservoir-to-reservoir connections.
    /// </summary>
    [AiDotNet.Attributes.FrozenParameter]
    private Matrix<T> _reservoirWeights;

    /// <summary>
    /// Cached transposes of the fixed reservoir matrices. ESN input and reservoir weights never train,
    /// so rebuilding these matrices on every settling iteration only creates avoidable work and GC churn.
    /// </summary>
    [Scratch]
    private Matrix<T> _inputWeightsTransposed = null!;
    [Scratch]
    private Matrix<T> _reservoirWeightsTransposed = null!;

    /// <summary>
    /// The weight matrix for reservoir-to-output connections.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputWeights;

    /// <summary>
    /// The bias vector for the reservoir.
    /// </summary>
    [AiDotNet.Attributes.FrozenParameter]
    private Vector<T> _reservoirBias;

    /// <summary>
    /// The bias vector for the output layer.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _outputBias;

    /// <summary>
    /// The current state of the reservoir.
    /// </summary>
    [AiDotNet.Attributes.Buffer]
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
    [Scratch]
    private List<Vector<T>> _collectedStates;

    /// <summary>
    /// Collected targets during training for regression.
    /// </summary>
    [Scratch]
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
    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public EchoStateNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 1),
            reservoirSize: 100, warmupPeriod: 0, reservoirInputVectorActivation: (IVectorActivationFunction<T>?)null)
    {
    }

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
        _spectralRadius = NumOps.FromDouble(spectralRadius);
        _sparsity = NumOps.FromDouble(sparsity);
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
        _outputWeights = new Tensor<T>([_reservoirSize, _outputSize]);
        _reservoirBias = new Vector<T>(_reservoirSize);
        _outputBias = new Vector<T>(_outputSize);
        _currentState = new Vector<T>(_reservoirSize); // Must match reservoir size, not input size

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
        _spectralRadius = NumOps.FromDouble(spectralRadius);
        _sparsity = NumOps.FromDouble(sparsity);
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
        _outputWeights = new Tensor<T>([_reservoirSize, _outputSize]);
        _reservoirBias = new Vector<T>(_reservoirSize);
        _outputBias = new Vector<T>(_outputSize);
        _currentState = new Vector<T>(_reservoirSize); // Must match reservoir size, not input size

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
        _outputWeights = new Tensor<T>([_reservoirSize, _outputSize]);
        _reservoirBias = new Vector<T>(_reservoirSize);
        _outputBias = new Vector<T>(_outputSize);
        _currentState = new Vector<T>(_reservoirSize); // Start with zero state

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
                if (_random.NextDouble() < NumOps.ToDouble(_sparsity))
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
        _reservoirWeights = ScaleToSpectralRadius(_reservoirWeights, NumOps.ToDouble(_spectralRadius));

        // Initialize output weights with small random values (Xavier-like initialization).
        // Zero initialization causes Predict to always return zero before training,
        // which makes DifferentInputs/ScaledInput tests fail.
        double scale = 1.0 / Math.Sqrt(_reservoirSize);
        for (int i = 0; i < _reservoirSize; i++)
        {
            for (int j = 0; j < _outputSize; j++)
            {
                _outputWeights[i, j] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * scale);
            }
        }

        for (int i = 0; i < _outputSize; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }

        // Initialize collection lists for training
        _collectedStates = new List<Vector<T>>();
        _collectedTargets = new List<Vector<T>>();

        RefreshReservoirWeightCaches();
    }

    /// <summary>
    /// Rebuilds derived matrix layouts after the fixed reservoir weights are initialized or restored.
    /// </summary>
    private void RefreshReservoirWeightCaches()
    {
        _inputWeightsTransposed = Engine.MatrixTranspose(_inputWeights);
        _reservoirWeightsTransposed = Engine.MatrixTranspose(_reservoirWeights);
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
        double currentRadius = NumOps.ToDouble(CalculateSpectralRadius(matrix));

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
    private T CalculateSpectralRadius(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> x = new Vector<T>(n);

        // Initialize with a random vector
        for (int i = 0; i < n; i++)
        {
            x[i] = NumOps.FromDouble(_random.NextDouble());
        }

        // Normalize
        x = VectorHelper.Normalize(x);

        // Iterate using power method (typically 100 iterations is sufficient)
        for (int iter = 0; iter < 100; iter++)
        {
            Vector<T> y = matrix.Multiply(x);
            x = VectorHelper.Normalize(y);
        }

        // Calculate Rayleigh quotient
        Vector<T> Ax = matrix.Multiply(x);
        T rayleighQuotient = Engine.DotProduct(Ax, x);

        return NumOps.Abs(rayleighQuotient);
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// <summary>
    /// Updates the reservoir state based on the input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    private void UpdateReservoirState(Vector<T> input)
    {
        // The fixed weight layouts are transposed once during initialization/deserialization rather than
        // once per settling step. A single prediction may need all 200 steps.
        Vector<T> inputContribution = Engine.MatrixVectorMultiply(_inputWeightsTransposed, input);

        Vector<T> reservoirContribution = Engine.MatrixVectorMultiply(_reservoirWeightsTransposed, _currentState);

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
    /// Resets the reservoir and drives the (static) input until the reservoir settles onto its
    /// input-driven fixed point, leaving the result in <see cref="_currentState"/>.
    /// </summary>
    /// <remarks>
    /// This is the SINGLE canonical state-derivation procedure shared by both <see cref="Train"/>
    /// and <see cref="Predict"/>. ESN ridge-regression (Jaeger 2001 §3.4) is only correct when the
    /// state the readout is fitted on is the same state inference reproduces for that input. The
    /// previous implementation broke this: <see cref="Train"/> drove the reservoir CONTINUOUSLY
    /// across calls (collecting the full transient trajectory x₁…x_N) while <see cref="Predict"/>
    /// reset and took a single step (x₁). Because x₁ is only one of N collected constraints, more
    /// training diluted its weight in the fitted readout, so the readout fit x₁ ever more loosely
    /// and Predict's loss ROSE with more iterations — the MoreData_ShouldNotDegrade failure.
    /// Deriving the state by reset + settle-to-fixed-point in BOTH paths makes the collected state a
    /// deterministic function of the input alone (x*), so the readout is fitted on and evaluated at
    /// exactly the same state and repeated training monotonically refines the same fit. The echo
    /// state property (spectral radius &lt; 1) guarantees convergence; <c>_warmupPeriod</c> (when
    /// set) forces a minimum settle so the recurrent reservoir contributes, and the cap bounds cost.
    /// </remarks>
    private void SettleReservoirState(Vector<T> input)
    {
        for (int i = 0; i < _reservoirSize; i++)
            _currentState[i] = NumOps.Zero;

        const int maxSettleSteps = 200;
        int minSteps = Math.Max(1, _warmupPeriod);
        T convergenceTol = NumOps.FromDouble(1e-6);

        for (int step = 0; step < maxSettleSteps; step++)
        {
            Vector<T> previousState = _currentState.Clone();
            UpdateReservoirState(input);

            if (step + 1 < minSteps) continue;

            T maxDelta = NumOps.Zero;
            for (int i = 0; i < _reservoirSize; i++)
            {
                T delta = NumOps.Abs(NumOps.Subtract(_currentState[i], previousState[i]));
                if (NumOps.GreaterThan(delta, maxDelta)) maxDelta = delta;
            }
            if (NumOps.LessThan(maxDelta, convergenceTol)) break;
        }
    }

    /// <summary>
    /// Computes the output based on the current reservoir state.
    /// </summary>
    /// <returns>The output vector.</returns>
    private Vector<T> ComputeOutput()
    {
        // Vectorized output: transpose(output_weights) * reservoir_state + output_bias.
        // _outputWeights is [reservoirSize, outputSize], so state-as-row @ W gives [1, outputSize]
        // directly -- the same product the MatrixTranspose + MatrixVectorMultiply pair computed,
        // without materialising the transpose. It is a tensor because the readout is this model's
        // only trainable parameter block and the base restores by writing through declared tensors.
        var stateRow = new Tensor<T>([1, _reservoirSize], _currentState);
        var linearRow = Engine.TensorMatMul(stateRow, _outputWeights);
        Vector<T> linearOutput = new Vector<T>(_outputSize);
        for (int j = 0; j < _outputSize; j++)
        {
            linearOutput[j] = linearRow[0, j];
        }
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
                spectralRadius: NumOps.ToDouble(_spectralRadius),
                sparsity: NumOps.ToDouble(_sparsity)
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

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
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
    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    protected override Tensor<T> PredictCore(Tensor<T> input)
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

        // Derive the reservoir state via the SAME reset+settle procedure Train collects on, so the
        // readout is evaluated on exactly the state it was fitted on (see SettleReservoirState).
        SettleReservoirState(inputVector);

        // Compute output
        Vector<T> outputVector = ComputeOutput();

        // Create and return output tensor
        return new Tensor<T>(new[] { 1, _outputSize }, outputVector);
    }

    /// <summary>
    /// Reports this network's two real computation stages: the settled reservoir state and the
    /// linear readout.
    /// </summary>
    /// <remarks>
    /// <para>
    /// NEITHER BASE STRATEGY CAN SEE AN ECHO STATE NETWORK. An ESN computes with raw
    /// <see cref="Matrix{T}"/> / <see cref="Vector{T}"/> algebra -- <c>_inputWeights</c>,
    /// <c>_reservoirWeights</c>, <c>_outputWeights</c> -- and holds no <c>ILayer</c> instances at
    /// all. So <see cref="Layers"/> is empty and the sequential fold reports nothing, and because no
    /// <c>LayerBase.Forward</c> is ever invoked the observer fallback records nothing either. The
    /// base returned an empty dictionary, which it documents as a failure to answer rather than an
    /// answer of "no activations".
    /// </para>
    /// <para>
    /// Overriding is the established route for exactly this shape: SwinTransformer does the same to
    /// surface its <c>ExtractFeatures</c> stages. Both stages here come from the real forward path
    /// (<c>SettleReservoirState</c> then <c>ComputeOutput</c>, the same pair
    /// <see cref="PredictCore"/> runs), so the reported activations are what the model actually
    /// computes, not a reconstruction.
    /// </para>
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        Vector<T> inputVector = input.ToVector();
        if (inputVector.Length != _inputSize)
        {
            throw new ArgumentException(
                $"Input vector length ({inputVector.Length}) does not match expected input size ({_inputSize}).",
                nameof(input));
        }

        // The managed path deliberately, not TryForwardGpuOptimized: the intermediate reservoir
        // state is the point of this method, and the fused GPU path only yields the final output.
        SettleReservoirState(inputVector);

        // CLONE, do not alias. Tensor<T>(shape, vector) wraps the vector it is handed, and
        // SettleReservoirState zeroes _currentState IN PLACE on its next call -- so a previously
        // returned "Reservoir" tensor would silently change underneath a caller that kept it, which
        // is exactly what an activation snapshot must not do. ComputeOutput already returns a fresh
        // vector, so only the reservoir state needs copying.
        var reservoirState = new Tensor<T>([1, _reservoirSize], _currentState.Clone());
        var readout = new Tensor<T>([1, _outputSize], ComputeOutput());

        return new Dictionary<string, Tensor<T>>
        {
            ["Reservoir"] = reservoirState,
            ["Readout"] = readout,
        };
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
    /// <summary>
    /// This model has no parameter gradients: it is trained by ridge regression on the linear readout,
    /// not by gradient descent.
    /// </summary>
    /// <remarks>
    /// Saying so explicitly is the honest answer, and the one the gradient-accessor contract asks
    /// for -- populate the surface from the tape, or state that it cannot be populated. Returning
    /// zeros here would claim every parameter had a vanishing gradient, which is a different and
    /// false statement. Jaeger 2001: the reservoir is fixed and only the readout W_out is fitted, in closed form.
    /// </remarks>
    /// <exception cref="NotSupportedException">Always, by design.</exception>
    public override Vector<T> GetParameterGradients() =>
        throw new NotSupportedException(
            $"{nameof(EchoStateNetwork<T>)} is trained by ridge regression on the linear readout, not gradient "
            + "descent, so it has no parameter gradients to report.");

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // ESN training per Jaeger 2001 "The 'echo state' approach" §3.4 /
        // Lukoševičius 2012 §4: the reservoir is FIXED random; the only
        // trainable weights are the linear readout W_out, which is solved
        // via ridge regression
        //     W_out = (X X^T + λI)^{-1} X Y^T
        // where X stacks reservoir states and Y stacks targets across the
        // training sequence. The previous implementation did per-call SGD
        // with lr=0.01 on the readout, which diverged under reservoir
        // feedback (the recurrent state evolves between iterations, so SGD
        // chased a moving target and loss grew rather than shrank — that
        // tripped Training_ShouldReduceLoss). We restore paper fidelity by
        // accumulating (state, target) pairs and re-solving ridge
        // regression on every call so the readout converges to the
        // closed-form optimum after each step.
        if (!_isTraining)
        {
            _isTraining = true;
            _collectedStates.Clear();
            _collectedTargets.Clear();
        }

        Vector<T> inputVector = input.ToVector();
        Vector<T> targetVector = expectedOutput.ToVector();

        if (inputVector.Length != _inputSize)
        {
            throw new ArgumentException($"Input vector length ({inputVector.Length}) does not match expected input size ({_inputSize}).");
        }
        if (targetVector.Length != _outputSize)
        {
            throw new ArgumentException($"Target vector length ({targetVector.Length}) does not match expected output size ({_outputSize}).");
        }

        // Derive this sample's reservoir state with the SAME reset+settle procedure Predict uses
        // (see SettleReservoirState). Resetting per sample makes the collected state a deterministic
        // function of the input — the exact state inference reproduces — instead of the continuous,
        // accumulation-order-dependent trajectory the previous implementation collected (which
        // Predict never reproduced, so loss grew with more training). The settle discards the
        // initial transient per Lukoševičius 2012 §6.4.
        SettleReservoirState(inputVector);
        _collectedStates.Add(_currentState.Clone());
        _collectedTargets.Add(targetVector.Clone());

        // Resolve the readout from all collected samples so far. This is
        // the same closed-form solve <see cref="FinalizeTraining"/> runs
        // at the end; we just don't clear the collection so further
        // Train() calls keep accumulating samples and re-solving against
        // the growing dataset.
        SolveReadoutRidgeRegression();

        // Report the post-solve training loss. ESN Train() is a closed-form
        // readout solve rather than a gradient step, so callers expect the
        // public loss to describe the fitted readout after this sample has
        // joined the regression set.
        Vector<T> prediction = ComputeOutput();
        LastLoss = _lossFunction.CalculateLoss(prediction, targetVector);
    }

    /// <summary>
    /// Solves the closed-form ridge regression for <see cref="_outputWeights"/>
    /// (and <see cref="_outputBias"/>) given the currently collected
    /// reservoir states and targets. Extracted so both <see cref="Train"/>
    /// and <see cref="FinalizeTraining"/> share the same paper-faithful
    /// solver and can't drift apart.
    /// </summary>
    private void SolveReadoutRidgeRegression()
    {
        int numSamples = _collectedStates.Count;
        if (numSamples == 0) return;

        int readoutFeatureCount = _reservoirSize + 1; // reservoir state plus bias feature

        // Ridge regression per Jaeger 2001 §3.4 / Lukoševičius 2012 §6.2, PRIMAL normal-equation
        // form  W = (XᵀX + λI)⁻¹ XᵀY, accumulated and solved in DOUBLE precision regardless of T.
        //
        // Two reasons for the primal form + double solve:
        //  • Primal inverts the (readoutFeatureCount × readoutFeatureCount) normal matrix XᵀX + λI
        //    (size fixed by the reservoir, NOT the sample count). With state derived by reset+settle
        //    (SettleReservoirState), repeated identical inputs add IDENTICAL rows; the dual Gram
        //    X Xᵀ + λI (numSamples²) then becomes rank-deficient and grows with every Train call,
        //    producing garbage readouts. XᵀX + λI is positive definite for any λ > 0 regardless of
        //    duplicate rows (λI lifts the null space).
        //  • Forming the normal matrix squares the condition number, and with many duplicate
        //    fixed-point rows the conditioning reaches ~‖x*‖²·N/λ ≈ 1e8 — beyond float's ~7 digits,
        //    which made the readout explode (memorization loss 1e-5 → 2.4). Accumulating and solving
        //    in double gives ~15 digits, comfortably covering that range; the result is cast back
        //    to T. The constant bias feature is solved jointly (standard ESN readout design matrix).
        int n = readoutFeatureCount;
        double lambda = Convert.ToDouble(_regularization);
        double[,] a = new double[n, n];        // XᵀX + λI
        double[,] b = new double[n, _outputSize]; // XᵀY

        for (int s = 0; s < numSamples; s++)
        {
            var state = _collectedStates[s];
            var tgt = _collectedTargets[s];
            // Build the augmented feature row x̃ = [state…, 1] in double.
            for (int i = 0; i < n; i++)
            {
                double xi = i < _reservoirSize ? Convert.ToDouble(state[i]) : 1.0;
                for (int j = i; j < n; j++)
                {
                    double xj = j < _reservoirSize ? Convert.ToDouble(state[j]) : 1.0;
                    a[i, j] += xi * xj;
                }
                for (int j = 0; j < _outputSize; j++)
                    b[i, j] += xi * Convert.ToDouble(tgt[j]);
            }
        }
        // Mirror the symmetric upper triangle into the lower, then add λ to the diagonal.
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
                a[j, i] = a[i, j];
            a[i, i] += lambda;
        }

        double[,]? weights = SolveLinearSystemDouble(a, b); // (n × outputSize)
        if (weights is null) return; // singular/degenerate — keep the previous readout

        // Safety net: a degenerate solve can still surface NaN/Inf. Keep the previous readout
        // rather than poisoning the model.
        for (int i = 0; i < n; i++)
            for (int j = 0; j < _outputSize; j++)
                if (double.IsNaN(weights[i, j]) || double.IsInfinity(weights[i, j]))
                    return;

        for (int i = 0; i < _reservoirSize; i++)
            for (int j = 0; j < _outputSize; j++)
                _outputWeights[i, j] = NumOps.FromDouble(weights[i, j]);

        for (int j = 0; j < _outputSize; j++)
            _outputBias[j] = NumOps.FromDouble(weights[_reservoirSize, j]);
    }

    /// <summary>
    /// Solves the symmetric positive-definite system <c>A · W = B</c> in double precision via
    /// Gauss-Jordan elimination with partial pivoting, returning <c>W</c> (or <c>null</c> if A is
    /// singular). Used by the ESN ridge readout solve so the conditioning of the (squared) normal
    /// matrix is handled at full double precision even when the model's numeric type T is float.
    /// </summary>
    private static double[,]? SolveLinearSystemDouble(double[,] a, double[,] b)
    {
        int n = a.GetLength(0);
        int m = b.GetLength(1);

        // Augment [A | B] and run Gauss-Jordan with partial pivoting.
        double[,] aug = new double[n, n + m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) aug[i, j] = a[i, j];
            for (int j = 0; j < m; j++) aug[i, n + j] = b[i, j];
        }

        for (int col = 0; col < n; col++)
        {
            int pivotRow = col;
            double maxAbs = Math.Abs(aug[col, col]);
            for (int r = col + 1; r < n; r++)
            {
                double v = Math.Abs(aug[r, col]);
                if (v > maxAbs) { maxAbs = v; pivotRow = r; }
            }
            if (maxAbs < 1e-300) return null; // singular

            if (pivotRow != col)
                for (int j = 0; j < n + m; j++)
                    (aug[col, j], aug[pivotRow, j]) = (aug[pivotRow, j], aug[col, j]);

            double pivot = aug[col, col];
            for (int j = 0; j < n + m; j++) aug[col, j] /= pivot;

            for (int r = 0; r < n; r++)
            {
                if (r == col) continue;
                double factor = aug[r, col];
                if (factor == 0.0) continue;
                for (int j = 0; j < n + m; j++)
                    aug[r, j] -= factor * aug[col, j];
            }
        }

        double[,] result = new double[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                result[i, j] = aug[i, n + j];
        return result;
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

        // Delegate to the shared dual-form ridge solver. The previous body
        // here inlined the primal form `(X^T X + λI)^{-1} X^T Y`, which
        // can numerically explode when reservoir states are nearly
        // collinear (a regime feedback drives toward over long sequences
        // — same failure mode that caused per-call Train SGD to diverge).
        // The shared helper uses the dual form `X^T (X X^T + λI)^{-1} Y`
        // (Jaeger 2001 §3.4 / Lukoševičius 2012 §6.2) which stays
        // well-conditioned for any λ > 0; without this delegation a caller
        // running incremental Train() followed by FinalizeTraining() would
        // overwrite the stable readout with the numerically weaker path
        // this PR was introduced to remove.
        SolveReadoutRidgeRegression();

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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ReservoirSize", _reservoirSize },
                { "SpectralRadius", NumOps.ToDouble(_spectralRadius) },
                { "Sparsity", NumOps.ToDouble(_sparsity) },
                { "InputSize", _inputSize },
                { "OutputSize", _outputSize },
                { "LeakingRate", Convert.ToDouble(_leakingRate) },
                { "Regularization", Convert.ToDouble(_regularization) },
                { "WarmupPeriod", _warmupPeriod }
            },
            ModelData = SerializeForMetadata()
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


    /// <summary>
    /// Serializes a matrix to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="matrix">The matrix to serialize.</param>
    /// <summary>
    /// Writes a 2-D tensor in EXACTLY the layout <see cref="SerializeMatrix"/> uses (rows, then
    /// columns, then row-major doubles), so a checkpoint written before the readout became a
    /// tensor still reads back correctly.
    /// </summary>
    private void SerializeTensor2D(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape[0]);
        writer.Write(tensor.Shape[1]);

        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                writer.Write(Convert.ToDouble(tensor[i, j]));
            }
        }
    }

    /// <summary>Reads a 2-D tensor written by <see cref="SerializeTensor2D"/>.</summary>
    private Tensor<T> DeserializeTensor2D(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int columns = reader.ReadInt32();

        var tensor = new Tensor<T>([rows, columns]);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                tensor[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        return tensor;
    }

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
}
