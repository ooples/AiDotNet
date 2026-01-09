using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected (dense) layer in a neural network.
/// </summary>
/// <remarks>
/// <para>
/// A dense layer connects every input neuron to every output neuron, with each connection having
/// a learnable weight. This is the most basic and widely used type of neural network layer.
/// Dense layers are capable of learning complex patterns by adjusting these weights during training.
/// </para>
/// <para><b>For Beginners:</b> A dense layer is like a voting system where every input gets to vote on every output.
///
/// Think of it like this:
/// - Each input sends information to every output
/// - Each connection has a different "importance" (weight)
/// - The layer learns which connections should be strong and which should be weak
///
/// For example, in an image recognition task:
/// - One input might detect a curved edge
/// - Another might detect a straight line
/// - The dense layer combines these features to recognize higher-level patterns
///
/// Dense layers are the building blocks of many neural networks because they can learn
/// almost any relationship between inputs and outputs, given enough neurons and training data.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This layer is not thread-safe. Each layer instance maintains internal state
/// during forward and backward passes. If you need concurrent execution, use separate layer instances
/// per thread or synchronize access to shared instances.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DenseLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (weight regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Weight regularization adds a penalty based on the magnitude of the weights to prevent overfitting.
    /// This helps the network generalize better to unseen data by discouraging overly complex models.
    /// </para>
    /// <para><b>For Beginners:</b> Weight regularization is like encouraging simplicity in your model.
    ///
    /// Why use regularization:
    /// - Prevents the network from memorizing training data (overfitting)
    /// - Encourages the network to learn general patterns instead of specific details
    /// - Makes the model work better on new, unseen data
    ///
    /// Think of it like learning to recognize cats:
    /// - Without regularization: "This cat has exactly 157 whiskers" (too specific)
    /// - With regularization: "Cats have fur, whiskers, and pointy ears" (general pattern)
    ///
    /// Regularization is especially helpful when you have limited training data.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the regularization auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much the regularization penalty contributes to the total loss.
    /// The total loss is: main_loss + (auxiliary_weight * regularization_loss).
    /// Typical values range from 0.0001 to 0.1.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the network should prefer simple models.
    ///
    /// The weight determines the balance between:
    /// - Fitting the training data well (main loss)
    /// - Keeping the model simple (regularization loss)
    ///
    /// Common values:
    /// - 0.01 (default): Moderate regularization
    /// - 0.001-0.005: Light regularization
    /// - 0.05-0.1: Strong regularization
    ///
    /// Higher values make the network simpler but might underfit the data.
    /// Lower values allow more complexity but might overfit.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the type of regularization to apply.
    /// </summary>
    public RegularizationType Regularization { get; set; } = RegularizationType.None;

    /// <summary>
    /// Gets or sets the L1 regularization strength (used when Regularization is L1 or L1L2).
    /// </summary>
    public T L1Strength { get; set; }

    /// <summary>
    /// Gets or sets the L2 regularization strength (used when Regularization is L2 or L1L2).
    /// </summary>
    public T L2Strength { get; set; }

    private T _lastRegularizationLoss;

    /// <summary>
    /// Tracks whether lazy initialization has been completed.
    /// </summary>
    private bool _isInitialized;

    /// <inheritdoc />
    public override bool IsInitialized => _isInitialized;

    /// <summary>
    /// The weight matrix that connects input neurons to output neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix represents the strength of connections between input and output neurons.
    /// Shape: [inputSize, outputSize] (industry standard convention).
    /// Each row corresponds to an input neuron, and each column corresponds to an output neuron.
    /// The value at position [i, j] is the weight of the connection from input i to output j.
    /// </para>
    /// <para><b>For Beginners:</b> The weights matrix is like a table of importance scores.
    ///
    /// Imagine a table where:
    /// - Each row represents one input neuron
    /// - Each column represents one output neuron
    /// - Each cell contains a number (weight) showing how strongly that input affects that output
    ///
    /// During training, these numbers change to help the network make better predictions.
    /// Positive weights strengthen connections, negative weights create inhibitory connections,
    /// and weights close to zero mean the connection is weak or unimportant.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights;

    /// <summary>
    /// The bias values added to each output neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains a bias value for each output neuron. Biases allow the network to shift
    /// the activation function, enabling it to fit the data better. Each bias is added to the weighted
    /// sum of inputs before applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like default values or thresholds for each output.
    ///
    /// Think of biases as:
    /// - A starting point or base value for each output
    /// - A way to adjust how easily an output neuron can "activate" or "fire"
    /// - Added after all the weighted inputs are summed up
    ///
    /// For example, a high bias might make an output neuron activate even with weak input signals,
    /// while a negative bias would require stronger input signals to activate.
    /// </para>
    /// </remarks>
    private Tensor<T> _biases;

    /// <summary>
    /// Temporary storage for weight gradients during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), this matrix stores the calculated gradients for the weights.
    /// These gradients indicate how much and in which direction each weight should be adjusted to reduce
    /// the network's error.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the "improvement directions" for all the weights.
    /// 
    /// When training the network:
    /// - The layer calculates how each weight should change
    /// - These changes are stored here temporarily
    /// - They're applied to the actual weights during the update step
    /// 
    /// It's like having a notepad where you write down all the adjustments you need to make
    /// before actually making them.
    /// </para>
    /// </remarks>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Temporary storage for bias gradients during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), this tensor stores the calculated gradients for the biases.
    /// These gradients indicate how much and in which direction each bias should be adjusted to reduce
    /// the network's error.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the "improvement directions" for all the biases.
    ///
    /// When training the network:
    /// - The layer calculates how each bias should change
    /// - These changes are stored here temporarily
    /// - They're applied to the actual biases during the update step
    ///
    /// It works together with the weight gradients to update all the layer's parameters.
    /// </para>
    /// </remarks>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Stored input data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), the layer needs access to the input data from the forward
    /// pass to calculate the gradients for the weights. This tensor stores that input data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the network's "short-term memory" of what it just processed.
    /// 
    /// The layer remembers:
    /// - The last batch of data it saw
    /// - So it can calculate exactly how to improve
    /// 
    /// Without this stored input, the layer wouldn't know which inputs contributed to
    /// errors in the output, making learning impossible.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput; // Pre-activation output for proper gradient computation

    // GPU-resident cached tensors for GPU training pipeline
    private IGpuTensor<T>? _lastInputGpu;
    private IGpuTensor<T>? _lastPreActivationGpu; // Pre-activation for GPU backward pass
    private IGpuTensor<T>? _lastOutputGpu; // Post-activation for sigmoid/tanh backward
    private int[]? _gpuOriginalInputShape;


    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>
    /// The sum of the number of weights and biases in the layer.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the total number of trainable parameters in the layer, which is the sum
    /// of the number of elements in the weights matrix and the biases vector. This is useful for
    /// understanding the complexity of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many individual numbers the layer can adjust during training.
    ///
    /// The parameter count:
    /// - Equals (number of inputs × number of outputs) + number of outputs
    /// - First part counts the weights, second part counts the biases
    /// - Higher numbers mean more flexibility but also more risk of overfitting
    ///
    /// For example, a dense layer with 100 inputs and 50 outputs would have
    /// 100 × 50 = 5,000 weights plus 50 biases, for a total of 5,050 parameters.
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // For lazy initialization, compute from InputShape/OutputShape if not yet initialized
            if (!_isInitialized)
            {
                return (InputShape[0] * OutputShape[0]) + OutputShape[0];
            }
            return (_weights.Shape[0] * _weights.Shape[1]) + _biases.Shape[0];
        }
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> for dense layers, as they contain trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation. Dense
    /// layers have trainable parameters (weights and biases), so they support training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// For dense layers:
    /// - The value is always true
    /// - This means the layer can adjust its weights and biases during training
    /// - It will improve its performance as it sees more examples
    /// 
    /// Some other layer types might not have trainable parameters and would return false here.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseLayer{T}"/> class with the specified 
    /// input and output sizes and a scalar activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="activationFunction">The activation function to apply. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a dense layer with the specified number of input and output neurons.
    /// The weights are initialized using Xavier/Glorot initialization, which scales the random values
    /// based on the number of input and output neurons. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method creates a new dense layer with specific dimensions.
    /// 
    /// When creating the layer, you specify:
    /// - How many inputs it will receive (inputSize)
    /// - How many outputs it will produce (outputSize)
    /// - What mathematical function to apply to the results (activation)
    /// 
    /// For example, a layer with inputSize=784 and outputSize=10 could connect the flattened
    /// pixels of a 28×28 image to 10 output neurons (one for each digit 0-9).
    /// 
    /// The layer automatically initializes all the weights and biases with carefully chosen
    /// starting values that help with training.
    /// </para>
    /// </remarks>
    public DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        if (inputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size must be greater than zero.");
        }

        if (outputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output size must be greater than zero.");
        }

        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        L1Strength = NumOps.FromDouble(0.01);
        L2Strength = NumOps.FromDouble(0.01);
        _lastRegularizationLoss = NumOps.Zero;

        // Store the initialization strategy
        InitializationStrategy = initializationStrategy;

        // Check if using lazy initialization
        if (initializationStrategy is { IsLazy: true })
        {
            // Defer weight allocation until first Forward() call
            // Create placeholder tensors with zero size - they'll be properly allocated in EnsureInitialized
            _weights = new Tensor<T>([0, 0]);
            _biases = new Tensor<T>([0]);
            _isInitialized = false;
        }
        else
        {
            // Eager initialization - allocate and initialize immediately
            _weights = new Tensor<T>([inputSize, outputSize]);
            _biases = new Tensor<T>([outputSize]);

            // Use strategy if provided, otherwise use default Xavier initialization
            if (initializationStrategy is not null)
            {
                initializationStrategy.InitializeWeights(_weights, inputSize, outputSize);
                initializationStrategy.InitializeBiases(_biases);
            }
            else
            {
                InitializeParameters();
            }

            // Register trainable parameters with the engine for GPU persistence
            RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
            _isInitialized = true;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseLayer{T}"/> class with the specified
    /// input and output sizes and a vector activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="vectorActivation">The vector activation function to apply (required to disambiguate from IActivationFunction overload).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a dense layer with the specified number of input and output neurons
    /// and a vector activation function. Vector activation functions operate on entire vectors at once,
    /// which can be more efficient for certain operations.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method is similar to the previous one, but uses a different type of
    /// activation function.
    ///
    /// A vector activation function:
    /// - Works on all outputs at once instead of one at a time
    /// - Can be more efficient for certain calculations
    /// - Might capture relationships between different outputs
    ///
    /// Most of the time, you'll use the standard constructor, but this one gives you
    /// flexibility if you need special activation functions that work on the entire
    /// output vector at once.
    /// </para>
    /// <para>
    /// <b>Note:</b> If your activation function implements both IActivationFunction and IVectorActivationFunction,
    /// use <see cref="WithActivation"/> or <see cref="WithVectorActivation"/> factory methods to avoid ambiguity.
    /// </para>
    /// </remarks>
    public DenseLayer(int inputSize, int outputSize, IVectorActivationFunction<T> vectorActivation,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputSize], [outputSize], vectorActivation)
    {
        if (inputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size must be greater than zero.");
        }

        if (outputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output size must be greater than zero.");
        }

        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        L1Strength = NumOps.FromDouble(0.01);
        L2Strength = NumOps.FromDouble(0.01);
        _lastRegularizationLoss = NumOps.Zero;

        // Store the initialization strategy
        InitializationStrategy = initializationStrategy;

        // Check if using lazy initialization
        if (initializationStrategy is { IsLazy: true })
        {
            // Defer weight allocation until first Forward() call
            _weights = new Tensor<T>([0, 0]);
            _biases = new Tensor<T>([0]);
            _isInitialized = false;
        }
        else
        {
            // Eager initialization - allocate and initialize immediately
            _weights = new Tensor<T>([inputSize, outputSize]);
            _biases = new Tensor<T>([outputSize]);

            if (initializationStrategy is not null)
            {
                initializationStrategy.InitializeWeights(_weights, inputSize, outputSize);
                initializationStrategy.InitializeBiases(_biases);
            }
            else
            {
                InitializeParameters();
            }

            RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
            _isInitialized = true;
        }
    }

    /// <summary>
    /// Ensures that weights are allocated and initialized for lazy initialization.
    /// </summary>
    protected override void EnsureInitialized()
    {
        if (_isInitialized) return;

        lock (InitializationLock)
        {
            if (_isInitialized) return;

            int inputSize = InputShape[0];
            int outputSize = OutputShape[0];

            // Allocate weights and biases
            _weights = new Tensor<T>([inputSize, outputSize]);
            _biases = new Tensor<T>([outputSize]);

            // Initialize using strategy or default
            if (InitializationStrategy is not null)
            {
                InitializationStrategy.InitializeWeights(_weights, inputSize, outputSize);
                InitializationStrategy.InitializeBiases(_biases);
            }
            else
            {
                InitializeParameters();
            }

            // Register trainable parameters with the engine for GPU persistence
            RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

            _isInitialized = true;
        }
    }

    /// <summary>
    /// Initializes the weights and biases with appropriate values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using Xavier/Glorot initialization, which scales the random
    /// values based on the number of input and output neurons. This helps prevent the vanishing or
    /// exploding gradient problem during training. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for all connections in the layer.
    /// 
    /// When initializing:
    /// - Weights are set to small random values (not all zero)
    /// - The range of these random values is carefully chosen
    /// - Biases start at zero
    /// 
    /// Good initialization is important because:
    /// - It helps the network learn faster
    /// - It prevents training problems (like vanishing or exploding gradients)
    /// - It gives each neuron a different starting point
    /// 
    /// This uses a technique called "Xavier/Glorot initialization" which works well
    /// for most neural networks.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // === Vectorized Xavier/Glorot Initialization (Phase B: US-GPU-015) ===
        // Initialize weights with random values scaled by Xavier initialization
        // Initialize biases to zero using vectorized operation

        T scaleT = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(NumOps.FromDouble(2.0), NumOps.FromDouble(InputShape[0] + OutputShape[0])));
        var scale = Convert.ToDouble(scaleT);

        // Initialize weights (still requires loop for individual random values)
        for (int i = 0; i < _weights.Shape[0]; i++)
        {
            for (int j = 0; j < _weights.Shape[1]; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble() * scale - scale / 2);
            }
        }

        // Vectorized bias initialization - set all biases to zero at once
        for (int i = 0; i < _biases.Shape[0]; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for weight regularization (L1, L2, or both).
    /// </summary>
    /// <returns>The computed regularization auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the regularization loss based on the magnitude of the weights.
    /// L1 regularization computes the sum of absolute values of weights.
    /// L2 regularization computes the sum of squared values of weights.
    /// L1L2 combines both penalties.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how "complex" the layer's weights are.
    ///
    /// Different regularization types:
    /// 1. L1 (Lasso): Σ|weight|
    ///    - Encourages many weights to become exactly zero
    ///    - Creates sparse networks (many connections turned off)
    ///    - Good for feature selection
    ///
    /// 2. L2 (Ridge): Σ(weight²)
    ///    - Encourages all weights to be small
    ///    - Prevents any single weight from dominating
    ///    - Smooths the network's behavior
    ///
    /// 3. L1L2 (Elastic Net): Combines both
    ///    - Gets benefits of both L1 and L2
    ///    - More flexible regularization
    ///
    /// The loss is added to the main loss during training to discourage large weights.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || Regularization == RegularizationType.None)
        {
            _lastRegularizationLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Ensure weights are initialized (supports lazy initialization)
        EnsureInitialized();

        T regularizationLoss = NumOps.Zero;

        // === Vectorized L1 Regularization: Σ|w| (Phase B: US-GPU-015) ===
        if (Regularization == RegularizationType.L1 || Regularization == RegularizationType.ElasticNet)
        {
            // Use vectorized abs and sum operations
            var absWeights = Engine.TensorAbs(_weights);
            T l1Loss = Engine.TensorSum(absWeights);
            l1Loss = NumOps.Multiply(L1Strength, l1Loss);
            regularizationLoss = NumOps.Add(regularizationLoss, l1Loss);
        }

        // === Vectorized L2 Regularization: Σ(w²) (Phase B: US-GPU-015) ===
        if (Regularization == RegularizationType.L2 || Regularization == RegularizationType.ElasticNet)
        {
            // Use vectorized element-wise multiply and sum operations
            var weightsSquared = Engine.TensorMultiply(_weights, _weights);
            T l2Loss = Engine.TensorSum(weightsSquared);
            // L2 regularization is typically 0.5 * lambda * Σ(w²)
            l2Loss = NumOps.Multiply(L2Strength, l2Loss);
            l2Loss = NumOps.Multiply(NumOps.FromDouble(0.5), l2Loss);
            regularizationLoss = NumOps.Add(regularizationLoss, l2Loss);
        }

        _lastRegularizationLoss = regularizationLoss;
        return regularizationLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the weight regularization auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about the weight regularization, including
    /// the computed regularization loss, type of regularization, strengths, and whether it's enabled.
    /// This information is useful for monitoring training progress and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how regularization is affecting the layer.
    ///
    /// The diagnostics include:
    /// - Total regularization loss (penalty for large weights)
    /// - Type of regularization being used (L1, L2, L1L2, or None)
    /// - Strength parameters for L1 and L2
    /// - Weight applied to the regularization loss
    /// - Whether regularization is enabled
    ///
    /// This helps you:
    /// - Monitor if regularization is helping prevent overfitting
    /// - Debug issues with model complexity
    /// - Understand the impact of different regularization settings
    ///
    /// You can use this information to adjust regularization parameters for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalRegularizationLoss", _lastRegularizationLoss?.ToString() ?? "0" },
            { "RegularizationType", Regularization.ToString() },
            { "L1Strength", L1Strength?.ToString() ?? "0.01" },
            { "L2Strength", L2Strength?.ToString() ?? "0.01" },
            { "RegularizationWeight", AuxiliaryLossWeight?.ToString() ?? "0.01" },
            { "UseRegularization", UseAuxiliaryLoss.ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Sets the weights of the layer to specified values.
    /// </summary>
    /// <param name="weights">The weight matrix to set.</param>
    /// <exception cref="ArgumentNullException">Thrown when the weights parameter is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the weights matrix has incorrect dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method allows direct setting of the weight matrix, which can be useful for transfer learning,
    /// weight initialization with custom algorithms, or loading pre-trained models. The dimensions of the
    /// provided matrix must match the layer's input and output dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you directly set all connection strengths at once.
    ///
    /// You might use this to:
    /// - Load pre-trained weights from another model
    /// - Test the layer with specific weight values
    /// - Implement custom initialization strategies
    ///
    /// The weight matrix must have exactly the right dimensions:
    /// - Rows equal to the number of inputs (inputSize)
    /// - Columns equal to the number of outputs (outputSize)
    ///
    /// If the dimensions don't match, the method will throw an error.
    /// </para>
    /// </remarks>
    protected override void SetWeights(Tensor<T> weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        // Ensure weights are initialized before validation (supports lazy initialization)
        EnsureInitialized();

        // Validate dimensions against current weights: [inputSize, outputSize]
        if (weights.Shape[0] != _weights.Shape[0] || weights.Shape[1] != _weights.Shape[1])
        {
            throw new ArgumentException(
                $"Weight tensor dimensions must be {_weights.Shape[0]}x{_weights.Shape[1]}, but got {weights.Shape[0]}x{weights.Shape[1]}");
        }

        // Set the weights directly
        _weights = weights;

        // Update input shape if needed - Shape[0] is inputSize in new convention
        if (InputShape.Length == 0 || InputShape[0] != weights.Shape[0])
        {
            UpdateInputShape([weights.Shape[0]]);
        }

        // Notify engine that weights have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_weights);
    }

    /// <summary>
    /// Gets the weights tensor of the layer.
    /// </summary>
    /// <returns>The weight tensor connecting input neurons to output neurons.</returns>
    public override Tensor<T> GetWeights()
    {
        // Ensure weights are initialized (supports lazy initialization)
        EnsureInitialized();
        return _weights;
    }

    /// <summary>
    /// Gets the biases tensor of the layer.
    /// </summary>
    /// <returns>The bias values added to each output neuron.</returns>
    public override Tensor<T> GetBiases()
    {
        // Ensure biases are initialized (supports lazy initialization)
        EnsureInitialized();
        return _biases;
    }

    /// <summary>
    /// The original shape of the input tensor, used to restore shape after forward pass.
    /// </summary>
    private int[] _originalInputShape = [];

    /// <summary>
    /// Processes the input data through the dense layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after applying the dense layer transformation and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the forward pass of the dense layer. It multiplies the input by the weights,
    /// adds the biases, and applies the activation function. The result is a tensor where each element
    /// represents the activation of an output neuron.
    /// </para>
    /// <para>
    /// <b>Industry Standard:</b> Like PyTorch's nn.Linear, this layer supports any-rank input tensors.
    /// The transformation is applied to the last dimension, preserving all batch/sequence dimensions.
    /// For example, input [..., inputSize] produces output [..., outputSize].
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms input data into output data.
    ///
    /// During the forward pass:
    /// - The input values are multiplied by their corresponding weights
    /// - All weighted inputs for each output neuron are added together
    /// - The bias is added to each sum
    /// - The activation function is applied to each result
    ///
    /// For example, if your inputs represent image features, the outputs might represent
    /// the probability of the image belonging to different categories.
    ///
    /// This is where the actual "thinking" happens in the neural network.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Ensure weights are initialized (for lazy initialization)
        EnsureInitialized();

        _lastInput = input;
        _originalInputShape = input.Shape;

        // Industry standard: Support any-rank input tensors [..., inputSize]
        // Transformation is applied to the last dimension
        // Output shape: [..., outputSize]

        int actualInputSize = input.Shape[^1]; // Last dimension
        int expectedInputSize = _weights.Shape[0]; // Weights are [inputSize, outputSize]

        // Dynamic input size adaptation: resize weights if input size doesn't match
        if (actualInputSize != expectedInputSize)
        {
            EnsureWeightShapeForInput(actualInputSize);
        }

        int inputSize = actualInputSize;

        Tensor<T> flattenedInput;
        int batchDim;

        if (input.Rank == 1)
        {
            // 1D input [features]: reshape to [1, features]
            flattenedInput = input.Reshape(1, inputSize);
            batchDim = 1;
        }
        else if (input.Rank == 2)
        {
            // 2D input [batch, features]: use directly
            flattenedInput = input;
            batchDim = flattenedInput.Shape[0];
        }
        else
        {
            // ND input [..., features]: flatten batch dimensions
            // E.g., [batch, seq, features] -> [batch*seq, features]
            batchDim = 1;
            for (int i = 0; i < input.Rank - 1; i++)
            {
                batchDim *= input.Shape[i];
            }
            flattenedInput = input.Reshape(batchDim, inputSize);
        }

        // Forward: output = Activation(input @ weights + biases)
        // input: [batchDim, inputSize]
        // weights: [inputSize, outputSize] (industry standard - no transpose needed)
        // result: [batchDim, outputSize]

        // Get the fused activation type for the engine
        var fusedActivation = GetFusedActivationType();

        Tensor<T> result;

        if (fusedActivation != FusedActivationType.None)
        {
            // Use IEngine's FusedLinear for optimal GPU/CPU performance
            // Engine handles: GPU kernel fusion, persistent tensor caching, CPU SIMD fallback
            result = Engine.FusedLinear(flattenedInput, _weights, _biases, fusedActivation);

            // For fused operations, pre-activation is only needed for gradient computation during training.
            // Skip this expensive GPU operation during inference to avoid 50% overhead.
            if (IsTrainingMode)
            {
                _lastOutput = Engine.FusedLinear(flattenedInput, _weights, _biases, FusedActivationType.None);
            }
            // Note: During inference, _lastOutput is NOT set because:
            // 1. It's not needed for backprop (no Backward() call expected during inference)
            // 2. Setting it to activated values would produce incorrect gradients if Backward() were called
            // 3. Keeping it null/stale makes the intent clear and aligns with other layer patterns
        }
        else
        {
            // For unsupported activations, use FusedLinear without activation then apply separately
            var preActivation = Engine.FusedLinear(flattenedInput, _weights, _biases, FusedActivationType.None);
            _lastOutput = preActivation;
            result = ApplyActivation(preActivation);
        }

        // Reshape back to original shape with outputSize as last dimension
        // E.g., [batch*seq, outputSize] -> [batch, seq, outputSize]
        if (input.Rank == 1)
        {
            // 1D input: return 1D output [outputSize]
            result = result.Reshape(OutputShape[0]);
        }
        else if (input.Rank > 2)
        {
            // ND input: restore original batch dimensions with new last dim
            var outputShape = new int[input.Rank];
            for (int i = 0; i < input.Rank - 1; i++)
            {
                outputShape[i] = _originalInputShape[i];
            }
            outputShape[^1] = OutputShape[0];
            result = result.Reshape(outputShape);
        }
        // 2D input: result is already [batch, outputSize]

        return result;
    }

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs a GPU-resident forward pass, keeping tensors on GPU.
    /// Use this for chained layer execution to avoid CPU round-trips.
    /// Supports any-rank tensor input (1D, 2D, or ND), matching CPU Forward behavior.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensors (uses first input). Last dimension is features.</param>
    /// <returns>GPU-resident output tensor with same batch dimensions, outputSize as last dim.</returns>
    /// <exception cref="InvalidOperationException">Thrown if GPU execution is not available.</exception>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        EnsureInitialized();

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];

        // Store for potential backward pass
        _originalInputShape = input.Shape;

        int actualInputSize = input.Shape[^1]; // Last dimension is always features
        int expectedInputSize = _weights.Shape[0];

        // Dynamic input size adaptation
        if (actualInputSize != expectedInputSize)
        {
            EnsureWeightShapeForInput(actualInputSize);
        }

        int outputSize = OutputShape[0];

        // Determine if reshape is needed and compute the effective batch dimension
        int batchDim;
        bool needsReshape = false;
        int[] originalBatchDims = Array.Empty<int>();

        if (input.Shape.Length == 1)
        {
            // 1D input [features] -> treat as single sample
            batchDim = 1;
            needsReshape = true;
        }
        else if (input.Shape.Length == 2)
        {
            // 2D input [batch, features] -> standard case, no reshape needed
            batchDim = input.Shape[0];
            needsReshape = false;
        }
        else
        {
            // ND input [dim0, dim1, ..., features] -> flatten batch dims, then reshape back
            needsReshape = true;
            originalBatchDims = new int[input.Shape.Length - 1];
            batchDim = 1;
            for (int i = 0; i < input.Shape.Length - 1; i++)
            {
                originalBatchDims[i] = input.Shape[i];
                batchDim *= input.Shape[i];
            }
        }

        // Reshape ND input to 2D [totalBatch, features] for matrix multiply
        IGpuTensor<T> input2D = input;
        if (needsReshape && input.Shape.Length > 2)
        {
            input2D = input.CreateView(0, [batchDim, actualInputSize]);
        }
        else if (needsReshape && input.Shape.Length == 1)
        {
            input2D = input.CreateView(0, [1, actualInputSize]);
        }

        // Get the fused activation type
        var fusedActivation = GetFusedActivationType();

        // Use GPU-resident FusedLinear - NO CPU round-trip
        // Result is [batchDim, outputSize]
        var result = gpuEngine.FusedLinearGpu(input2D, _weights, _biases, fusedActivation);

        // Cache state for backward pass only during training - KEEP ON GPU for GPU-resident training
        if (IsTrainingMode)
        {
            // Store GPU-resident tensors for BackwardGpu (no CPU roundtrip)
            _lastInputGpu = input2D;
            _gpuOriginalInputShape = input.Shape.ToArray();

            // For fused activations, we need pre-activation for gradient computation
            if (fusedActivation != FusedActivationType.None)
            {
                _lastPreActivationGpu = gpuEngine.FusedLinearGpu(input2D, _weights, _biases, FusedActivationType.None);
                _lastOutputGpu = result; // Post-activation for sigmoid/tanh backward
            }
            else
            {
                _lastPreActivationGpu = result;
                _lastOutputGpu = result;
            }

            // Also download to CPU for hybrid CPU/GPU backward compatibility
            _lastInput = input.ToTensor();
            _lastOutput = _lastPreActivationGpu.ToTensor();
        }

        // Reshape output back to original batch dimensions if needed
        if (input.Shape.Length == 1)
        {
            // 1D input -> 1D output [outputSize]
            result = result.CreateView(0, [outputSize]);
        }
        else if (input.Shape.Length > 2)
        {
            // ND input -> ND output [dim0, dim1, ..., outputSize]
            int[] outputShape = new int[originalBatchDims.Length + 1];
            for (int i = 0; i < originalBatchDims.Length; i++)
            {
                outputShape[i] = originalBatchDims[i];
            }
            outputShape[^1] = outputSize;
            result = result.CreateView(0, outputShape);
        }
        // 2D input: result is already [batch, outputSize]

        return result;
    }

    /// <summary>
    /// Performs GPU-resident backward pass for the dense layer.
    /// Computes gradients for weights, biases, and input entirely on GPU - no CPU roundtrip.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient from the next layer.</param>
    /// <returns>GPU-resident gradient to pass to the previous layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown if ForwardGpu was not called first.</exception>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine");

        if (_lastInputGpu == null || _lastPreActivationGpu == null || _gpuOriginalInputShape == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu");

        // Ensure gradient is 2D for computation
        int batchDim = _lastInputGpu.Shape[0];
        int outputSize = OutputShape[0];
        int inputSize = _weights.Shape[0];

        IGpuTensor<T> gradient2D;
        if (outputGradient.Shape.Length == 1)
        {
            gradient2D = outputGradient.CreateView(0, [1, outputSize]);
        }
        else if (outputGradient.Shape.Length == 2)
        {
            gradient2D = outputGradient;
        }
        else
        {
            // Flatten ND gradient to 2D
            int flatBatch = 1;
            for (int i = 0; i < outputGradient.Shape.Length - 1; i++)
                flatBatch *= outputGradient.Shape[i];
            gradient2D = outputGradient.CreateView(0, [flatBatch, outputSize]);
        }

        // Step 1: Compute activation gradient using GPU-resident activation backward
        IGpuTensor<T> activationGradient = ComputeActivationGradientGpu(gpuEngine, gradient2D);

        // Step 2: Compute weight gradient: dW = input^T @ activationGrad
        // input: [batchDim, inputSize], activationGrad: [batchDim, outputSize]
        // input^T: [inputSize, batchDim], result: [inputSize, outputSize]
        var inputTransposed = gpuEngine.TransposeGpu<T>(_lastInputGpu);
        var weightsGradGpu = gpuEngine.MatMulGpuTensors<T>(inputTransposed, activationGradient);

        // Download weight gradient to CPU for UpdateParameters
        _weightsGradient = weightsGradGpu.ToTensor();

        // Step 3: Compute bias gradient: dB = sum(activationGrad, axis=0)
        // Result: [1, outputSize] -> reshape to [outputSize]
        var biasGradGpu = gpuEngine.SumAxisGpu<T>(activationGradient, 0);
        var biasGradTensor = biasGradGpu.ToTensor();
        _biasesGradient = biasGradTensor.Reshape([outputSize]);

        // Step 4: Compute input gradient: dX = activationGrad @ W^T
        // activationGrad: [batchDim, outputSize], W: [inputSize, outputSize]
        // W^T: [outputSize, inputSize], result: [batchDim, inputSize]
        var weightsGpu = gpuEngine.UploadToGpu(_weights, GpuTensorRole.Weight);
        var weightsTransposed = gpuEngine.TransposeGpu<T>(weightsGpu);
        var inputGradient = gpuEngine.MatMulGpuTensors<T>(activationGradient, weightsTransposed);

        // Reshape input gradient back to original shape if needed
        if (_gpuOriginalInputShape.Length == 1)
        {
            return inputGradient.CreateView(0, [inputSize]);
        }
        else if (_gpuOriginalInputShape.Length > 2)
        {
            return inputGradient.CreateView(0, _gpuOriginalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Computes activation gradient using GPU-resident backward operations.
    /// </summary>
    private IGpuTensor<T> ComputeActivationGradientGpu(DirectGpuTensorEngine gpuEngine, IGpuTensor<T> gradOutput)
    {
        // Determine activation type and apply appropriate backward
        var fusedActivation = GetFusedActivationType();

        return fusedActivation switch
        {
            FusedActivationType.ReLU => gpuEngine.ReluBackwardGpu<T>(gradOutput, _lastPreActivationGpu!),
            FusedActivationType.Sigmoid => gpuEngine.SigmoidBackwardGpu<T>(gradOutput, _lastOutputGpu!),
            FusedActivationType.Tanh => gpuEngine.TanhBackwardGpu<T>(gradOutput, _lastOutputGpu!),
            FusedActivationType.GELU => gpuEngine.GeluBackwardGpu<T>(gradOutput, _lastPreActivationGpu!),
            FusedActivationType.Swish => gpuEngine.SwishBackwardGpu<T>(gradOutput, _lastPreActivationGpu!),
            FusedActivationType.LeakyReLU => gpuEngine.LeakyReluBackwardGpu<T>(gradOutput, _lastPreActivationGpu!, 0.01f),
            FusedActivationType.Softmax => gpuEngine.SoftmaxBackwardGpu<T>(gradOutput, _lastOutputGpu!),
            FusedActivationType.None => gradOutput, // Identity activation - gradient passes through unchanged
            _ => gradOutput // Fallback for unsupported activations
        };
    }

    private void EnsureWeightShapeForInput(int actualInputSize)
    {
        // Weights are [inputSize, outputSize]
        if (_weights.Shape[0] == actualInputSize)
        {
            return;
        }

        int existingInputSize = _weights.Shape[0];
        int outputSize = _weights.Shape[1];
        var resizedWeights = new Tensor<T>([actualInputSize, outputSize]);

        int sharedInputSize = Math.Min(existingInputSize, actualInputSize);
        for (int i = 0; i < sharedInputSize; i++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                resizedWeights[i, o] = _weights[i, o];
            }
        }

        if (actualInputSize > sharedInputSize)
        {
            T scale = NumOps.FromDouble(Math.Sqrt(2.0 / (actualInputSize + outputSize)));
            var random = RandomHelper.CreateSecureRandom();
            for (int i = sharedInputSize; i < actualInputSize; i++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    resizedWeights[i, o] = NumOps.Multiply(scale, NumOps.FromDouble(random.NextDouble() * 2 - 1));
                }
            }
        }

        _weights = resizedWeights;
        _weightsGradient = null;
        UpdateInputShape([actualInputSize]);
    }

    /// <summary>
    /// Calculates gradients for the input, weights, and biases during backpropagation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the backward pass of the dense layer during training. It calculates
    /// the gradient of the loss with respect to the input, weights, and biases. The calculated
    /// gradients for weights and biases are stored for the subsequent parameter update, and the
    /// input gradient is returned for propagation to earlier layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps the layer learn from its mistakes.
    ///
    /// During the backward pass:
    /// - The layer receives information about how wrong its output was
    /// - It calculates how to adjust its weights and biases to be more accurate
    /// - It prepares the adjustments but doesn't apply them yet
    /// - It passes information back to previous layers so they can learn too
    ///
    /// This is where the actual "learning" happens. The layer figures out which connections
    /// should be strengthened and which should be weakened based on the error in its output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Calculate activation gradient: dL/dz = dL/dy * f'(z)
        // The activation was applied to _lastOutput (pre-activation), so use it for derivative computation
        bool shapeMatches = outputGradient.Rank == _lastOutput.Rank;
        if (shapeMatches)
        {
            for (int i = 0; i < _lastOutput.Shape.Length; i++)
            {
                if (_lastOutput.Shape[i] != outputGradient.Shape[i])
                {
                    shapeMatches = false;
                    break;
                }
            }
        }

        if (!shapeMatches && outputGradient.Length == _lastOutput.Length)
        {
            outputGradient = outputGradient.Reshape(_lastOutput.Shape);
        }

        Tensor<T> activationGradient;

        if (UsingVectorActivation && VectorActivation != null)
        {
            activationGradient = VectorActivation.Backward(_lastOutput, outputGradient);
        }
        else if (ScalarActivation != null)
        {
            activationGradient = ScalarActivation.Backward(_lastOutput, outputGradient);
        }
        else
        {
            activationGradient = outputGradient; // Identity
        }

        // Handle any-rank input: flatten to 2D for gradient computation
        int inputSize = _lastInput.Shape[^1];
        int batchDim;

        Tensor<T> flattenedInput;
        if (_lastInput.Rank == 1)
        {
            batchDim = 1;
            flattenedInput = _lastInput.Reshape(1, inputSize);
        }
        else if (_lastInput.Rank == 2)
        {
            batchDim = _lastInput.Shape[0];
            flattenedInput = _lastInput;
        }
        else
        {
            // ND input: flatten batch dimensions
            batchDim = 1;
            for (int i = 0; i < _lastInput.Rank - 1; i++)
            {
                batchDim *= _lastInput.Shape[i];
            }
            flattenedInput = _lastInput.Reshape(batchDim, inputSize);
        }

        // Flatten gradient to 2D [batchDim, outputSize] for tensor operations
        Tensor<T> flattenedGradient;
        if (activationGradient.Rank == 1)
        {
            flattenedGradient = activationGradient.Reshape(1, OutputShape[0]);
        }
        else if (activationGradient.Rank == 2)
        {
            flattenedGradient = activationGradient;
        }
        else
        {
            flattenedGradient = activationGradient.Reshape(batchDim, OutputShape[0]);
        }

        // 2. Compute Weight Gradients: dW = input^T @ dL/dz
        // Weights are [inputSize, outputSize], so gradient must have same shape
        // [inputSize, batchDim] @ [batchDim, outputSize] -> [inputSize, outputSize]
        var inputTransposed = Engine.TensorTranspose(flattenedInput);
        _weightsGradient = Engine.TensorMatMul(inputTransposed, flattenedGradient);

        // 3. Compute Bias Gradients: dB = sum(dL/dz, axis=0)
        // Sum gradients across the batch dimension
        _biasesGradient = flattenedGradient.Sum([0]);

        // 4. Compute Input Gradient: dX = dL/dz @ W^T
        // Weights are [inputSize, outputSize], need transpose for backward pass
        // [batchDim, outputSize] @ [outputSize, inputSize] -> [batchDim, inputSize]
        var weightsTransposed = Engine.TensorTranspose(_weights);
        var inputGradient = Engine.TensorMatMul(flattenedGradient, weightsTransposed);

        // Reshape back to original input shape
        return inputGradient.Reshape(_originalInputShape);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It's slower than the
    /// manual implementation but can be useful for:
    /// - Verifying gradient correctness
    /// - Rapid prototyping with custom modifications
    /// - Research and experimentation
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Handle any-rank input: flatten to 2D for gradient computation
        int inputSize = _lastInput.Shape[^1];
        int batchDim;

        if (_lastInput.Rank == 1)
        {
            batchDim = 1;
        }
        else if (_lastInput.Rank == 2)
        {
            batchDim = _lastInput.Shape[0];
        }
        else
        {
            batchDim = 1;
            for (int i = 0; i < _lastInput.Rank - 1; i++)
            {
                batchDim *= _lastInput.Shape[i];
            }
        }

        var flattenedInput = _lastInput.Reshape(batchDim, inputSize);

        // Create computation nodes directly from tensors
        var input = Autodiff.TensorOperations<T>.Variable(flattenedInput, "input", requiresGradient: true);
        var weights = Autodiff.TensorOperations<T>.Variable(_weights, "weights", requiresGradient: true);
        var biases = Autodiff.TensorOperations<T>.Variable(_biases, "biases", requiresGradient: true);

        // Forward computation using autodiff ops
        // output = input @ weights + biases (industry standard: no transpose needed)
        // Weights are [inputSize, outputSize]
        var matmul = Autodiff.TensorOperations<T>.MatrixMultiply(input, weights);

        // Add biases directly - autodiff Add operation handles broadcasting and gradient reduction
        // matmul is [batchSize, outputSize], biases is [outputSize]
        // Add broadcasts biases and reduces gradients automatically
        var output = Autodiff.TensorOperations<T>.Add(matmul, biases);

        // Apply activation using autodiff
        var activated = ApplyActivationAutodiff(output);

        // Manually propagate gradients using the output gradient we received
        // Flatten gradient to 2D for computation
        Tensor<T> flattenedOutputGradient;
        if (outputGradient.Rank == 1)
        {
            flattenedOutputGradient = outputGradient.Reshape(1, OutputShape[0]);
        }
        else if (outputGradient.Rank == 2)
        {
            flattenedOutputGradient = outputGradient;
        }
        else
        {
            flattenedOutputGradient = outputGradient.Reshape(batchDim, OutputShape[0]);
        }
        activated.Gradient = flattenedOutputGradient;

        // Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((activated, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients
        if (weights.Gradient == null)
            throw new InvalidOperationException("Weights gradient is null after backward pass");
        if (biases.Gradient == null)
            throw new InvalidOperationException("Biases gradient is null after backward pass");
        if (input.Gradient == null)
            throw new InvalidOperationException("Input gradient is null after backward pass");

        _weightsGradient = weights.Gradient;
        _biasesGradient = biases.Gradient;

        // Reshape back to original input shape
        return input.Gradient.Reshape(_originalInputShape);
    }

    /// <summary>
    /// Applies activation function using autodiff operations.
    /// </summary>
    private Autodiff.ComputationNode<T> ApplyActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        if (ScalarActivation is ReLUActivation<T>)
        {
            return Autodiff.TensorOperations<T>.ReLU(input);
        }
        else if (ScalarActivation is SigmoidActivation<T>)
        {
            return Autodiff.TensorOperations<T>.Sigmoid(input);
        }
        else if (ScalarActivation is TanhActivation<T>)
        {
            return Autodiff.TensorOperations<T>.Tanh(input);
        }
        else
        {
            // For unsupported activations, return input unchanged
            // This is a limitation of autodiff - not all activations are implemented yet
            return input;
        }
    }

    private Tensor<T>? _weightsVelocity;
    private Tensor<T>? _biasesVelocity;

    /// <summary>
    /// Updates the layer's parameters (weights and biases) using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (weights and biases) based on the gradients
    /// calculated during the backward pass. The learning rate controls the step size of the update.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the lessons learned during training.
    /// 
    /// When updating parameters:
    /// - The learning rate controls how big each adjustment is
    /// - Small learning rate = small, careful changes
    /// - Large learning rate = big, faster changes (but might overshoot)
    /// 
    /// The weights and biases are adjusted by subtracting the gradient multiplied by the learning rate.
    /// This moves them in the direction that reduces the error the most.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            // Initialize velocity tensors if needed (lazily)
            if (_weightsVelocity == null)
            {
                _weightsVelocity = new Tensor<T>(_weights.Shape);
                _weightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_weightsVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_biasesVelocity == null)
            {
                _biasesVelocity = new Tensor<T>(_biases.Shape);
                _biasesVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_biasesVelocity, PersistentTensorRole.OptimizerState);
            }

            // Perform GPU-resident SGD update
            gpuEngine.SgdMomentumUpdateGpu(_weights, _weightsGradient, _weightsVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_biases, _biasesGradient, _biasesVelocity, lr, 0.0f, 0.0f);
        }
        else
        {
            _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
            _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));

            // Notify engine that weights/biases have changed (for GPU cache invalidation)
            Engine.InvalidatePersistentTensor(_weights);
            Engine.InvalidatePersistentTensor(_biases);
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all trainable parameters (weights and biases) from the layer
    /// and returns them as a single vector. This is useful for optimization algorithms that operate
    /// on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method gathers all the learned values from the layer.
    ///
    /// The parameters include:
    /// - All weight values (connections between inputs and outputs)
    /// - All bias values (base values for each output)
    ///
    /// These are combined into a single long list (vector), which can be used for:
    /// - Saving the model
    /// - Sharing parameters between layers
    /// - Advanced optimization techniques
    ///
    /// This provides access to all the "knowledge" the layer has learned.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Ensure weights and biases are initialized (supports lazy initialization)
        EnsureInitialized();
        return Vector<T>.Concatenate(new Vector<T>(_weights.ToArray()), new Vector<T>(_biases.ToArray()));
    }

    /// <summary>
    /// Gets the gradients of all trainable parameters in this layer.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            return new Vector<T>(ParameterCount);
        }

        return Vector<T>.Concatenate(
            new Vector<T>(_weightsGradient.ToArray()),
            new Vector<T>(_biasesGradient.ToArray()));
    }

    /// <summary>
    /// Sets all trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (weights and biases) of the layer from a single
    /// vector. The vector must have the exact length required for all parameters of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learned values at once.
    ///
    /// When setting parameters:
    /// - The vector must have exactly the right number of values
    /// - The values are assigned to the weights and biases in a specific order
    ///
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Copying parameters from another model
    /// - Setting parameters that were optimized externally
    ///
    /// It's like replacing all the "knowledge" in the layer with new information.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Ensure weights and biases are initialized (supports lazy initialization)
        EnsureInitialized();

        int expected = _weights.Length + _biases.Length;
        if (parameters.Length != expected)
        {
            throw new ArgumentException($"Expected {expected} parameters, but got {parameters.Length}");
        }

        int index = 0;
        _weights = new Tensor<T>(_weights.Shape, parameters.Slice(index, _weights.Length));
        index += _weights.Length;
        _biases = new Tensor<T>(_biases.Shape, parameters.Slice(index, _biases.Length));

        // Notify engine that weights/biases have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Clears stored gradients for weights and biases.
    /// </summary>
    public override void ClearGradients()
    {
        if (_weightsGradient != null)
        {
            _weightsGradient.Fill(NumOps.Zero);
        }

        if (_biasesGradient != null)
        {
            _biasesGradient.Fill(NumOps.Zero);
        }

        base.ClearGradients();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the cached input values from the most recent forward pass and the gradients
    /// calculated during the backward pass. This is useful when starting to process a new batch or
    /// when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The layer forgets the last input it processed
    /// - It clears any calculated gradients
    /// 
    /// This is useful for:
    /// - Processing a new, unrelated set of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like wiping a whiteboard clean before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes (CPU)
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;

        // Clear GPU-resident cached tensors
        _lastInputGpu = null;
        _lastPreActivationGpu = null;
        _lastOutputGpu = null;
        _gpuOriginalInputShape = null;
    }

    /// <summary>
    /// Creates a deep copy of the layer with the same configuration and parameters.
    /// </summary>
    /// <returns>A new instance of the <see cref="DenseLayer{T}"/> class with the same configuration and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the dense layer, including its configuration and parameters.
    /// This is useful when you need multiple instances of the same layer, such as in ensemble methods or
    /// when implementing layer factories.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of the layer.
    /// 
    /// The copy:
    /// - Has the same input and output dimensions
    /// - Has the same weights and biases
    /// - Is completely independent from the original
    /// 
    /// This is useful for:
    /// - Creating multiple similar layers
    /// - Experimenting with variations of a layer
    /// - Implementing certain advanced techniques
    /// 
    /// Think of it like making a perfect clone that starts exactly where the original is.
    /// </para>
    /// </remarks>
    public override LayerBase<T> Clone()
    {
        DenseLayer<T> copy;

        if (UsingVectorActivation && VectorActivation is not null)
        {
            copy = new DenseLayer<T>(InputShape[0], OutputShape[0], VectorActivation);
        }
        else
        {
            copy = new DenseLayer<T>(InputShape[0], OutputShape[0], ScalarActivation);
        }

        copy.SetParameters(GetParameters());
        return copy;
    }

    /// <summary>
    /// Exports the dense layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes (input data, weights, biases).</param>
    /// <returns>The output computation node representing the layer's prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph that mirrors the layer's forward pass logic.
    /// The graph uses TensorOperations which now integrates with IEngine for GPU acceleration
    /// where supported (e.g., Add operations use IEngine.TensorAdd).
    /// </para>
    /// <para>
    /// Current IEngine integration status:
    /// - Addition operations: Fully GPU-accelerated via IEngine.TensorAdd
    /// - Matrix multiplication: Uses Tensor.MatrixMultiply (pending IEngine integration)
    /// - Transpose operations: Uses Tensor.Transpose (pending IEngine integration)
    /// </para>
    /// <para>
    /// The computation graph enables:
    /// - JIT compilation for optimized inference
    /// - Operation fusion and dead code elimination
    /// - Automatic differentiation via backpropagation
    /// - Deferred execution with GPU acceleration
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Validate parameters
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Ensure weights and biases are initialized (supports lazy initialization)
        EnsureInitialized();

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (!CanActivationBeJitted())
        {
            var activationType = ScalarActivation?.GetType().Name ?? VectorActivation?.GetType().Name ?? "unknown";
            throw new NotSupportedException(
                $"Activation function '{activationType}' is not supported for JIT compilation yet. " +
                "Supported activations: ReLU, Sigmoid, Tanh, Softmax");
        }

        // Input shape: [batchSize, inputSize]
        // Weights are [inputSize, outputSize] in industry standard convention
        int inputSize = _weights.Shape[0];

        // Create placeholder for input data
        // Note: Using batch size 1 for placeholder; actual batch size is determined at runtime
        var inputPlaceholder = new Tensor<T>(new int[] { 1, inputSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "input");

        // Create constant nodes for weights and biases
        // Weights shape: [inputSize, outputSize] (industry standard - no transpose needed)
        var weightsNode = TensorOperations<T>.Variable(_weights, "weights");

        // Biases shape: [outputSize]
        var biasesNode = TensorOperations<T>.Variable(_biases, "biases");

        // Add input nodes in order: input, weights, biases
        inputNodes.Add(inputNode);
        inputNodes.Add(weightsNode);
        inputNodes.Add(biasesNode);

        // Build computation graph: output = (input x weights) + biases
        // Industry standard: no transpose needed with [inputSize, outputSize] weights

        // Step 1: Matrix multiply: input x weights
        var matmulResult = TensorOperations<T>.MatrixMultiply(inputNode, weightsNode);

        // Step 2: Add biases (uses IEngine.TensorAdd for GPU acceleration!)
        var outputNode = TensorOperations<T>.Add(matmulResult, biasesNode);

        // Step 3: Apply activation function
        var activatedOutput = ApplyActivationToGraph(outputNode);

        return activatedOutput;
    }

    /// <summary>
    /// Gets whether this layer currently supports JIT compilation.
    /// </summary>
    /// <value>
    /// True if the layer's activation function is supported for JIT compilation.
    /// Supported activations: ReLU, Sigmoid, Tanh, Softmax, Identity.
    /// </value>
    public override bool SupportsJitCompilation => CanActivationBeJitted();

    /// <summary>
    /// Releases resources used by this layer, including GPU tensor handles.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
    /// <remarks>
    /// <para>
    /// This method releases GPU memory allocated for persistent weight tensors.
    /// It is called by the base class Dispose() method.
    /// </para>
    /// <para><b>For Beginners:</b> GPU memory is limited and precious.
    ///
    /// When you're done with a layer:
    /// - Call Dispose() or use a 'using' statement
    /// - This frees up GPU memory for other operations
    /// - Failing to dispose can cause memory leaks on the GPU
    ///
    /// Example:
    /// <code>
    /// using var layer = new DenseLayer&lt;float&gt;(784, 128);
    /// // ... use layer ...
    /// // Automatically disposed when out of scope
    /// </code>
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Release GPU handles for persistent tensors
            Engine.InvalidatePersistentTensor(_weights);
            Engine.InvalidatePersistentTensor(_biases);

            // Clear other managed resources (CPU)
            _weightsGradient = null;
            _biasesGradient = null;
            _lastInput = null;
            _lastOutput = null;

            // Clear GPU-resident cached tensors
            _lastInputGpu = null;
            _lastPreActivationGpu = null;
            _lastOutputGpu = null;
            _gpuOriginalInputShape = null;
        }

        base.Dispose(disposing);
    }
}
