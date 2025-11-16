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
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DenseLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Specifies the type of regularization to apply to the layer's weights.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines how the network prevents overfitting in this layer.
    ///
    /// Regularization types:
    /// - None: No regularization (default)
    /// - L1: Encourages weights to become exactly zero (creates sparse networks)
    /// - L2: Encourages weights to be small but not necessarily zero (smooths the network)
    /// - L1L2: Combines both L1 and L2 regularization
    /// </para>
    /// </remarks>
    public enum RegularizationType
    {
        None,
        L1,
        L2,
        L1L2
    }

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
    /// The weight matrix that connects input neurons to output neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix represents the strength of connections between input and output neurons. Each row
    /// corresponds to an output neuron, and each column corresponds to an input neuron. The value at
    /// position [i, j] is the weight of the connection from input j to output i.
    /// </para>
    /// <para><b>For Beginners:</b> The weights matrix is like a table of importance scores.
    /// 
    /// Imagine a table where:
    /// - Each row represents one output neuron
    /// - Each column represents one input neuron
    /// - Each cell contains a number (weight) showing how strongly that input affects that output
    /// 
    /// During training, these numbers change to help the network make better predictions.
    /// Positive weights strengthen connections, negative weights create inhibitory connections,
    /// and weights close to zero mean the connection is weak or unimportant.
    /// </para>
    /// </remarks>
    private Matrix<T> _weights;

    /// <summary>
    /// The bias values added to each output neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains a bias value for each output neuron. Biases allow the network to shift
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
    private Vector<T> _biases;

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
    private Matrix<T>? _weightsGradient;

    /// <summary>
    /// Temporary storage for bias gradients during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), this vector stores the calculated gradients for the biases.
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
    private Vector<T>? _biasesGradient;

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
    public override int ParameterCount => (_weights.Rows * _weights.Columns) + _biases.Length;

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
    public DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        L1Strength = NumOps.FromDouble(0.01);
        L2Strength = NumOps.FromDouble(0.01);
        _lastRegularizationLoss = NumOps.Zero;

        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseLayer{T}"/> class with the specified 
    /// input and output sizes and a vector activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="vectorActivation">The vector activation function to apply. Defaults to ReLU if not specified.</param>
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
    /// </remarks>
    public DenseLayer(int inputSize, int outputSize, IVectorActivationFunction<T>? vectorActivation = null)
        : base([inputSize], [outputSize], vectorActivation ?? new ReLUActivation<T>())
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        L1Strength = NumOps.FromDouble(0.01);
        L2Strength = NumOps.FromDouble(0.01);
        _lastRegularizationLoss = NumOps.Zero;

        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
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
        // Initialize weights and biases (e.g., using Xavier/Glorot initialization)
        var random = new Random();
        var scale = Math.Sqrt(2.0 / (InputShape[0] + OutputShape[0]));

        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble() * scale - scale / 2);
            }

            _biases[i] = NumOps.Zero; // Initialize biases to zero
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

        T regularizationLoss = NumOps.Zero;

        // Compute L1 regularization: Σ|w|
        if (Regularization == RegularizationType.L1 || Regularization == RegularizationType.L1L2)
        {
            T l1Loss = NumOps.Zero;
            for (int i = 0; i < _weights.Rows; i++)
            {
                for (int j = 0; j < _weights.Columns; j++)
                {
                    T absWeight = NumOps.Abs(_weights[i, j]);
                    l1Loss = NumOps.Add(l1Loss, absWeight);
                }
            }
            l1Loss = NumOps.Multiply(L1Strength, l1Loss);
            regularizationLoss = NumOps.Add(regularizationLoss, l1Loss);
        }

        // Compute L2 regularization: Σ(w²)
        if (Regularization == RegularizationType.L2 || Regularization == RegularizationType.L1L2)
        {
            T l2Loss = NumOps.Zero;
            for (int i = 0; i < _weights.Rows; i++)
            {
                for (int j = 0; j < _weights.Columns; j++)
                {
                    T squaredWeight = NumOps.Multiply(_weights[i, j], _weights[i, j]);
                    l2Loss = NumOps.Add(l2Loss, squaredWeight);
                }
            }
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
    /// - Rows equal to the number of outputs
    /// - Columns equal to the number of inputs
    /// 
    /// If the dimensions don't match, the method will throw an error.
    /// </para>
    /// </remarks>
    public void SetWeights(Matrix<T> weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        // Validate dimensions
        if (weights.Rows != OutputShape[0] || weights.Columns != InputShape[0])
        {
            throw new ArgumentException($"Weight matrix dimensions must be {OutputShape[0]}x{InputShape[0]}, but got {weights.Rows}x{weights.Columns}");
        }

        // Set the weights directly
        _weights = weights;
    }

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
        _lastInput = input;
        int batchSize = input.Shape[0];

        var flattenedInput = input.Reshape(batchSize, input.Shape[1]);
        var output = flattenedInput.Multiply(_weights.Transpose()).Add(_biases);

        if (UsingVectorActivation)
        {
            return VectorActivation!.Activate(output);
        }
        else
        {
            return ApplyActivation(output);
        }
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
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        Tensor<T> activationGradient;
        if (UsingVectorActivation)
        {
            activationGradient = VectorActivation!.Derivative(outputGradient);
        }
        else
        {
            // Apply scalar activation derivative element-wise
            activationGradient = new Tensor<T>(outputGradient.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                activationGradient[i] = ScalarActivation!.Derivative(outputGradient[i]);
            }
        }

        var flattenedInput = _lastInput.Reshape(batchSize, _lastInput.Shape[1]);

        _weightsGradient = activationGradient.Transpose([1, 0]).ToMatrix().Multiply(flattenedInput.ToMatrix());
        _biasesGradient = activationGradient.Sum([0]).ToMatrix().ToColumnVector();

        var inputGradient = activationGradient.Multiply(_weights);

        return inputGradient.Reshape(_lastInput.Shape);
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

        int batchSize = _lastInput.Shape[0];
        var flattenedInput = _lastInput.Reshape(batchSize, _lastInput.Shape[1]);

        // Convert to computation nodes
        var weightsTensor = MatrixToTensor(_weights);
        var biasesTensor = VectorToTensor(_biases);

        var input = Autodiff.TensorOperations<T>.Variable(flattenedInput, "input", requiresGradient: true);
        var weights = Autodiff.TensorOperations<T>.Variable(weightsTensor, "weights", requiresGradient: true);
        var biases = Autodiff.TensorOperations<T>.Variable(biasesTensor, "biases", requiresGradient: true);

        // Forward computation using autodiff ops
        // output = input @ weights.T + biases
        var weightsTransposed = Autodiff.TensorOperations<T>.Transpose(weights);
        var matmul = Autodiff.TensorOperations<T>.MatrixMultiply(input, weightsTransposed);

        // Add biases directly - autodiff Add operation handles broadcasting and gradient reduction
        // matmul is [batchSize, outputSize], biases is [outputSize]
        // Add broadcasts biases and reduces gradients automatically
        var output = Autodiff.TensorOperations<T>.Add(matmul, biases);

        // Apply activation using autodiff
        var activated = ApplyActivationAutodiff(output);

        // Manually propagate gradients using the output gradient we received
        // Set the gradient at the output and call backward functions manually
        var flattenedOutputGradient = outputGradient.Reshape(batchSize, outputGradient.Shape[1]);
        activated.Gradient = flattenedOutputGradient;

        // Perform topological sort and backward pass
        var topoOrder = GetTopologicalOrder(activated);

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
        _weightsGradient = TensorToMatrix(weights.Gradient!);
        _biasesGradient = TensorToVector(biases.Gradient!);

        return input.Gradient!.Reshape(_lastInput.Shape);
    }

    /// <summary>
    /// Gets the topological order of nodes in the computation graph.
    /// </summary>
    private List<Autodiff.ComputationNode<T>> GetTopologicalOrder(Autodiff.ComputationNode<T> root)
    {
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var result = new List<Autodiff.ComputationNode<T>>();

        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));

                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        return result;
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

    /// <summary>
    /// Broadcasts biases across the batch dimension.
    /// </summary>
    private Tensor<T> BroadcastBiases(Tensor<T> biases, int batchSize)
    {
        var biasLength = biases.Length;
        var broadcasted = new Tensor<T>(new int[] { batchSize, biasLength });

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < biasLength; j++)
            {
                broadcasted[i, j] = biases[j];
            }
        }

        return broadcasted;
    }

    /// <summary>
    /// Converts a Matrix to a 2D Tensor.
    /// </summary>
    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = new Tensor<T>(new int[] { matrix.Rows, matrix.Columns });
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                tensor[i, j] = matrix[i, j];
            }
        }
        return tensor;
    }

    /// <summary>
    /// Converts a Vector to a 1D Tensor.
    /// </summary>
    private Tensor<T> VectorToTensor(Vector<T> vector)
    {
        var tensor = new Tensor<T>(new int[] { vector.Length });
        for (int i = 0; i < vector.Length; i++)
        {
            tensor[i] = vector[i];
        }
        return tensor;
    }

    /// <summary>
    /// Converts a 2D Tensor to a Matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        if (tensor.Rank != 2)
            throw new ArgumentException("Tensor must be 2D to convert to Matrix");

        var matrix = new Matrix<T>(tensor.Shape[0], tensor.Shape[1]);
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                matrix[i, j] = tensor[i, j];
            }
        }
        return matrix;
    }

    /// <summary>
    /// Converts a 1D Tensor to a Vector.
    /// </summary>
    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        // Handle both 1D and 2D tensors (for column vectors)
        int length = tensor.Length;
        var vector = new Vector<T>(length);

        for (int i = 0; i < length; i++)
        {
            vector[i] = tensor[i];
        }

        return vector;
    }

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

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
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
        // Calculate total number of parameters
        int totalParams = _weights.Rows * _weights.Columns + _biases.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy weight parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }
    
        // Copy bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }
    
        return parameters;
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
        if (parameters.Length != _weights.Rows * _weights.Columns + _biases.Length)
        {
            throw new ArgumentException($"Expected {_weights.Rows * _weights.Columns + _biases.Length} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weight parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }
    
        // Set bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
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
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _weightsGradient = null;
        _biasesGradient = null;
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
        
        if (UsingVectorActivation)
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
}