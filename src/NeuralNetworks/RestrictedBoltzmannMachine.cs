namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Restricted Boltzmann Machine, which is a type of neural network that learns probability distributions over its inputs.
/// </summary>
/// <remarks>
/// <para>
/// A Restricted Boltzmann Machine (RBM) is a two-layer neural network that learns to reconstruct its input data.
/// Unlike feedforward networks, RBMs are generative models that learn the probability distribution of the training data.
/// They consist of a visible layer (representing the input data) and a hidden layer (representing features), with
/// connections between layers but no connections within a layer (hence "restricted"). RBMs are trained using an
/// algorithm called Contrastive Divergence, which involves both forward and backward passes between layers.
/// </para>
/// <para><b>For Beginners:</b> A Restricted Boltzmann Machine is like a two-way translator between data and features.
/// 
/// Think of it like this:
/// - The visible layer is like words in English
/// - The hidden layer is like words in French
/// - The network learns how to translate back and forth between the languages
/// 
/// When you train an RBM:
/// - It learns to recognize patterns in your data (translate English to French)
/// - It also learns to recreate the original data from those patterns (translate French back to English)
/// 
/// For example, if you train an RBM on images of faces:
/// - The visible layer represents the pixel values of the images
/// - The hidden layer might learn to recognize features like "has a mustache" or "is smiling"
/// - Once trained, you could activate certain hidden units to generate new face images with specific features
/// 
/// RBMs can be used for dimensionality reduction, feature learning, pattern completion, and even generating
/// new data samples similar to the training data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RestrictedBoltzmannMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the bias values for the visible layer neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The visible biases adjust the activation probability of each visible neuron regardless of input.
    /// They act as offsets that can make certain visible units more or less likely to be active.
    /// </para>
    /// <para><b>For Beginners:</b> Visible biases are like the default preferences for each visible neuron.
    /// 
    /// Think of each bias as a starting point:
    /// - A higher bias means that neuron tends to be "on" more often
    /// - A lower bias means that neuron tends to be "off" more often
    /// 
    /// For example, if your data has pixels that are almost always white in a certain area,
    /// the visible bias for those pixels might become high, making them default to white
    /// unless there's strong evidence they should be another color.
    /// </para>
    /// </remarks>
    private Vector<T> _visibleBiases { get; set; }

    /// <summary>
    /// Gets or sets the bias values for the hidden layer neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hidden biases adjust the activation probability of each hidden neuron regardless of input.
    /// They act as offsets that can make certain hidden features more or less likely to be detected.
    /// </para>
    /// <para><b>For Beginners:</b> Hidden biases are like the threshold for each hidden neuron to activate.
    /// 
    /// Each hidden neuron looks for a specific pattern:
    /// - A higher bias means that pattern is assumed to be common, so the neuron activates more easily
    /// - A lower bias means the pattern is assumed to be rare, so the neuron needs stronger evidence to activate
    /// 
    /// For example, if a hidden neuron detects "smiles" in face images, its bias might be high if most
    /// faces in your training data are smiling, making it more likely to detect smiles.
    /// </para>
    /// </remarks>
    private Vector<T> _hiddenBiases { get; set; }

    /// <summary>
    /// Gets or sets the weight matrix representing connections between visible and hidden neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The weights matrix defines the strength of connections between each visible neuron and each hidden neuron.
    /// Positive weights increase the probability that connected neurons activate together, while negative weights
    /// decrease this probability. These weights capture the patterns and relationships in the training data.
    /// </para>
    /// <para><b>For Beginners:</b> _weights are like the importance of connections between neurons.
    /// 
    /// Each weight tells you:
    /// - How strongly a visible neuron influences a hidden neuron
    /// - Whether the influence is positive (they tend to be active together) or negative (when one is active, the other tends not to be)
    /// 
    /// For example:
    /// - If certain pixels always appear together in a pattern, the weights between them and a hidden neuron might become strongly positive
    /// - If certain pixels never appear together, the weights might become negative
    /// 
    /// The weights matrix is where most of the learning happens - it's like the "knowledge" the RBM has about patterns in your data.
    /// </para>
    /// </remarks>
    private Matrix<T> _weights { get; set; }

    /// <summary>
    /// Gets the number of neurons in the visible layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The visible size determines the dimensionality of the input data that the RBM can process.
    /// It should match the number of features in the input data (e.g., the number of pixels in an image).
    /// </para>
    /// <para><b>For Beginners:</b> This is how many input values the RBM can accept.
    ///
    /// For example:
    /// - If processing 28×28 pixel images, VisibleSize would be 784 (28×28)
    /// - If processing customer data with 15 attributes, VisibleSize would be 15
    ///
    /// Think of it as the number of "sensors" the network has to observe the input data.
    /// </para>
    /// </remarks>
    public int VisibleSize { get; private set; }

    /// <summary>
    /// Gets the number of neurons in the hidden layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hidden size determines the capacity of the RBM to learn patterns and features from the input data.
    /// A larger hidden size allows the RBM to learn more complex representations but may require more data and
    /// time to train effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many pattern detectors or features the RBM can learn.
    /// 
    /// Choosing the right hidden size is important:
    /// - Too small: The RBM won't be able to capture all important patterns in your data
    /// - Too large: The RBM might "memorize" the training data instead of learning general patterns
    /// 
    /// For example, if analyzing face images:
    /// - HiddenSize = 10 might only let the RBM learn very basic features
    /// - HiddenSize = 100 might allow it to learn more subtle patterns like facial expressions
    /// 
    /// Think of it as the number of "concepts" the network can understand about your data.
    /// </para>
    /// </remarks>
    public int HiddenSize { get; private set; }

    /// <summary>
    /// Gets the total number of parameters (weights and biases) in the RBM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The parameter count includes:
    /// - Weights matrix: HiddenSize × VisibleSize parameters
    /// - Visible biases: VisibleSize parameters
    /// - Hidden biases: HiddenSize parameters
    /// </para>
    /// <para><b>For Beginners:</b> This tells you the total number of learnable values in the RBM.
    /// More parameters means the RBM can learn more complex patterns, but also requires more data and computation.
    /// </para>
    /// </remarks>
    public override int ParameterCount => (HiddenSize * VisibleSize) + VisibleSize + HiddenSize;

    /// <summary>
    /// Gets or sets the scalar activation function used in the RBM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The scalar activation function applies non-linearity to individual neuron activations, determining
    /// the probability of neuron activation given its inputs. In RBMs, this is typically a sigmoid function
    /// that maps values to the range [0,1] to represent probabilities.
    /// </para>
    /// <para><b>For Beginners:</b> This function converts the raw input signal to an activation probability.
    /// 
    /// The activation function:
    /// - Takes the weighted sum of inputs to a neuron
    /// - Transforms it into a probability (a value between 0 and 1)
    /// - Determines how likely the neuron is to activate
    /// 
    /// In RBMs, the sigmoid function is commonly used, which creates an S-shaped curve that
    /// smoothly transitions from 0 to 1. This models the probability of a neuron being "on" or "off".
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _scalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the RBM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The vector activation function applies non-linearity to entire vectors of neuron activations at once.
    /// This can be more efficient than applying scalar functions to each neuron individually, and may
    /// support more sophisticated activation patterns that consider relationships between neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This function converts multiple raw input signals to activation probabilities all at once.
    /// 
    /// Unlike the scalar activation that processes one neuron at a time:
    /// - The vector activation processes an entire layer of neurons together
    /// - It can be more efficient for computation
    /// - It might capture relationships between different neurons' activations
    /// 
    /// This is an alternative to the scalar activation - an RBM will use either one or the other, not both.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _vectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for Contrastive Divergence training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate controls how quickly the RBM updates its weights and biases during training.
    /// A higher learning rate leads to faster but potentially less stable learning, while a lower
    /// learning rate provides more stable but slower learning.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate controls how big each learning step is.
    /// 
    /// Think of it like adjusting a dial:
    /// - Higher values (like 0.1) make bigger adjustments but might overshoot
    /// - Lower values (like 0.001) make smaller, more careful adjustments
    /// 
    /// Finding the right learning rate is important - too high and the network might never
    /// converge to good weights; too low and it might take too long to learn anything useful.
    /// </para>
    /// </remarks>
    private T _learningRate;

    /// <summary>
    /// Gets or sets the number of steps to run the Gibbs sampling chain during Contrastive Divergence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter determines how many times the RBM alternates between sampling hidden units and
    /// reconstructing visible units during the Contrastive Divergence algorithm. Higher values provide
    /// more accurate gradient estimates but are computationally more expensive.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many back-and-forth cycles the RBM does during training.
    /// 
    /// During training, the RBM:
    /// - Starts with real data (visible layer)
    /// - Computes hidden layer activations
    /// - Reconstructs the visible layer from those hidden activations
    /// - Repeats this process several times
    /// 
    /// The cdSteps parameter controls how many times this cycle repeats for each training example.
    /// Most often, just 1 step (called CD-1) works well, but more steps can sometimes give better results.
    /// </para>
    /// </remarks>
    private int _cdSteps;

    /// <summary>
    /// Initializes a new instance of the <see cref="RestrictedBoltzmannMachine{T}"/> class with the specified architecture, sizes, and scalar activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the RBM.</param>
    /// <param name="visibleSize">The number of neurons in the visible layer.</param>
    /// <param name="hiddenSize">The number of neurons in the hidden layer.</param>
    /// <param name="scalarActivation">The scalar activation function to use. If null, a default activation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Restricted Boltzmann Machine with the specified visible and hidden layer sizes,
    /// using the provided scalar activation function. It initializes weights to small random values and biases to zero,
    /// which is a common starting point for training RBMs.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the RBM with specific dimensions and an activation function that works on one neuron at a time.
    /// 
    /// When creating a new RBM this way:
    /// - You specify how many visible neurons (input values) you have
    /// - You specify how many hidden neurons (feature detectors) you want
    /// - You can optionally provide a specific activation function
    /// 
    /// The constructor sets up:
    /// - A weights matrix connecting all visible neurons to all hidden neurons
    /// - Bias values for all neurons (initially set to zero)
    /// - The specified scalar activation function
    /// 
    /// This prepares the RBM for training, but it won't actually learn anything until you train it with data.
    /// </para>
    /// </remarks>
    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture, int visibleSize, int hiddenSize, double learningRate = 0.01, int cdSteps = 1,
        IActivationFunction<T>? scalarActivation = null, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        VisibleSize = visibleSize;
        HiddenSize = hiddenSize;
        _weights = Matrix<T>.CreateRandom(hiddenSize, visibleSize);
        _visibleBiases = Vector<T>.CreateDefault(visibleSize, NumOps.Zero);
        _hiddenBiases = Vector<T>.CreateDefault(hiddenSize, NumOps.Zero);
        _scalarActivation = scalarActivation ?? new SigmoidActivation<T>();
        _learningRate = NumOps.FromDouble(learningRate);
        _cdSteps = cdSteps;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RestrictedBoltzmannMachine{T}"/> class with the specified architecture, sizes, and vector activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the RBM.</param>
    /// <param name="visibleSize">The number of neurons in the visible layer.</param>
    /// <param name="hiddenSize">The number of neurons in the hidden layer.</param>
    /// <param name="vectorActivation">The vector activation function to use. If null, a default activation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Restricted Boltzmann Machine with the specified visible and hidden layer sizes,
    /// using the provided vector activation function. It initializes weights to small random values and biases to zero,
    /// which is a common starting point for training RBMs. The vector activation function operates on entire layers
    /// at once, which may be more efficient for certain implementations.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the RBM with specific dimensions and an activation function that works on many neurons at once.
    /// 
    /// When creating a new RBM this way:
    /// - You specify how many visible neurons (input values) you have
    /// - You specify how many hidden neurons (feature detectors) you want
    /// - You can optionally provide a specific vector activation function
    /// 
    /// The constructor sets up:
    /// - A weights matrix connecting all visible neurons to all hidden neurons
    /// - Bias values for all neurons (initially set to zero)
    /// - The specified vector activation function
    /// 
    /// The main difference from the previous constructor is that this one uses an activation function
    /// that can process all neurons in a layer simultaneously, which can be more efficient.
    /// </para>
    /// </remarks>
    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture, int visibleSize, int hiddenSize, double learningRate = 0.01, int cdSteps = 1,
        IVectorActivationFunction<T>? vectorActivation = null, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        VisibleSize = visibleSize;
        HiddenSize = hiddenSize;
        _weights = Matrix<T>.CreateRandom(hiddenSize, visibleSize);
        _visibleBiases = Vector<T>.CreateDefault(visibleSize, NumOps.Zero);
        _hiddenBiases = Vector<T>.CreateDefault(hiddenSize, NumOps.Zero);
        _vectorActivation = vectorActivation ?? new SigmoidActivation<T>();
        _learningRate = NumOps.FromDouble(learningRate);
        _cdSteps = cdSteps;
    }

    /// <summary>
    /// Initializes the neural network layers. In an RBM, this method is typically empty as RBMs use direct weight and bias parameters rather than standard neural network layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RBMs differ from feedforward neural networks in that they don't use a layer-based computation model.
    /// Instead, they directly manipulate weights and biases for the visible and hidden units. Therefore,
    /// this method is typically empty or performs specialized initialization for RBMs.
    /// </para>
    /// <para><b>For Beginners:</b> RBMs work differently from standard neural networks.
    /// 
    /// While standard neural networks process data through sequential layers:
    /// - RBMs work by going back and forth between just two layers
    /// - They don't use the same layer concept as feedforward networks
    /// - They operate directly on the weights and biases connecting the visible and hidden layers
    /// 
    /// That's why this method is empty - the RBM initializes its weights and biases directly
    /// rather than creating a sequence of layers like a standard neural network.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // RBM doesn't use layers in the same way as feedforward networks
        // Instead, we'll initialize the weights and biases directly
    }

    /// <summary>
    /// Initializes the weights and biases of the RBM with appropriate starting values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the RBM parameters with suitable values for training. Biases are set to zero,
    /// and weights are initialized to small random values, which is a common practice for training RBMs.
    /// Small initial weight values help prevent saturation of the activation functions and promote better learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets the starting values for the RBM's weights and biases.
    /// 
    /// When initializing parameters:
    /// - Biases start at zero (neutral)
    /// - _weights start with small random values (typically between -0.05 and 0.05)
    /// 
    /// These starting values are important because:
    /// - Zero biases don't favor any particular activation state initially
    /// - Small random weights break symmetry (if all weights were the same, all hidden units would learn the same thing)
    /// - Small weights prevent the network from getting stuck at extreme activation values
    /// 
    /// This creates a good starting point from which the RBM can learn meaningful patterns during training.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Initialize biases to zero and weights to small random values
        for (int i = 0; i < VisibleSize; i++)
        {
            _visibleBiases[i] = NumOps.Zero;
            for (int j = 0; j < HiddenSize; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1 - 0.05);
            }
        }

        for (int j = 0; j < HiddenSize; j++)
        {
            _hiddenBiases[j] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Calculates the activation probabilities of the hidden layer given the visible layer.
    /// </summary>
    /// <param name="visibleLayer">The visible layer tensor.</param>
    /// <returns>A tensor containing the activation probabilities of the hidden layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the activation probabilities of each hidden unit given the state of the visible layer.
    /// It calculates the weighted sum of visible unit values for each hidden unit, adds the hidden bias, and
    /// applies the activation function to obtain the probability of activation.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds which patterns or features are present in the input data.
    /// 
    /// When calculating hidden layer activations:
    /// - Each hidden neuron receives input from all visible neurons
    /// - The inputs are weighted by the connection strengths
    /// - The hidden neuron's bias is added
    /// - An activation function converts this sum to a probability
    /// 
    /// This is like asking each feature detector: "Based on what you see in the input data,
    /// how confident are you that your specific pattern is present?"
    /// 
    /// The result is a set of probabilities for each hidden neuron, indicating how strongly
    /// each feature is detected in the current input.
    /// </para>
    /// </remarks>
    public Tensor<T> GetHiddenLayerActivation(Tensor<T> visibleLayer)
    {
        // Convert input tensor to a column matrix for matrix multiplication
        // Input shape: [visibleSize] or [batchSize, visibleSize] -> need [visibleSize, 1] or [visibleSize, batchSize]
        Matrix<T> visibleMatrix;
        if (visibleLayer.Rank == 1)
        {
            // 1D tensor: reshape to column matrix [visibleSize, 1]
            visibleMatrix = new Matrix<T>(visibleLayer.Shape[0], 1);
            for (int i = 0; i < visibleLayer.Shape[0]; i++)
            {
                visibleMatrix[i, 0] = visibleLayer[i];
            }
        }
        else if (visibleLayer.Rank == 2)
        {
            // 2D tensor: transpose if needed to get [visibleSize, batchSize]
            if (visibleLayer.Shape[0] == _weights.Columns)
            {
                // Already in correct orientation [visibleSize, batchSize]
                visibleMatrix = visibleLayer.ToMatrix();
            }
            else if (visibleLayer.Shape[1] == _weights.Columns)
            {
                // Need to transpose from [batchSize, visibleSize] to [visibleSize, batchSize]
                var temp = visibleLayer.ToMatrix();
                visibleMatrix = (Matrix<T>)temp.Transpose();
            }
            else
            {
                throw new ArgumentException($"Visible layer shape {string.Join(",", visibleLayer.Shape)} is incompatible with weights shape [{_weights.Rows},{_weights.Columns}]");
            }
        }
        else
        {
            throw new ArgumentException($"Visible layer must be 1D or 2D, got {visibleLayer.Rank}D");
        }

        var hiddenActivations = _weights.Multiply(visibleMatrix).Add(_hiddenBiases.ToColumnMatrix());

        if (_vectorActivation != null)
        {
            return _vectorActivation.Activate(Tensor<T>.FromRowMatrix(hiddenActivations));
        }
        else if (_scalarActivation != null)
        {
            return Tensor<T>.FromRowMatrix(hiddenActivations.Transform((x, _, _) => _scalarActivation.Activate(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    /// <summary>
    /// Updates the parameters of the RBM. This method is not typically used in RBMs and throws a NotImplementedException.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <exception cref="NotImplementedException">Always thrown as this method is not implemented for RBMs.</exception>
    /// <remarks>
    /// <para>
    /// RBMs typically use specialized training algorithms like Contrastive Divergence rather than the generic
    /// parameter update approach used by other neural networks. This method throws a NotImplementedException
    /// to indicate that RBMs should be trained using the Train method instead.
    /// </para>
    /// <para><b>For Beginners:</b> This method is not used in RBMs because they train differently.
    /// 
    /// While standard neural networks update their parameters based on error gradients:
    /// - RBMs use a different approach called Contrastive Divergence
    /// - They compare "reality" (input data) with "imagination" (reconstructions)
    /// - They directly adjust weights based on this comparison
    /// 
    /// Instead of using this method, you should use the Train method to train an RBM.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        int weightCount = HiddenSize * VisibleSize;
        int totalLength = weightCount + VisibleSize + HiddenSize;
        var parameters = new Vector<T>(totalLength);

        int paramIndex = 0;

        // Extract weights (HiddenSize × VisibleSize)
        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < VisibleSize; j++)
            {
                parameters[paramIndex++] = _weights[i, j];
            }
        }

        // Extract visible biases
        for (int i = 0; i < VisibleSize; i++)
        {
            parameters[paramIndex++] = _visibleBiases[i];
        }

        // Extract hidden biases
        for (int i = 0; i < HiddenSize; i++)
        {
            parameters[paramIndex++] = _hiddenBiases[i];
        }

        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        UpdateParameters(parameters);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int weightCount = HiddenSize * VisibleSize;
        int expectedLength = weightCount + VisibleSize + HiddenSize;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Parameter vector length mismatch. Expected {expectedLength} parameters but got {parameters.Length}.", nameof(parameters));
        }

        int paramIndex = 0;

        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < VisibleSize; j++)
            {
                _weights[i, j] = parameters[paramIndex++];
            }
        }

        for (int i = 0; i < VisibleSize; i++)
        {
            _visibleBiases[i] = parameters[paramIndex++];
        }

        for (int i = 0; i < HiddenSize; i++)
        {
            _hiddenBiases[i] = parameters[paramIndex++];
        }
    }

    /// <summary>
    /// Makes predictions using the RBM by computing hidden layer activations.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The hidden layer activations as a tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the RBM, mapping the input data to its corresponding
    /// hidden representation. For RBMs, "prediction" typically means extracting features or transforming
    /// the input data to a different representation.
    /// </para>
    /// <para><b>For Beginners:</b> This method extracts patterns or features from the input data.
    /// 
    /// Unlike standard neural networks that might predict a class or value:
    /// - RBMs transform input data into a representation of detected patterns
    /// - The output tells you which features or patterns were found in the input
    /// - This can be used for feature extraction or dimensionality reduction
    /// 
    /// For example, if your RBM has learned to recognize features in face images,
    /// this method would tell you which of those features (like "has glasses" or
    /// "is smiling") are present in a new face image you provide.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For RBMs, "prediction" is typically extracting the hidden layer representation

        // Ensure input has the right shape
        if (input.Shape.Length != 2 || input.Shape[1] != VisibleSize)
        {
            // Reshape input if needed
            var reshapedInput = new Tensor<T>(new[] { 1, VisibleSize });
            Vector<T> inputVector = input.ToVector();

            for (int i = 0; i < Math.Min(inputVector.Length, VisibleSize); i++)
            {
                reshapedInput[0, i] = inputVector[i];
            }

            input = reshapedInput;
        }

        // Get hidden layer activations
        return GetHiddenLayerActivation(input);
    }

    /// <summary>
    /// Samples binary states from activation probabilities.
    /// </summary>
    /// <param name="activations">Tensor of activation probabilities.</param>
    /// <returns>Tensor of binary states (0 or 1).</returns>
    /// <remarks>
    /// <para>
    /// This method converts activation probabilities to binary states using stochastic sampling.
    /// Each unit has a probability of being active (1) equal to its activation value, and inactive (0) otherwise.
    /// </para>
    /// <para><b>For Beginners:</b> This converts probabilities to binary on/off states.
    /// 
    /// For each neuron:
    /// - Its activation is a probability between 0 and 1
    /// - We randomly decide if it should be "on" (1) or "off" (0)
    /// - The higher the probability, the more likely it will be "on"
    /// 
    /// This introduces randomness into the RBM, which is important for its training
    /// process and allows it to explore different configurations.
    /// </para>
    /// </remarks>
    private Tensor<T> SampleBinaryStates(Tensor<T> activations)
    {
        var result = new Tensor<T>(activations.Shape);
        var random = RandomHelper.CreateSecureRandom();

        for (int i = 0; i < activations.Length; i++)
        {
            double probability = Convert.ToDouble(activations.GetFlatIndexValue(i));
            T state = NumOps.FromDouble(random.NextDouble() < probability ? 1.0 : 0.0);
            result.SetFlatIndex(i, state);
        }

        return result;
    }

    /// <summary>
    /// Calculates the activation probabilities of the visible layer given the hidden layer.
    /// </summary>
    /// <param name="hiddenLayer">The hidden layer tensor.</param>
    /// <returns>A tensor containing the activation probabilities of the visible layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the activation probabilities of each visible unit given the state of the hidden layer.
    /// It calculates the weighted sum of hidden unit values for each visible unit, adds the visible bias, and
    /// applies the activation function to obtain the probability of activation.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs the input data based on detected patterns.
    /// 
    /// When calculating visible layer activations:
    /// - Each visible neuron receives input from all hidden neurons
    /// - The inputs are weighted by the connection strengths
    /// - The visible neuron's bias is added
    /// - An activation function converts this sum to a probability
    /// 
    /// This is like asking each input neuron: "Based on the patterns the network detected,
    /// what's the probability that you should be active?"
    /// 
    /// The result is a reconstruction of the input data based on the patterns detected,
    /// which might not be identical to the original input.
    /// </para>
    /// </remarks>
    public Tensor<T> GetVisibleLayerActivation(Tensor<T> hiddenLayer)
    {
        // Convert input tensor to a column matrix for matrix multiplication
        // Input shape: [hiddenSize] or [batchSize, hiddenSize] -> need [hiddenSize, 1] or [hiddenSize, batchSize]
        Matrix<T> hiddenMatrix;
        int hiddenSize = _weights.Rows; // weights is [hiddenSize, visibleSize]

        if (hiddenLayer.Rank == 1)
        {
            // 1D tensor: reshape to column matrix [hiddenSize, 1]
            hiddenMatrix = new Matrix<T>(hiddenLayer.Shape[0], 1);
            for (int i = 0; i < hiddenLayer.Shape[0]; i++)
            {
                hiddenMatrix[i, 0] = hiddenLayer[i];
            }
        }
        else if (hiddenLayer.Rank == 2)
        {
            // 2D tensor: transpose if needed to get [hiddenSize, batchSize]
            if (hiddenLayer.Shape[0] == hiddenSize)
            {
                // Already in correct orientation [hiddenSize, batchSize]
                hiddenMatrix = hiddenLayer.ToMatrix();
            }
            else if (hiddenLayer.Shape[1] == hiddenSize)
            {
                // Need to transpose from [batchSize, hiddenSize] to [hiddenSize, batchSize]
                var temp = hiddenLayer.ToMatrix();
                hiddenMatrix = (Matrix<T>)temp.Transpose();
            }
            else
            {
                throw new ArgumentException($"Hidden layer shape {string.Join(",", hiddenLayer.Shape)} is incompatible with hidden size {hiddenSize}");
            }
        }
        else
        {
            throw new ArgumentException($"Hidden layer must be 1D or 2D, got {hiddenLayer.Rank}D");
        }

        // We need to transpose the weights matrix for the reverse direction
        var visibleActivations = _weights.Transpose().Multiply(hiddenMatrix).Add(_visibleBiases.ToColumnMatrix());

        if (_vectorActivation != null)
        {
            return _vectorActivation.Activate(Tensor<T>.FromRowMatrix(visibleActivations));
        }
        else if (_scalarActivation != null)
        {
            return Tensor<T>.FromRowMatrix(visibleActivations.Transform((x, _, _) => _scalarActivation.Activate(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    /// <summary>
    /// Trains the RBM using Contrastive Divergence.
    /// </summary>
    /// <param name="input">The input data tensor.</param>
    /// <param name="expectedOutput">Not used for RBMs as they are unsupervised models.</param>
    /// <remarks>
    /// <para>
    /// This method implements Contrastive Divergence (CD) training for the RBM. It compares the correlation
    /// between visible and hidden units when driven by the data to the correlation when driven by the model's
    /// own reconstructions, and updates the weights and biases accordingly.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the RBM to recognize patterns in your data.
    /// 
    /// The training process works like this:
    /// 1. Start with real data (the visible layer)
    /// 2. Compute which patterns (hidden layer) are activated by this data
    /// 3. Reconstruct an approximation of the data from these patterns
    /// 4. See what patterns this reconstruction would activate
    /// 5. Update the weights based on the difference between steps 2 and 4
    /// 
    /// The goal is for the RBM to generate reconstructions that are statistically
    /// similar to the real data, which means it has learned the underlying patterns.
    /// 
    /// Note that unlike supervised learning, RBMs don't use expected outputs - they
    /// learn the structure of the input data on their own.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // For RBMs, we ignore expectedOutput as they are unsupervised models

        // Reshape input to batch format if needed
        Tensor<T> batchInput;
        if (input.Shape.Length == 1)
        {
            // Single sample, reshape to [1, visibleSize]
            batchInput = new Tensor<T>([1, VisibleSize]);
            for (int i = 0; i < Math.Min(input.Shape[0], VisibleSize); i++)
            {
                batchInput[0, i] = input[i];
            }
        }
        else
        {
            // Assume batch format [batchSize, visibleSize]
            batchInput = input;
        }

        int batchSize = batchInput.Shape[0];

        // Initialize weight and bias updates
        var weightUpdates = new Matrix<T>(HiddenSize, VisibleSize);
        var visibleBiasUpdates = new Vector<T>(VisibleSize);
        var hiddenBiasUpdates = new Vector<T>(HiddenSize);

        // Process each sample in the batch
        for (int b = 0; b < batchSize; b++)
        {
            // Extract single sample
            var visibleSample = new Tensor<T>([1, VisibleSize]);
            for (int i = 0; i < VisibleSize; i++)
            {
                visibleSample[0, i] = batchInput[b, i];
            }

            // Positive phase
            var hiddenActivations = GetHiddenLayerActivation(visibleSample);
            var hiddenStates = SampleBinaryStates(hiddenActivations);

            // Store positive associations
            var positiveAssociations = ComputeAssociations(visibleSample, hiddenActivations);

            // Negative phase (Contrastive Divergence)
            var visibleReconstruction = visibleSample;
            var hiddenReactivations = hiddenActivations;

            // Run Gibbs sampling chain for _cdSteps steps
            for (int step = 0; step < _cdSteps; step++)
            {
                // Reconstruct visible layer
                visibleReconstruction = GetVisibleLayerActivation(hiddenStates);

                // Compute hidden activations from reconstruction
                hiddenReactivations = GetHiddenLayerActivation(visibleReconstruction);

                // Sample hidden states for next step (except last step)
                if (step < _cdSteps - 1)
                {
                    hiddenStates = SampleBinaryStates(hiddenReactivations);
                }
            }

            // Store negative associations
            var negativeAssociations = ComputeAssociations(visibleReconstruction, hiddenReactivations);

            // Compute updates for this sample
            var sampleWeightUpdates = positiveAssociations.Subtract(negativeAssociations);

            // Accumulate updates
            weightUpdates = weightUpdates.Add(sampleWeightUpdates);

            // Update visible bias (visible data - visible reconstruction)
            for (int i = 0; i < VisibleSize; i++)
            {
                visibleBiasUpdates[i] = NumOps.Add(
                    visibleBiasUpdates[i],
                    NumOps.Subtract(visibleSample[0, i], visibleReconstruction[0, i])
                );
            }

            // Update hidden bias (hidden activations - hidden reactivations)
            for (int j = 0; j < HiddenSize; j++)
            {
                hiddenBiasUpdates[j] = NumOps.Add(
                    hiddenBiasUpdates[j],
                    NumOps.Subtract(hiddenActivations[0, j], hiddenReactivations[0, j])
                );
            }
        }

        // Apply updates, normalized by batch size
        T batchNormalization = NumOps.FromDouble(1.0 / batchSize);
        T learningFactor = NumOps.Multiply(_learningRate, batchNormalization);

        // Update weights
        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < VisibleSize; j++)
            {
                T update = NumOps.Multiply(weightUpdates[i, j], learningFactor);
                _weights[i, j] = NumOps.Add(_weights[i, j], update);
            }
        }

        // Update biases
        for (int i = 0; i < VisibleSize; i++)
        {
            T update = NumOps.Multiply(visibleBiasUpdates[i], learningFactor);
            _visibleBiases[i] = NumOps.Add(_visibleBiases[i], update);
        }

        for (int i = 0; i < HiddenSize; i++)
        {
            T update = NumOps.Multiply(hiddenBiasUpdates[i], learningFactor);
            _hiddenBiases[i] = NumOps.Add(_hiddenBiases[i], update);
        }

        // Calculate and set the reconstruction error as the loss
        LastLoss = ComputeReconstructionError(batchInput);
    }

    /// <summary>
    /// Computes the outer product of visible and hidden activations to get association matrix.
    /// </summary>
    /// <param name="visible">The visible layer tensor.</param>
    /// <param name="hidden">The hidden layer tensor.</param>
    /// <returns>Matrix of associations between visible and hidden units.</returns>
    private Matrix<T> ComputeAssociations(Tensor<T> visible, Tensor<T> hidden)
    {
        var associations = new Matrix<T>(HiddenSize, VisibleSize);

        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < VisibleSize; j++)
            {
                associations[i, j] = NumOps.Multiply(hidden[0, i], visible[0, j]);
            }
        }

        return associations;
    }

    /// <summary>
    /// Generates samples from the RBM by starting with a random visible state and performing Gibbs sampling.
    /// </summary>
    /// <param name="numSamples">The number of samples to generate.</param>
    /// <param name="numSteps">The number of Gibbs sampling steps to perform.</param>
    /// <returns>Tensor containing the generated samples.</returns>
    /// <remarks>
    /// <para>
    /// This method generates new data samples that follow the distribution learned by the RBM.
    /// It starts with random visible units, then repeatedly samples the hidden and visible layers
    /// in a process called Gibbs sampling to get samples from the model's learned distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates new data samples based on patterns the RBM has learned.
    /// 
    /// The generation process works like this:
    /// 1. Start with random values for the visible layer
    /// 2. Compute hidden layer activations based on these visible values
    /// 3. Reconstruct a new visible layer from the hidden activations
    /// 4. Repeat steps 2-3 multiple times (Gibbs sampling)
    /// 5. Return the final visible layer as a generated sample
    /// 
    /// This allows the RBM to "dream up" new data that resembles the training data.
    /// For example, if trained on face images, it might generate new faces that
    /// don't exist but look realistic.
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateSamples(int numSamples, int numSteps = 1000)
    {
        var samples = new Tensor<T>(new[] { numSamples, VisibleSize });
        var random = RandomHelper.CreateSecureRandom();

        for (int s = 0; s < numSamples; s++)
        {
            // Start with random visible state
            var visibleState = new Tensor<T>(new[] { 1, VisibleSize });
            for (int i = 0; i < VisibleSize; i++)
            {
                visibleState[0, i] = NumOps.FromDouble(random.NextDouble() > 0.5 ? 1.0 : 0.0);
            }

            // Run Gibbs sampling
            for (int step = 0; step < numSteps; step++)
            {
                // Visible -> Hidden
                var hiddenActivations = GetHiddenLayerActivation(visibleState);
                var hiddenState = SampleBinaryStates(hiddenActivations);

                // Hidden -> Visible
                var visibleActivations = GetVisibleLayerActivation(hiddenState);
                visibleState = SampleBinaryStates(visibleActivations);
            }

            // Store the final sample
            for (int i = 0; i < VisibleSize; i++)
            {
                samples[s, i] = visibleState[0, i];
            }
        }

        return samples;
    }

    public T ComputeReconstructionError(Tensor<T> input)
    {
        // Reshape input if needed
        Tensor<T> reshapedInput;
        if (input.Shape.Length == 1)
        {
            reshapedInput = new Tensor<T>(new[] { 1, input.Shape[0] });
            for (int i = 0; i < input.Shape[0]; i++)
            {
                reshapedInput[0, i] = input[i];
            }
        }
        else
        {
            reshapedInput = input;
        }

        int batchSize = reshapedInput.Shape[0];
        T totalError = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            // Extract single sample
            var visibleSample = new Tensor<T>(new[] { 1, VisibleSize });
            for (int i = 0; i < VisibleSize; i++)
            {
                visibleSample[0, i] = reshapedInput[b, i];
            }

            // Forward pass to hidden layer
            var hiddenActivations = GetHiddenLayerActivation(visibleSample);

            // Reconstruct visible layer
            var visibleReconstruction = GetVisibleLayerActivation(hiddenActivations);

            // Compute mean squared error
            T sampleError = NumOps.Zero;
            for (int i = 0; i < VisibleSize; i++)
            {
                T diff = NumOps.Subtract(visibleSample[0, i], visibleReconstruction[0, i]);
                sampleError = NumOps.Add(sampleError, NumOps.Multiply(diff, diff));
            }

            // Average error over features
            sampleError = NumOps.Divide(sampleError, NumOps.FromDouble(VisibleSize));

            // Add to total error
            totalError = NumOps.Add(totalError, sampleError);
        }

        // Average error over batch
        return NumOps.Divide(totalError, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Gets metadata about the RBM model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the RBM.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the RBM, including its architecture,
    /// layer sizes, and other relevant parameters. This information is useful for model
    /// management, tracking experiments, and reporting.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your RBM.
    /// 
    /// The metadata includes:
    /// - The sizes of visible and hidden layers
    /// - Information about the activation functions used
    /// - The total number of parameters (weights and biases)
    /// - Other configuration details
    /// 
    /// This information is useful for documentation, comparing different RBM configurations,
    /// and understanding the structure of your model at a glance.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Count total parameters
        int totalParams = (VisibleSize * HiddenSize) + VisibleSize + HiddenSize;

        // Determine activation type
        string activationType = _vectorActivation != null
            ? _vectorActivation.GetType().Name
            : (_scalarActivation != null ? _scalarActivation.GetType().Name : "None");

        return new ModelMetadata<T>
        {
            ModelType = ModelType.RestrictedBoltzmannMachine,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "VisibleSize", VisibleSize },
                { "HiddenSize", HiddenSize },
                { "TotalParameters", totalParams },
                { "WeightCount", VisibleSize * HiddenSize },
                { "VisibleBiasCount", VisibleSize },
                { "HiddenBiasCount", HiddenSize },
                { "ActivationType", activationType },
                { "LearningRate", Convert.ToDouble(_learningRate) },
                { "CDSteps", _cdSteps }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes the RBM-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves RBM-specific data to the binary stream, including the weights, biases,
    /// and configuration parameters like learning rate and CD steps.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the RBM's learned knowledge to a file.
    /// 
    /// The serialization process saves:
    /// - All weights between visible and hidden neurons
    /// - All bias values for both layers
    /// - Configuration settings like learning rate
    /// 
    /// This allows you to save a trained RBM and reload it later without having to
    /// retrain it from scratch, which can be time-consuming.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write layer sizes
        writer.Write(VisibleSize);
        writer.Write(HiddenSize);

        // Write weights
        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < VisibleSize; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }

        // Write visible biases
        for (int i = 0; i < VisibleSize; i++)
        {
            writer.Write(Convert.ToDouble(_visibleBiases[i]));
        }

        // Write hidden biases
        for (int i = 0; i < HiddenSize; i++)
        {
            writer.Write(Convert.ToDouble(_hiddenBiases[i]));
        }

        // Write configuration parameters
        writer.Write(Convert.ToDouble(_learningRate));
        writer.Write(_cdSteps);

        // Write activation type
        bool hasVectorActivation = _vectorActivation != null;
        writer.Write(hasVectorActivation);

        if (hasVectorActivation)
        {
            writer.Write(_vectorActivation!.GetType().FullName ?? "Unknown");
        }
        else if (_scalarActivation != null)
        {
            writer.Write(_scalarActivation.GetType().FullName ?? "Unknown");
        }
        else
        {
            writer.Write("None");
        }
    }

    /// <summary>
    /// Deserializes RBM-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads RBM-specific data from the binary stream, including the weights, biases,
    /// and configuration parameters like learning rate and CD steps. It restores the RBM to the
    /// exact state it was in when serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads all the RBM's saved knowledge from a file.
    /// 
    /// The deserialization process loads:
    /// - All weights between visible and hidden neurons
    /// - All bias values for both layers
    /// - Configuration settings like learning rate
    /// 
    /// This allows you to restore a previously trained RBM exactly as it was,
    /// without needing to retrain it from scratch.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read layer sizes (and validate they match)
        int storedVisibleSize = reader.ReadInt32();
        int storedHiddenSize = reader.ReadInt32();

        if (storedVisibleSize != VisibleSize || storedHiddenSize != HiddenSize)
        {
            throw new InvalidOperationException(
                $"Size mismatch during deserialization. Expected {VisibleSize}x{HiddenSize}, " +
                $"but found {storedVisibleSize}x{storedHiddenSize}."
            );
        }

        // Read weights
        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < VisibleSize; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Read visible biases
        for (int i = 0; i < VisibleSize; i++)
        {
            _visibleBiases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read hidden biases
        for (int i = 0; i < HiddenSize; i++)
        {
            _hiddenBiases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read configuration parameters
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
        _cdSteps = reader.ReadInt32();

        // Read activation type
        bool hasVectorActivation = reader.ReadBoolean();
        string activationType = reader.ReadString();

        if (hasVectorActivation)
        {
            // Default to sigmoid if the exact type can't be recreated
            if (_vectorActivation == null)
            {
                _vectorActivation = new SigmoidActivation<T>();
            }
        }
        else if (activationType != "None" && _scalarActivation == null)
        {
            // Default to sigmoid if the exact type can't be recreated
            _scalarActivation = new SigmoidActivation<T>();
        }
    }

    /// <summary>
    /// Sets the training parameters for the RBM.
    /// </summary>
    /// <param name="learningRate">The learning rate for weight updates.</param>
    /// <param name="cdSteps">The number of Contrastive Divergence steps.</param>
    /// <remarks>
    /// <para>
    /// This method configures the learning rate and the number of Contrastive Divergence steps
    /// used during training. The learning rate controls how quickly the RBM updates its weights,
    /// while the CD steps control how many Gibbs sampling steps are performed in each update.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you adjust how the RBM learns.
    /// 
    /// You can configure:
    /// - Learning rate: How big each learning step is (typical values: 0.001 to 0.1)
    /// - CD steps: How many back-and-forth cycles to run during training (often 1, sometimes more)
    /// 
    /// These parameters affect learning quality and speed:
    /// - Higher learning rates learn faster but may be less stable
    /// - More CD steps give more accurate updates but take longer
    /// 
    /// Finding the right balance for your specific data is important for effective training.
    /// </para>
    /// </remarks>
    public void SetTrainingParameters(T learningRate, int cdSteps = 1)
    {
        _learningRate = learningRate;
        _cdSteps = Math.Max(1, cdSteps); // Ensure at least 1 CD step
    }

    /// <summary>
    /// Extracts features from input data using the trained RBM.
    /// </summary>
    /// <param name="input">The input data tensor.</param>
    /// <param name="binarize">Whether to binarize the hidden activations.</param>
    /// <returns>The hidden layer features as a tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms input data into features learned by the RBM's hidden layer.
    /// It can be used for feature extraction, dimensionality reduction, or as a pre-processing
    /// step before using the data with another algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts raw data into abstract features.
    /// 
    /// When extracting features:
    /// - The input data is passed to the visible layer
    /// - The hidden layer activations represent learned features
    /// - These features can capture important patterns in the data
    /// 
    /// You can choose to get:
    /// - Probability values (binarize=false) showing how strongly each feature is detected
    /// - Binary values (binarize=true) indicating whether each feature is present or not
    /// 
    /// This is useful for:
    /// - Reducing data dimensionality (e.g., compressing 784 pixels to 100 features)
    /// - Extracting meaningful patterns for other algorithms to use
    /// - Pre-processing data for classification or other tasks
    /// </para>
    /// </remarks>
    public Tensor<T> ExtractFeatures(Tensor<T> input, bool binarize = false)
    {
        // For RBMs, feature extraction is simply getting the hidden layer activations
        var hiddenActivations = Predict(input);

        // Optionally binarize the features
        if (binarize)
        {
            return SampleBinaryStates(hiddenActivations);
        }

        return hiddenActivations;
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Determine which constructor to use based on whether we're using scalar or vector activations
        if (_vectorActivation != null)
        {
            // Use the vector activation constructor
            return new RestrictedBoltzmannMachine<T>(
                Architecture, VisibleSize, HiddenSize, Convert.ToDouble(_learningRate), _cdSteps, _vectorActivation, LossFunction);
        }
        else
        {
            // Use the scalar activation constructor
            return new RestrictedBoltzmannMachine<T>(
                Architecture, VisibleSize, HiddenSize, Convert.ToDouble(_learningRate), _cdSteps, _scalarActivation, LossFunction);
        }
    }
}
