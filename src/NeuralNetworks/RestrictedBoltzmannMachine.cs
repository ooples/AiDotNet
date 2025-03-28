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
    private IActivationFunction<T>? _scalarActivation { get; }

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
    private IVectorActivationFunction<T>? _vectorActivation { get; }

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
    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture, int visibleSize, int hiddenSize, IActivationFunction<T>? scalarActivation = null) : 
        base(architecture)
    {
        VisibleSize = visibleSize;
        HiddenSize = hiddenSize;
        _weights = Matrix<T>.CreateRandom(hiddenSize, visibleSize);
        _visibleBiases = Vector<T>.CreateDefault(visibleSize, NumOps.Zero);
        _hiddenBiases = Vector<T>.CreateDefault(hiddenSize, NumOps.Zero);
        _scalarActivation = scalarActivation;
        _vectorActivation = null;
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
    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture, int visibleSize, int hiddenSize, IVectorActivationFunction<T>? vectorActivation = null) : 
        base(architecture)
    {
        VisibleSize = visibleSize;
        HiddenSize = hiddenSize;
        _weights = Matrix<T>.CreateRandom(hiddenSize, visibleSize);
        _visibleBiases = Vector<T>.CreateDefault(visibleSize, NumOps.Zero);
        _hiddenBiases = Vector<T>.CreateDefault(hiddenSize, NumOps.Zero);
        _scalarActivation = null;
        _vectorActivation = vectorActivation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RestrictedBoltzmannMachine{T}"/> class using layer sizes from the provided architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the RBM, containing layer size information.</param>
    /// <exception cref="ArgumentException">Thrown when the architecture does not contain exactly two layers or if layer sizes are invalid.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Restricted Boltzmann Machine by extracting the visible and hidden layer sizes
    /// from the provided architecture. The architecture must define exactly two layers (visible and hidden),
    /// and both layer sizes must be positive.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the RBM using the layer sizes defined in the architecture.
    /// 
    /// When creating a new RBM this way:
    /// - You don't directly specify the layer sizes
    /// - Instead, the architecture object contains this information
    /// - The RBM extracts the sizes and validates that they make sense for an RBM
    /// 
    /// This constructor expects:
    /// - Exactly two layers defined in the architecture (visible and hidden)
    /// - Both layer sizes to be greater than zero
    /// 
    /// It will throw an error if these conditions aren't met. This approach is useful when you're
    /// creating an RBM as part of a larger system that uses architecture objects to define networks.
    /// </para>
    /// </remarks>
    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        // Get the layer sizes
        int[] layerSizes = architecture.GetLayerSizes();

        // Check if we have exactly two layers (visible and hidden)
        if (layerSizes.Length != 2)
        {
            throw new ArgumentException("RBM requires exactly two layers (visible and hidden units).");
        }

        VisibleSize = layerSizes[0];
        HiddenSize = layerSizes[1];

        if (VisibleSize <= 0 || HiddenSize <= 0)
        {
            throw new ArgumentException("Both visible and hidden unit counts must be positive for RBM.");
        }

        _visibleBiases = new Vector<T>(VisibleSize);
        _hiddenBiases = new Vector<T>(HiddenSize);
        _weights = new Matrix<T>(HiddenSize, VisibleSize);

        InitializeParameters();
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
    /// Processes the input through the RBM to produce a reconstructed output.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The reconstructed output vector after processing through the network.</returns>
    /// <remarks>
    /// <para>
    /// In an RBM, prediction typically means reconstructing the visible layer from the hidden layer activations
    /// derived from the input. This method passes the input to the hidden layer, samples the hidden states,
    /// and then passes those hidden states back to the visible layer to produce a reconstruction of the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method shows what the RBM would reconstruct from your input data.
    /// 
    /// The prediction process has two steps:
    /// 1. Forward pass: Convert the input data to hidden layer activations (find the features)
    /// 2. Backward pass: Convert the hidden layer activations back to visible layer values (reconstruct the data)
    /// 
    /// For example, if you input an image of a face:
    /// - The RBM first identifies which features it detects in the face
    /// - Then it tries to reconstruct the face using just those features
    /// 
    /// In a well-trained RBM, the reconstructed output should closely match the original input.
    /// This is different from classification networks that output a category or label.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        // In an RBM, prediction is typically done by reconstructing the visible layer
        Vector<T> hiddenProbs = SampleHiddenGivenVisible(input);
        return SampleVisibleGivenHidden(hiddenProbs);
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
        var hiddenActivations = _weights.Multiply(visibleLayer.ToMatrix()).Add(_hiddenBiases.ToColumnMatrix());
            
        if (_vectorActivation != null)
        {
            return _vectorActivation.Activate(Tensor<T>.FromMatrix(hiddenActivations));
        }
        else if (_scalarActivation != null)
        {
            return Tensor<T>.FromMatrix(hiddenActivations.Transform((x, _, _) => _scalarActivation.Activate(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    /// <summary>
    /// Samples the hidden layer states given the visible layer states.
    /// </summary>
    /// <param name="visible">The visible layer vector.</param>
    /// <returns>A vector containing the probabilities of hidden layer activations.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the activation probabilities of the hidden layer given the visible layer states.
    /// For each hidden unit, it calculates the weighted sum of visible unit values plus the hidden unit's bias,
    /// and then applies a sigmoid activation function to obtain the probability of activation.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines which features the RBM detects in the input.
    /// 
    /// When sampling the hidden layer:
    /// - The RBM looks at each input value
    /// - It calculates how strongly each hidden neuron should respond
    /// - It applies the sigmoid function to get a probability between 0 and 1
    /// 
    /// This is like translating from the "language of data" (visible layer)
    /// to the "language of features" (hidden layer).
    /// 
    /// The result tells you, for each feature the RBM knows about, how likely that
    /// feature is present in the current input.
    /// </para>
    /// </remarks>
    private Vector<T> SampleHiddenGivenVisible(Vector<T> visible)
    {
        Vector<T> hiddenProbs = new Vector<T>(HiddenSize);
        for (int j = 0; j < HiddenSize; j++)
        {
            T activation = _hiddenBiases[j];
            for (int i = 0; i < VisibleSize; i++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(_weights[i, j], visible[i]));
            }

            hiddenProbs[j] = new SigmoidActivation<T>().Activate(activation);
        }

        return hiddenProbs;
    }

    /// <summary>
    /// Reconstructs the visible layer given the hidden layer.
    /// </summary>
    /// <param name="hiddenLayer">The hidden layer tensor.</param>
    /// <returns>A tensor containing the reconstructed visible layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the visible layer based on the hidden layer states. It computes the weighted sum
    /// of hidden unit activations for each visible unit, adds the visible bias, and applies the activation function
    /// to obtain the reconstruction. This process is the inverse of the hidden layer activation calculation.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs the original data from the detected features.
    /// 
    /// When reconstructing the visible layer:
    /// - Each visible neuron receives input from all hidden neurons
    /// - The inputs are weighted by the connection strengths
    /// - The visible neuron's bias is added
    /// - An activation function converts this sum to a probability
    /// 
    /// This is like translating back from the "language of features" (hidden layer)
    /// to the "language of data" (visible layer).
    /// 
    /// The result is what the RBM thinks the original data should look like,
    /// based on the features it detected. In a well-trained RBM, this reconstruction
    /// should closely match the original input.
    /// </para>
    /// </remarks>
    private Tensor<T> GetVisibleLayerReconstruction(Tensor<T> hiddenLayer)
    {
        var visibleActivations = _weights.Transpose().Multiply(hiddenLayer.ToMatrix()).Add(_visibleBiases.ToColumnMatrix());
            
        if (_vectorActivation != null)
        {
            return _vectorActivation.Activate(Tensor<T>.FromMatrix(visibleActivations));
        }
        else if (_scalarActivation != null)
        {
            return Tensor<T>.FromMatrix(visibleActivations.Transform((x, _, _) => _scalarActivation.Activate(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    /// <summary>
    /// Samples the visible layer states given the hidden layer states.
    /// </summary>
    /// <param name="hidden">The hidden layer vector.</param>
    /// <returns>A vector containing the probabilities of visible layer activations.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the activation probabilities of the visible layer given the hidden layer states.
    /// For each visible unit, it calculates the weighted sum of hidden unit values plus the visible unit's bias,
    /// and then applies a sigmoid activation function to obtain the probability of activation.
    /// </para>
    /// <para><b>For Beginners:</b> This method recreates the input data based on the detected features.
    /// 
    /// When sampling the visible layer:
    /// - The RBM looks at each feature (hidden neuron activation)
    /// - It calculates how each feature affects each visible neuron
    /// - It applies the sigmoid function to get a probability between 0 and 1
    /// 
    /// This is like asking: "If these features are present, what would the original data look like?"
    /// 
    /// The result is a reconstruction of the input data based solely on the features the RBM detected.
    /// The closer this reconstruction is to the original input, the better the RBM has learned.
    /// </para>
    /// </remarks>
    private Vector<T> SampleVisibleGivenHidden(Vector<T> hidden)
    {
        Vector<T> visibleProbs = new Vector<T>(VisibleSize);
        for (int i = 0; i < VisibleSize; i++)
        {
            T activation = _visibleBiases[i];
            for (int j = 0; j < HiddenSize; j++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(_weights[i, j], hidden[j]));
            }

            visibleProbs[i] = new SigmoidActivation<T>().Activate(activation);
        }

        return visibleProbs;
    }

    /// <summary>
    /// Trains the RBM on the provided data using Contrastive Divergence.
    /// </summary>
    /// <param name="data">The training data tensor.</param>
    /// <param name="epochs">The number of training epochs.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method implements the Contrastive Divergence algorithm to train the RBM. For each epoch,
    /// it performs a positive phase (processing real data), a negative phase (processing reconstructed data),
    /// and updates the weights and biases based on the difference between these phases. This approach
    /// approximates the gradient of the log-likelihood, making the RBM increasingly better at modeling
    /// the probability distribution of the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This method is the heart of RBM training - it's where the actual learning happens.
    /// 
    /// For each training epoch (complete pass through the data):
    /// 
    /// 1. Positive phase - working with real data:
    ///    - Pass the training data through the RBM
    ///    - Calculate which hidden features activate
    ///    - Record the connection strengths between visible and hidden units
    /// 
    /// 2. Negative phase - working with the RBM's "imagination":
    ///    - Let the RBM reconstruct the data from the hidden features
    ///    - See which hidden features would activate from this reconstruction
    ///    - Record the connection strengths in this imagined scenario
    /// 
    /// 3. Update the network:
    ///    - Strengthen connections that appear strongly in the real data but not in the reconstruction
    ///    - Weaken connections that appear strongly in the reconstruction but not in the real data
    ///    - Update the bias values to make each neuron's activation more accurate
    /// 
    /// The learning rate controls how quickly the network parameters change - too high and training
    /// might be unstable, too low and training might take too long.
    /// </para>
    /// </remarks>
    public void Train(Tensor<T> data, int epochs, T learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Positive phase
            Tensor<T> visibleLayer = data;
            Tensor<T> hiddenLayer = GetHiddenLayerActivation(visibleLayer);
            Matrix<T> posGradient = TensorOuterProduct(visibleLayer, hiddenLayer);

            // Negative phase
            Tensor<T> visibleReconstruction = GetVisibleLayerReconstruction(hiddenLayer);
            Tensor<T> hiddenReconstruction = GetHiddenLayerActivation(visibleReconstruction);
            Matrix<T> negGradient = TensorOuterProduct(visibleReconstruction, hiddenReconstruction);

            // Update weights and biases
            _weights = _weights.Add(posGradient.Subtract(negGradient).Multiply(learningRate));
    
            // Update visible biases
            Vector<T> visibleBiasGradient = TensorToVector(visibleLayer.Subtract(visibleReconstruction));
            T visibleBiasMean = NumOps.Divide(visibleBiasGradient.Sum(), NumOps.FromDouble(visibleBiasGradient.Length));
            _visibleBiases = _visibleBiases.Add(Vector<T>.CreateDefault(VisibleSize, NumOps.Multiply(visibleBiasMean, learningRate)));

            // Update hidden biases
            Vector<T> hiddenBiasGradient = TensorToVector(hiddenLayer.Subtract(hiddenReconstruction));
            T hiddenBiasMean = NumOps.Divide(hiddenBiasGradient.Sum(), NumOps.FromDouble(hiddenBiasGradient.Length));
            _hiddenBiases = _hiddenBiases.Add(Vector<T>.CreateDefault(HiddenSize, NumOps.Multiply(hiddenBiasMean, learningRate)));
        }
    }

    /// <summary>
    /// Computes the outer product of two tensors.
    /// </summary>
    /// <param name="t1">The first tensor.</param>
    /// <param name="t2">The second tensor.</param>
    /// <returns>A matrix representing the outer product.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the outer product of two tensors, which is a matrix where each element is the
    /// product of corresponding elements from the flattened tensors. The outer product is used in the Contrastive
    /// Divergence algorithm to compute weight updates based on the correlation between visible and hidden units.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the relationship between every possible pair of neurons.
    /// 
    /// The outer product:
    /// - Takes two sets of neuron activations (from the visible and hidden layers)
    /// - Creates a matrix showing how each neuron in the first set relates to each neuron in the second set
    /// - Each value in the matrix is the product of a visible neuron's value and a hidden neuron's value
    /// 
    /// This is used during training to understand which connections between neurons need to be
    /// strengthened or weakened, based on the patterns in the data.
    /// </para>
    /// </remarks>
    private Matrix<T> TensorOuterProduct(Tensor<T> t1, Tensor<T> t2)
    {
        Vector<T> v1 = TensorToVector(t1);
        Vector<T> v2 = TensorToVector(t2);
        return OuterProduct(v1, v2);
    }

    /// <summary>
    /// Converts a tensor to a vector by flattening.
    /// </summary>
    /// <param name="tensor">The tensor to convert.</param>
    /// <returns>A flattened vector representation of the tensor.</returns>
    /// <remarks>
    /// <para>
    /// This utility method converts a multi-dimensional tensor to a one-dimensional vector by flattening it.
    /// This is useful for operations that require vector inputs, such as computing outer products.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts multi-dimensional data into a simple list.
    /// 
    /// For example, if you have a 2x3 matrix like:
    /// [1, 2, 3]
    /// [4, 5, 6]
    /// 
    /// This method converts it to a single vector:
    /// [1, 2, 3, 4, 5, 6]
    /// 
    /// This makes it easier to perform certain mathematical operations that expect one-dimensional data.
    /// </para>
    /// </remarks>
    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        return tensor.ToMatrix().Flatten();
    }

    /// <summary>
    /// Computes the outer product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>A matrix representing the outer product.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the outer product of two vectors, which is a matrix where the element at position
    /// (i,j) is the product of the i-th element of the first vector and the j-th element of the second vector.
    /// The outer product is used in the weight update calculations during RBM training.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a multiplication table between two lists of numbers.
    /// 
    /// If you have two vectors:
    /// v1 = [a, b, c]
    /// v2 = [x, y]
    /// 
    /// The outer product is a matrix where each cell is the product of one element from v1 and one from v2:
    /// [a*x, a*y]
    /// [b*x, b*y]
    /// [c*x, c*y]
    /// 
    /// This is used in RBM training to understand how visible and hidden neurons relate to each other,
    /// which helps the RBM learn the patterns in your data.
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
    public override void UpdateParameters(Vector<T> parameters)
    {
        // This method is not typically used in RBMs
        throw new NotImplementedException("UpdateParameters is not implemented for Restricted Boltzmann Machines.");
    }

    /// <summary>
    /// Saves the state of the Restricted Boltzmann Machine to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the RBM, including the visible and hidden layer sizes,
    /// visible and hidden biases, and weight matrix. It converts all numeric values to doubles before
    /// writing them to ensure consistency across different numeric types.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the RBM to a file.
    /// 
    /// When serializing:
    /// - The sizes of both visible and hidden layers are saved
    /// - All bias values for both layers are saved
    /// - All connection weights between neurons are saved
    /// 
    /// This is useful for:
    /// - Saving a trained model to use later
    /// - Sharing a model with others
    /// - Creating backups during long training processes
    /// 
    /// Think of it like taking a complete snapshot of the RBM that can be restored later.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(VisibleSize);
        writer.Write(HiddenSize);

        for (int i = 0; i < VisibleSize; i++)
        {
            writer.Write(Convert.ToDouble(_visibleBiases[i]));
        }

        for (int j = 0; j < HiddenSize; j++)
        {
            writer.Write(Convert.ToDouble(_hiddenBiases[j]));
        }

        for (int i = 0; i < VisibleSize; i++)
        {
            for (int j = 0; j < HiddenSize; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    /// <summary>
    /// Loads the state of the Restricted Boltzmann Machine from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the RBM from a binary reader. It reads the visible and hidden
    /// layer sizes, recreates the biases and weight matrix, and populates them with the values read from the
    /// reader. This allows a previously saved RBM state to be restored.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved RBM state from a file.
    /// 
    /// When deserializing:
    /// - The sizes of both layers are read first
    /// - New vectors and matrices are created with these sizes
    /// - All bias values and weights are read and restored
    /// 
    /// This allows you to:
    /// - Load a previously trained model
    /// - Continue using or training a model from where you left off
    /// - Use models created by others
    /// 
    /// Think of it like restoring a complete snapshot of an RBM that was saved earlier.
    /// Once loaded, the RBM will be in exactly the same state as when it was saved,
    /// ready to make predictions or continue training.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        VisibleSize = reader.ReadInt32();
        HiddenSize = reader.ReadInt32();

        _visibleBiases = new Vector<T>(VisibleSize);
        _hiddenBiases = new Vector<T>(HiddenSize);
        _weights = new Matrix<T>(VisibleSize, HiddenSize);

        for (int i = 0; i < VisibleSize; i++)
        {
            _visibleBiases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int j = 0; j < HiddenSize; j++)
        {
            _hiddenBiases[j] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int i = 0; i < VisibleSize; i++)
        {
            for (int j = 0; j < HiddenSize; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}