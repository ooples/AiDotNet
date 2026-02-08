global using AiDotNet.LossFunctions;
using AiDotNet.Extensions;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an autoencoder neural network that can compress data into a lower-dimensional representation and reconstruct it.
/// </summary>
/// <remarks>
/// <para>
/// An autoencoder is a type of neural network designed to learn efficient data encodings in an unsupervised manner.
/// It consists of an encoder that compresses the input data into a lower-dimensional representation (the latent space)
/// and a decoder that reconstructs the original data from this representation. Autoencoders are trained to minimize
/// the difference between the original input and the reconstructed output.
/// </para>
/// <para><b>For Beginners:</b> An autoencoder is like a sophisticated compression and decompression system.
/// 
/// Think of it like this:
/// - The encoder part takes your original data (like an image) and compresses it into a smaller representation
/// - The middle layer (latent space) holds this compressed version of your data
/// - The decoder part takes this compressed version and tries to recreate the original data
///
/// For example, with images:
/// - You might compress a 256x256 pixel image (65,536 values) into just 100 numbers
/// - The network learns which features are most important to preserve
/// - It then learns to reconstruct the image from only those 100 numbers
/// 
/// This is useful for:
/// - Data compression
/// - Noise reduction (by removing noise during reconstruction)
/// - Feature learning (the compressed representation often contains meaningful features)
/// - Anomaly detection (unusual data is reconstructed poorly)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Autoencoder<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly AutoencoderOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the size of the encoded representation (latent space).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates the dimensionality of the encoded representation, which is the size of the output from
    /// the middle (bottleneck) layer of the autoencoder. This is typically smaller than the input size, forcing the
    /// autoencoder to learn a compressed representation of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the compressed representation.
    /// 
    /// For example:
    /// - If your input data has 1000 dimensions (like a 1000-pixel image)
    /// - And EncodedSize is 50
    /// - Then your data is being compressed to 5% of its original size
    /// 
    /// The smaller this value:
    /// - The more compression is happening
    /// - The more the network has to be selective about what information to keep
    /// - The more efficient but potentially less accurate the reconstruction becomes
    /// </para>
    /// </remarks>
    public int EncodedSize { get; private set; }

    /// <summary>
    /// The learning rate used for training the autoencoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate determines the step size at each iteration while moving toward a minimum of the loss function.
    /// It influences how quickly the model adapts to the problem.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the learning rate as the size of the steps the model takes when learning.
    /// 
    /// - A larger learning rate means bigger steps, potentially learning faster but risking overshooting the optimal solution.
    /// - A smaller learning rate means smaller steps, learning more slowly but potentially finding a more precise solution.
    /// 
    /// Typical values range from 0.1 to 0.0001, depending on the specific problem and model architecture.
    /// </para>
    /// </remarks>
    private T _learningRate;

    /// <summary>
    /// The number of training epochs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An epoch is one complete pass through the entire training dataset. The number of epochs determines
    /// how many times the learning algorithm will work through the entire dataset.
    /// </para>
    /// <para><b>For Beginners:</b> Epochs are like complete study sessions of your data.
    /// 
    /// - If _epochs is 10, it means the autoencoder will study the entire dataset 10 times.
    /// - More epochs often lead to better learning, but too many can cause overfitting (memorizing instead of generalizing).
    /// - The right number of epochs depends on your data and problem. It's common to start with 10-100 and adjust based on results.
    /// </para>
    /// </remarks>
    private int _epochs;

    /// <summary>
    /// The size of each batch used in training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Batch size determines how many training examples are processed together before updating the model's parameters.
    /// It affects both the speed of training and the model's ability to generalize.
    /// </para>
    /// <para><b>For Beginners:</b> Think of batch size as how many examples the model looks at before making adjustments.
    /// 
    /// - A smaller batch size (e.g., 32) means more frequent updates, potentially leading to faster convergence but with more fluctuations.
    /// - A larger batch size (e.g., 256) means more stable updates but potentially slower learning.
    /// 
    /// Common batch sizes are powers of 2, like 32, 64, or 128, due to memory considerations in GPUs.
    /// </para>
    /// </remarks>
    private int _batchSize;

    /// <summary>
    /// The loss function used to measure the difference between the input and the reconstructed output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The loss function quantifies how well the autoencoder is performing by measuring the difference between
    /// the original input and the reconstructed output. It guides the learning process by providing a metric to minimize.
    /// </para>
    /// <para><b>For Beginners:</b> The loss function is like a score that tells the autoencoder how well it's doing.
    /// 
    /// - A lower score means the reconstruction is closer to the original input.
    /// - The autoencoder tries to minimize this score during training.
    /// - Common loss functions for autoencoders include Mean Squared Error (MSE) and Binary Cross-Entropy.
    /// 
    /// The choice of loss function depends on the nature of your data and the specific goals of your autoencoder.
    /// </para>
    /// </remarks>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Target sparsity parameter (desired average activation level).
    /// Default is 0.05 (5% of neurons should be active on average).
    /// </summary>
    private T _sparsityParameter;

    /// <summary>
    /// Stores the last encoder activations for auxiliary loss computation.
    /// </summary>
    private Tensor<T>? _lastEncoderActivations;

    /// <summary>
    /// Stores the last computed sparsity loss for diagnostics.
    /// </summary>
    private T _lastSparsityLoss;

    /// <summary>
    /// Stores the average activation level for diagnostics.
    /// </summary>
    private T _averageActivation;

    /// <summary>
    /// Gets or sets whether to use auxiliary loss (sparsity penalty) during training.
    /// Default is false. Enable for sparse autoencoders.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the sparsity penalty.
    /// Default is 0.001. Typical range: 0.0001 to 0.01.
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Autoencoder{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture specification for the autoencoder.</param>
    /// <param name="learningRate">The learning rate for training the autoencoder.</param>
    /// <param name="epochs">The number of training epochs to perform.</param>
    /// <param name="batchSize">The batch size to use during training.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, Mean Squared Error will be used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an autoencoder with the specified architecture and training parameters.
    /// It initializes the layers according to the architecture specification or uses default layers
    /// if none are provided. It also sets the EncodedSize property based on the size of the middle layer.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new autoencoder with the specified settings.
    /// 
    /// The architecture parameter determines the structure of your network, while:
    /// - learningRate controls how quickly the model learns (typically 0.001 to 0.1)
    /// - epochs specifies how many times to process the entire dataset (often 10-100)
    /// - batchSize determines how many examples to process at once (typically 32-256)
    /// 
    /// These parameters balance learning speed against stability and accuracy.
    /// </para>
    /// </remarks>
    public Autoencoder(NeuralNetworkArchitecture<T> architecture, T learningRate, int epochs = 1, int batchSize = 32, ILossFunction<T>? lossFunction = null, AutoencoderOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new AutoencoderOptions();
        Options = _options;

        EncodedSize = 0;
        _learningRate = learningRate;
        _epochs = epochs;
        _batchSize = batchSize;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // Initialize fields that require NumOps (must be done in constructor, not field initializers)
        _sparsityParameter = NumOps.FromDouble(0.05);
        _lastSparsityLoss = NumOps.Zero;
        _averageActivation = NumOps.Zero;
        AuxiliaryLossWeight = NumOps.FromDouble(0.001);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the autoencoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the layers of the autoencoder either by using the layers provided by the user in the
    /// architecture specification or by creating default autoencoder layers if none are provided. It also sets the
    /// EncodedSize property based on the size of the middle layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the autoencoder.
    /// 
    /// It does one of two things:
    /// 1. If you provided specific layers in the architecture, it uses those
    /// 2. If you didn't provide layers, it creates a default set of layers appropriate for an autoencoder
    /// 
    /// The default layers typically create a "bottleneck" shape:
    /// - Starting with larger layers (the input)
    /// - Getting progressively smaller (the encoder)
    /// - Reaching a small middle layer (the latent space)
    /// - Getting progressively larger again (the decoder)
    /// - Ending with a layer the same size as the input (the output)
    /// 
    /// It also sets the EncodedSize property to the size of the middle layer.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAutoEncoderLayers(Architecture));
        }

        // Set EncodedSize based on the middle layer
        EncodedSize = Layers[Layers.Count / 2].GetOutputShape()[0];
    }

    /// <summary>
    /// Validates that the custom layers conform to autoencoder requirements.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <exception cref="ArgumentException">Thrown when the layers do not conform to autoencoder requirements.</exception>
    /// <remarks>
    /// <para>
    /// This method validates that the provided layers conform to the requirements of an autoencoder:
    /// 1. There must be at least 3 layers (input, encoded, and output).
    /// 2. The input and output layers must have the same size.
    /// 3. The architecture must be symmetric.
    /// 4. The activation functions must be symmetric.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the layers you provided will work for an autoencoder.
    /// 
    /// It makes sure:
    /// - You have at least 3 layers (input ? encoded ? output)
    /// - The input and output layers are the same size (since an autoencoder reconstructs its input)
    /// - The network is symmetric (decoder mirrors encoder)
    /// - The activation functions are symmetric (same functions used in corresponding encoder/decoder layers)
    /// 
    /// If any of these requirements aren't met, it shows an error explaining what's wrong.
    /// This helps ensure your autoencoder is structured correctly before you start training it.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
        {
            throw new ArgumentException("Autoencoder must have at least 3 layers (input, encoded, and output).");
        }

        // Check if input and output layers have the same size
        if (!Enumerable.SequenceEqual(layers[0].GetInputShape(), layers[layers.Count - 1].GetOutputShape()))
        {
            throw new ArgumentException("Input and output layer sizes must be the same for an autoencoder.");
        }

        // Ensure the architecture is symmetric
        for (int i = 0; i < layers.Count / 2; i++)
        {
            if (!Enumerable.SequenceEqual(layers[i].GetOutputShape(), layers[layers.Count - 1 - i].GetInputShape()))
            {
                throw new ArgumentException($"Layer sizes must be symmetric. Mismatch at position {i} and {layers.Count - i - 1}");
            }
        }

        // Validate activation functions
        for (int i = 0; i < layers.Count / 2; i++)
        {
            var leftActivation = layers[i].GetActivationTypes();
            var rightActivation = layers[layers.Count - 1 - i].GetActivationTypes();

            if (!Enumerable.SequenceEqual(leftActivation, rightActivation))
            {
                throw new ArgumentException($"Activation functions must be symmetric. Mismatch at position {i} and {layers.Count - i - 1}");
            }
        }
    }

    /// <summary>
    /// Encodes the input data into the latent space representation.
    /// </summary>
    /// <param name="input">The input tensor to encode.</param>
    /// <returns>The encoded tensor (latent space representation).</returns>
    /// <remarks>
    /// <para>
    /// This method performs the encoding part of the autoencoder, passing the input data through the encoder layers
    /// to produce the latent space representation. It stops at the middle layer of the autoencoder.
    /// </para>
    /// <para><b>For Beginners:</b> This method compresses your data into the smaller representation.
    /// 
    /// When you call Encode:
    /// - Your data is processed by the first half of the network (the encoder part)
    /// - It passes through progressively smaller layers
    /// - It stops at the middle layer (the bottleneck)
    /// - You get back the compressed representation
    /// 
    /// This is useful when you:
    /// - Want the compressed representation for other tasks
    /// - Need to reduce the dimensionality of your data
    /// - Want to extract the key features learned by the autoencoder
    /// 
    /// For example, you might encode images into a small vector to cluster similar images together.
    /// </para>
    /// </remarks>
    public Tensor<T> Encode(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Decodes the latent space representation back to the original space.
    /// </summary>
    /// <param name="encodedInput">The encoded tensor (latent space representation) to decode.</param>
    /// <returns>The reconstructed tensor in the original space.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the decoding part of the autoencoder, passing the latent space representation through
    /// the decoder layers to reconstruct the original data. It starts from the middle layer of the autoencoder.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs your original data from the compressed representation.
    /// 
    /// When you call Decode:
    /// - Your compressed data is processed by the second half of the network (the decoder part)
    /// - It passes through progressively larger layers
    /// - It expands the compressed representation back to the original size
    /// - You get back a reconstruction of the original data
    /// 
    /// This is useful when you:
    /// - Want to see what information was preserved during compression
    /// - Need to generate new data from the latent space
    /// - Want to remove noise or fill in missing data
    /// 
    /// For example, you might create a new face image by decoding a point in the latent space.
    /// </para>
    /// </remarks>
    public Tensor<T> Decode(Tensor<T> encodedInput)
    {
        var current = encodedInput;
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Calculates the loss between the predicted output and the expected output.
    /// </summary>
    /// <param name="predicted">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The calculated loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the loss between the predicted output and the expected output using the
    /// loss function specified during initialization. For autoencoders, this typically measures how well
    /// the network reconstructs the input data.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how different the reconstructed output is from the expected output.
    /// 
    /// It uses the loss function that was specified when creating the autoencoder to calculate:
    /// - How far off the reconstruction is from the original
    /// - A single number representing the total error
    /// 
    /// Lower values mean better reconstruction (less difference between original and reconstructed data).
    /// This value guides the training process to improve the network's performance.
    /// </para>
    /// </remarks>
    protected T CalculateLoss(Tensor<T> predicted, Tensor<T> expected)
    {
        // Flatten the tensors to vectors for the loss function
        Vector<T> predictedVector = predicted.ToVector();
        Vector<T> expectedVector = expected.ToVector();

        // Calculate loss using the specified loss function
        return _lossFunction.CalculateLoss(predictedVector, expectedVector);
    }

    /// <summary>
    /// Calculates the gradient of the loss function with respect to the network output.
    /// </summary>
    /// <param name="predicted">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the gradient of the loss function with respect to the network output.
    /// This gradient is then used as the starting point for backpropagation to update the network weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to adjust the network to reduce the reconstruction error.
    /// 
    /// It works by:
    /// - Comparing the predicted output with the expected output
    /// - Calculating how the output needs to change to reduce the error
    /// - Creating a "gradient" that shows the direction and amount of recommended change
    /// 
    /// This gradient is then used to update the network's parameters during training,
    /// helping the autoencoder gradually improve its reconstruction ability.
    /// </para>
    /// </remarks>
    protected Tensor<T> CalculateOutputGradient(Tensor<T> predicted, Tensor<T> expected)
    {
        // Flatten the tensors to vectors for the loss function
        Vector<T> predictedVector = predicted.ToVector();
        Vector<T> expectedVector = expected.ToVector();

        // Calculate the derivative of the loss function
        Vector<T> gradientVector = _lossFunction.CalculateDerivative(predictedVector, expectedVector);

        // Reshape the gradient back to the original tensor shape
        return new Tensor<T>(predicted.Shape, gradientVector);
    }

    /// <summary>
    /// Sets the target sparsity parameter for sparse autoencoder training.
    /// </summary>
    /// <param name="sparsity">Target average activation level (typically 0.01 to 0.1).</param>
    /// <remarks>
    /// <para>
    /// The sparsity parameter controls how "active" the encoder's hidden units should be on average.
    /// Lower values enforce stronger sparsity (fewer active neurons), while higher values allow more neurons to be active.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how selective the autoencoder is about which features to use.
    ///
    /// Sparsity means:
    /// - Lower values (e.g., 0.01): Only 1% of neurons active on average - very selective, learns distinct features
    /// - Medium values (e.g., 0.05): 5% active - good balance (default)
    /// - Higher values (e.g., 0.1): 10% active - less sparse, more distributed representations
    ///
    /// Sparse representations often learn more interpretable and meaningful features.
    /// </para>
    /// </remarks>
    public void SetSparsityParameter(T sparsity)
    {
        _sparsityParameter = sparsity;
    }

    /// <summary>
    /// Computes the auxiliary loss for sparse autoencoders, which penalizes non-sparse activations.
    /// </summary>
    /// <returns>The sparsity loss value.</returns>
    /// <remarks>
    /// <para>
    /// The sparsity loss uses KL divergence between the target sparsity and actual average activation.
    /// Formula: KL(ρ || ρ̂) = ρ * log(ρ/ρ̂) + (1-ρ) * log((1-ρ)/(1-ρ̂))
    /// where ρ is target sparsity and ρ̂ is actual average activation.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates a penalty for having too many neurons active at once.
    ///
    /// The sparsity loss:
    /// - Measures how far the actual neuron activations are from the target sparsity
    /// - Encourages only a few neurons to be active for each input
    /// - Forces the autoencoder to learn more selective, meaningful features
    /// - Results in better feature learning and interpretability
    ///
    /// Without sparsity:
    /// - All neurons might activate for every input
    /// - Features become less distinct and harder to interpret
    /// - The autoencoder might just learn to memorize rather than extract key features
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastEncoderActivations == null)
        {
            return NumOps.Zero;
        }

        // Compute average activation across the batch
        T sumActivation = NumOps.Zero;
        int totalElements = _lastEncoderActivations.Length;

        for (int i = 0; i < totalElements; i++)
        {
            sumActivation = NumOps.Add(sumActivation, _lastEncoderActivations[i]);
        }

        _averageActivation = NumOps.Divide(sumActivation, NumOps.FromDouble(totalElements));

        // Compute KL divergence: KL(ρ || ρ̂) = ρ * log(ρ/ρ̂) + (1-ρ) * log((1-ρ)/(1-ρ̂))
        T epsilon = NumOps.FromDouble(1e-10); // Small value to prevent log(0)

        // Clamp average activation to prevent numerical issues
        T rhoHat = _averageActivation;
        if (NumOps.LessThan(rhoHat, epsilon))
        {
            rhoHat = epsilon;
        }
        T oneMinusRhoHat = NumOps.Subtract(NumOps.One, rhoHat);
        if (NumOps.LessThan(oneMinusRhoHat, epsilon))
        {
            oneMinusRhoHat = epsilon;
        }

        // KL divergence calculation
        T term1 = NumOps.Multiply(
            _sparsityParameter,
            NumOps.Log(NumOps.Divide(_sparsityParameter, rhoHat))
        );

        T oneMinusRho = NumOps.Subtract(NumOps.One, _sparsityParameter);
        T term2 = NumOps.Multiply(
            oneMinusRho,
            NumOps.Log(NumOps.Divide(oneMinusRho, oneMinusRhoHat))
        );

        _lastSparsityLoss = NumOps.Add(term1, term2);
        return _lastSparsityLoss;
    }

    /// <summary>
    /// Computes the gradient of the sparsity loss with respect to encoder activations.
    /// </summary>
    /// <returns>A tensor containing the sparsity loss gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the KL divergence sparsity loss:
    /// d(KL)/da = (1/n) * [-ρ/ρ̂ + (1-ρ)/(1-ρ̂)] * AuxiliaryLossWeight
    /// where ρ is target sparsity, ρ̂ is average activation, n is number of activations.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeSparsityGradient()
    {
        if (_lastEncoderActivations == null)
        {
            throw new InvalidOperationException("No encoder activations available for gradient computation.");
        }

        // Compute the gradient coefficient: -ρ/ρ̂ + (1-ρ)/(1-ρ̂)
        T epsilon = NumOps.FromDouble(1e-10);
        T rhoHat = _averageActivation;
        if (NumOps.LessThan(rhoHat, epsilon))
        {
            rhoHat = epsilon;
        }
        T oneMinusRhoHat = NumOps.Subtract(NumOps.One, rhoHat);
        if (NumOps.LessThan(oneMinusRhoHat, epsilon))
        {
            oneMinusRhoHat = epsilon;
        }

        T oneMinusRho = NumOps.Subtract(NumOps.One, _sparsityParameter);

        // d(KL)/d(ρ̂) = -ρ/ρ̂ + (1-ρ)/(1-ρ̂)
        T term1 = NumOps.Divide(NumOps.Negate(_sparsityParameter), rhoHat);
        T term2 = NumOps.Divide(oneMinusRho, oneMinusRhoHat);
        T dKL_drhoHat = NumOps.Add(term1, term2);

        // d(ρ̂)/da = 1/n for each activation
        int n = _lastEncoderActivations.Length;
        T scalingFactor = NumOps.Divide(dKL_drhoHat, NumOps.FromDouble(n));

        // Apply auxiliary loss weight
        scalingFactor = NumOps.Multiply(scalingFactor, AuxiliaryLossWeight);

        // Create gradient tensor with the same shape as encoder activations
        var gradient = new Tensor<T>(_lastEncoderActivations.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = scalingFactor;
        }

        return gradient;
    }

    /// <summary>
    /// Gets diagnostic information about the sparsity loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about sparsity.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into the autoencoder's sparsity behavior, including:
    /// - Current sparsity loss value
    /// - Average activation level
    /// - Target sparsity parameter
    /// - Whether auxiliary loss is enabled
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to track sparsity during training.
    ///
    /// The diagnostics include:
    /// - Sparsity Loss: How far from the target sparsity (lower is better)
    /// - Average Activation: Current average neuron activity level
    /// - Target Sparsity: The desired average activity level
    /// - Sparsity Weight: How much the sparsity penalty influences training
    ///
    /// These values help you:
    /// - Verify that sparsity is being enforced
    /// - Tune the sparsity parameter and weight
    /// - Detect if neurons are dying (too much sparsity) or too active (too little)
    /// - Monitor training health and feature learning quality
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "SparsityLoss", System.Convert.ToString(_lastSparsityLoss) ?? "0" },
            { "AverageActivation", System.Convert.ToString(_averageActivation) ?? "0" },
            { "TargetSparsity", System.Convert.ToString(_sparsityParameter) ?? "0.05" },
            { "SparsityWeight", System.Convert.ToString(AuxiliaryLossWeight) ?? "0.001" },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Trains the autoencoder on the provided data.
    /// </summary>
    /// <param name="input">The input data to train on.</param>
    /// <param name="expectedOutput">The expected output, typically the same as the input for standard autoencoders.</param>
    /// <remarks>
    /// <para>
    /// This method trains the autoencoder to reproduce the input as its output. For standard autoencoders,
    /// the expectedOutput is typically the same as the input. For denoising autoencoders, the input might be
    /// a noisy version while expectedOutput is the clean version.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the autoencoder to compress and reconstruct your data effectively.
    /// 
    /// During training:
    /// - The autoencoder processes your input data through all its layers
    /// - It compares the reconstructed output with the expected output
    /// - It calculates how to adjust its internal parameters to make the reconstruction better
    /// - It updates all parameters to gradually improve performance
    /// 
    /// For a standard autoencoder, the expected output is the same as the input (it learns to recreate what it sees).
    /// For specialized autoencoders like denoising autoencoders, the input could be noisy data while the expected output is clean data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Basic validation
        if (input.Shape[1] != Layers[0].GetInputShape()[0])
        {
            throw new ArgumentException($"Input shape {input.Shape[1]} does not match expected input shape {Layers[0].GetInputShape()[0]}");
        }

        if (expectedOutput.Shape[1] != Layers[Layers.Count - 1].GetOutputShape()[0])
        {
            throw new ArgumentException($"Expected output shape {expectedOutput.Shape[1]} does not match network output shape {Layers[Layers.Count - 1].GetOutputShape()[0]}");
        }

        // Training loop
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            T epochLoss = NumOps.Zero;

            // Process in batches
            for (int i = 0; i < input.Shape[0]; i += _batchSize)
            {
                int currentBatchSize = Math.Min(_batchSize, input.Shape[0] - i);
                var batchInput = input.Slice(0, i, 0, i + currentBatchSize);
                var batchExpected = expectedOutput.Slice(0, i, 0, i + currentBatchSize);

                // Forward pass
                var current = batchInput;
                List<Tensor<T>> layerOutputs = new List<Tensor<T>>(Layers.Count + 1) { batchInput };

                for (int j = 0; j < Layers.Count; j++)
                {
                    current = Layers[j].Forward(current);
                    layerOutputs.Add(current);

                    // Store encoder activations for sparsity loss (at the middle layer)
                    if (j == Layers.Count / 2)
                    {
                        _lastEncoderActivations = current;
                    }
                }

                // Calculate reconstruction loss
                var reconstructionLoss = CalculateLoss(current, batchExpected);

                // Calculate auxiliary loss (sparsity) if enabled
                T auxiliaryLoss = NumOps.Zero;
                if (UseAuxiliaryLoss)
                {
                    var sparsityLoss = ComputeAuxiliaryLoss();
                    auxiliaryLoss = NumOps.Multiply(sparsityLoss, AuxiliaryLossWeight);
                }

                // Total loss combines reconstruction and sparsity
                var loss = NumOps.Add(reconstructionLoss, auxiliaryLoss);
                epochLoss = NumOps.Add(epochLoss, loss);

                // Backward pass (calculate gradients)
                var outputGradient = CalculateOutputGradient(current, batchExpected);

                // Backpropagate from output to middle layer (decoder layers)
                int middleLayerIndex = Layers.Count / 2;
                for (int j = Layers.Count - 1; j > middleLayerIndex; j--)
                {
                    outputGradient = Layers[j].Backward(outputGradient);
                }

                // Add sparsity gradient at the middle layer (encoder output)
                if (UseAuxiliaryLoss && _lastEncoderActivations != null)
                {
                    var sparsityGradient = ComputeSparsityGradient();
                    // Add weighted sparsity gradient to the reconstruction gradient
                    outputGradient = outputGradient.Add(sparsityGradient);
                }

                // Continue backpropagation through encoder layers
                for (int j = middleLayerIndex; j >= 0; j--)
                {
                    outputGradient = Layers[j].Backward(outputGradient);
                }

                // Update parameters
                for (int j = 0; j < Layers.Count; j++)
                {
                    if (Layers[j].SupportsTraining)
                    {
                        Layers[j].UpdateParameters(_learningRate);
                    }
                }
            }

            // Calculate average loss for the epoch
            epochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(input.Shape[0]));

            // Store the last loss value
            LastLoss = epochLoss;

            // Report progress
            Console.WriteLine($"Epoch {epoch + 1}/{_epochs}, Loss: {epochLoss}");
        }
    }

    /// <summary>
    /// Generates new data samples by sampling points in the latent space and decoding them.
    /// </summary>
    /// <param name="count">The number of samples to generate.</param>
    /// <param name="mean">The mean value for random sampling (typically 0).</param>
    /// <param name="stdDev">The standard deviation for random sampling (typically 1).</param>
    /// <returns>A tensor containing the generated samples.</returns>
    /// <remarks>
    /// <para>
    /// This method generates new data samples by randomly sampling points in the latent space
    /// and decoding them. It's useful for creative applications and exploring the learned data manifold.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates new data samples similar to what the autoencoder has seen.
    /// 
    /// It works by:
    /// - Creating random points in the compressed representation space
    /// - Passing these points through the decoder part of the network
    /// - Producing new, synthetic data samples that resemble the training data
    /// 
    /// This can be used for:
    /// - Generating new content (like images or music)
    /// - Data augmentation (creating additional training examples)
    /// - Exploring what the autoencoder has learned
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateSamples(int count, double mean = 0, double stdDev = 1)
    {
        // Create a random normal distribution in the latent space
        var random = RandomHelper.CreateSecureRandom();
        var latentSamples = new Matrix<T>(count, EncodedSize);

        // Generate random points in the latent space
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < EncodedSize; j++)
            {
                // Apply mean and standard deviation
                double value = mean + random.NextGaussian() * stdDev;
                latentSamples[i, j] = NumOps.FromDouble(value);
            }
        }

        // Convert to tensor
        var latentTensor = new Tensor<T>([count, EncodedSize], latentSamples);

        // Decode the latent samples
        return Decode(latentTensor);
    }

    /// <summary>
    /// Makes a prediction using the autoencoder by encoding and then decoding the input.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The reconstructed output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a full forward pass through the autoencoder, encoding the input into the latent space
    /// and then decoding it back to the original space. The goal of training is to make this reconstructed output
    /// as close as possible to the original input.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs your data through the entire autoencoder (compress then reconstruct).
    /// 
    /// It's essentially a shortcut that:
    /// 1. Calls Encode() to compress your data
    /// 2. Then calls Decode() to reconstruct it
    /// 
    /// When the autoencoder is well-trained:
    /// - The output should closely resemble the input
    /// - But with some differences based on what the network learned was important
    /// 
    /// For example, if you feed in a noisy image, you might get back a cleaner version
    /// as the autoencoder learns to focus on the important features and ignore the noise.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Process input as a tensor through the network
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Updates the parameters of the autoencoder.
    /// </summary>
    /// <param name="parameters">The parameters to update the network with.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the autoencoder with the provided parameter values.
    /// It distributes the parameters to each layer based on the number of parameters in each layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's internal values to improve its performance.
    /// 
    /// During training:
    /// - The learning algorithm calculates how the parameters should change
    /// - This method applies those changes to the actual parameters
    /// - Each layer gets its own portion of the parameter updates
    /// 
    /// Think of it like fine-tuning all the components of the autoencoder based on feedback:
    /// - The encoder learns better ways to compress the data
    /// - The decoder learns better ways to reconstruct the data
    /// - Together they improve at preserving the most important information
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Gets metadata about the autoencoder model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// This method returns metadata about the autoencoder, including the model type, input/output dimensions,
    /// encoded size, and layer configuration. This information is useful for model management, serialization,
    /// and transfer learning.
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Autoencoder,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", Layers[0].GetInputShape()[0] },
                { "EncodedSize", EncodedSize },
                { "LayerCount", Layers.Count },
                { "IsSymmetric", true },
                { "LayerSizes", Layers.Select(l => l.GetOutputShape()[0]).ToArray() }
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data for the Autoencoder.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific configuration and state of the Autoencoder to a binary stream.
    /// It includes network-specific parameters that are essential for later reconstruction of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the unique settings of your Autoencoder.
    /// 
    /// It writes:
    /// - The encoded size (size of the compressed representation)
    /// - The learning rate used for training
    /// - The number of epochs and batch size used in training
    /// - The configuration of each layer
    /// 
    /// Saving these details allows you to recreate the exact same network structure later.
    /// It's like writing down a detailed recipe so you can make the same dish again in the future.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(EncodedSize);
        writer.Write(Convert.ToDouble(_learningRate));
        writer.Write(_epochs);
        writer.Write(_batchSize);
    }

    /// <summary>
    /// Deserializes network-specific data for the Autoencoder.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific configuration and state of the Autoencoder from a binary stream.
    /// It reconstructs the network-specific parameters to match the state of the network when it was serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the unique settings of your Autoencoder.
    /// 
    /// It reads:
    /// - The encoded size (size of the compressed representation)
    /// - The learning rate used for training
    /// - The number of epochs and batch size used in training
    /// - The configuration of each layer
    /// 
    /// Loading these details allows you to recreate the exact same network structure that was previously saved.
    /// It's like following a detailed recipe to recreate a dish exactly as it was made before.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        EncodedSize = reader.ReadInt32();
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
        _epochs = reader.ReadInt32();
        _batchSize = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of the autoencoder model.
    /// </summary>
    /// <returns>A new instance of the autoencoder model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the autoencoder model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data. The new instance will have the same architecture, learning rate, epochs, batch size,
    /// and loss function as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the model structure without copying the learned data.
    /// 
    /// Think of it like creating a blueprint of the autoencoder:
    /// - It copies the same overall design (how many layers, how they're arranged)
    /// - It preserves settings like learning rate and batch size
    /// - It keeps the same encoded size (compression level)
    /// - But it doesn't copy any of the learned knowledge yet
    /// 
    /// This is primarily used when saving or loading models, creating a framework that the saved parameters
    /// can be loaded into later.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Autoencoder<T>(
            Architecture,
            _learningRate,
            _epochs,
            _batchSize,
            _lossFunction
        );
    }
}
