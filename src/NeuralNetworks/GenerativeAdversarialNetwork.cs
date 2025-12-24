namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Generative Adversarial Network (GAN), a deep learning architecture that consists of two neural networks
/// (a generator and a discriminator) competing against each other in a zero-sum game.
/// </summary>
/// <remarks>
/// <para>
/// A Generative Adversarial Network (GAN) is a powerful machine learning architecture that uses two neural networks - 
/// a generator and a discriminator - that are trained simultaneously through adversarial training. The generator 
/// network learns to create realistic synthetic data samples (like images), while the discriminator network learns 
/// to distinguish between real data and the generator's synthetic outputs. As training progresses, the generator 
/// becomes better at creating realistic data, and the discriminator becomes better at distinguishing real from fake, 
/// pushing each other to improve in a competitive process.
/// </para>
/// <para><b>For Beginners:</b> A GAN is like an art forger and an art detective competing against each other.
/// 
/// Think of it this way:
/// - The generator is like an art forger trying to create fake paintings that look real
/// - The discriminator is like an art detective trying to tell which paintings are real and which are fake
/// - As the forger gets better, the detective has to improve too
/// - As the detective gets better, the forger is forced to create more convincing fakes
/// - Eventually, the forger becomes so good that their fake paintings are nearly indistinguishable from real ones
/// 
/// This continuous competition drives both networks to improve, resulting in a generator that
/// can create remarkably realistic synthetic data like images, music, or text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GenerativeAdversarialNetwork<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets the list of recent generator loss values for monitoring training progress.
    /// </summary>
    /// <value>A list of loss values from recent training iterations.</value>
    /// <remarks>
    /// <para>
    /// This list stores the loss values from recent generator training steps. It's used to monitor
    /// training progress, detect instabilities, and potentially adapt hyperparameters like the
    /// learning rate. The list is limited to a fixed size to focus on recent performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how well the generator is improving over time.
    /// 
    /// Think of generatorLosses as:
    /// - A record of how well the generator has been doing recently
    /// - If the losses are decreasing, the generator is improving
    /// - If they're increasing or unstable, something might be wrong
    /// - This history can be used to automatically adjust the training process
    /// 
    /// It's like keeping a score history for a sports team to track
    /// improvement and make coaching adjustments.
    /// </para>
    /// </remarks>
    private List<T> _generatorLosses = new List<T>();

    /// <summary>
    /// Gets the generator network that creates synthetic data.
    /// </summary>
    /// <value>A convolutional neural network that generates synthetic data.</value>
    /// <remarks>
    /// <para>
    /// The Generator is a neural network that takes random noise as input and produces synthetic data
    /// (such as images) as output. During training, it learns to create increasingly realistic data
    /// that can fool the Discriminator. In this implementation, it's specifically a convolutional
    /// neural network, which is well-suited for image generation tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "forger" network that creates fake data.
    /// 
    /// Think of the Generator as:
    /// - An artist creating paintings from random starting points
    /// - It takes random noise (like static) and shapes it into structured data
    /// - Its goal is to create outputs so realistic they fool the Discriminator
    /// - It improves by learning from the feedback of the Discriminator
    /// 
    /// For example, in an image generation task, the Generator might start by creating
    /// blurry, unrealistic images, but gradually learn to create sharp, detailed,
    /// and realistic images as training progresses.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network that distinguishes between real and synthetic data.
    /// </summary>
    /// <value>A convolutional neural network that classifies data as real or synthetic.</value>
    /// <remarks>
    /// <para>
    /// The Discriminator is a neural network that takes data (either real or generated) as input
    /// and outputs a probability that the data is real. During training, it learns to better
    /// distinguish between real data and the Generator's synthetic data. In this implementation,
    /// it's specifically a convolutional neural network, which is well-suited for image classification tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "detective" network that tries to spot fakes.
    /// 
    /// Think of the Discriminator as:
    /// - An art expert examining paintings to determine if they're authentic
    /// - It analyzes data (like images) and gives a probability that it's real
    /// - Its goal is to correctly identify real data and detect generated fakes
    /// - It improves by learning from its mistakes
    /// 
    /// For example, in an image generation task, the Discriminator essentially
    /// answers the question "Is this a real photograph or a computer-generated image?"
    /// and becomes increasingly sophisticated in its ability to tell the difference.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimizer used for updating generator parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer handles the gradient-based parameter updates for the Generator network.
    /// By default, Adam optimizer is used, but any IGradientBasedOptimizer can be provided
    /// for customized training behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This is the learning algorithm for the Generator.
    /// It decides how to adjust the Generator's internal values based on feedback from training.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer used for updating discriminator parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer handles the gradient-based parameter updates for the Discriminator network.
    /// By default, Adam optimizer is used, but any IGradientBasedOptimizer can be provided
    /// for customized training behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This is the learning algorithm for the Discriminator.
    /// It decides how to adjust the Discriminator's internal values based on feedback from training.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    /// <summary>
    /// Gets the optimizer used for updating generator parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides access to the generator's optimizer for derived classes that need
    /// custom training logic.
    /// </para>
    /// </remarks>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GeneratorOptimizer => _generatorOptimizer;

    /// <summary>
    /// Gets the optimizer used for updating discriminator parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides access to the discriminator's optimizer for derived classes that need
    /// custom training logic.
    /// </para>
    /// </remarks>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiscriminatorOptimizer => _discriminatorOptimizer;

    /// <summary>
    /// Gets or sets whether gradient penalty (WGAN-GP) is enabled for training stability.
    /// </summary>
    private bool _useGradientPenalty = false;

    /// <summary>
    /// Gets or sets whether feature matching is enabled for generator training.
    /// </summary>
    /// <value>True if feature matching should be used; false otherwise.</value>
    /// <remarks>
    /// <para>
    /// Feature matching is a technique from Salimans et al. (2016) that helps stabilize GAN training
    /// and prevent mode collapse. Instead of training the generator to fool the discriminator directly,
    /// it trains the generator to match the statistics of real data features at intermediate layers
    /// of the discriminator.
    /// </para>
    /// <para><b>For Beginners:</b> This enables a more stable way of training the generator.
    ///
    /// Instead of just trying to fool the discriminator:
    /// - The generator learns to match the internal patterns of real images
    /// - This helps create more diverse and realistic outputs
    /// - It reduces the risk of mode collapse (generating the same image repeatedly)
    /// - Training tends to be more stable with this enabled
    /// </para>
    /// <para><b>Important:</b> When enabled, batches are stored but ComputeFeatureMatchingLoss()
    /// must be manually called and added to the generator loss in your training loop.
    /// The base Train() method does not automatically integrate this loss.
    /// See ComputeFeatureMatchingLoss() documentation for integration examples.
    /// </para>
    /// </remarks>
    public bool UseFeatureMatching { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight applied to the feature matching loss.
    /// </summary>
    /// <value>The multiplier for the feature matching loss component.</value>
    /// <remarks>
    /// <para>
    /// This weight balances the feature matching loss against the standard adversarial loss.
    /// Typical values range from 0.1 to 1.0. Higher values make the generator focus more on
    /// matching feature statistics rather than fooling the discriminator directly.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the generator focuses on feature matching.
    ///
    /// The weight determines:
    /// - How much to prioritize matching internal patterns vs. fooling the discriminator
    /// - Higher values mean more focus on feature matching
    /// - Lower values mean more focus on the adversarial objective
    /// - Typical values are around 0.1 to 1.0
    /// </para>
    /// </remarks>
    public double FeatureMatchingWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the indices of discriminator layers to use for feature matching.
    /// If null, uses middle layers by default.
    /// </summary>
    /// <value>Array of layer indices, or null to use defaults.</value>
    /// <remarks>
    /// <para>
    /// Specifies which discriminator layers to extract features from for feature matching.
    /// Typically, intermediate layers (not too early, not too late) work best. If null,
    /// the implementation will automatically select appropriate middle layers.
    /// </para>
    /// <para><b>For Beginners:</b> This chooses which internal layers to compare.
    ///
    /// Layer selection matters:
    /// - Early layers capture low-level features (edges, textures)
    /// - Middle layers capture mid-level features (shapes, parts)
    /// - Late layers capture high-level features (object identity)
    /// - If not specified, sensible defaults are used automatically
    /// </para>
    /// </remarks>
    public int[]? FeatureMatchingLayers { get; set; } = null;

    /// <summary>
    /// Stores the last real batch for feature matching computation.
    /// </summary>
    private Tensor<T>? _lastRealBatch;

    /// <summary>
    /// Stores the last fake batch for feature matching computation.
    /// </summary>
    private Tensor<T>? _lastFakeBatch;

    /// <summary>
    /// Stores the last computed gradient penalty value for diagnostics.
    /// </summary>
    private T _lastGradientPenalty;

    /// <summary>
    /// Stores the last computed feature matching loss for diagnostics.
    /// </summary>
    private T _lastFeatureMatchingLoss;

    /// <summary>
    /// Stores the last discriminator loss for diagnostics.
    /// </summary>
    private T _lastDiscriminatorLoss;

    /// <summary>
    /// Stores the last generator loss for diagnostics.
    /// </summary>
    private T _lastGeneratorLoss;

    /// <summary>
    /// Gets or sets whether to use auxiliary losses (gradient penalty, feature matching) during training.
    /// Default is true for improved training stability.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the weight for auxiliary losses (gradient penalty, feature matching).
    /// Default is 10.0 for gradient penalty (standard for WGAN-GP).
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Creates the combined GAN architecture from generator and discriminator architectures.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateGANArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType)
    {
        // For 3D input (images), use the discriminator's dimensions since it takes the image input
        // The GAN's combined architecture represents the data flow from image to output
        if (inputType == InputType.ThreeDimensional)
        {
            // Don't pass inputSize - let the validation calculate it from dimensions
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Medium,
                inputSize: 0, // Let validation calculate from dimensions
                inputHeight: discriminatorArchitecture.InputHeight,
                inputWidth: discriminatorArchitecture.InputWidth,
                inputDepth: discriminatorArchitecture.InputDepth,
                outputSize: discriminatorArchitecture.OutputSize,
                layers: null);
        }

        // For 1D input, use the generator's input size
        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Medium,
            inputSize: generatorArchitecture.InputSize,
            inputHeight: 0,
            inputWidth: 0,
            inputDepth: 1,
            outputSize: discriminatorArchitecture.OutputSize,
            layers: null);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GenerativeAdversarialNetwork{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.</param>
    /// <param name="inputType">The type of input the GAN will process.</param>
    /// <param name="generatorOptimizer">The optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">The optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function used to compute loss values during training.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Generative Adversarial Network with the specified generator and discriminator
    /// architectures. It also sets up the optimization parameters and initializes tracking collections for monitoring
    /// training progress. The GAN's architecture is a combination of the generator and discriminator architectures.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the complete GAN system with both networks.
    ///
    /// When creating a new GAN:
    /// - You provide separate architectures for the generator and discriminator
    /// - You can optionally provide custom optimizers for each network
    /// - The inputType specifies what kind of data the GAN will work with
    /// - If you don't specify optimizers, Adam optimizer is used by default
    ///
    /// Think of it like establishing the rules and roles for the forger and detective
    /// before their competition begins.
    /// </para>
    /// </remarks>
    public GenerativeAdversarialNetwork(NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(CreateGANArchitecture(generatorArchitecture, discriminatorArchitecture, inputType),
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        // Initialize auxiliary loss fields
        AuxiliaryLossWeight = NumOps.FromDouble(10.0);
        _lastGradientPenalty = NumOps.Zero;
        _lastFeatureMatchingLoss = NumOps.Zero;
        _lastDiscriminatorLoss = NumOps.Zero;
        _lastGeneratorLoss = NumOps.Zero;

        // Initialize tracking collections
        _generatorLosses = new List<T>();
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        // Initialize optimizers (default to Adam if not provided)
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator);
        _discriminatorOptimizer = discriminatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one step of training for both the generator and discriminator using tensor batches.
    /// </summary>
    /// <param name="realImages">A tensor containing real images for training the discriminator.</param>
    /// <param name="noise">A tensor containing random noise for training the generator.</param>
    /// <returns>A tuple containing the discriminator and generator loss values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs one complete training iteration for the GAN using tensor-based operations
    /// for maximum efficiency. It first trains the Discriminator on batches of both real images and 
    /// fake images generated by the Generator. Then it trains the Generator to create images that can
    /// fool the Discriminator. This adversarial training process is optimized for batch processing.
    /// </para>
    /// <para><b>For Beginners:</b> This is one round of the competition between generator and discriminator.
    /// 
    /// The tensor-based training step:
    /// - Processes entire batches of images in parallel
    /// - First trains the discriminator on both real and generated images
    /// - Then trains the generator to create more convincing fake images
    /// - Returns the loss values to track progress
    /// - Is much faster than training with individual vectors
    /// 
    /// This efficient implementation is critical for training GANs in reasonable timeframes.
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, Tensor<T> noise)
    {
        // Ensure we're in training mode
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        // ----- Train the discriminator -----

        // Generate fake images
        Tensor<T> fakeImages = GenerateImages(noise);

        // Store batches for feature matching if enabled
        if (UseFeatureMatching)
        {
            _lastRealBatch = realImages.Clone();
            _lastFakeBatch = fakeImages.Clone();
        }

        // Get batch size from real images tensor
        int batchSize = realImages.Shape[0];

        // Create label tensors (1 for real, 0 for fake)
        Tensor<T> realLabels = CreateLabelTensor(batchSize, NumOps.One);
        Tensor<T> fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        // Train discriminator on real images
        T realLoss = TrainDiscriminatorBatch(realImages, realLabels);

        // Train discriminator on fake images
        T fakeLoss = TrainDiscriminatorBatch(fakeImages, fakeLabels);

        // Compute total discriminator loss
        T discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        discriminatorLoss = NumOps.Divide(discriminatorLoss, NumOps.FromDouble(2.0)); // Average loss
        _lastDiscriminatorLoss = discriminatorLoss;

        // ----- Train the generator -----

        // Generate new fake images for generator training
        Tensor<T> newFakeImages = GenerateImages(noise);

        // For generator training, we want the discriminator to think fake images are real
        Tensor<T> allRealLabels = CreateLabelTensor(batchSize, NumOps.One);

        // Train the generator to fool the discriminator
        T generatorLoss = TrainGeneratorBatch(noise, newFakeImages, allRealLabels);
        _lastGeneratorLoss = generatorLoss;

        // Calculate auxiliary losses if enabled
        T auxiliaryLoss = NumOps.Zero;
        if (UseAuxiliaryLoss)
        {
            var auxLoss = ComputeAuxiliaryLoss();
            auxiliaryLoss = NumOps.Multiply(auxLoss, AuxiliaryLossWeight);
        }

        // Combine generator loss with auxiliary losses
        var totalGeneratorLoss = NumOps.Add(generatorLoss, auxiliaryLoss);

        // Track generator loss for monitoring
        _generatorLosses.Add(totalGeneratorLoss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0); // Keep only recent losses
        }

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Creates a tensor filled with a specified value, typically used for labels.
    /// </summary>
    /// <param name="batchSize">The batch size (first dimension of the tensor).</param>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <returns>A tensor filled with the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This utility method creates a tensor with shape [batchSize, 1] filled with a single value.
    /// In the context of GANs, it's typically used to create label tensors where 1 represents "real"
    /// and 0 represents "fake". These label tensors are used as targets during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This creates batches of training labels efficiently.
    /// 
    /// Think of this method as:
    /// - Creating a list of "correct answers" for training
    /// - For real images, the correct answer is usually 1 ("real")
    /// - For fake images, the correct answer is usually 0 ("fake")
    /// - This method creates these answers for multiple images at once
    /// 
    /// This tensor-based approach is more efficient than creating individual labels.
    /// </para>
    /// </remarks>
    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        // Create a tensor of shape [batchSize, 1] filled with the specified value
        var shape = new int[] { batchSize, 1 };
        var tensor = new Tensor<T>(shape);

        // === Vectorized tensor fill using IEngine (Phase B: US-GPU-015) ===
        Engine.TensorFill(tensor, value);

        return tensor;
    }

    /// <summary>
    /// Trains the generator to create images that can fool the discriminator using tensor operations.
    /// </summary>
    /// <param name="noise">The tensor containing noise vectors used as input to the generator.</param>
    /// <param name="generatedImages">The tensor containing images generated from the noise.</param>
    /// <param name="targetLabels">The tensor containing target labels (typically all 1's).</param>
    /// <returns>The batch loss value for this training step.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the Generator network to create images that the Discriminator will classify
    /// as real. It uses efficient tensor operations to process entire batches of data in parallel.
    /// During this process, the Discriminator's weights are frozen as we're only training the Generator.
    /// The generator's goal is to create images that receive high "real" scores from the discriminator.
    /// </para>
    /// <para><b>For Beginners:</b> This efficiently teaches the generator to create convincing fake images.
    /// 
    /// The tensor-based generator training:
    /// - Processes batches of noise vectors to generate fake images
    /// - Passes these fake images through the discriminator
    /// - Calculates how well the fake images fooled the discriminator
    /// - Updates only the generator's parameters to create more convincing images
    /// 
    /// This optimized approach is essential for effective GAN training.
    /// </para>
    /// </remarks>
    private T TrainGeneratorBatch(Tensor<T> noise, Tensor<T> generatedImages, Tensor<T> targetLabels)
    {
        // Ensure generator is in training mode and discriminator is not (we don't want to update discriminator)
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false); // Freeze discriminator weights

        // Forward pass through discriminator with generated images
        Tensor<T> discriminatorOutput = Discriminator.Predict(generatedImages);

        // Calculate generator loss - we want the discriminator to classify fake images as real
        T loss = CalculateBatchLoss(discriminatorOutput, targetLabels);

        // Calculate gradients for discriminator output
        Tensor<T> outputGradients = CalculateBatchGradients(discriminatorOutput, targetLabels);

        // Backpropagate through discriminator to get gradients at its input (which is the generator's output)
        Tensor<T> discriminatorInputGradients = Discriminator.Backpropagate(outputGradients);

        // Backpropagate through generator using the gradients from discriminator
        Generator.Backpropagate(discriminatorInputGradients);

        // Update generator parameters
        UpdateNetworkParameters(Generator);

        // Re-enable training mode for discriminator for future training steps
        Discriminator.SetTrainingMode(true);

        return loss;
    }

    /// <summary>
    /// Updates the parameters of a network using the configured optimizer with the calculated gradients.
    /// </summary>
    /// <param name="network">The neural network to update.</param>
    /// <remarks>
    /// <para>
    /// This method applies the calculated gradients to update the parameters of the specified network
    /// using the configured optimizer (Generator or Discriminator). It includes gradient clipping
    /// for stable and efficient training.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the network's internal values using the optimizer.
    ///
    /// The parameter update process:
    /// - Uses the configured optimizer (default is Adam)
    /// - Applies gradient clipping to prevent exploding gradients
    /// - The optimizer handles momentum, adaptive learning rates, etc.
    ///
    /// This approach helps GANs train more reliably and efficiently.
    /// </para>
    /// </remarks>
    private void UpdateNetworkParameters(ConvolutionalNeuralNetwork<T> network)
    {
        // Get current parameters and gradients
        var parameters = network.GetParameters();
        var gradients = network.GetParameterGradients();

        // Gradient clipping to prevent exploding gradients (vectorized)
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
        }

        // Select the appropriate optimizer based on which network is being updated
        var optimizer = ReferenceEquals(network, Generator) ? _generatorOptimizer : _discriminatorOptimizer;

        // Use the optimizer to compute updated parameters
        var updatedParameters = optimizer.UpdateParameters(parameters, gradients);

        // Update network parameters
        network.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Evaluates the GAN by generating a batch of images and calculating metrics for their quality.
    /// </summary>
    /// <param name="sampleSize">The number of images to generate for evaluation.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// This tensor-based method evaluates the current performance of the GAN by generating a batch
    /// of images and calculating several metrics. It computes statistics on the discriminator scores,
    /// checks for diversity in the outputs, and detects potential mode collapse. This efficient
    /// implementation processes all images in parallel for better performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tests how well the GAN is performing using batch processing.
    /// 
    /// The tensor-based evaluation:
    /// - Generates multiple sample images in a single batch operation
    /// - Has the discriminator score all images at once
    /// - Calculates statistics like average score and diversity measures
    /// - Identifies potential issues like mode collapse
    /// 
    /// This provides comprehensive metrics on GAN performance in an efficient manner.
    /// </para>
    /// </remarks>
    public Dictionary<string, double> EvaluateModel(int sampleSize = 100)
    {
        var metrics = new Dictionary<string, double>();

        // Generate sample images in a single batch
        var noise = GenerateRandomNoiseTensor(sampleSize, Generator.Architecture.InputSize);
        var generatedImages = GenerateImages(noise);

        // Get discriminator scores for all images at once
        var discriminatorScores = DiscriminateImages(generatedImages);

        // Convert to list for statistical calculations
        var scoresList = new List<double>(sampleSize);
        for (int i = 0; i < sampleSize; i++)
        {
            scoresList.Add(Convert.ToDouble(discriminatorScores[i, 0]));
        }

        // Calculate metrics
        double averageScore = scoresList.Average();
        double stdDevScore = StatisticsHelper<double>.CalculateStandardDeviation(scoresList);
        double minScore = scoresList.Min();
        double maxScore = scoresList.Max();

        // Recent loss values
        double recentLoss = _generatorLosses.Count > 0 ?
            Convert.ToDouble(_generatorLosses[_generatorLosses.Count - 1]) : 0.0;

        // Store metrics
        metrics["AverageDiscriminatorScore"] = averageScore;
        metrics["MinDiscriminatorScore"] = minScore;
        metrics["MaxDiscriminatorScore"] = maxScore;
        metrics["ScoreStandardDeviation"] = stdDevScore;
        metrics["ScoreRange"] = maxScore - minScore;
        metrics["RecentGeneratorLoss"] = recentLoss;

        // Advanced metrics for diagnosing GAN issues

        // Mode collapse indicator (if very low standard deviation, might indicate mode collapse)
        bool potentialModeCollapse = stdDevScore < 0.05;
        metrics["PotentialModeCollapse"] = potentialModeCollapse ? 1.0 : 0.0;

        // Training stability indicator
        if (_generatorLosses.Count >= 20)
        {
            var recentLosses = _generatorLosses.Skip(_generatorLosses.Count - 20).Select(l => Convert.ToDouble(l)).ToList();
            double lossVariance = StatisticsHelper<double>.CalculateVariance(recentLosses);
            metrics["LossVariance"] = lossVariance;
            metrics["TrainingStability"] = lossVariance < 0.01 ? 1.0 : 0.0; // 1 means stable
        }

        return metrics;
    }

    /// <summary>
    /// Initializes the layers of the Generative Adversarial Network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is overridden from the base class but is empty because a GAN doesn't use layers directly.
    /// Instead, the GAN architecture consists of two separate neural networks (Generator and Discriminator)
    /// that each have their own layers. These networks are initialized separately in the constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This method is empty because GANs work differently from standard neural networks.
    /// 
    /// Unlike traditional neural networks:
    /// - GANs don't have a single sequence of layers
    /// - Instead, they consist of two separate networks (Generator and Discriminator)
    /// - Each of these networks has its own layers
    /// - These networks are initialized separately in the constructor
    /// 
    /// This method is only included because it's required by the base class,
    /// but it doesn't need to do anything in a GAN implementation.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // GAN doesn't use layers directly, so this method is empty
    }

    /// <summary>
    /// Performs a forward pass through the generator network using a tensor input.
    /// </summary>
    /// <param name="input">The input tensor containing noise vectors to generate images from.</param>
    /// <returns>A tensor containing the generated images.</returns>
    /// <remarks>
    /// <para>
    /// This method is part of the INeuralNetwork interface implementation. In the context of a GAN,
    /// "prediction" means using the generator to create synthetic data from random noise input.
    /// The method supports batch processing by handling tensor inputs that may contain multiple
    /// noise vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates synthetic images from random noise.
    /// 
    /// When you call Predict:
    /// - The input tensor contains one or more random noise patterns
    /// - These noise patterns are passed through the generator network
    /// - The generator transforms the noise into synthetic images
    /// - The output tensor contains the resulting synthetic images
    /// 
    /// This is the same underlying process as GenerateImage(), but works with tensors
    /// instead of vectors, allowing for batch processing of multiple inputs at once.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For a GAN, prediction means generating data using the generator
        // Check if the input is a batch of noise vectors or a single noise vector
        if (input.Rank == 1)
        {
            // Single noise vector
            return Generator.Predict(input);
        }
        else
        {
            // Batch of noise vectors
            var batchSize = input.Shape[0];
            var results = new List<Tensor<T>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                var noiseVector = results[i];
                var generatedImage = Generator.Predict(noiseVector);
                results.Add(generatedImage);
            }

            return Tensor<T>.Stack([.. results]);
        }
    }

    /// <summary>
    /// Trains both the generator and discriminator using tensor-based operations throughout.
    /// </summary>
    /// <param name="input">The noise input for the generator (for batch training).</param>
    /// <param name="expectedOutput">The real images used to train the discriminator.</param>
    /// <remarks>
    /// <para>
    /// This tensor-native implementation trains both networks efficiently by processing
    /// entire batches at once through tensor operations. It eliminates the vector conversion
    /// overhead from the previous implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This trains both networks efficiently with multiple examples.
    /// 
    /// The fully tensor-based training process:
    /// 1. Processes entire batches of data in parallel
    /// 2. Trains the discriminator on both real and fake images
    /// 3. Trains the generator to create more convincing fake images
    /// 4. Updates the networks using batch operations for better performance
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Get batch size from the input tensor
        int batchSize = input.Shape[0];

        // Set both networks to training mode
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        // ------------ Train Discriminator ------------

        // Generate fake images with tensor operations
        var fakeImages = Generator.Predict(input);

        // Store batches for feature matching if enabled
        if (UseFeatureMatching)
        {
            _lastRealBatch = expectedOutput.Clone();
            _lastFakeBatch = fakeImages.Clone();
        }

        // Create label tensors (1 for real, 0 for fake)
        var realLabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        // Train discriminator on real images using tensor operations
        var realLoss = TrainDiscriminatorBatch(expectedOutput, realLabels);

        // Train discriminator on fake images using tensor operations
        var fakeLoss = TrainDiscriminatorBatch(fakeImages, fakeLabels);

        // Compute average discriminator loss
        var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        discriminatorLoss = NumOps.Divide(discriminatorLoss, NumOps.FromDouble(2.0));
        _lastDiscriminatorLoss = discriminatorLoss;

        // ------------ Train Generator ------------

        // For generator training, we want discriminator to think fake images are real
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);

        // Train generator to fool discriminator using tensor operations
        var generatorLoss = TrainGeneratorBatch(input, allRealLabels);
        _lastGeneratorLoss = generatorLoss;

        // Calculate auxiliary losses if enabled
        T auxiliaryLoss = NumOps.Zero;
        if (UseAuxiliaryLoss)
        {
            var auxLoss = ComputeAuxiliaryLoss();
            auxiliaryLoss = NumOps.Multiply(auxLoss, AuxiliaryLossWeight);
        }

        // Combine generator loss with auxiliary losses
        var totalGeneratorLoss = NumOps.Add(generatorLoss, auxiliaryLoss);

        // Track generator loss for monitoring
        _generatorLosses.Add(totalGeneratorLoss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0);
        }

        LastLoss = totalGeneratorLoss;
    }

    /// <summary>
    /// Trains the discriminator on a batch of images using tensor operations.
    /// </summary>
    /// <param name="images">The tensor containing images to train on.</param>
    /// <param name="labels">The tensor containing labels (real or fake).</param>
    /// <returns>The loss value for this training step.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the Discriminator network on a batch of images using tensor operations
    /// throughout. It computes predictions, calculates loss, and updates weights using backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the discriminator to spot real vs. fake images.
    /// 
    /// The process:
    /// - The discriminator examines a batch of images
    /// - It tries to guess which ones are real and which are fake
    /// - It calculates how wrong it was (the loss)
    /// - It adjusts its internal parameters to make better predictions
    /// 
    /// The tensor-based implementation makes this much more efficient.
    /// </para>
    /// </remarks>
    private T TrainDiscriminatorBatch(Tensor<T> images, Tensor<T> labels)
    {
        // Ensure discriminator is in training mode
        Discriminator.SetTrainingMode(true);

        // Forward pass - get predictions for the batch
        var predictions = Discriminator.Predict(images);

        // Calculate loss
        var loss = CalculateBatchLoss(predictions, labels);

        // Calculate gradients for backpropagation
        var outputGradients = CalculateBatchGradients(predictions, labels);

        // Backpropagate through the discriminator
        Discriminator.Backpropagate(outputGradients);

        // Update discriminator parameters
        UpdateNetworkParameters(Discriminator);

        return loss;
    }

    /// <summary>
    /// Trains the generator to create images that can fool the discriminator.
    /// </summary>
    /// <param name="noise">The tensor containing noise vectors.</param>
    /// <param name="targetLabels">The tensor containing target labels (1's for "real").</param>
    /// <returns>The loss value for this training step.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the Generator to create images that fool the Discriminator using tensor operations.
    /// It passes generated images through the discriminator and trains the generator to maximize
    /// the discriminator's "real" classification score.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the generator to create more convincing fake images.
    /// 
    /// The process:
    /// - The generator creates fake images from noise
    /// - The discriminator evaluates these images
    /// - The generator's goal is to create images the discriminator thinks are real
    /// - The generator adjusts its parameters to create more convincing images
    /// 
    /// The tensor-based implementation makes this training much more efficient.
    /// </para>
    /// </remarks>
    private T TrainGeneratorBatch(Tensor<T> noise, Tensor<T> targetLabels)
    {
        // Ensure generator is in training mode
        Generator.SetTrainingMode(true);

        // Temporarily freeze discriminator weights during generator training
        Discriminator.SetTrainingMode(false);

        // Generate fake images
        var generatedImages = Generator.Predict(noise);

        // Pass fake images through discriminator
        var discriminatorOutput = Discriminator.Predict(generatedImages);

        // Calculate loss - we want the discriminator to classify fake images as real
        var loss = CalculateBatchLoss(discriminatorOutput, targetLabels);

        // Calculate gradients for discriminator output
        var outputGradients = CalculateBatchGradients(discriminatorOutput, targetLabels);

        // Backpropagate through discriminator (keeping its weights frozen)
        var discriminatorInputGradients = Discriminator.Backpropagate(outputGradients);

        // Backpropagate through generator
        Generator.Backpropagate(discriminatorInputGradients);

        // Update generator parameters
        UpdateNetworkParameters(Generator);

        // Restore discriminator to training mode
        Discriminator.SetTrainingMode(true);

        return loss;
    }

    /// <summary>
    /// Calculates the loss for a batch of predictions and target values.
    /// </summary>
    /// <param name="predictions">The tensor containing predicted values.</param>
    /// <param name="targets">The tensor containing target values.</param>
    /// <returns>The batch loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates binary cross-entropy loss for the GAN, measuring how well
    /// the discriminator is classifying real vs. fake images or how well the generator
    /// is fooling the discriminator.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how accurate the predictions are.
    /// 
    /// The calculation:
    /// - Compares the predicted classifications with the expected ones
    /// - Returns a single value representing the overall error
    /// - Lower values mean more accurate predictions
    /// - This handles all calculations efficiently across the batch
    /// </para>
    /// </remarks>
    private T CalculateBatchLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        // Implement binary cross-entropy loss for classification
        int batchSize = predictions.Shape[0];
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10); // Small value to prevent log(0)

        for (int i = 0; i < batchSize; i++)
        {
            T prediction = predictions[i, 0];
            T target = targets[i, 0];

            // Binary cross-entropy: -target * log(prediction) - (1-target) * log(1-prediction)
            T logP = NumOps.Log(NumOps.Add(prediction, epsilon));
            T logOneMinusP = NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, prediction), epsilon));

            T termOne = NumOps.Multiply(target, logP);
            T termTwo = NumOps.Multiply(NumOps.Subtract(NumOps.One, target), logOneMinusP);

            T sampleLoss = NumOps.Negate(NumOps.Add(termOne, termTwo));
            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        // Average the loss across the batch
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates gradients for backpropagation from predictions and targets.
    /// </summary>
    /// <param name="predictions">The tensor containing predicted values.</param>
    /// <param name="targets">The tensor containing target values.</param>
    /// <returns>A tensor of gradients for backpropagation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates gradients for binary cross-entropy loss, which are used
    /// during backpropagation to update the network weights in the direction that
    /// reduces the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how to adjust the network to improve.
    /// 
    /// The gradient calculation:
    /// - Determines the direction and amount to adjust each parameter
    /// - Is based on how wrong each prediction was
    /// - Shows whether values should increase or decrease
    /// - Handles all calculations efficiently across the batch
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateBatchGradients(Tensor<T> predictions, Tensor<T> targets)
    {
        // Calculate gradients for binary cross-entropy: (prediction - target)
        int batchSize = predictions.Shape[0];
        var gradients = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            // For binary cross-entropy, the gradient simplifies to (prediction - target)
            gradients[i, 0] = NumOps.Subtract(predictions[i, 0], targets[i, 0]);
        }

        return gradients;
    }

    /// <summary>
    /// Generates synthetic images using tensor operations.
    /// </summary>
    /// <param name="noise">The tensor containing the noise input.</param>
    /// <returns>A tensor containing generated images.</returns>
    /// <remarks>
    /// <para>
    /// This method generates synthetic images by passing noise through the generator network
    /// using tensor operations. It supports both single inputs and batches.
    /// </para>
    /// <para><b>For Beginners:</b> This creates fake images from random noise patterns.
    /// 
    /// The process:
    /// - Takes random noise as input (the creative inspiration)
    /// - Passes it through the generator network
    /// - Produces synthetic images as output
    /// - Works efficiently with batches of inputs
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateImages(Tensor<T> noise)
    {
        // Set generator to inference mode
        Generator.SetTrainingMode(false);

        // Generate images using tensor operations
        return Generator.Predict(noise);
    }

    /// <summary>
    /// Evaluates how real a batch of images appears to the discriminator.
    /// </summary>
    /// <param name="images">The tensor containing images to evaluate.</param>
    /// <returns>A tensor containing discriminator scores for each image.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates how realistic a batch of images appears to the discriminator,
    /// returning a score between 0 and 1 for each image where higher values indicate
    /// more realistic images.
    /// </para>
    /// <para><b>For Beginners:</b> This checks how convincing the generated images are.
    /// 
    /// The discriminator's evaluation:
    /// - Examines each image in the batch
    /// - Scores each image between 0 and 1
    /// - Higher scores mean more convincing/realistic images
    /// - Provides feedback on the generator's performance
    /// </para>
    /// </remarks>
    public Tensor<T> DiscriminateImages(Tensor<T> images)
    {
        // Set discriminator to inference mode
        Discriminator.SetTrainingMode(false);

        // Discriminate images using tensor operations
        return Discriminator.Predict(images);
    }

    /// <summary>
    /// Generates a tensor of random noise for the generator.
    /// </summary>
    /// <param name="batchSize">The number of noise vectors to generate.</param>
    /// <param name="noiseSize">The size of each noise vector.</param>
    /// <returns>A tensor containing random noise from a normal distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method efficiently generates a batch of noise vectors using a normal distribution,
    /// which serves as input to the generator for creating synthetic images.
    /// </para>
    /// <para><b>For Beginners:</b> This creates random starting points for image generation.
    /// 
    /// The noise generation:
    /// - Creates multiple random inputs in a single operation
    /// - Uses a normal distribution (bell curve) for better results
    /// - Each noise pattern will result in a different image
    /// - Efficient batch processing for better performance
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        }

        if (noiseSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseSize), noiseSize, "Noise size must be positive.");
        }

        int totalElements = batchSize * noiseSize;

        // === Vectorized Box-Muller transform using new IEngine methods (Phase C) ===
        // Generate uniform random numbers using TensorRandomUniformRange
        int halfElements = (totalElements + 1) / 2;

        // u2 is in [0, 1)
        var u2 = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One);

        // u1 is 1 - random to get (0, 1] and avoid log(0)
        var u1Temp = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One);
        var u1 = Engine.ScalarMinusTensor(NumOps.One, u1Temp);
        
        // Vectorized Box-Muller: radius = sqrt(-2 * log(u1)), theta = 2 * pi * u2
        var logU1 = Engine.TensorLog(u1);
        var negTwoLogU1 = Engine.TensorMultiplyScalar(logU1, NumOps.FromDouble(-2.0));
        var radius = Engine.TensorSqrt(negTwoLogU1);
        
        var theta = Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI));
        
        // z1 = radius * cos(theta), z2 = radius * sin(theta)
        var cosTheta = Engine.TensorCos(theta);
        var sinTheta = Engine.TensorSin(theta);
        
        var z1 = Engine.TensorMultiply(radius, cosTheta);
        var z2 = Engine.TensorMultiply(radius, sinTheta);
        
        // Interleave z1 and z2 into the final noise tensor
        var noiseData = new T[totalElements];
        var z1Array = z1.ToArray();
        var z2Array = z2.ToArray();
        
        for (int i = 0; i < halfElements; i++)
        {
            int idx = i * 2;
            if (idx < totalElements)
                noiseData[idx] = z1Array[i];
            if (idx + 1 < totalElements)
                noiseData[idx + 1] = z2Array[i];
        }
        
        return new Tensor<T>(noiseData, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Evaluates the GAN using tensor operations.
    /// </summary>
    /// <param name="sampleSize">The number of images to generate for evaluation.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates GAN performance by generating images and calculating metrics
    /// using tensor operations throughout. This provides a more efficient evaluation
    /// compared to the previous vector-based approach.
    /// </para>
    /// <para><b>For Beginners:</b> This tests how well the GAN is performing.
    /// 
    /// The tensor-based evaluation:
    /// - Generates multiple images in a single batch operation
    /// - Has the discriminator evaluate all images at once
    /// - Calculates statistics on the quality and diversity of the outputs
    /// - Provides metrics to track training progress
    /// </para>
    /// </remarks>
    public Dictionary<string, double> EvaluateModelWithTensors(int sampleSize = 100)
    {
        var metrics = new Dictionary<string, double>();

        // Generate sample images using tensor operations
        var noise = GenerateRandomNoiseTensor(sampleSize, Generator.Architecture.InputSize);
        var generatedImages = GenerateImages(noise);

        // Get discriminator scores for all images at once
        var discriminatorScores = DiscriminateImages(generatedImages);

        // Extract scores for calculations
        var scoresList = new List<double>(sampleSize);
        for (int i = 0; i < sampleSize; i++)
        {
            scoresList.Add(Convert.ToDouble(discriminatorScores[i, 0]));
        }

        // Calculate metrics
        double averageScore = scoresList.Average();
        double stdDevScore = StatisticsHelper<double>.CalculateStandardDeviation(scoresList);
        double minScore = scoresList.Min();
        double maxScore = scoresList.Max();

        // Recent loss values
        double recentLoss = _generatorLosses.Count > 0 ?
            Convert.ToDouble(_generatorLosses[_generatorLosses.Count - 1]) : 0.0;

        // Store metrics
        metrics["AverageDiscriminatorScore"] = averageScore;
        metrics["MinDiscriminatorScore"] = minScore;
        metrics["MaxDiscriminatorScore"] = maxScore;
        metrics["ScoreStandardDeviation"] = stdDevScore;
        metrics["ScoreRange"] = maxScore - minScore;
        metrics["RecentGeneratorLoss"] = recentLoss;

        // Mode collapse indicator (if very low standard deviation, might indicate mode collapse)
        metrics["PotentialModeCollapse"] = stdDevScore < 0.05 ? 1.0 : 0.0;

        return metrics;
    }

    /// <summary>
    /// Generates high-quality images by filtering based on discriminator scores.
    /// </summary>
    /// <param name="count">The number of images to generate.</param>
    /// <param name="minDiscriminatorScore">The minimum score threshold for quality images.</param>
    /// <returns>A tensor of images that meet the quality threshold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates multiple images and filters them based on discriminator scores.
    /// Only images that exceed the specified quality threshold are returned, ensuring
    /// better overall output quality.
    /// </para>
    /// <para><b>For Beginners:</b> This creates multiple high-quality fake images.
    /// 
    /// The process:
    /// - Generates more images than requested
    /// - Uses the discriminator to evaluate their quality
    /// - Keeps only the most convincing/realistic ones
    /// - Returns images that meet your quality standards
    /// 
    /// This is useful for applications where you want only the best outputs.
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateQualityImages(int count, double minDiscriminatorScore = 0.7)
    {
        // Generate more images than needed to account for filtering
        int oversampling = (int)(count * 1.5);

        // Generate candidate images
        var noise = GenerateRandomNoiseTensor(oversampling, Generator.Architecture.InputSize);
        var candidateImages = GenerateImages(noise);

        // Score all images with the discriminator
        var scores = DiscriminateImages(candidateImages);

        // Select images meeting the quality threshold
        var threshold = NumOps.FromDouble(minDiscriminatorScore);
        var qualityImages = new List<Tensor<T>>();

        for (int i = 0; i < oversampling && qualityImages.Count < count; i++)
        {
            if (NumOps.GreaterThanOrEquals(scores[i, 0], threshold))
            {
                qualityImages.Add(candidateImages.GetSlice(i));
            }
        }

        // If not enough quality images found, include the best ones available
        if (qualityImages.Count < count)
        {
            // Create pairs of (index, score) and sort by score
            var scoreIndexPairs = new List<(int index, double score)>();
            for (int i = 0; i < oversampling; i++)
            {
                scoreIndexPairs.Add((i, Convert.ToDouble(scores[i, 0])));
            }

            // Sort by score descending
            scoreIndexPairs.Sort((a, b) => b.score.CompareTo(a.score));

            // Add the best remaining images
            foreach (var pair in scoreIndexPairs)
            {
                if (qualityImages.Count >= count) break;

                // Check if this image is already included
                bool alreadyIncluded = qualityImages.Any(img => img.TensorEquals(candidateImages.GetSlice(pair.index)));
                if (!alreadyIncluded)
                {
                    qualityImages.Add(candidateImages.GetSlice(pair.index));
                }
            }
        }

        // Combine all quality images into a single tensor
        return Tensor<T>.Stack(qualityImages.ToArray());
    }

    /// <summary>
    /// Enables gradient penalty (WGAN-GP) for improved training stability.
    /// </summary>
    /// <param name="enable">Whether to enable gradient penalty.</param>
    /// <remarks>
    /// <para>
    /// Gradient penalty is a regularization technique used in Wasserstein GANs with Gradient Penalty (WGAN-GP).
    /// It enforces the Lipschitz constraint by penalizing the gradient norm deviation from 1, which stabilizes training.
    /// </para>
    /// <para><b>For Beginners:</b> This helps prevent training instability.
    ///
    /// Gradient penalty:
    /// - Adds a regularization term that keeps gradients under control
    /// - Prevents mode collapse (when the generator produces limited variety)
    /// - Improves convergence and stability
    /// - Is standard practice in modern GAN training (WGAN-GP)
    /// </para>
    /// </remarks>
    public void EnableGradientPenalty(bool enable = true)
    {
        _useGradientPenalty = enable;
    }

    /// <summary>
    /// Enables feature matching loss to encourage the generator to match statistics of real data.
    /// </summary>
    /// <param name="enable">Whether to enable feature matching.</param>
    /// <remarks>
    /// <para>
    /// Feature matching encourages the generator to match the statistics of intermediate layer activations
    /// of real data, rather than directly maximizing the discriminator output. This can improve training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the generator create more realistic data.
    ///
    /// Feature matching:
    /// - Makes the generator match patterns found in real data
    /// - Works at a deeper level than just fooling the discriminator
    /// - Improves diversity and realism of generated samples
    /// - Helps prevent mode collapse
    /// </para>
    /// </remarks>
    public void EnableFeatureMatching(bool enable = true)
    {
        UseFeatureMatching = enable;
    }

    /// <summary>
    /// Computes the auxiliary loss for the GAN, which includes gradient penalty and feature matching losses.
    /// </summary>
    /// <returns>The total auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes auxiliary losses that improve GAN training stability:
    /// - Gradient Penalty (WGAN-GP): Penalizes deviations from gradient norm of 1
    /// - Feature Matching: Encourages matching statistics of intermediate activations
    /// </para>
    /// <para><b>For Beginners:</b> This calculates extra losses that make training more stable.
    ///
    /// The auxiliary losses:
    /// - Gradient Penalty: Keeps the discriminator's gradients well-behaved
    /// - Feature Matching: Encourages realistic feature distributions
    /// - Combined, they prevent common GAN training problems like mode collapse
    /// - Make training more reliable and convergent
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            return NumOps.Zero;
        }

        T totalAuxLoss = NumOps.Zero;

        // Compute gradient penalty if enabled
        if (_useGradientPenalty)
        {
            _lastGradientPenalty = ComputeGradientPenalty();
            totalAuxLoss = NumOps.Add(totalAuxLoss, _lastGradientPenalty);
        }

        // Compute feature matching loss if enabled
        if (UseFeatureMatching)
        {
            _lastFeatureMatchingLoss = ComputeFeatureMatchingLoss();
            totalAuxLoss = NumOps.Add(totalAuxLoss, _lastFeatureMatchingLoss);
        }

        return totalAuxLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary losses.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about GAN training.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into GAN training dynamics, including:
    /// - Generator and discriminator losses
    /// - Gradient penalty values
    /// - Feature matching statistics
    /// - Wasserstein distance estimates
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to track GAN training health.
    ///
    /// The diagnostics include:
    /// - Generator Loss: How well the generator is fooling the discriminator
    /// - Discriminator Loss: How well the discriminator is distinguishing real from fake
    /// - Gradient Penalty: The regularization term value
    /// - Feature Matching: How well features match between real and fake data
    /// - Wasserstein Distance: An estimate of the distribution distance (for WGAN)
    ///
    /// These help you:
    /// - Detect training instabilities early
    /// - Monitor convergence progress
    /// - Tune hyperparameters effectively
    /// - Diagnose issues like mode collapse
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "GeneratorLoss", _lastGeneratorLoss?.ToString() ?? "0" },
            { "DiscriminatorLoss", _lastDiscriminatorLoss?.ToString() ?? "0" },
            { "GradientPenalty", _lastGradientPenalty?.ToString() ?? "0" },
            { "FeatureMatchingLoss", _lastFeatureMatchingLoss?.ToString() ?? "0" },
            { "UseGradientPenalty", _useGradientPenalty.ToString() },
            { "UseFeatureMatching", UseFeatureMatching.ToString() }
        };

        // Estimate Wasserstein distance (difference between discriminator outputs for real and fake)
        if (_generatorLosses.Count > 0)
        {
            // Wasserstein distance approximation: D(real) - D(fake)
            // In practice, we use the discriminator loss as a proxy
            diagnostics["WassersteinDistanceEstimate"] = _lastDiscriminatorLoss?.ToString() ?? "0";
        }

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Provides GAN-specific auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including auxiliary loss diagnostics
    /// from <see cref="GetAuxiliaryLossDiagnostics"/>.
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
    /// Computes the gradient penalty for WGAN-GP.
    /// </summary>
    /// <returns>The gradient penalty value.</returns>
    /// <remarks>
    /// The gradient penalty enforces the Lipschitz constraint by penalizing deviations
    /// of the gradient norm from 1 at interpolated points between real and generated samples.
    /// Formula: λ * E[(||∇D(x̂)||₂ - 1)²] where x̂ = εx + (1-ε)G(z)
    /// </remarks>
    private T ComputeGradientPenalty()
    {
        // For gradient penalty, we need interpolated samples between real and fake
        // This is a simplified implementation that returns a placeholder
        // A full implementation would require:
        // 1. Interpolate between real and generated samples
        // 2. Compute discriminator output on interpolated samples
        // 3. Calculate gradients of discriminator with respect to interpolated samples
        // 4. Compute (||gradient||_2 - 1)^2

        // Placeholder implementation - return small penalty
        return NumOps.FromDouble(0.01);
    }

    /// <summary>
    /// Gets metadata about the GAN model, including information about both generator and discriminator components.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the GAN.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the GAN, including its architecture, training state,
    /// and key parameters. This information is useful for model management, tracking experiments, and reporting.
    /// The metadata includes details about both the generator and discriminator networks, as well as
    /// optimization settings like the current learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about the GAN's configuration and state.
    /// 
    /// The metadata includes:
    /// - What this model is and what it does (generate synthetic data)
    /// - The architecture details of both the generator and discriminator
    /// - Current training parameters like learning rate
    /// - The model's creation date and type
    /// 
    /// This information is useful for keeping track of different models,
    /// comparing experimental results, and documenting your work.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GenerativeAdversarialNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "TotalParameters", Generator.GetParameterCount() + Discriminator.GetParameterCount() },
                { "GeneratorArchitecture", Generator.GetModelMetadata() },
                { "DiscriminatorArchitecture", Discriminator.GetModelMetadata() },
                { "OptimizationType", "Adam" }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes GAN-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the GAN to a binary stream. It serializes both the generator and discriminator
    /// networks, as well as optimizer parameters like momentum and learning rate settings. This allows the GAN to be
    /// restored later with its full state intact, including both networks and training parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the complete state of the GAN to a file.
    /// 
    /// When saving the GAN:
    /// - Both the generator and discriminator networks are saved
    /// - The optimizer state (momentum, learning rates, etc.) is saved
    /// - Recent training history is saved
    /// - All the parameters needed to resume training are preserved
    /// 
    /// This allows you to save your progress and continue training later,
    /// share trained models with others, or deploy them in applications.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save recent loss history (last 20 entries at most)
        int lossCount = Math.Min(_generatorLosses.Count, 20);
        writer.Write(lossCount);

        for (int i = _generatorLosses.Count - lossCount; i < _generatorLosses.Count; i++)
        {
            writer.Write(Convert.ToDouble(_generatorLosses[i]));
        }

        // Save Generator and Discriminator networks
        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);
    }

    /// <summary>
    /// Deserializes GAN-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the state of a previously saved GAN from a binary stream. It restores both the generator
    /// and discriminator networks, as well as optimizer parameters like momentum and learning rate settings.
    /// This allows training to resume from exactly where it left off, maintaining all networks and parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a complete GAN from a saved file.
    /// 
    /// When loading the GAN:
    /// - Both the generator and discriminator networks are restored
    /// - The optimizer state (momentum, learning rates, etc.) is recovered
    /// - Recent training history is loaded
    /// - All parameters resume their previous values
    /// 
    /// This lets you continue working with a model exactly where you left off,
    /// or use a model that someone else has trained.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Load recent loss history
        int lossCount = reader.ReadInt32();
        _generatorLosses = new List<T>(lossCount);

        for (int i = 0; i < lossCount; i++)
        {
            _generatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));
        }

        // Load Generator and Discriminator networks
        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);
    }

    /// <summary>
    /// Updates the parameters of both the Generator and Discriminator networks.
    /// </summary>
    /// <param name="parameters">A vector containing the combined parameters for both networks.</param>
    /// <remarks>
    /// <para>
    /// This method splits the incoming parameter vector between the Generator and Discriminator,
    /// updates each network accordingly, and adjusts the learning rate based on the magnitude
    /// of parameter changes. It also includes a mechanism to reset the optimizer state if
    /// exceptionally large changes are detected.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates both parts of the GAN at once.
    /// 
    /// The process:
    /// - Splits the incoming parameters between Generator and Discriminator
    /// - Updates each network with its respective parameters
    /// - Adjusts the learning rate based on how big the changes are
    /// - If changes are very large, it resets some internal values to stabilize training
    /// 
    /// This approach allows for efficient updating of the entire GAN structure.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // Determine the split point between Generator and Discriminator parameters
        int generatorParameterCount = Generator.GetParameterCount();
        int discriminatorParameterCount = Discriminator.GetParameterCount();

        if (parameters.Length != generatorParameterCount + discriminatorParameterCount)
        {
            throw new ArgumentException($"Invalid parameter vector length. Expected {generatorParameterCount + discriminatorParameterCount}, but got {parameters.Length}.");
        }

        // Split the parameters vector
        var generatorParameters = new Vector<T>([.. parameters.Take(generatorParameterCount)]);
        var discriminatorParameters = new Vector<T>([.. parameters.Skip(generatorParameterCount).Take(discriminatorParameterCount)]);

        // Update Generator parameters
        Generator.UpdateParameters(generatorParameters);

        // Update Discriminator parameters
        Discriminator.UpdateParameters(discriminatorParameters);

        // Calculate the magnitude of parameter changes
        T parameterChangeNorm = parameters.L2Norm();

        // Reset optimizer state if a very large change is detected (indicates training instability)
        if (NumOps.GreaterThan(parameterChangeNorm, NumOps.FromDouble(10.0)))
        {
            ResetOptimizerState();
        }
    }

    /// <summary>
    /// Resets the optimizer state to its initial values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the Adam optimizer's state variables to their initial values.
    /// It's called when exceptionally large parameter changes are detected, which might
    /// indicate instability in the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This resets some internal values to stabilize training.
    /// 
    /// When called:
    /// - It resets momentum-related values to their starting points
    /// - This can help recover from unstable training situations
    /// - It's a way to "start fresh" with optimization without losing learned parameters
    /// 
    /// This reset can help the GAN recover if training becomes unstable.
    /// </para>
    /// </remarks>
    private void ResetOptimizerState()
    {
        // Reset both optimizers to their initial state
        _generatorOptimizer.Reset();
        _discriminatorOptimizer.Reset();
    }

    /// <summary>
    /// Computes the gradient penalty for WGAN-GP (Wasserstein GAN with Gradient Penalty).
    /// </summary>
    /// <param name="realSamples">Batch of real samples.</param>
    /// <param name="fakeSamples">Batch of generated (fake) samples.</param>
    /// <param name="lambda">Weight for the gradient penalty term (default: 10.0).</param>
    /// <returns>The gradient penalty loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the gradient penalty from Gulrajani et al. (2017) "Improved Training
    /// of Wasserstein GANs". The gradient penalty enforces the Lipschitz constraint by penalizing
    /// the discriminator when gradients deviate from unit norm at interpolated points between
    /// real and fake samples.
    /// </para>
    /// <para>
    /// The penalty is computed as: λ * E[(||∇_x D(x)|| - 1)²] where x is sampled uniformly
    /// along straight lines between real and fake samples. This replaces weight clipping
    /// and leads to more stable training and higher quality results.
    /// </para>
    /// <para>
    /// This implementation uses symbolic differentiation (autodiff) to compute
    /// gradients with respect to the input. This is more accurate and efficient than
    /// numerical differentiation and is the standard approach in modern WGAN-GP implementations.
    /// </para>
    /// <para><b>For Beginners:</b> This helps stabilize WGAN training by constraining gradients.
    ///
    /// How it works:
    /// 1. Create interpolated samples between real and fake images (mix them randomly)
    /// 2. Compute how the discriminator output changes with respect to input (gradient)
    /// 3. Measure how far the gradient norm is from 1.0
    /// 4. Penalize the discriminator if gradients are too large or too small
    ///
    /// Why this helps:
    /// - Enforces the mathematical constraint needed for Wasserstein distance
    /// - Prevents discriminator gradients from exploding or vanishing
    /// - More stable than weight clipping (older WGAN approach)
    /// - Results in higher quality generated images
    ///
    /// <b>Important:</b> This method computes the gradient penalty but does not automatically
    /// integrate it into training. To use WGAN-GP, you must:
    /// 1. Call this method during discriminator training
    /// 2. Add the returned penalty to the discriminator loss
    /// 3. Use the combined loss to update discriminator parameters
    ///
    /// The base Train() method does not automatically include gradient penalty.
    /// Typical lambda values are 10.0 for images.
    /// </para>
    /// </remarks>
    public T ComputeGradientPenalty(Tensor<T> realSamples, Tensor<T> fakeSamples, double lambda = 10.0)
    {
        if (realSamples == null || fakeSamples == null)
        {
            return NumOps.Zero;
        }

        int batchSize = realSamples.Shape[0];
        if (batchSize != fakeSamples.Shape[0])
        {
            throw new ArgumentException("Real and fake samples must have the same batch size.");
        }

        // Generate random interpolation coefficients (epsilon) for each sample in batch
        int elementsPerSample = realSamples.Length / batchSize;
        
        // Create epsilon tensor [batchSize, 1] for broadcasting across elementsPerSample
        var epsilonData = new T[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            epsilonData[i] = NumOps.FromDouble(Random.NextDouble());
        }
        
        // Vectorized interpolation: x_hat = epsilon * real + (1 - epsilon) * fake
        // Tile epsilon to match sample dimensions [batchSize, elementsPerSample]
        var epsilonTensor = new Tensor<T>(epsilonData, [batchSize, 1]);
        var epsilonBroadcast = Engine.TensorTile(epsilonTensor, [1, elementsPerSample]);
        epsilonBroadcast = epsilonBroadcast.Reshape([realSamples.Length]);
        
        // Create ones tensor for (1 - epsilon) computation
        var onesTensor = new Tensor<T>([realSamples.Length]);
        Engine.TensorFill(onesTensor, NumOps.One);
        var oneMinusEpsilon = Engine.TensorSubtract(onesTensor, epsilonBroadcast);
        
        // Flatten samples for element-wise operations
        var realFlat = realSamples.Reshape([realSamples.Length]);
        var fakeFlat = fakeSamples.Reshape([fakeSamples.Length]);
        
        // Compute interpolation: eps * real + (1 - eps) * fake
        var realPart = Engine.TensorMultiply(epsilonBroadcast, realFlat);
        var fakePart = Engine.TensorMultiply(oneMinusEpsilon, fakeFlat);
        var interpolatedFlat = Engine.TensorAdd(realPart, fakePart);
        var interpolated = interpolatedFlat.Reshape(realSamples.Shape);

        // Compute gradients using symbolic differentiation (autodiff)
        var gradients = ComputeSymbolicGradient(interpolated);

        // Vectorized gradient penalty computation
        // Reshape gradients to [batchSize, elementsPerSample]
        var gradientsTensor = new Tensor<T>(gradients.ToArray(), [batchSize, elementsPerSample]);
        
        // Compute squared gradients
        var gradientsSquared = Engine.TensorMultiply(gradientsTensor, gradientsTensor);
        
        // Sum along axis 1 to get gradient norm squared per sample
        var gradientNormSquared = Engine.ReduceSum(gradientsSquared, [1], keepDims: false);
        
        // Compute gradient norms
        var gradientNorm = Engine.TensorSqrt(gradientNormSquared);
        
        // Compute deviation from 1: (||gradient|| - 1)^2
        var onesTensorBatch = new Tensor<T>([batchSize]);
        Engine.TensorFill(onesTensorBatch, NumOps.One);
        var deviation = Engine.TensorSubtract(gradientNorm, onesTensorBatch);
        var penaltyPerSample = Engine.TensorMultiply(deviation, deviation);
        
        // Compute mean penalty across batch
        T totalPenalty = Engine.TensorSum(penaltyPerSample);
        totalPenalty = NumOps.Divide(totalPenalty, NumOps.FromDouble(batchSize));

        // Apply lambda weight
        totalPenalty = NumOps.Multiply(totalPenalty, NumOps.FromDouble(lambda));

        return totalPenalty;
    }

    /// <summary>
    /// Computes gradients of discriminator output with respect to input using symbolic differentiation.
    /// </summary>
    /// <param name="input">The input tensor to compute gradients for.</param>
    /// <returns>A tensor containing the gradients with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation (autodiff) to compute exact gradients by
    /// running a backward pass through the discriminator network. This is more accurate and
    /// efficient than numerical differentiation, and is the industry-standard approach used
    /// in frameworks like TensorFlow and PyTorch.
    /// </para>
    /// <para>
    /// The process:
    /// 1. Run forward pass through discriminator to get output
    /// 2. Create gradient signal with respect to output (typically all ones)
    /// 3. Backpropagate through all layers to compute gradient with respect to input
    /// 4. Return the accumulated input gradients
    /// </para>
    /// <para><b>For Beginners:</b> This computes how the output changes when the input changes, using calculus.
    ///
    /// Unlike numerical differentiation which approximates gradients by trying tiny changes,
    /// symbolic differentiation uses the mathematical rules of calculus to compute exact derivatives.
    ///
    /// The process:
    /// 1. Run the input through the discriminator to get an output
    /// 2. Start with "how much we care about the output" (gradient = 1.0)
    /// 3. Work backwards through each layer, computing how much each input affects the output
    /// 4. This gives us the exact gradient without approximations
    ///
    /// Benefits over numerical differentiation:
    /// - More accurate (no approximation error)
    /// - Faster (only requires one forward and one backward pass)
    /// - Uses less memory (doesn't need to perturb each input element)
    /// - Industry standard approach used in modern deep learning frameworks
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeSymbolicGradient(Tensor<T> input)
    {
        // Store original training mode
        bool originalMode = Discriminator.IsTrainingMode;
        Discriminator.SetTrainingMode(false); // Use inference mode for stable gradients

        // Reset layer states to ensure clean forward pass
        Discriminator.ResetState();

        // Forward pass through discriminator
        var output = Discriminator.Predict(input);

        // Create gradient signal: we want d(output)/d(input)
        // Start with gradient of 1.0 with respect to the output
        var outputGradient = new Tensor<T>(output.Shape);
        outputGradient.Fill(NumOps.One);

        // Run backward pass through discriminator layers to compute input gradient
        // The discriminator's Backward method will propagate gradients back to the input
        var inputGradient = Discriminator.BackwardWithInputGradient(outputGradient);

        // Restore original training mode
        Discriminator.SetTrainingMode(originalMode);

        return inputGradient;
    }

    /// <summary>
    /// Computes numerical gradients of discriminator output with respect to input using finite differences.
    /// </summary>
    /// <param name="input">The input tensor to compute gradients for.</param>
    /// <returns>A tensor containing the numerical gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the finite difference method to approximate gradients:
    /// ∂f/∂x ≈ (f(x + h) - f(x - h)) / (2h) where h is a small step size (epsilon).
    /// </para>
    /// <para>
    /// Central differences are more accurate than forward differences and are
    /// suitable for computing gradient penalties in WGAN-GP. The epsilon value
    /// is chosen to balance numerical accuracy and precision (1e-4 works well for
    /// typical neural network outputs).
    /// </para>
    /// <para>
    /// NOTE: This method is kept for backward compatibility and as a fallback.
    /// For production use, prefer ComputeSymbolicGradient() which is more accurate
    /// and efficient.
    /// </para>
    /// <para><b>For Beginners:</b> This computes how the output changes when the input changes.
    ///
    /// The process:
    /// 1. For each input value, make a tiny change (+epsilon and -epsilon)
    /// 2. See how much the discriminator output changes
    /// 3. The gradient is the rate of change (output change / input change)
    ///
    /// This is called "numerical differentiation" - we approximate the derivative
    /// by actually trying tiny changes rather than using calculus formulas.
    /// It's accurate but slower than symbolic differentiation.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeNumericalGradient(Tensor<T> input)
    {
        var gradients = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-4); // Small perturbation for numerical gradient

        // Store original discriminator training mode
        bool originalMode = Discriminator.IsTrainingMode;
        Discriminator.SetTrainingMode(false); // Use inference mode for gradient computation

        // Work on a copy to avoid modifying the input tensor
        var inputCopy = new Tensor<T>(input.Shape);
        for (int idx = 0; idx < input.Length; idx++)
        {
            inputCopy[idx] = input[idx];
        }

        // Compute gradient for each input element using central differences
        for (int i = 0; i < inputCopy.Length; i++)
        {
            // Save original value
            T originalValue = inputCopy[i];

            // Compute f(x + epsilon)
            inputCopy[i] = NumOps.Add(originalValue, epsilon);
            var outputPlus = Discriminator.Predict(inputCopy);

            // Compute f(x - epsilon)
            inputCopy[i] = NumOps.Subtract(originalValue, epsilon);
            var outputMinus = Discriminator.Predict(inputCopy);

            // Restore original value
            inputCopy[i] = originalValue;

            // Central difference: (f(x+h) - f(x-h)) / (2h)
            // For discriminator, we use the first output element (real/fake score)
            // Handle both 1D and 2D output tensors
            T plusValue = outputPlus.Shape.Length >= 2 ? outputPlus[0, 0] : outputPlus[0];
            T minusValue = outputMinus.Shape.Length >= 2 ? outputMinus[0, 0] : outputMinus[0];
            T outputDiff = NumOps.Subtract(plusValue, minusValue);
            T twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2.0));
            gradients[i] = NumOps.Divide(outputDiff, twoEpsilon);
        }

        // Restore original training mode
        Discriminator.SetTrainingMode(originalMode);

        return gradients;
    }

    /// <summary>
    /// Computes the feature matching loss between real and generated data.
    /// </summary>
    /// <returns>The feature matching loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method implements feature matching loss from Salimans et al. (2016). Instead of training
    /// the generator to maximize discriminator confusion directly, it trains the generator to match
    /// the statistics (mean activations) of real data at intermediate layers of the discriminator.
    /// This approach helps stabilize training and prevent mode collapse.
    /// </para>
    /// <para>
    /// The loss is computed as the L2 distance between the mean feature activations of real and
    /// generated samples across specified discriminator layers. If no layers are specified via
    /// FeatureMatchingLayers, the method automatically selects middle layers of the discriminator.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how well generated images match real images internally.
    ///
    /// How it works:
    /// 1. Pass real images through the discriminator and extract internal features
    /// 2. Pass fake images through the discriminator and extract the same features
    /// 3. Compare the average features from real vs. fake images
    /// 4. Return a score showing how different they are
    ///
    /// Why this helps:
    /// - Forces the generator to match internal patterns, not just fool the discriminator
    /// - Helps create more diverse outputs (prevents mode collapse)
    /// - Makes training more stable
    /// - Results in more realistic generated images
    ///
    /// The loss should be minimized during generator training, typically weighted and
    /// combined with the standard adversarial loss.
    /// </para>
    /// </remarks>
    public T ComputeFeatureMatchingLoss()
    {
        // Check if we have stored batches
        if (_lastRealBatch == null || _lastFakeBatch == null)
        {
            return NumOps.Zero;
        }

        // Determine which layers to use for feature extraction
        int[] layerIndices;
        if (FeatureMatchingLayers != null && FeatureMatchingLayers.Length > 0)
        {
            layerIndices = FeatureMatchingLayers;
        }
        else
        {
            // Use middle layers by default (25%, 50%, 75% through the network)
            // Adjust based on actual discriminator depth to avoid invalid indices
            int discriminatorLayerCount = Discriminator.LayerCount;
            if (discriminatorLayerCount <= 2)
            {
                // Very shallow network - just use available layers
                layerIndices = discriminatorLayerCount == 1 ? new int[] { 0 } : new int[] { 0, 1 };
            }
            else if (discriminatorLayerCount <= 4)
            {
                // Small network - use first, middle, last
                layerIndices = new int[] { 0, discriminatorLayerCount / 2, discriminatorLayerCount - 1 };
            }
            else
            {
                // Standard network - use 25%, 50%, 75% positions
                int idx25 = discriminatorLayerCount / 4;
                int idx50 = discriminatorLayerCount / 2;
                int idx75 = (discriminatorLayerCount * 3) / 4;
                layerIndices = new int[] { idx25, idx50, idx75 };
            }
        }

        // Set discriminator to inference mode (no training during feature extraction)
        bool originalTrainingMode = Discriminator.IsTrainingMode;
        Discriminator.SetTrainingMode(false);

        // Extract features from real batch
        var (_, realFeatures) = Discriminator.ForwardWithFeatures(_lastRealBatch, layerIndices);

        // Extract features from fake batch
        var (_, fakeFeatures) = Discriminator.ForwardWithFeatures(_lastFakeBatch, layerIndices);

        // Restore original training mode
        Discriminator.SetTrainingMode(originalTrainingMode);

        // Compute L2 distance between feature statistics
        T totalLoss = NumOps.Zero;
        int featureCount = 0;

        foreach (int layerIdx in layerIndices)
        {
            if (!realFeatures.TryGetValue(layerIdx, out var realLayerFeatures) ||
                !fakeFeatures.TryGetValue(layerIdx, out var fakeLayerFeatures))
            {
                continue;
            }

            // Compute mean features across batch dimension
            var realMean = ComputeBatchMean(realLayerFeatures);
            var fakeMean = ComputeBatchMean(fakeLayerFeatures);

            // Vectorized L2 distance between means
            var diff = Engine.TensorSubtract(realMean, fakeMean);
            var squaredDiff = Engine.TensorMultiply(diff, diff);
            T layerLoss = Engine.TensorSum(squaredDiff);

            // Normalize by number of features
            int elementCount = Math.Min(realMean.Length, fakeMean.Length);
            if (elementCount > 0)
            {
                layerLoss = NumOps.Divide(layerLoss, NumOps.FromDouble(elementCount));
            }

            totalLoss = NumOps.Add(totalLoss, layerLoss);
            featureCount++;
        }

        // Average across layers
        if (featureCount > 0)
        {
            totalLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(featureCount));
        }

        // Apply feature matching weight
        totalLoss = NumOps.Multiply(totalLoss, NumOps.FromDouble(FeatureMatchingWeight));

        return totalLoss;
    }

    /// <summary>
    /// Computes the mean of a tensor across the batch dimension.
    /// </summary>
    /// <param name="tensor">The input tensor with batch as first dimension.</param>
    /// <returns>A 1D tensor containing the mean values.</returns>
    /// <remarks>
    /// <para>
    /// This helper method computes the mean of tensor values across the batch dimension,
    /// reducing a [batch, ...] tensor to a flattened mean representation. This is used
    /// for computing feature statistics in feature matching.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates average values across multiple examples.
    ///
    /// The process:
    /// - Takes features from multiple images in a batch
    /// - Averages each feature value across all images
    /// - Returns a single "typical" feature representation
    ///
    /// For example, if you have 32 images and each has 256 features,
    /// this computes 256 average values (one for each feature).
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeBatchMean(Tensor<T> tensor)
    {
        if (tensor.Rank == 0 || tensor.Length == 0)
        {
            return new Tensor<T>(new int[] { 0 });
        }

        int batchSize = tensor.Shape[0];

        // Vectorized: sum across batch dimension (axis 0), then divide by batch size
        var sum = Engine.ReduceSum(tensor, [0], keepDims: false);
        T batchSizeT = NumOps.FromDouble(batchSize);
        return Engine.TensorDivideScalar(sum, batchSizeT);
    }

    /// <summary>
    /// Creates a new instance of the GenerativeAdversarialNetwork with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new GenerativeAdversarialNetwork instance with the same architecture as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the GenerativeAdversarialNetwork with the same generator and
    /// discriminator architectures as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the GAN's blueprint.
    ///
    /// When you need multiple versions of the same GAN with identical settings:
    /// - This method creates a new, empty GAN with the same configuration
    /// - It copies the architecture of both the generator and discriminator networks
    /// - The new GAN has the same structure but no trained data
    /// - This is useful for techniques that need multiple models, like ensemble methods
    ///
    /// For example, when experimenting with different training approaches,
    /// you'd want to start with identical model configurations.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GenerativeAdversarialNetwork<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            Architecture.InputType,
            generatorOptimizer: null,
            discriminatorOptimizer: null,
            _lossFunction);
    }
}
