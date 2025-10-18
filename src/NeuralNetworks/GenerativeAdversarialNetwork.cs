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
public class GenerativeAdversarialNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the momentum values for the optimizer.
    /// </summary>
    /// <value>A vector of momentum values for each parameter.</value>
    /// <remarks>
    /// <para>
    /// Momentum is an optimization technique that helps accelerate gradient descent in the relevant direction
    /// and dampens oscillations. It does this by adding a fraction of the previous parameter update to the
    /// current update. This helps the optimizer converge faster and avoid getting stuck in local minima.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the networks learn more smoothly.
    /// 
    /// Think of momentum as:
    /// - A ball rolling down a hill that builds up speed
    /// - It helps the network keep moving in a consistent direction
    /// - It smooths out the learning process, preventing wild changes
    /// - This makes training more stable and often faster
    /// 
    /// For example, if the network is consistently trying to adjust a parameter in
    /// the same direction, momentum helps it make bigger adjustments over time.
    /// </para>
    /// </remarks>
    private Vector<T> _momentum = default!;

    /// <summary>
    /// Gets or sets the second moment estimates for the Adam optimizer.
    /// </summary>
    /// <value>A vector of second moment values for each parameter.</value>
    /// <remarks>
    /// <para>
    /// The second moment estimates are used by the Adam optimizer to adapt the learning rate for each parameter.
    /// They track the squared gradients, providing a measure of how quickly each parameter is changing.
    /// Parameters that change more rapidly get smaller learning rates, and vice versa, which helps stabilize training.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the networks adjust their learning speed for different parts.
    /// 
    /// Think of second moments as:
    /// - A record of how wildly each part of the network has been changing
    /// - Parts that change a lot get smaller updates (to avoid instability)
    /// - Parts that change little get larger updates (to learn faster)
    /// - This adaptive approach helps GANs train more reliably
    /// 
    /// It's like automatically adjusting the sensitivity of different controls
    /// based on how jumpy they've been in the past.
    /// </para>
    /// </remarks>
    private Vector<T> _secondMoment = default!;

    /// <summary>
    /// Gets or sets the current value of beta1 raised to the power of the iteration count for Adam optimizer.
    /// </summary>
    /// <value>The current beta1 power value.</value>
    /// <remarks>
    /// <para>
    /// This value is used for bias correction in the Adam optimizer. The beta1 parameter controls the
    /// exponential decay rate for the first moment estimates (momentum). The power value is updated at
    /// each training step and helps correct the bias in the early stages of training.
    /// </para>
    /// <para><b>For Beginners:</b> This is a technical value that helps the optimizer work correctly.
    /// 
    /// Think of beta1Power as:
    /// - A correction factor for the optimizer
    /// - It helps make the early stages of training more accurate
    /// - Without it, the network might learn too slowly at the beginning
    /// - It's automatically adjusted during training
    /// 
    /// This is part of what makes modern optimizers like Adam so effective for
    /// training complex models like GANs.
    /// </para>
    /// </remarks>
    private T _beta1Power = default!;

    /// <summary>
    /// Gets or sets the current value of beta2 raised to the power of the iteration count for Adam optimizer.
    /// </summary>
    /// <value>The current beta2 power value.</value>
    /// <remarks>
    /// <para>
    /// This value is used for bias correction in the Adam optimizer. The beta2 parameter controls the
    /// exponential decay rate for the second moment estimates. The power value is updated at
    /// each training step and helps correct the bias in the early stages of training.
    /// </para>
    /// <para><b>For Beginners:</b> This is another technical value that helps the optimizer work correctly.
    /// 
    /// Think of beta2Power as:
    /// - A companion to beta1Power for the second moment estimates
    /// - It ensures the adaptive learning rates are accurate from the start
    /// - Without it, the adaptation might be too aggressive or too conservative initially
    /// - Like beta1Power, it's automatically adjusted during training
    /// 
    /// These correction factors are what make Adam one of the preferred optimizers
    /// for training GANs.
    /// </para>
    /// </remarks>
    private T _beta2Power = default!;

    /// <summary>
    /// Gets or sets the current learning rate for the optimizer.
    /// </summary>
    /// <value>A double representing the current learning rate.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines the step size at each iteration while moving toward a minimum of the loss function.
    /// In this implementation, the learning rate can decay over time and can be adapted based on training progress.
    /// Finding the right learning rate is critical for effective GAN training.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how big the adjustments are during training.
    /// 
    /// Think of the learning rate as:
    /// - The size of the steps the networks take when learning
    /// - Too large, and they might overshoot and never find the best solution
    /// - Too small, and training will take forever
    /// - In this implementation, it gradually decreases over time
    /// 
    /// The learning rate is one of the most important hyperparameters to tune
    /// when training GANs, as they can be notoriously unstable.
    /// </para>
    /// </remarks>
    private double _currentLearningRate = 0.001;

    /// <summary>
    /// Gets or sets the initial learning rate for the optimizer.
    /// </summary>
    /// <value>A double representing the initial learning rate.</value>
    /// <remarks>
    /// <para>
    /// The initial learning rate is the starting point for the optimizer's step size. It determines
    /// how large the initial parameter updates are during training. This value is typically reduced
    /// over time using learning rate decay to fine-tune the model as it approaches convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This sets the starting speed of learning for the networks.
    /// 
    /// Think of the initial learning rate as:
    /// - The initial step size the networks take when learning
    /// - A larger value means bigger initial steps (faster initial learning, but potentially unstable)
    /// - A smaller value means smaller initial steps (slower initial learning, but potentially more stable)
    /// - It's often reduced over time as the networks fine-tune their performance
    /// 
    /// Finding the right initial learning rate is crucial for effective GAN training,
    /// as it impacts both the speed of convergence and the stability of the training process.
    /// </para>
    /// </remarks>
    private double _initialLearningRate = 0.001;

    /// <summary>
    /// Gets or sets the rate at which the learning rate decays during training.
    /// </summary>
    /// <value>A double representing the learning rate decay factor.</value>
    /// <remarks>
    /// <para>
    /// The learning rate decay factor determines how quickly the learning rate decreases over time.
    /// A value close to 1 means the learning rate decreases very slowly, while a smaller value
    /// causes it to decrease more rapidly. Decreasing the learning rate over time can help the
    /// model converge to a more optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the step size shrinks during training.
    /// 
    /// Think of learning rate decay as:
    /// - A factor that gradually reduces the learning rate
    /// - At the beginning, large steps help explore the solution space quickly
    /// - As training progresses, smaller steps help fine-tune the solution
    /// - A value of 0.9999 means the learning rate decreases very slowly
    /// 
    /// This is like starting with bold brush strokes when painting, then gradually
    /// switching to finer brushes for the details.
    /// </para>
    /// </remarks>
    private double _learningRateDecay = 0.9999;

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
    private List<T> _generatorLosses = [];

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
    /// Initializes a new instance of the <see cref="GenerativeAdversarialNetwork{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.</param>
    /// <param name="initialLearningRate">The initial learning rate for the optimizer. Default is 0.001.</param>
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
    /// - The fitnessCalculator determines how performance is measured
    /// - The inputType specifies what kind of data the GAN will work with
    /// - The initialLearningRate controls how quickly the networks learn initially
    /// 
    /// Think of it like establishing the rules and roles for the forger and detective
    /// before their competition begins.
    /// </para>
    /// </remarks>
    public GenerativeAdversarialNetwork(NeuralNetworkArchitecture<T> generatorArchitecture, 
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.001)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative, 
            NetworkComplexity.Medium, 
            generatorArchitecture.InputSize, 
            discriminatorArchitecture.OutputSize, 
            0, 0, 0, 
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;
    
        // Initialize optimizer parameters
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;
    
        // Initialize tracking collections
        _generatorLosses = [];
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

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
    
        // ----- Train the generator -----
    
        // Generate new fake images for generator training
        Tensor<T> newFakeImages = GenerateImages(noise);
    
        // For generator training, we want the discriminator to think fake images are real
        Tensor<T> allRealLabels = CreateLabelTensor(batchSize, NumOps.One);
    
        // Train the generator to fool the discriminator
        T generatorLoss = TrainGeneratorBatch(noise, newFakeImages, allRealLabels);
    
        // Track generator loss for monitoring
        _generatorLosses.Add(generatorLoss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0); // Keep only recent losses
        }
    
        // ----- Adaptive learning rate adjustment -----
    
        // Adapt learning rate based on recent performance
        if (_generatorLosses.Count >= 20)
        {
            var recentAverage = _generatorLosses.Skip(_generatorLosses.Count - 10).Average(l => Convert.ToDouble(l));
            var previousAverage = _generatorLosses.Skip(_generatorLosses.Count - 20).Take(10).Average(l => Convert.ToDouble(l));
        
            // If loss is not improving or worsening, adjust learning rate
            if (recentAverage > previousAverage * 0.95)
            {
                _currentLearningRate *= 0.95; // Reduce learning rate by 5%
            }
            else if (recentAverage < previousAverage * 0.8)
            {
                // Loss is improving significantly, we can potentially increase learning rate slightly
                _currentLearningRate = Math.Min(_currentLearningRate * 1.05, 0.001); // Increase but cap
            }
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
    
        // Fill with the specified value
        for (int i = 0; i < batchSize; i++)
        {
            tensor[i, 0] = value;
        }
    
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
    /// Updates the parameters of a network using the Adam optimizer with the calculated gradients.
    /// </summary>
    /// <param name="network">The neural network to update.</param>
    /// <remarks>
    /// <para>
    /// This method applies the calculated gradients to update the parameters of the specified network
    /// using the Adam optimizer. It includes gradient clipping, momentum, and adaptive learning rates
    /// for stable and efficient training. This tensor-based implementation handles all parameters
    /// at once for better performance.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the network's internal values using an advanced algorithm.
    /// 
    /// The parameter update process:
    /// - Uses the Adam optimizer, which adapts to the training dynamics
    /// - Applies momentum to smooth updates and avoid oscillations
    /// - Includes adaptive learning rates for different parameters
    /// - Prevents excessively large updates that could destabilize training
    /// 
    /// This approach helps GANs train more reliably and efficiently.
    /// </para>
    /// </remarks>
    private void UpdateNetworkParameters(ConvolutionalNeuralNetwork<T> network)
    {
        // Get current parameters and gradients
        var parameters = network.GetParameters();
        var gradients = network.GetParameterGradients();
    
        // Initialize optimizer state if not already done
        if (_momentum == null || _momentum.Length != parameters.Length)
        {
            _momentum = new Vector<T>(parameters.Length);
            _momentum.Fill(NumOps.Zero);
        }
    
        if (_secondMoment == null || _secondMoment.Length != parameters.Length)
        {
            _secondMoment = new Vector<T>(parameters.Length);
            _secondMoment.Fill(NumOps.Zero);
        }
    
        // Gradient clipping to prevent exploding gradients
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);
    
        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
            }
        }
    
        // Adam optimizer parameters
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        var beta1 = NumOps.FromDouble(0.9);  // Momentum coefficient
        var beta2 = NumOps.FromDouble(0.999); // RMS coefficient
        var epsilon = NumOps.FromDouble(1e-8);
    
        // Updated parameters vector
        var updatedParameters = new Vector<T>(parameters.Length);
    
        // Apply Adam updates to all parameters at once
        for (int i = 0; i < parameters.Length; i++)
        {
            // Update momentum (first moment)
            _momentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _momentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );
        
            // Update second moment
            _secondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _secondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );
        
            // Bias correction
            var momentumCorrected = NumOps.Divide(_momentum[i], NumOps.Subtract(NumOps.One, _beta1Power));
            var secondMomentCorrected = NumOps.Divide(_secondMoment[i], NumOps.Subtract(NumOps.One, _beta2Power));
        
            // Adam update
            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(secondMomentCorrected), epsilon)
            );
        
            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, momentumCorrected)
            );
        }
    
        // Update beta powers for next iteration
        _beta1Power = NumOps.Multiply(_beta1Power, beta1);
        _beta2Power = NumOps.Multiply(_beta2Power, beta2);
    
        // Apply learning rate decay
        _currentLearningRate *= _learningRateDecay;
    
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
        metrics["CurrentLearningRate"] = _currentLearningRate;

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

        // ------------ Train Generator ------------

        // For generator training, we want discriminator to think fake images are real
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);

        // Train generator to fool discriminator using tensor operations
        var generatorLoss = TrainGeneratorBatch(input, allRealLabels);

        // Track generator loss for monitoring
        _generatorLosses.Add(generatorLoss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0);
        }

        // Adapt learning rate based on recent performance
        if (_generatorLosses.Count >= 20)
        {
            var recentAverage = _generatorLosses.Skip(_generatorLosses.Count - 10).Average(l => Convert.ToDouble(l));
            var previousAverage = _generatorLosses.Skip(_generatorLosses.Count - 20).Take(10).Average(l => Convert.ToDouble(l));

            // If loss is not improving or worsening, adjust learning rate
            if (recentAverage > previousAverage * 0.95)
            {
                _currentLearningRate *= 0.95; // Reduce learning rate by 5%
            }
            else if (recentAverage < previousAverage * 0.8)
            {
                // Loss is improving significantly, we can potentially increase learning rate slightly
                _currentLearningRate = Math.Min(_currentLearningRate * 1.05, 0.001); // Increase but cap
            }
        }

        LastLoss = generatorLoss;
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
        var random = new Random();
        var shape = new int[] { batchSize, noiseSize };
        var noise = new Tensor<T>(shape);
    
        // Generate normally distributed random numbers using Box-Muller transform
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i += 2)
            {
                double u1 = random.NextDouble(); // Uniform(0,1) random number
                double u2 = random.NextDouble(); // Uniform(0,1) random number
            
                // Box-Muller transformation
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;
            
                double z1 = radius * Math.Cos(theta);
                noise[b, i] = NumOps.FromDouble(z1);
            
                // If we're not at the last element, generate the second value
                if (i + 1 < noiseSize)
                {
                    double z2 = radius * Math.Sin(theta);
                    noise[b, i + 1] = NumOps.FromDouble(z2);
                }
            }
        }
    
        return noise;
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
        metrics["CurrentLearningRate"] = _currentLearningRate;
    
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
        return Tensor<T>.Stack([.. qualityImages]);
    }

    /// <summary>
    /// Gets metadata about the GAN model, including information about both generator and discriminator components.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the GAN.</returns>
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
        // Save learning rate parameters
        writer.Write(_currentLearningRate);
        writer.Write(_learningRateDecay);
    
        // Save optimizer state
        writer.Write(_beta1Power != null ? Convert.ToDouble(_beta1Power) : 1.0);
        writer.Write(_beta2Power != null ? Convert.ToDouble(_beta2Power) : 1.0);
    
        // Save momentum and second moment vectors
        if (_momentum != null && _momentum.Length > 0)
        {
            writer.Write(true); // Flag indicating momentum exists
            writer.Write(_momentum.Length);
        
            for (int i = 0; i < _momentum.Length; i++)
            {
                writer.Write(Convert.ToDouble(_momentum[i]));
            }
        }
        else
        {
            writer.Write(false); // No momentum saved
        }
    
        if (_secondMoment != null && _secondMoment.Length > 0)
        {
            writer.Write(true); // Flag indicating second moment exists
            writer.Write(_secondMoment.Length);
        
            for (int i = 0; i < _secondMoment.Length; i++)
            {
                writer.Write(Convert.ToDouble(_secondMoment[i]));
            }
        }
        else
        {
            writer.Write(false); // No second moment saved
        }
    
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
        // Load learning rate parameters
        _currentLearningRate = reader.ReadDouble();
        _learningRateDecay = reader.ReadDouble();

        // Load optimizer state
        _beta1Power = NumOps.FromDouble(reader.ReadDouble());
        _beta2Power = NumOps.FromDouble(reader.ReadDouble());

        // Load momentum if it exists
        bool hasMomentum = reader.ReadBoolean();
        if (hasMomentum)
        {
            int momentumLength = reader.ReadInt32();
            _momentum = new Vector<T>(momentumLength);
    
            for (int i = 0; i < momentumLength; i++)
            {
                _momentum[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
        else
        {
            _momentum = new Vector<T>(0);
        }

        // Load second moment if it exists
        bool hasSecondMoment = reader.ReadBoolean();
        if (hasSecondMoment)
        {
            int secondMomentLength = reader.ReadInt32();
            _secondMoment = new Vector<T>(secondMomentLength);
    
            for (int i = 0; i < secondMomentLength; i++)
            {
                _secondMoment[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
        else
        {
            _secondMoment = new Vector<T>(0);
        }

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

        // Adjust learning rate based on parameter change magnitude
        if (NumOps.GreaterThan(parameterChangeNorm, NumOps.FromDouble(1.0)))
        {
            _currentLearningRate *= 0.95; // Reduce learning rate if changes are large
        }
        else if (NumOps.LessThan(parameterChangeNorm, NumOps.FromDouble(0.01)))
        {
            _currentLearningRate = Math.Min(_currentLearningRate * 1.05, _initialLearningRate); // Increase learning rate if changes are small, but cap it at initial rate
        }

        // Reset optimizer state if a very large change is detected
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
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;
        _momentum = new Vector<T>(Generator.GetParameterCount() + Discriminator.GetParameterCount());
        _secondMoment = new Vector<T>(Generator.GetParameterCount() + Discriminator.GetParameterCount());
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
            _lossFunction,
            _initialLearningRate);
    }
}