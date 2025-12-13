namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Wasserstein GAN with Gradient Penalty (WGAN-GP), an improved version of WGAN
/// that uses gradient penalty instead of weight clipping to enforce the Lipschitz constraint.
/// </summary>
/// <remarks>
/// <para>
/// WGAN-GP improves upon WGAN by:
/// - Replacing weight clipping with a gradient penalty term
/// - Providing smoother and more stable training
/// - Avoiding pathological behavior caused by weight clipping
/// - Achieving better performance and convergence
/// - Eliminating the need to tune the clipping threshold
/// </para>
/// <para><b>For Beginners:</b> WGAN-GP is an enhanced version of WGAN with better training stability.
///
/// Key improvements over WGAN:
/// - Uses a "gradient penalty" instead of hard weight limits
/// - This penalty gently guides the critic to behave correctly
/// - More stable and reliable training
/// - Produces higher quality results
/// - Easier to use (fewer hyperparameters to tune)
///
/// The gradient penalty ensures the critic learns smoothly without the problems
/// that weight clipping can cause.
///
/// Reference: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class WGANGP<T> : NeuralNetworkBase<T>
{
    private Vector<T> _generatorMomentum;
    private Vector<T> _generatorSecondMoment;
    private Vector<T> _criticMomentum;
    private Vector<T> _criticSecondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate;
    private double _initialLearningRate;
    private readonly List<T> _criticLosses = [];
    private readonly List<T> _generatorLosses = [];

    /// <summary>
    /// The coefficient for the gradient penalty term in the loss function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The gradient penalty coefficient (lambda) controls how strongly the gradient penalty
    /// is enforced. A typical value is 10.0. Higher values enforce the Lipschitz constraint
    /// more strictly, while lower values allow more flexibility.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strict the gradient penalty is.
    ///
    /// Gradient penalty coefficient:
    /// - Typical value is 10.0
    /// - Higher values = stricter enforcement of the constraint
    /// - Lower values = more flexibility for the critic
    /// - The paper recommends 10.0 as a good default
    /// </para>
    /// </remarks>
    private double _gradientPenaltyCoefficient = 10.0;

    /// <summary>
    /// The number of critic training iterations per generator iteration.
    /// </summary>
    private int _criticIterations = 5;

    /// <summary>
    /// Gets the generator network that creates synthetic data.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the critic network that evaluates data quality.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Critic { get; private set; }

    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="WGANGP{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="criticArchitecture">The neural network architecture for the critic.</param>
    /// <param name="inputType">The type of input the WGAN-GP will process.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">The initial learning rate. Default is 0.0001.</param>
    /// <param name="gradientPenaltyCoefficient">The gradient penalty coefficient (lambda). Default is 10.0.</param>
    /// <param name="criticIterations">Number of critic iterations per generator iteration. Default is 5.</param>
    public WGANGP(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0001,
        double gradientPenaltyCoefficient = 10.0,
        int criticIterations = 5)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            generatorArchitecture.InputSize,
            criticArchitecture.OutputSize,
            0, 0, 0,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;
        _gradientPenaltyCoefficient = gradientPenaltyCoefficient;
        _criticIterations = criticIterations;

        // Initialize optimizer parameters
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Critic = new ConvolutionalNeuralNetwork<T>(criticArchitecture);

        _generatorMomentum = Vector<T>.Empty();
        _generatorSecondMoment = Vector<T>.Empty();
        _criticMomentum = Vector<T>.Empty();
        _criticSecondMoment = Vector<T>.Empty();

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for the WGAN-GP using tensor batches.
    /// </summary>
    /// <param name="realImages">A tensor containing real images.</param>
    /// <param name="noise">A tensor containing random noise for the generator.</param>
    /// <returns>A tuple containing the critic loss (including gradient penalty) and generator loss.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the WGAN-GP training algorithm:
    /// 1. Train the critic multiple times with gradient penalty
    /// 2. For each critic update, compute the gradient penalty on interpolated samples
    /// 3. Train the generator once to maximize the critic's score on fake images
    /// </para>
    /// <para><b>For Beginners:</b> One training round for WGAN-GP.
    ///
    /// The training process:
    /// - Trains the critic several times with gradient penalty
    /// - The gradient penalty keeps the critic well-behaved
    /// - Trains the generator once to improve
    /// - Returns loss values for monitoring progress
    /// </para>
    /// </remarks>
    public (T criticLoss, T generatorLoss) TrainStep(Tensor<T> realImages, Tensor<T> noise)
    {
        Generator.SetTrainingMode(true);
        Critic.SetTrainingMode(true);

        T totalCriticLoss = NumOps.Zero;

        // Train critic multiple times
        for (int i = 0; i < _criticIterations; i++)
        {
            // Generate fake images
            Tensor<T> fakeImages = GenerateImages(noise);

            // Get batch size
            int batchSize = realImages.Shape[0];

            // Train critic and get losses
            var (criticLoss, _) = TrainCriticBatchWithGP(realImages, fakeImages, batchSize);

            totalCriticLoss = NumOps.Add(totalCriticLoss, criticLoss);
        }

        // Average critic loss
        T avgCriticLoss = NumOps.Divide(totalCriticLoss, NumOps.FromDouble(_criticIterations));

        // Train generator
        Tensor<T> newNoise = GenerateRandomNoiseTensor(noise.Shape[0], Generator.Architecture.InputSize);
        T generatorLoss = TrainGeneratorBatch(newNoise);

        // Track losses
        _criticLosses.Add(avgCriticLoss);
        _generatorLosses.Add(generatorLoss);

        if (_criticLosses.Count > 100)
        {
            _criticLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (avgCriticLoss, generatorLoss);
    }

    /// <summary>
    /// Trains the critic on a batch with gradient penalty.
    /// </summary>
    /// <param name="realImages">The tensor containing real images.</param>
    /// <param name="fakeImages">The tensor containing generated fake images.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>A tuple containing the critic loss and gradient penalty value.</returns>
    /// <remarks>
    /// <para>
    /// The gradient penalty is computed on interpolated samples between real and fake images.
    /// For each sample, we interpolate between a real and fake image, then compute the gradient
    /// of the critic's output with respect to this interpolated input. The penalty encourages
    /// the norm of this gradient to be close to 1, which enforces the Lipschitz constraint.
    /// </para>
    /// <para><b>For Beginners:</b> This trains the critic with the gradient penalty.
    ///
    /// The gradient penalty process:
    /// - Creates "in-between" images by mixing real and fake
    /// - Checks how the critic responds to these mixed images
    /// - Penalizes the critic if its response is too extreme
    /// - This keeps the critic smooth and well-behaved
    /// </para>
    /// </remarks>
    private (T criticLoss, T gradientPenalty) TrainCriticBatchWithGP(
        Tensor<T> realImages,
        Tensor<T> fakeImages,
        int batchSize)
    {
        Critic.SetTrainingMode(true);

        // Forward pass on real images to compute scores
        var realScores = Critic.Predict(realImages);
        T realScore = NumOps.Zero;
        for (int i = 0; i < batchSize; i++)
        {
            realScore = NumOps.Add(realScore, realScores[i, 0]);
        }
        realScore = NumOps.Divide(realScore, NumOps.FromDouble(batchSize));

        // Forward pass on fake images to compute scores
        var fakeScores = Critic.Predict(fakeImages);
        T fakeScore = NumOps.Zero;
        for (int i = 0; i < batchSize; i++)
        {
            fakeScore = NumOps.Add(fakeScore, fakeScores[i, 0]);
        }
        fakeScore = NumOps.Divide(fakeScore, NumOps.FromDouble(batchSize));

        // Compute gradient penalty (this calls Predict on interpolated images which overwrites cache)
        var (gradientPenalty, gpParameterGradients) = ComputeGradientPenaltyWithGradients(realImages, fakeImages, batchSize);

        // Wasserstein loss with gradient penalty: -E[D(real)] + E[D(fake)] + lambda * GP
        T wassersteinDistance = NumOps.Subtract(realScore, fakeScore);
        T gpTerm = NumOps.Multiply(NumOps.FromDouble(_gradientPenaltyCoefficient), gradientPenalty);
        T criticLoss = NumOps.Add(NumOps.Negate(wassersteinDistance), gpTerm);

        // Create gradients for real images (maximize score)
        var realGradients = new Tensor<T>(realScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            realGradients[i, 0] = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        }

        // IMPORTANT: Re-run forward pass on real images before backprop
        // The GP computation called Predict(interpolated) which overwrote the cached activations
        Critic.Predict(realImages);
        Critic.Backpropagate(realGradients);
        var realParameterGradients = Critic.GetParameterGradients().Clone();

        // Create gradients for fake images (minimize score)
        var fakeGradients = new Tensor<T>(fakeScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            fakeGradients[i, 0] = NumOps.Divide(NumOps.Negate(NumOps.One), NumOps.FromDouble(batchSize));
        }

        // IMPORTANT: Re-run forward pass on fake images before backprop
        Critic.Predict(fakeImages);
        Critic.Backpropagate(fakeGradients);
        var fakeParameterGradients = Critic.GetParameterGradients().Clone();

        // Combine all gradients: real gradients + fake gradients + scaled GP gradients
        var combinedGradients = new Vector<T>(realParameterGradients.Length);
        T gpScale = NumOps.FromDouble(_gradientPenaltyCoefficient);
        for (int i = 0; i < combinedGradients.Length; i++)
        {
            // Sum all gradient contributions
            T realGrad = realParameterGradients[i];
            T fakeGrad = fakeParameterGradients[i];
            T gpGrad = NumOps.Multiply(gpScale, gpParameterGradients[i]);
            combinedGradients[i] = NumOps.Add(NumOps.Add(realGrad, fakeGrad), gpGrad);
        }

        // Update critic parameters with combined gradients
        UpdateCriticParametersWithGradients(combinedGradients);

        return (criticLoss, gradientPenalty);
    }

    /// <summary>
    /// Computes the gradient penalty and returns both the penalty value and the parameter gradients.
    /// </summary>
    private (T penalty, Vector<T> parameterGradients) ComputeGradientPenaltyWithGradients(
        Tensor<T> realImages,
        Tensor<T> fakeImages,
        int batchSize)
    {
        var random = new Random();

        // Create interpolated images
        var interpolatedImages = new Tensor<T>(realImages.Shape);

        // Compute number of elements per sample (excludes batch dimension)
        int sampleSize = realImages.Length / batchSize;

        for (int b = 0; b < batchSize; b++)
        {
            T epsilon = NumOps.FromDouble(random.NextDouble());

            for (int i = 0; i < sampleSize; i++)
            {
                int flatIdx = b * sampleSize + i;
                T realValue = realImages.GetFlat(flatIdx);
                T fakeValue = fakeImages.GetFlat(flatIdx);

                T interpolated = NumOps.Add(
                    NumOps.Multiply(epsilon, realValue),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, epsilon), fakeValue)
                );

                interpolatedImages.SetFlat(flatIdx, interpolated);
            }
        }

        // Forward pass through critic
        var interpolatedScores = Critic.Predict(interpolatedImages);

        // Create gradients of all ones (we want to compute d(score)/d(input))
        var ones = new Tensor<T>(interpolatedScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            ones[i, 0] = NumOps.One;
        }

        // Backpropagate to get gradients with respect to input
        var inputGradients = Critic.Backpropagate(ones);

        // Capture the parameter gradients from this backprop
        var gpParameterGradients = Critic.GetParameterGradients().Clone();

        // Compute L2 norm of gradients for each sample
        T totalPenalty = NumOps.Zero;
        int gradientSampleSize = inputGradients.Length / batchSize;

        for (int b = 0; b < batchSize; b++)
        {
            T gradNormSquared = NumOps.Zero;

            for (int i = 0; i < gradientSampleSize; i++)
            {
                int flatIdx = b * gradientSampleSize + i;
                T gradValue = inputGradients.GetFlat(flatIdx);
                gradNormSquared = NumOps.Add(gradNormSquared, NumOps.Multiply(gradValue, gradValue));
            }

            T gradNorm = NumOps.Sqrt(gradNormSquared);
            T deviation = NumOps.Subtract(gradNorm, NumOps.One);
            T penalty = NumOps.Multiply(deviation, deviation);

            totalPenalty = NumOps.Add(totalPenalty, penalty);
        }

        return (NumOps.Divide(totalPenalty, NumOps.FromDouble(batchSize)), gpParameterGradients);
    }

    /// <summary>
    /// Updates critic parameters with pre-computed combined gradients.
    /// </summary>
    private void UpdateCriticParametersWithGradients(Vector<T> gradients)
    {
        var parameters = Critic.GetParameters();

        if (_criticMomentum == null || _criticMomentum.Length != parameters.Length)
        {
            _criticMomentum = new Vector<T>(parameters.Length);
            _criticMomentum.Fill(NumOps.Zero);
        }

        if (_criticSecondMoment == null || _criticSecondMoment.Length != parameters.Length)
        {
            _criticSecondMoment = new Vector<T>(parameters.Length);
            _criticSecondMoment.Fill(NumOps.Zero);
        }

        // Gradient clipping
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

        // Adam optimizer with beta1=0 (paper recommendation)
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.9);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            _criticMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _criticMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _criticSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _criticSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(_criticSecondMoment[i]), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, _criticMomentum[i])
            );
        }

        Critic.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Computes the gradient penalty for WGAN-GP.
    /// </summary>
    /// <param name="realImages">The tensor containing real images.</param>
    /// <param name="fakeImages">The tensor containing fake images.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>The gradient penalty value.</returns>
    /// <remarks>
    /// <para>
    /// The gradient penalty is computed as follows:
    /// 1. Create interpolated samples between real and fake images
    /// 2. Compute the critic's gradient with respect to these interpolated samples
    /// 3. Compute the L2 norm of the gradients
    /// 4. Penalize deviation from norm = 1: (||grad|| - 1)^2
    /// </para>
    /// <para><b>For Beginners:</b> This measures how well-behaved the critic is.
    ///
    /// The gradient penalty:
    /// - Creates mixed images between real and fake
    /// - Checks how sensitive the critic is to small changes
    /// - Wants this sensitivity to be "just right" (not too much, not too little)
    /// - Returns a penalty if the critic is misbehaving
    /// </para>
    /// </remarks>
    private T ComputeGradientPenalty(Tensor<T> realImages, Tensor<T> fakeImages, int batchSize)
    {
        var random = new Random();

        // Create interpolated images
        var interpolatedImages = new Tensor<T>(realImages.Shape);

        // For each sample in the batch, create an interpolated version
        for (int b = 0; b < batchSize; b++)
        {
            // Random interpolation coefficient (epsilon) for this sample
            T epsilon = NumOps.FromDouble(random.NextDouble());

            // Interpolate: x_hat = epsilon * real + (1 - epsilon) * fake
            for (int i = 1; i < realImages.Shape.Length; i++)
            {
                T realValue = realImages[b, i - 1];
                T fakeValue = fakeImages[b, i - 1];

                T interpolated = NumOps.Add(
                    NumOps.Multiply(epsilon, realValue),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, epsilon), fakeValue)
                );

                interpolatedImages[b, i - 1] = interpolated;
            }
        }

        // Forward pass through critic
        var interpolatedScores = Critic.Predict(interpolatedImages);

        // Create gradients of all ones (we want to compute d(score)/d(input))
        var ones = new Tensor<T>(interpolatedScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            ones[i, 0] = NumOps.One;
        }

        // Backpropagate to get gradients with respect to input
        var inputGradients = Critic.Backpropagate(ones);

        // Compute L2 norm of gradients for each sample
        T totalPenalty = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            T gradNormSquared = NumOps.Zero;

            // Compute squared norm for this sample
            for (int i = 1; i < inputGradients.Shape.Length; i++)
            {
                T gradValue = inputGradients[b, i - 1];
                gradNormSquared = NumOps.Add(gradNormSquared, NumOps.Multiply(gradValue, gradValue));
            }

            T gradNorm = NumOps.Sqrt(gradNormSquared);

            // Penalty: (||grad|| - 1)^2
            T deviation = NumOps.Subtract(gradNorm, NumOps.One);
            T penalty = NumOps.Multiply(deviation, deviation);

            totalPenalty = NumOps.Add(totalPenalty, penalty);
        }

        // Average penalty across batch
        return NumOps.Divide(totalPenalty, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Trains the generator to fool the critic.
    /// </summary>
    private T TrainGeneratorBatch(Tensor<T> noise)
    {
        Generator.SetTrainingMode(true);
        // Keep Critic in training mode - required for backpropagation
        // We just don't call UpdateCriticParameters() during generator training

        // Generate fake images
        var generatedImages = Generator.Predict(noise);

        // Get critic scores
        var criticScores = Critic.Predict(generatedImages);

        // Calculate average score (generator wants to maximize this)
        int batchSize = noise.Shape[0];
        T totalScore = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            totalScore = NumOps.Add(totalScore, criticScores[i, 0]);
        }

        T avgScore = NumOps.Divide(totalScore, NumOps.FromDouble(batchSize));
        T loss = NumOps.Negate(avgScore);

        // Create gradients
        var gradients = new Tensor<T>(criticScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        }

        // Backpropagate through critic to get gradients for generator
        var criticInputGradients = Critic.BackwardWithInputGradient(gradients);

        // Backpropagate through generator
        Generator.Backward(criticInputGradients);

        // Update generator parameters
        UpdateGeneratorParameters();

        return loss;
    }

    /// <summary>
    /// Updates generator parameters using Adam optimizer.
    /// </summary>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_generatorMomentum == null || _generatorMomentum.Length != parameters.Length)
        {
            _generatorMomentum = new Vector<T>(parameters.Length);
            _generatorMomentum.Fill(NumOps.Zero);
        }

        if (_generatorSecondMoment == null || _generatorSecondMoment.Length != parameters.Length)
        {
            _generatorSecondMoment = new Vector<T>(parameters.Length);
            _generatorSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        var beta1 = NumOps.FromDouble(0.0); // Use beta1=0 for WGAN-GP (paper recommendation)
        var beta2 = NumOps.FromDouble(0.9);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update moments
            _generatorMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _generatorMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _generatorSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _generatorSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            // Adam update
            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(_generatorSecondMoment[i]), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, _generatorMomentum[i])
            );
        }

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates critic parameters using Adam optimizer.
    /// </summary>
    private void UpdateCriticParameters()
    {
        var parameters = Critic.GetParameters();
        var gradients = Critic.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_criticMomentum == null || _criticMomentum.Length != parameters.Length)
        {
            _criticMomentum = new Vector<T>(parameters.Length);
            _criticMomentum.Fill(NumOps.Zero);
        }

        if (_criticSecondMoment == null || _criticSecondMoment.Length != parameters.Length)
        {
            _criticSecondMoment = new Vector<T>(parameters.Length);
            _criticSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer with beta1=0 (paper recommendation)
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.9);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            _criticMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _criticMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _criticSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _criticSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(_criticSecondMoment[i]), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, _criticMomentum[i])
            );
        }

        Critic.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Generates synthetic images using the generator.
    /// </summary>
    public Tensor<T> GenerateImages(Tensor<T> noise)
    {
        Generator.SetTrainingMode(false);
        return Generator.Predict(noise);
    }

    /// <summary>
    /// Generates a tensor of random noise.
    /// </summary>
    public Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
    {
        var random = new Random();
        var shape = new int[] { batchSize, noiseSize };
        var noise = new Tensor<T>(shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i += 2)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();

                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;

                double z1 = radius * Math.Cos(theta);
                noise[b, i] = NumOps.FromDouble(z1);

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
    /// Evaluates the WGAN-GP by generating images and calculating metrics.
    /// </summary>
    public Dictionary<string, double> EvaluateModel(int sampleSize = 100)
    {
        var metrics = new Dictionary<string, double>();

        var noise = GenerateRandomNoiseTensor(sampleSize, Generator.Architecture.InputSize);
        var generatedImages = GenerateImages(noise);

        Critic.SetTrainingMode(false);
        var criticScores = Critic.Predict(generatedImages);

        var scoresList = new List<double>(sampleSize);
        for (int i = 0; i < sampleSize; i++)
        {
            scoresList.Add(Convert.ToDouble(criticScores[i, 0]));
        }

        metrics["AverageCriticScore"] = scoresList.Average();
        metrics["MinCriticScore"] = scoresList.Min();
        metrics["MaxCriticScore"] = scoresList.Max();
        metrics["CriticScoreStdDev"] = StatisticsHelper<double>.CalculateStandardDeviation(scoresList);
        metrics["CurrentLearningRate"] = _currentLearningRate;
        metrics["GradientPenaltyCoefficient"] = _gradientPenaltyCoefficient;

        if (_generatorLosses.Count > 0)
        {
            metrics["RecentGeneratorLoss"] = Convert.ToDouble(_generatorLosses[_generatorLosses.Count - 1]);
        }

        if (_criticLosses.Count > 0)
        {
            metrics["RecentCriticLoss"] = Convert.ToDouble(_criticLosses[_criticLosses.Count - 1]);
        }

        return metrics;
    }

    protected override void InitializeLayers()
    {
        // WGAN-GP doesn't use layers directly
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(expectedOutput, input);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.WassersteinGANGP,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "CriticParameters", Critic.GetParameterCount() },
                { "GradientPenaltyCoefficient", _gradientPenaltyCoefficient },
                { "CriticIterations", _criticIterations }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_currentLearningRate);
        writer.Write(_gradientPenaltyCoefficient);
        writer.Write(_criticIterations);

        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var criticBytes = Critic.Serialize();
        writer.Write(criticBytes.Length);
        writer.Write(criticBytes);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _currentLearningRate = reader.ReadDouble();
        _gradientPenaltyCoefficient = reader.ReadDouble();
        _criticIterations = reader.ReadInt32();

        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int criticDataLength = reader.ReadInt32();
        byte[] criticData = reader.ReadBytes(criticDataLength);
        Critic.Deserialize(criticData);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new WGANGP<T>(
            Generator.Architecture,
            Critic.Architecture,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
            _gradientPenaltyCoefficient,
            _criticIterations);
    }

    /// <summary>
    /// Updates the parameters of both the generator and critic networks.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for both networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorParameterCount = Generator.GetParameterCount();
        int criticParameterCount = Critic.GetParameterCount();

        if (parameters.Length != generatorParameterCount + criticParameterCount)
        {
            throw new ArgumentException(
                $"Expected {generatorParameterCount + criticParameterCount} parameters, " +
                $"but received {parameters.Length}.",
                nameof(parameters));
        }

        // Split and update Generator parameters
        var generatorParameters = new Vector<T>(generatorParameterCount);
        for (int i = 0; i < generatorParameterCount; i++)
        {
            generatorParameters[i] = parameters[i];
        }
        Generator.UpdateParameters(generatorParameters);

        // Split and update Critic parameters
        var criticParameters = new Vector<T>(criticParameterCount);
        for (int i = 0; i < criticParameterCount; i++)
        {
            criticParameters[i] = parameters[generatorParameterCount + i];
        }
        Critic.UpdateParameters(criticParameters);
    }
}
