using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;

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
    private readonly List<T> _criticLosses = new List<T>();
    private readonly List<T> _generatorLosses = new List<T>();

    /// <summary>
    /// The optimizer for the generator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer for the critic network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _criticOptimizer;

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
    /// Creates the combined WGAN-GP architecture with correct dimension handling.
    /// </summary>
    /// <param name="generatorArchitecture">The generator architecture.</param>
    /// <param name="criticArchitecture">The critic architecture.</param>
    /// <param name="inputType">The type of input.</param>
    /// <returns>The combined architecture for the WGAN-GP.</returns>
    private static NeuralNetworkArchitecture<T> CreateWGANGPArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Medium,
                inputSize: 0,
                inputHeight: criticArchitecture.InputHeight,
                inputWidth: criticArchitecture.InputWidth,
                inputDepth: criticArchitecture.InputDepth,
                outputSize: criticArchitecture.OutputSize,
                layers: null);
        }

        // For OneDimensional and TwoDimensional, use simple constructor
        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Medium,
            inputSize: generatorArchitecture.InputSize,
            outputSize: criticArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="WGANGP{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="criticArchitecture">The neural network architecture for the critic.</param>
    /// <param name="inputType">The type of input the WGAN-GP will process.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="criticOptimizer">Optional optimizer for the critic. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="gradientPenaltyCoefficient">The gradient penalty coefficient (lambda). Default is 10.0.</param>
    /// <param name="criticIterations">Number of critic iterations per generator iteration. Default is 5.</param>
    /// <remarks>
    /// <para>
    /// The WGAN-GP constructor initializes both the generator and critic networks along with their
    /// respective optimizers. The gradient penalty coefficient controls the strength of the
    /// Lipschitz constraint enforcement.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the WGAN-GP with sensible defaults.
    ///
    /// Key parameters:
    /// - Generator/critic architectures define the network structures
    /// - Optimizers control how the networks learn
    /// - Gradient penalty coefficient (10.0) controls constraint strength
    /// - Critic iterations (5) means the critic trains 5 times per generator update
    /// </para>
    /// </remarks>
    public WGANGP(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? criticOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        double gradientPenaltyCoefficient = 10.0,
        int criticIterations = 5)
        : base(CreateWGANGPArchitecture(generatorArchitecture, criticArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        // Input validation
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }
        if (criticArchitecture is null)
        {
            throw new ArgumentNullException(nameof(criticArchitecture), "Critic architecture cannot be null.");
        }
        if (gradientPenaltyCoefficient <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gradientPenaltyCoefficient), gradientPenaltyCoefficient, "Gradient penalty coefficient must be positive.");
        }
        if (criticIterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(criticIterations), criticIterations, "Critic iterations must be positive.");
        }

        _gradientPenaltyCoefficient = gradientPenaltyCoefficient;
        _criticIterations = criticIterations;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Critic = new ConvolutionalNeuralNetwork<T>(criticArchitecture);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        // Initialize optimizers (default to Adam if not provided)
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator);
        _criticOptimizer = criticOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Critic);

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
        if (realImages is null)
        {
            throw new ArgumentNullException(nameof(realImages), "Real images tensor cannot be null.");
        }

        if (noise is null)
        {
            throw new ArgumentNullException(nameof(noise), "Noise tensor cannot be null.");
        }

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

        // Forward pass on real images to compute scores using vectorized reduction
        var realScores = Critic.Predict(realImages);
        T realScore = NumOps.Divide(Engine.TensorSum(realScores), NumOps.FromDouble(batchSize));

        // Forward pass on fake images to compute scores using vectorized reduction
        var fakeScores = Critic.Predict(fakeImages);
        T fakeScore = NumOps.Divide(Engine.TensorSum(fakeScores), NumOps.FromDouble(batchSize));

        // Compute gradient penalty (this calls Predict on interpolated images which overwrites cache)
        var (gradientPenalty, gpParameterGradients) = ComputeGradientPenaltyWithGradients(realImages, fakeImages, batchSize);

        // Wasserstein loss with gradient penalty: -E[D(real)] + E[D(fake)] + lambda * GP
        T wassersteinDistance = NumOps.Subtract(realScore, fakeScore);
        T gpTerm = NumOps.Multiply(NumOps.FromDouble(_gradientPenaltyCoefficient), gradientPenalty);
        T criticLoss = NumOps.Add(NumOps.Negate(wassersteinDistance), gpTerm);

        // Create gradients for real images (maximize score) using vectorized fill
        var realGradients = new Tensor<T>(realScores.Shape);
        Engine.TensorFill(realGradients, NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize)));

        // IMPORTANT: Re-run forward pass on real images before backprop
        // The GP computation called Predict(interpolated) which overwrote the cached activations
        Critic.Predict(realImages);
        Critic.Backpropagate(realGradients);
        var realParameterGradients = Critic.GetParameterGradients().Clone();

        // Create gradients for fake images (minimize score) using vectorized fill
        var fakeGradients = new Tensor<T>(fakeScores.Shape);
        Engine.TensorFill(fakeGradients, NumOps.Divide(NumOps.Negate(NumOps.One), NumOps.FromDouble(batchSize)));

        // IMPORTANT: Re-run forward pass on fake images before backprop
        Critic.Predict(fakeImages);
        Critic.Backpropagate(fakeGradients);
        var fakeParameterGradients = Critic.GetParameterGradients().Clone();

        // Combine all gradients using vectorized operations: real + fake + scaled GP
        T gpScale = NumOps.FromDouble(_gradientPenaltyCoefficient);
        var scaledGpGradients = Engine.Multiply(gpParameterGradients, gpScale);
        var realPlusFake = Engine.Add(realParameterGradients, fakeParameterGradients);
        var combinedGradients = Engine.Add(realPlusFake, scaledGpGradients);

        // Update critic parameters with combined gradients using optimizer
        UpdateCriticWithOptimizer(combinedGradients);

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
        // Create interpolated images using vectorized operations
        // Formula: interpolated = epsilon * real + (1 - epsilon) * fake
        // Generate random epsilon values per sample and broadcast to full shape

        // Compute number of elements per sample (excludes batch dimension)
        int sampleSize = realImages.Length / batchSize;

        // Generate random epsilon values [batchSize, 1, ...] and tile to match image shape
        var epsilonShape = new int[realImages.Shape.Length];
        epsilonShape[0] = batchSize;
        for (int d = 1; d < epsilonShape.Length; d++) epsilonShape[d] = 1;
        var epsilonBase = Engine.TensorRandomUniform<T>(epsilonShape);

        // Tile epsilon to match full image shape
        var tileFactors = new int[realImages.Shape.Length];
        tileFactors[0] = 1;
        for (int d = 1; d < tileFactors.Length; d++) tileFactors[d] = realImages.Shape[d];
        var epsilon = Engine.TensorTile(epsilonBase, tileFactors);

        // Compute (1 - epsilon)
        var onesTensor = new Tensor<T>(epsilon.Shape);
        Engine.TensorFill(onesTensor, NumOps.One);
        var oneMinusEpsilon = Engine.TensorSubtract(onesTensor, epsilon);

        // interpolated = epsilon * real + (1 - epsilon) * fake
        var epsilonTimesReal = Engine.TensorMultiply(epsilon, realImages);
        var oneMinusEpsilonTimesFake = Engine.TensorMultiply(oneMinusEpsilon, fakeImages);
        var interpolatedImages = Engine.TensorAdd(epsilonTimesReal, oneMinusEpsilonTimesFake);

        // Forward pass through critic
        var interpolatedScores = Critic.Predict(interpolatedImages);

        // Create gradients of all ones using vectorized fill
        var ones = new Tensor<T>(interpolatedScores.Shape);
        Engine.TensorFill(ones, NumOps.One);

        // Backpropagate to get gradients with respect to input
        var inputGradients = Critic.Backpropagate(ones);

        // Capture the parameter gradients from this backprop
        var gpParameterGradients = Critic.GetParameterGradients().Clone();

        // Compute L2 norm of gradients for each sample using vectorized operations
        int gradientSampleSize = inputGradients.Length / batchSize;
        var gradientsReshaped = inputGradients.Reshape([batchSize, gradientSampleSize]);

        // gradNormSquared[b] = sum(grad[b, i]^2) for each batch
        var gradSquared = Engine.TensorMultiply(gradientsReshaped, gradientsReshaped);
        var gradNormSquared = Engine.ReduceSum(gradSquared, [1], keepDims: false);

        // gradNorm = sqrt(gradNormSquared)
        var gradNorm = Engine.TensorSqrt(gradNormSquared);

        // deviation = gradNorm - 1
        var onesForDeviation = new Tensor<T>(gradNorm.Shape);
        Engine.TensorFill(onesForDeviation, NumOps.One);
        var deviation = Engine.TensorSubtract(gradNorm, onesForDeviation);

        // penalty = deviation^2
        var penalty = Engine.TensorMultiply(deviation, deviation);

        // totalPenalty = mean(penalty)
        T totalPenalty = NumOps.Divide(Engine.TensorSum(penalty), NumOps.FromDouble(batchSize));

        return (totalPenalty, gpParameterGradients);
    }

    /// <summary>
    /// Updates critic parameters using the configured optimizer with pre-computed gradients.
    /// </summary>
    /// <param name="gradients">The pre-computed combined gradients.</param>
    private void UpdateCriticWithOptimizer(Vector<T> gradients)
    {
        var parameters = Critic.GetParameters();

        // Gradient clipping using vectorized operations
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParameters = _criticOptimizer.UpdateParameters(parameters, gradients);
        Critic.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Trains the generator to fool the critic.
    /// </summary>
    /// <param name="noise">The tensor containing noise vectors.</param>
    /// <returns>The generator loss value.</returns>
    private T TrainGeneratorBatch(Tensor<T> noise)
    {
        Generator.SetTrainingMode(true);

        // Generate fake images
        var generatedImages = Generator.Predict(noise);

        // Get critic scores
        var criticScores = Critic.Predict(generatedImages);

        // Calculate average score using vectorized reduction (generator wants to maximize this)
        int batchSize = noise.Shape[0];
        T avgScore = NumOps.Divide(Engine.TensorSum(criticScores), NumOps.FromDouble(batchSize));
        T loss = NumOps.Negate(avgScore);

        // Create gradients using vectorized fill
        var gradients = new Tensor<T>(criticScores.Shape);
        Engine.TensorFill(gradients, NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize)));

        // Backpropagate through critic to get gradients for generator
        var criticInputGradients = Critic.BackwardWithInputGradient(gradients);

        // Backpropagate through generator
        Generator.Backward(criticInputGradients);

        // Update generator parameters using optimizer
        UpdateGeneratorWithOptimizer();

        return loss;
    }

    /// <summary>
    /// Updates generator parameters using the configured optimizer.
    /// </summary>
    private void UpdateGeneratorWithOptimizer()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Gradient clipping using vectorized operations
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParameters = _generatorOptimizer.UpdateParameters(parameters, gradients);
        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Generates synthetic images using the generator.
    /// </summary>
    /// <param name="noise">The noise tensor to generate images from.</param>
    /// <returns>A tensor containing the generated images.</returns>
    public Tensor<T> GenerateImages(Tensor<T> noise)
    {
        Generator.SetTrainingMode(false);
        return Generator.Predict(noise);
    }

    /// <summary>
    /// Generates a tensor of random noise for the generator.
    /// </summary>
    /// <param name="batchSize">The number of noise vectors to generate.</param>
    /// <param name="noiseSize">The dimensionality of each noise vector.</param>
    /// <returns>A tensor of random noise values.</returns>
    /// <remarks>
    /// <para>
    /// This method uses vectorized Gaussian noise generation for optimal performance.
    /// The generated noise has mean 0 and standard deviation 1, following the standard
    /// normal distribution recommended for GAN training.
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

        var totalElements = batchSize * noiseSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);
        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Evaluates the WGAN-GP by generating images and calculating metrics.
    /// </summary>
    /// <param name="sampleSize">The number of samples to generate for evaluation.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
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
            scoresList.Add(NumOps.ToDouble(criticScores[i, 0]));
        }

        metrics["AverageCriticScore"] = scoresList.Average();
        metrics["MinCriticScore"] = scoresList.Min();
        metrics["MaxCriticScore"] = scoresList.Max();
        metrics["CriticScoreStdDev"] = StatisticsHelper<double>.CalculateStandardDeviation(scoresList);
        metrics["GradientPenaltyCoefficient"] = _gradientPenaltyCoefficient;

        if (_generatorLosses.Count > 0)
        {
            metrics["RecentGeneratorLoss"] = NumOps.ToDouble(_generatorLosses[_generatorLosses.Count - 1]);
        }

        if (_criticLosses.Count > 0)
        {
            metrics["RecentCriticLoss"] = NumOps.ToDouble(_criticLosses[_criticLosses.Count - 1]);
        }

        return metrics;
    }

    /// <summary>
    /// Resets both optimizer states for a fresh training run.
    /// </summary>
    public void ResetOptimizerState()
    {
        _generatorOptimizer.Reset();
        _criticOptimizer.Reset();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // WGAN-GP doesn't use layers directly
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(expectedOutput, input);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_gradientPenaltyCoefficient);
        writer.Write(_criticIterations);

        // Serialize loss histories
        writer.Write(_generatorLosses.Count);
        foreach (var loss in _generatorLosses)
            writer.Write(NumOps.ToDouble(loss));

        writer.Write(_criticLosses.Count);
        foreach (var loss in _criticLosses)
            writer.Write(NumOps.ToDouble(loss));

        // Serialize networks
        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var criticBytes = Critic.Serialize();
        writer.Write(criticBytes.Length);
        writer.Write(criticBytes);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _gradientPenaltyCoefficient = reader.ReadDouble();
        _criticIterations = reader.ReadInt32();

        // Deserialize loss histories
        _generatorLosses.Clear();
        int genLossCount = reader.ReadInt32();
        for (int i = 0; i < genLossCount; i++)
            _generatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));

        _criticLosses.Clear();
        int criticLossCount = reader.ReadInt32();
        for (int i = 0; i < criticLossCount; i++)
            _criticLosses.Add(NumOps.FromDouble(reader.ReadDouble()));

        // Deserialize networks
        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int criticDataLength = reader.ReadInt32();
        byte[] criticData = reader.ReadBytes(criticDataLength);
        Critic.Deserialize(criticData);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new WGANGP<T>(
            Generator.Architecture,
            Critic.Architecture,
            Architecture.InputType,
            null, // Use default optimizer
            null, // Use default optimizer
            _lossFunction,
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
