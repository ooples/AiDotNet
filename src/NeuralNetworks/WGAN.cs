using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Wasserstein Generative Adversarial Network (WGAN), which uses the Wasserstein distance
/// (Earth Mover's distance) to measure the difference between the generated and real data distributions.
/// </summary>
/// <remarks>
/// <para>
/// WGAN addresses several training instabilities in vanilla GANs by:
/// - Using Wasserstein distance instead of Jensen-Shannon divergence
/// - Replacing the discriminator with a "critic" that doesn't output probabilities
/// - Enforcing a Lipschitz constraint through weight clipping
/// - Providing a loss that correlates with image quality
/// - Enabling more stable training and better convergence
/// </para>
/// <para><b>For Beginners:</b> WGAN is an improved GAN that solves many training problems.
///
/// Key improvements over vanilla GAN:
/// - More stable training (less likely to fail)
/// - The loss value actually tells you how well training is going
/// - No mode collapse issues (generating only a few types of outputs)
/// - Can train the discriminator (critic) many times without problems
///
/// The main change is using a different mathematical way to measure the difference
/// between real and fake images, which turns out to be much more stable.
///
/// Reference: Arjovsky et al., "Wasserstein GAN" (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class WGAN<T> : NeuralNetworkBase<T>
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
    /// The weight clipping threshold for enforcing the Lipschitz constraint.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In WGAN, the critic (discriminator) must be a 1-Lipschitz function. This is enforced
    /// by clipping the weights to lie within [-c, c] where c is the clipping threshold.
    /// A typical value is 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how large the critic's weights can get.
    ///
    /// Weight clipping:
    /// - Prevents the critic from becoming too powerful
    /// - Ensures the mathematical properties needed for Wasserstein distance
    /// - Typical value is 0.01 (weights stay between -0.01 and 0.01)
    /// - This is a simple but effective way to enforce the required constraint
    /// </para>
    /// </remarks>
    private double _weightClipValue = 0.01;

    /// <summary>
    /// The number of critic training iterations per generator iteration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// WGAN typically trains the critic multiple times for each generator update.
    /// This ensures the critic provides meaningful gradients to the generator.
    /// A typical value is 5.
    /// </para>
    /// <para><b>For Beginners:</b> How many times to train the critic vs. the generator.
    ///
    /// Training ratio:
    /// - The critic (discriminator) needs to be well-trained to guide the generator
    /// - For every 1 generator update, we do 5 critic updates
    /// - This ensures the critic always stays "ahead" and provides useful feedback
    /// - Unlike vanilla GANs, this doesn't cause training to fail
    /// </para>
    /// </remarks>
    private int _criticIterations = 5;

    /// <summary>
    /// Gets the generator network that creates synthetic data.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the critic network (called discriminator in vanilla GAN) that evaluates data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In WGAN, this is called a "critic" rather than "discriminator" because it doesn't
    /// output a probability. Instead, it outputs a score that estimates the Wasserstein distance.
    /// </para>
    /// <para><b>For Beginners:</b> The critic is like a discriminator but better.
    ///
    /// Critic vs. Discriminator:
    /// - Discriminator outputs probability (0-1): "Is this real?"
    /// - Critic outputs a score (any number): "How real is this?"
    /// - The critic's score directly relates to image quality
    /// - Higher scores mean more realistic images
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Critic { get; private set; }

    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined WGAN architecture with correct dimension handling.
    /// </summary>
    /// <param name="generatorArchitecture">The generator architecture.</param>
    /// <param name="criticArchitecture">The critic architecture.</param>
    /// <param name="inputType">The type of input.</param>
    /// <returns>The combined architecture for the WGAN.</returns>
    private static NeuralNetworkArchitecture<T> CreateWGANArchitecture(
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
    /// Initializes a new instance of the <see cref="WGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.
    /// The generator output size must match the critic input size.</param>
    /// <param name="criticArchitecture">The neural network architecture for the critic.
    /// The critic output size must be 1 (single Wasserstein score).</param>
    /// <param name="inputType">The type of input the WGAN will process.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, RMSprop optimizer is used (recommended for WGAN).</param>
    /// <param name="criticOptimizer">Optional optimizer for the critic. If null, RMSprop optimizer is used (recommended for WGAN).</param>
    /// <param name="lossFunction">Optional loss function. Defaults to <see cref="WassersteinLoss{T}"/> which
    /// implements the Wasserstein distance formula. WGAN training uses the critic scores directly for
    /// gradient computation, but the WassersteinLoss provides a consistent interface for computing
    /// loss values and serialization.</param>
    /// <param name="weightClipValue">The weight clipping threshold. Default is 0.01.</param>
    /// <param name="criticIterations">Number of critic iterations per generator iteration. Default is 5.</param>
    /// <remarks>
    /// <para>
    /// The WGAN constructor initializes both the generator and critic networks along with their
    /// respective optimizers. RMSprop is recommended over Adam for WGAN training stability.
    /// </para>
    /// <para>
    /// <b>Architecture Validation:</b>
    /// <list type="bullet">
    /// <item><description>Generator output size must match critic input size (generator produces images that critic evaluates)</description></item>
    /// <item><description>Critic output size must be 1 (outputs a Wasserstein score, not a probability)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>About the Loss Function:</b>
    /// Unlike traditional GANs that use binary cross-entropy loss, WGAN uses the Wasserstein distance
    /// (Earth Mover's distance). By default, WGAN uses <see cref="WassersteinLoss{T}"/> which implements
    /// this mathematically-correct loss function. The actual WGAN training optimizes critic outputs:
    /// <list type="bullet">
    /// <item><description>Critic loss: maximize E[critic(real)] - E[critic(fake)]</description></item>
    /// <item><description>Generator loss: maximize E[critic(fake)]</description></item>
    /// </list>
    /// The WassersteinLoss computes the same formula: -mean(predicted * label), where label is +1 for
    /// real samples and -1 for fake samples.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the WGAN with sensible defaults.
    ///
    /// Key parameters:
    /// - Generator/critic architectures define the network structures
    /// - Optimizers control how the networks learn (RMSprop is recommended for WGAN)
    /// - Weight clipping (0.01) enforces the mathematical constraints
    /// - Critic iterations (5) means the critic trains 5 times per generator update
    ///
    /// About the loss function: WGAN uses the "Wasserstein distance" (also called Earth Mover's
    /// distance) to measure how different real and fake images are. By default, we use
    /// WassersteinLoss which implements this mathematically. The critic's output is a score
    /// (higher = more real-looking), not a probability like in regular GANs. You don't need
    /// to specify a loss function - the default WassersteinLoss is the correct choice!
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when generatorArchitecture or criticArchitecture is null.</exception>
    /// <exception cref="ArgumentException">Thrown when architecture sizes are incompatible.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when weightClipValue or criticIterations is invalid.</exception>
    public WGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? criticOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        double weightClipValue = 0.01,
        int criticIterations = 5)
        : base(CreateWGANArchitecture(generatorArchitecture, criticArchitecture, inputType),
               lossFunction ?? new WassersteinLoss<T>())
    {
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }
        if (criticArchitecture is null)
        {
            throw new ArgumentNullException(nameof(criticArchitecture), "Critic architecture cannot be null.");
        }

        // Validate Generator/Critic architecture compatibility
        // The generator output size must match the critic input size for the WGAN to work
        if (generatorArchitecture.OutputSize != criticArchitecture.InputSize)
        {
            throw new ArgumentException(
                $"Generator output size ({generatorArchitecture.OutputSize}) must match critic input size ({criticArchitecture.InputSize}). " +
                "The generator produces images that the critic evaluates.",
                nameof(criticArchitecture));
        }

        // Validate critic output size - WGAN critic outputs a single scalar (not a probability)
        if (criticArchitecture.OutputSize != 1)
        {
            throw new ArgumentException(
                $"Critic output size must be 1 (Wasserstein score), but was {criticArchitecture.OutputSize}. " +
                "The critic outputs a real-valued score, not a probability distribution.",
                nameof(criticArchitecture));
        }

        if (weightClipValue <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weightClipValue), weightClipValue, "Weight clip value must be positive.");
        }
        if (criticIterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(criticIterations), criticIterations, "Critic iterations must be positive.");
        }

        _weightClipValue = weightClipValue;
        _criticIterations = criticIterations;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Critic = new ConvolutionalNeuralNetwork<T>(criticArchitecture);
        _lossFunction = lossFunction ?? new WassersteinLoss<T>();

        // Initialize optimizers (RMSprop is the recommended default for WGAN per the original paper)
        _generatorOptimizer = generatorOptimizer ?? new RootMeanSquarePropagationOptimizer<T, Tensor<T>, Tensor<T>>(Generator);
        _criticOptimizer = criticOptimizer ?? new RootMeanSquarePropagationOptimizer<T, Tensor<T>, Tensor<T>>(Critic);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for the WGAN using tensor batches.
    /// </summary>
    /// <param name="realImages">A tensor containing real images.</param>
    /// <param name="noise">A tensor containing random noise for the generator.</param>
    /// <returns>A tuple containing the critic and generator loss values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the WGAN training algorithm:
    /// 1. Train the critic multiple times (typically 5) with weight clipping
    /// 2. Train the generator once
    /// 3. The critic is trained to maximize the difference between real and fake scores
    /// 4. The generator is trained to maximize the critic's score on fake images
    /// </para>
    /// <para><b>For Beginners:</b> One training round for WGAN.
    ///
    /// The training process:
    /// - Trains the critic several times to make it really good at judging quality
    /// - Clips the critic's weights to keep it well-behaved
    /// - Trains the generator once to improve its outputs
    /// - Returns loss values that actually mean something (higher = better)
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

        if (realImages.Shape[0] != noise.Shape[0])
        {
            throw new ArgumentException(
                $"Batch size mismatch: realImages batch={realImages.Shape[0]} vs noise batch={noise.Shape[0]}.",
                nameof(noise));
        }

        Generator.SetTrainingMode(true);
        Critic.SetTrainingMode(true);

        T totalCriticLoss = NumOps.Zero;

        // Train critic multiple times
        for (int i = 0; i < _criticIterations; i++)
        {
            // Generate fake images
            Tensor<T> fakeImages = GenerateImages(noise);

            // Train critic on real images (maximize score)
            T realScore = TrainCriticBatch(realImages, isReal: true);

            // Train critic on fake images (minimize score)
            T fakeScore = TrainCriticBatch(fakeImages, isReal: false);

            // Wasserstein loss: E[D(real)] - E[D(fake)]
            T criticLoss = NumOps.Subtract(realScore, fakeScore);

            // We want to maximize this, so we negate for gradient descent
            criticLoss = NumOps.Negate(criticLoss);

            totalCriticLoss = NumOps.Add(totalCriticLoss, criticLoss);

            // Clip weights to enforce Lipschitz constraint
            ClipCriticWeights();
        }

        // Average critic loss across iterations
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
    /// Clips the critic's weights to enforce the Lipschitz constraint.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clips all weights in the critic network to lie within [-c, c] where c is
    /// the weight clipping threshold. This is the key mechanism in WGAN for enforcing the
    /// Lipschitz constraint required for the Wasserstein distance to be well-defined.
    /// </para>
    /// <para><b>For Beginners:</b> This keeps the critic's weights from getting too large.
    ///
    /// Weight clipping:
    /// - Goes through all the critic's weights
    /// - If any weight is above +0.01, clips it to +0.01
    /// - If any weight is below -0.01, clips it to -0.01
    /// - This ensures the mathematical properties needed for WGAN to work
    /// </para>
    /// </remarks>
    private void ClipCriticWeights()
    {
        var parameters = Critic.GetParameters();
        var clipMin = NumOps.FromDouble(-_weightClipValue);
        var clipMax = NumOps.FromDouble(_weightClipValue);

        // Vectorized weight clipping using Engine.Clamp
        var clippedParameters = Engine.Clamp(parameters, clipMin, clipMax);
        Critic.UpdateParameters(clippedParameters);
    }

    /// <summary>
    /// Trains the critic on a batch of images.
    /// </summary>
    /// <param name="images">The tensor containing images to train on.</param>
    /// <param name="isReal">True if the images are real, false if they are generated.</param>
    /// <returns>The average critic score for this batch.</returns>
    private T TrainCriticBatch(Tensor<T> images, bool isReal)
    {
        Critic.SetTrainingMode(true);

        // Forward pass through critic
        var criticScores = Critic.Predict(images);

        // Calculate average score using vectorized reduction
        int batchSize = images.Shape[0];
        T avgScore = NumOps.Divide(Engine.TensorSum(criticScores), NumOps.FromDouble(batchSize));

        // Create predicted scores from tensor (extract first column) and labels using vectorized fill
        // Labels: +1 for real samples, -1 for fake samples (Wasserstein convention)
        var predictedScores = new Vector<T>(batchSize);
        T labelValue = isReal ? NumOps.One : NumOps.Negate(NumOps.One);

        // Extract scores from tensor to vector (column 0)
        for (int i = 0; i < batchSize; i++)
        {
            predictedScores[i] = criticScores[i, 0];
        }
        var labels = Engine.Fill<T>(batchSize, labelValue);

        // Use the loss function to calculate gradients
        // WassersteinLoss.CalculateDerivative returns -labels[i] / n
        // For real (label=+1): gradient = -1/n (want to maximize score)
        // For fake (label=-1): gradient = +1/n (want to minimize score)
        var derivativeVector = _lossFunction.CalculateDerivative(predictedScores, labels);

        // Convert gradient vector back to tensor for backpropagation
        var gradients = new Tensor<T>(criticScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = derivativeVector[i];
        }

        // Backpropagate
        Critic.Backpropagate(gradients);

        // Update parameters using optimizer
        UpdateCriticWithOptimizer();

        return avgScore;
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

        // Get critic scores for generated images
        var criticScores = Critic.Predict(generatedImages);

        // Create predicted scores and labels vectors for loss function
        // For generator training, we want to maximize critic score on fakes
        // Use label +1 (treating fakes as "should be real") to get gradient -1/n
        // This minimizes the negative of the score, i.e., maximizes the score
        int batchSize = noise.Shape[0];
        var predictedScores = new Vector<T>(batchSize);

        // Extract scores from tensor to vector (column 0)
        for (int i = 0; i < batchSize; i++)
        {
            predictedScores[i] = criticScores[i, 0];
        }
        var labels = Engine.Fill<T>(batchSize, NumOps.One); // Want to maximize score (be more "real")

        // Calculate loss using the loss function: -mean(predicted * labels)
        // With labels = +1: loss = -mean(predicted), minimizing this maximizes scores
        T loss = _lossFunction.CalculateLoss(predictedScores, labels);

        // Use the loss function to calculate gradients
        // WassersteinLoss.CalculateDerivative with label=+1 returns -1/n
        var derivativeVector = _lossFunction.CalculateDerivative(predictedScores, labels);

        // Convert gradient vector back to tensor for backpropagation
        var gradients = new Tensor<T>(criticScores.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = derivativeVector[i];
        }

        // Backpropagate through critic to get gradients for generator output
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
    /// Updates critic parameters using the configured optimizer.
    /// </summary>
    private void UpdateCriticWithOptimizer()
    {
        var parameters = Critic.GetParameters();
        var gradients = Critic.GetParameterGradients();

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
    /// This method uses vectorized Gaussian noise generation from the Engine for optimal performance.
    /// The noise is sampled from a standard normal distribution (mean=0, stddev=1).
    /// </para>
    /// <para><b>For Beginners:</b> This creates random input values for the generator.
    ///
    /// The random noise serves as the "seed" for generating images:
    /// - Each batch contains multiple noise vectors
    /// - Each vector has a fixed size determined by the generator architecture
    /// - The values are randomly sampled from a bell curve (normal distribution)
    /// - Different random values will produce different generated images
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

        var totalElements = checked(batchSize * noiseSize);
        var mean = NumOps.Zero;
        var stddev = NumOps.One;
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);
        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Evaluates the WGAN by generating images and calculating metrics.
    /// </summary>
    /// <param name="sampleSize">The number of samples to generate for evaluation.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    public Dictionary<string, double> EvaluateModel(int sampleSize = 100)
    {
        var metrics = new Dictionary<string, double>();

        var noise = GenerateRandomNoiseTensor(sampleSize, Generator.Architecture.InputSize);
        var generatedImages = GenerateImages(noise);

        // Get critic scores
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
        // WGAN doesn't use layers directly
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For WGAN, the parameters have GAN-specific semantics:
    /// - <paramref name="input"/>: The real images tensor (training data)
    /// - <paramref name="expectedOutput"/>: The noise tensor for the generator
    /// </para>
    /// <para><b>For Beginners:</b> In WGAN training:
    /// - Pass your real images as the first parameter
    /// - Pass random noise vectors as the second parameter
    /// - The network will train both critic and generator in one step
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // input = realImages, expectedOutput = noise
        TrainStep(input, expectedOutput);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.WassersteinGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "CriticParameters", Critic.GetParameterCount() },
                { "WeightClipValue", _weightClipValue },
                { "CriticIterations", _criticIterations }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_weightClipValue);
        writer.Write(_criticIterations);

        // Serialize loss histories
        writer.Write(_generatorLosses.Count);
        foreach (var loss in _generatorLosses)
            writer.Write(NumOps.ToDouble(loss));

        writer.Write(_criticLosses.Count);
        foreach (var loss in _criticLosses)
            writer.Write(NumOps.ToDouble(loss));

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
        _weightClipValue = reader.ReadDouble();
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
        return new WGAN<T>(
            Generator.Architecture,
            Critic.Architecture,
            Architecture.InputType,
            null, // Use default optimizer
            null, // Use default optimizer
            _lossFunction,
            _weightClipValue,
            _criticIterations);
    }

    /// <summary>
    /// Updates the parameters of both the generator and critic networks.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for both networks.</param>
    /// <remarks>
    /// <para>
    /// The parameters vector is split between the generator and critic based on their
    /// respective parameter counts. Generator parameters come first, followed by critic parameters.
    /// </para>
    /// </remarks>
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
