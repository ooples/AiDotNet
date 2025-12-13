using System.IO;
using AiDotNet.Helpers;

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
    // Generator optimizer state (RMSprop)
    private Vector<T> _genMomentum;
    private Vector<T> _genSecondMoment;
    private double _genCurrentLearningRate;

    // Critic optimizer state (RMSprop)
    private Vector<T> _criticMomentum;
    private Vector<T> _criticSecondMoment;
    private double _criticCurrentLearningRate;

    private double _initialLearningRate;
    private double _learningRateDecay;
    private readonly List<T> _criticLosses = [];
    private readonly List<T> _generatorLosses = [];

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
    /// Initializes a new instance of the <see cref="WGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="criticArchitecture">The neural network architecture for the critic.</param>
    /// <param name="inputType">The type of input the WGAN will process.</param>
    /// <param name="lossFunction">Optional loss function (typically not used, as WGAN uses Wasserstein distance).</param>
    /// <param name="initialLearningRate">The initial learning rate. Default is 0.00005 (lower than vanilla GAN).</param>
    /// <param name="weightClipValue">The weight clipping threshold. Default is 0.01.</param>
    /// <param name="criticIterations">Number of critic iterations per generator iteration. Default is 5.</param>
    public WGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.00005,
        double weightClipValue = 0.01,
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
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture));
        }
        if (criticArchitecture is null)
        {
            throw new ArgumentNullException(nameof(criticArchitecture));
        }
        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }
        if (weightClipValue <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weightClipValue), weightClipValue, "Weight clip value must be positive.");
        }
        if (criticIterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(criticIterations), criticIterations, "Critic iterations must be positive.");
        }

        _initialLearningRate = initialLearningRate;
        _weightClipValue = weightClipValue;
        _criticIterations = criticIterations;
        _learningRateDecay = 0.9999;

        // Initialize Generator optimizer state (RMSprop)
        _genCurrentLearningRate = initialLearningRate;
        _genMomentum = Vector<T>.Empty();
        _genSecondMoment = Vector<T>.Empty();

        // Initialize Critic optimizer state (RMSprop)
        _criticCurrentLearningRate = initialLearningRate;
        _criticMomentum = Vector<T>.Empty();
        _criticSecondMoment = Vector<T>.Empty();

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Critic = new ConvolutionalNeuralNetwork<T>(criticArchitecture);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

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

        for (int i = 0; i < parameters.Length; i++)
        {
            // Clip to [-c, c]
            if (NumOps.GreaterThan(parameters[i], clipMax))
            {
                parameters[i] = clipMax;
            }
            else if (NumOps.LessThan(parameters[i], clipMin))
            {
                parameters[i] = clipMin;
            }
        }

        Critic.UpdateParameters(parameters);
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

        // Calculate average score
        int batchSize = images.Shape[0];
        T totalScore = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            totalScore = NumOps.Add(totalScore, criticScores[i, 0]);
        }

        T avgScore = NumOps.Divide(totalScore, NumOps.FromDouble(batchSize));

        // For WGAN, we want to:
        // - Maximize critic output for real images
        // - Minimize critic output for fake images
        // This is equivalent to maximizing (realScore - fakeScore)

        // Create gradients: +1 for real images, -1 for fake images
        var gradients = new Tensor<T>(criticScores.Shape);
        T gradientValue = isReal ? NumOps.One : NumOps.Negate(NumOps.One);

        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = NumOps.Divide(gradientValue, NumOps.FromDouble(batchSize));
        }

        // Backpropagate
        Critic.Backpropagate(gradients);

        // Update parameters
        UpdateCriticParameters();

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
        // Keep Critic in training mode - required for backpropagation
        // We just don't call UpdateCriticParameters() during generator training

        // Generate fake images
        var generatedImages = Generator.Predict(noise);

        // Get critic scores for generated images
        var criticScores = Critic.Predict(generatedImages);

        // Calculate average score (generator wants to maximize this)
        int batchSize = noise.Shape[0];
        T totalScore = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            totalScore = NumOps.Add(totalScore, criticScores[i, 0]);
        }

        T avgScore = NumOps.Divide(totalScore, NumOps.FromDouble(batchSize));

        // Loss is negative of the score (we minimize loss, which maximizes score)
        T loss = NumOps.Negate(avgScore);

        // Create gradients (we want to maximize critic output)
        var gradients = new Tensor<T>(criticScores.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        }

        // Backpropagate through critic to get gradients for generator output
        var criticInputGradients = Critic.BackwardWithInputGradient(gradients);

        // Backpropagate through generator
        Generator.Backward(criticInputGradients);

        // Update generator parameters
        UpdateGeneratorParameters();

        return loss;
    }

    /// <summary>
    /// Updates Generator parameters using RMSprop optimizer with vectorized operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the RMSprop optimizer with gradient clipping for stable training.
    /// All vector operations are performed using Engine methods for optimal performance.
    /// </para>
    /// <para><b>For Beginners:</b> This adjusts the generator's weights to make it better at fooling the critic.
    ///
    /// The update process:
    /// - Clips gradients to prevent extreme updates
    /// - Uses RMSprop to adaptively adjust the learning rate for each parameter
    /// - Applies learning rate decay to gradually slow down learning over time
    /// - Updates all generator parameters in a vectorized manner for efficiency
    /// </para>
    /// </remarks>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_genMomentum.Length != parameters.Length)
        {
            _genMomentum = new Vector<T>(parameters.Length);
            _genMomentum.Fill(NumOps.Zero);
        }

        if (_genSecondMoment.Length != parameters.Length)
        {
            _genSecondMoment = new Vector<T>(parameters.Length);
            _genSecondMoment.Fill(NumOps.Zero);
        }

        // Gradient clipping (vectorized)
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
        }

        // RMSprop parameters (recommended for WGAN)
        var learningRate = NumOps.FromDouble(_genCurrentLearningRate);
        var decay = NumOps.FromDouble(0.9);
        var epsilon = NumOps.FromDouble(1e-8);

        // Update second moment (RMSprop) - vectorized
        // secondMoment = decay * secondMoment + (1 - decay) * gradients^2
        var oneMinusDecay = NumOps.Subtract(NumOps.One, decay);
        var gradientsSquared = Engine.Multiply(gradients, gradients);
        var decayedSecondMoment = Engine.Multiply(_genSecondMoment, decay);
        var newSecondMomentTerm = Engine.Multiply(gradientsSquared, oneMinusDecay);
        _genSecondMoment = Engine.Add(decayedSecondMoment, newSecondMomentTerm);

        // RMSprop update - vectorized
        // adaptiveLR = learningRate / (sqrt(secondMoment) + epsilon)
        var sqrtSecondMoment = (Vector<T>)Engine.Sqrt(_genSecondMoment);
        var epsilonVector = Vector<T>.CreateDefault(sqrtSecondMoment.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(sqrtSecondMoment, epsilonVector);
        var learningRateVector = Vector<T>.CreateDefault(denominator.Length, learningRate);
        var adaptiveLR = (Vector<T>)Engine.Divide(learningRateVector, denominator);

        // parameters = parameters - adaptiveLR * gradients
        var update = (Vector<T>)Engine.Multiply(adaptiveLR, gradients);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        // Apply learning rate decay
        _genCurrentLearningRate *= _learningRateDecay;

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates Critic parameters using RMSprop optimizer with vectorized operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the RMSprop optimizer with gradient clipping for stable training.
    /// All vector operations are performed using Engine methods for optimal performance.
    /// </para>
    /// <para><b>For Beginners:</b> This adjusts the critic's weights to make it better at evaluating images.
    ///
    /// The update process:
    /// - Clips gradients to prevent extreme updates
    /// - Uses RMSprop to adaptively adjust the learning rate for each parameter
    /// - Applies learning rate decay to gradually slow down learning over time
    /// - Updates all critic parameters in a vectorized manner for efficiency
    /// </para>
    /// </remarks>
    private void UpdateCriticParameters()
    {
        var parameters = Critic.GetParameters();
        var gradients = Critic.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_criticMomentum.Length != parameters.Length)
        {
            _criticMomentum = new Vector<T>(parameters.Length);
            _criticMomentum.Fill(NumOps.Zero);
        }

        if (_criticSecondMoment.Length != parameters.Length)
        {
            _criticSecondMoment = new Vector<T>(parameters.Length);
            _criticSecondMoment.Fill(NumOps.Zero);
        }

        // Gradient clipping (vectorized)
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
        }

        // RMSprop parameters (recommended for WGAN)
        var learningRate = NumOps.FromDouble(_criticCurrentLearningRate);
        var decay = NumOps.FromDouble(0.9);
        var epsilon = NumOps.FromDouble(1e-8);

        // Update second moment (RMSprop) - vectorized
        // secondMoment = decay * secondMoment + (1 - decay) * gradients^2
        var oneMinusDecay = NumOps.Subtract(NumOps.One, decay);
        var gradientsSquared = Engine.Multiply(gradients, gradients);
        var decayedSecondMoment = Engine.Multiply(_criticSecondMoment, decay);
        var newSecondMomentTerm = Engine.Multiply(gradientsSquared, oneMinusDecay);
        _criticSecondMoment = Engine.Add(decayedSecondMoment, newSecondMomentTerm);

        // RMSprop update - vectorized
        // adaptiveLR = learningRate / (sqrt(secondMoment) + epsilon)
        var sqrtSecondMoment = (Vector<T>)Engine.Sqrt(_criticSecondMoment);
        var epsilonVector = Vector<T>.CreateDefault(sqrtSecondMoment.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(sqrtSecondMoment, epsilonVector);
        var learningRateVector = Vector<T>.CreateDefault(denominator.Length, learningRate);
        var adaptiveLR = (Vector<T>)Engine.Divide(learningRateVector, denominator);

        // parameters = parameters - adaptiveLR * gradients
        var update = (Vector<T>)Engine.Multiply(adaptiveLR, gradients);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        // Apply learning rate decay
        _criticCurrentLearningRate *= _learningRateDecay;

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
    /// Generates a tensor of random noise for the generator.
    /// </summary>
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

        var totalElements = batchSize * noiseSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);
        return Tensor<T>.FromVector(noiseVector, new int[] { batchSize, noiseSize });
    }

    /// <summary>
    /// Evaluates the WGAN by generating images and calculating metrics.
    /// </summary>
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
            scoresList.Add(Convert.ToDouble(criticScores[i, 0]));
        }

        metrics["AverageCriticScore"] = scoresList.Average();
        metrics["MinCriticScore"] = scoresList.Min();
        metrics["MaxCriticScore"] = scoresList.Max();
        metrics["CriticScoreStdDev"] = StatisticsHelper<double>.CalculateStandardDeviation(scoresList);
        metrics["GeneratorLearningRate"] = _genCurrentLearningRate;
        metrics["CriticLearningRate"] = _criticCurrentLearningRate;

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
        // WGAN doesn't use layers directly
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

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save both learning rates
        writer.Write(_genCurrentLearningRate);
        writer.Write(_criticCurrentLearningRate);
        writer.Write(_weightClipValue);
        writer.Write(_criticIterations);
        writer.Write(_initialLearningRate);
        writer.Write(_learningRateDecay);

        // Serialize generator optimizer state
        SerializationHelper<T>.SerializeVector(writer, _genMomentum);
        SerializationHelper<T>.SerializeVector(writer, _genSecondMoment);

        // Serialize critic optimizer state
        SerializationHelper<T>.SerializeVector(writer, _criticMomentum);
        SerializationHelper<T>.SerializeVector(writer, _criticSecondMoment);

        // Serialize generator and critic networks
        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var criticBytes = Critic.Serialize();
        writer.Write(criticBytes.Length);
        writer.Write(criticBytes);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read both learning rates
        _genCurrentLearningRate = reader.ReadDouble();
        _criticCurrentLearningRate = reader.ReadDouble();
        _weightClipValue = reader.ReadDouble();
        _criticIterations = reader.ReadInt32();
        _initialLearningRate = reader.ReadDouble();
        _learningRateDecay = reader.ReadDouble();

        // Deserialize generator optimizer state
        _genMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _genSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        // Deserialize critic optimizer state
        _criticMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _criticSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        // Deserialize generator and critic networks
        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int criticDataLength = reader.ReadInt32();
        byte[] criticData = reader.ReadBytes(criticDataLength);
        Critic.Deserialize(criticData);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new WGAN<T>(
            Generator.Architecture,
            Critic.Architecture,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
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
