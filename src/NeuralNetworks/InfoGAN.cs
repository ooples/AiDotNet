using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Information Maximizing Generative Adversarial Network (InfoGAN), which learns
/// disentangled representations in an unsupervised manner by maximizing mutual information
/// between latent codes and generated observations.
/// </summary>
/// <remarks>
/// <para>
/// InfoGAN extends the GAN framework by:
/// - Decomposing the input noise into incompressible noise (z) and latent codes (c)
/// - Maximizing the mutual information I(c; G(z,c)) between codes and generated images
/// - Learning interpretable and disentangled representations automatically
/// - Using an auxiliary network Q to approximate the posterior P(c|x)
/// - Enabling control over semantic features without labeled data
/// </para>
/// <para><b>For Beginners:</b> InfoGAN learns to separate different features automatically.
///
/// Key concept:
/// - Splits random input into two parts:
///   1. Random noise (z): provides variety
///   2. Latent codes (c): control specific features
/// - Learns what each code controls WITHOUT labels
/// - Example: For faces, might learn codes for:
///   * Code 1: controls rotation
///   * Code 2: controls width
///   * Code 3: controls lighting
///
/// How it works:
/// - Generator uses both z and c to create images
/// - Auxiliary network Q tries to predict c from the generated image
/// - If Q can predict c accurately, the codes are meaningful
/// - This forces codes to represent interpretable features
///
/// Use cases:
/// - Discover semantic features in datasets
/// - Disentangled representation learning
/// - Controllable image generation
/// - Feature manipulation (change one aspect, keep others)
///
/// Reference: Chen et al., "InfoGAN: Interpretable Representation Learning by
/// Information Maximizing Generative Adversarial Nets" (2016)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InfoGAN<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The optimizer used for training the generator network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the generator.
    /// The optimizer handles momentum, adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the generator
    /// learns from its mistakes and adjusts its parameters during training.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer used for training the discriminator network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the discriminator.
    /// The optimizer handles momentum, adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the discriminator
    /// learns to better distinguish real images from fake ones.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    /// <summary>
    /// The optimizer used for training the Q network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the Q network,
    /// which predicts latent codes from generated images. The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the Q network
    /// learns to predict which latent codes were used to generate an image.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _qNetworkOptimizer;

    /// <summary>
    /// List of recent generator losses for tracking training progress.
    /// </summary>
    private readonly List<T> _generatorLosses = new List<T>();

    /// <summary>
    /// List of recent discriminator losses for tracking training progress.
    /// </summary>
    private readonly List<T> _discriminatorLosses = new List<T>();

    /// <summary>
    /// The size of the latent code c.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The latent code represents interpretable factors of variation. A typical value is 10-20.
    /// This can include both discrete codes (for categorical features) and continuous codes
    /// (for continuous features like rotation angle).
    /// </para>
    /// <para><b>For Beginners:</b> How many controllable features to learn.
    ///
    /// - Larger values: more features to discover, but may be harder to train
    /// - Smaller values: fewer but potentially clearer features
    /// - Typical: 10 codes can capture many important features
    /// </para>
    /// </remarks>
    private int _latentCodeSize;

    /// <summary>
    /// The coefficient for the mutual information loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the trade-off between the standard GAN objective and the mutual information
    /// objective. A typical value is 1.0. Higher values enforce stronger disentanglement.
    /// </para>
    /// <para><b>For Beginners:</b> How important is the feature learning vs image quality.
    ///
    /// - Higher (e.g., 2.0): prioritize learning clear features
    /// - Lower (e.g., 0.5): prioritize image quality
    /// - Default (1.0): balanced approach
    /// </para>
    /// </remarks>
    private double _mutualInfoCoefficient;

    /// <summary>
    /// Gets the generator network.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    /// <summary>
    /// Gets the auxiliary Q network that predicts latent codes from images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Q network shares most of its parameters with the discriminator (up to the
    /// last layer). It outputs the predicted latent code distribution given an image.
    /// This network is key to maximizing mutual information.
    /// </para>
    /// <para><b>For Beginners:</b> The Q network is the "feature detector".
    ///
    /// - Takes an image as input
    /// - Outputs: "I think these codes were used to make this"
    /// - Training makes Q better at guessing codes
    /// - This forces generator to use codes meaningfully
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> QNetwork { get; private set; }

    /// <summary>
    /// Gets the total number of trainable parameters in the InfoGAN.
    /// </summary>
    public override int ParameterCount => Generator.GetParameterCount() + Discriminator.GetParameterCount() + QNetwork.GetParameterCount();

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined InfoGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateInfoGANArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType)
    {
        // Validate before base initializer to throw ArgumentNullException instead of NRE
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture));
        }
        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture));
        }

        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: discriminatorArchitecture.InputHeight,
                inputWidth: discriminatorArchitecture.InputWidth,
                inputDepth: discriminatorArchitecture.InputDepth,
                outputSize: discriminatorArchitecture.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: generatorArchitecture.InputSize,
            outputSize: discriminatorArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="InfoGAN{T}"/> class with the specified architecture and training parameters.
    /// </summary>
    /// <param name="generatorArchitecture">The architecture for the generator network.</param>
    /// <param name="discriminatorArchitecture">The architecture for the discriminator network.</param>
    /// <param name="qNetworkArchitecture">The architecture for the Q network (should output latentCodeSize values).</param>
    /// <param name="latentCodeSize">The size of the latent code (number of controllable features).</param>
    /// <param name="inputType">The type of input data (e.g., ThreeDimensional for images).</param>
    /// <param name="generatorOptimizer">
    /// Optional optimizer for the generator. If null, an Adam optimizer with default settings is created.
    /// </param>
    /// <param name="discriminatorOptimizer">
    /// Optional optimizer for the discriminator. If null, an Adam optimizer with default settings is created.
    /// </param>
    /// <param name="qNetworkOptimizer">
    /// Optional optimizer for the Q network. If null, an Adam optimizer with default settings is created.
    /// </param>
    /// <param name="lossFunction">Optional loss function. If null, the default loss function for generative tasks is used.</param>
    /// <param name="mutualInfoCoefficient">
    /// The coefficient for mutual information loss. Higher values prioritize feature learning. Default is 1.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an InfoGAN with three networks:
    /// - Generator: Creates images from noise and latent codes
    /// - Discriminator: Determines if images are real or fake
    /// - Q Network: Predicts latent codes from generated images
    ///
    /// The mutual information loss encourages the generator to use the latent codes in meaningful ways
    /// that can be recovered by the Q network.
    /// </para>
    /// <para><b>For Beginners:</b> InfoGAN learns controllable features automatically:
    /// - The generator creates images using random noise + controllable codes
    /// - The Q network tries to guess which codes were used
    /// - This forces the codes to represent real, interpretable features
    /// - After training, you can manipulate specific features by changing the codes
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown when any of the architecture parameters is null.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when latentCodeSize is not positive or mutualInfoCoefficient is negative.
    /// </exception>
    public InfoGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        NeuralNetworkArchitecture<T> qNetworkArchitecture,
        int latentCodeSize,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? qNetworkOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        double mutualInfoCoefficient = 1.0)
        : base(CreateInfoGANArchitecture(generatorArchitecture, discriminatorArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        if (generatorArchitecture is null)
            throw new ArgumentNullException(nameof(generatorArchitecture));
        if (discriminatorArchitecture is null)
            throw new ArgumentNullException(nameof(discriminatorArchitecture));
        if (qNetworkArchitecture is null)
            throw new ArgumentNullException(nameof(qNetworkArchitecture));
        if (latentCodeSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentCodeSize), latentCodeSize, "Latent code size must be positive.");
        if (mutualInfoCoefficient < 0)
            throw new ArgumentOutOfRangeException(nameof(mutualInfoCoefficient), mutualInfoCoefficient, "Mutual information coefficient must be non-negative.");

        _latentCodeSize = latentCodeSize;
        _mutualInfoCoefficient = mutualInfoCoefficient;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        QNetwork = new ConvolutionalNeuralNetwork<T>(qNetworkArchitecture);

        // Initialize optimizers - use provided optimizers or create default Adam optimizers
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator);
        _discriminatorOptimizer = discriminatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator);
        _qNetworkOptimizer = qNetworkOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(QNetwork);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for InfoGAN.
    /// </summary>
    /// <param name="realImages">Real images.</param>
    /// <param name="noise">Random noise (z).</param>
    /// <param name="latentCodes">Latent codes (c) to condition generation.</param>
    /// <returns>Tuple of (discriminator loss, generator loss, mutual info loss).</returns>
    /// <remarks>
    /// <para>
    /// InfoGAN training:
    /// 1. Train discriminator (standard GAN objective)
    /// 2. Train generator with GAN loss + mutual information loss
    /// 3. Train Q network to predict latent codes from generated images
    /// </para>
    /// <para><b>For Beginners:</b> One round of InfoGAN training.
    ///
    /// Steps:
    /// 1. Generate images using noise + latent codes
    /// 2. Train discriminator to spot fakes (standard GAN)
    /// 3. Train generator to fool discriminator
    /// 4. Train Q network to guess the codes from images
    /// 5. Make generator use codes that Q can predict
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss, T mutualInfoLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> noise,
        Tensor<T> latentCodes)
    {
        // Validate inputs are non-null and have consistent batch sizes
        if (realImages is null)
        {
            throw new ArgumentNullException(nameof(realImages));
        }
        if (noise is null)
        {
            throw new ArgumentNullException(nameof(noise));
        }
        if (latentCodes is null)
        {
            throw new ArgumentNullException(nameof(latentCodes));
        }

        if (realImages.Shape.Length == 0 || noise.Shape.Length == 0 || latentCodes.Shape.Length == 0)
        {
            throw new ArgumentException("Input tensors must have at least one dimension.");
        }

        int batchSize = realImages.Shape[0];
        if (noise.Shape[0] != batchSize || latentCodes.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch size mismatch: realImages has {batchSize}, " +
                $"noise has {noise.Shape[0]}, latentCodes has {latentCodes.Shape[0]}. " +
                "All inputs must have the same batch size.");
        }

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);
        QNetwork.SetTrainingMode(true);

        // ----- Train Discriminator -----

        // Concatenate noise and latent codes for generator
        var generatorInput = ConcatenateTensors(noise, latentCodes);

        // Generate fake images
        var fakeImages = Generator.Predict(generatorInput);

        // Real labels
        var realLabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        // Train on real images
        var realPredictions = Discriminator.Predict(realImages);
        T realLoss = CalculateBinaryLoss(realPredictions, realLabels, batchSize);

        var realGradients = CalculateBinaryGradients(realPredictions, realLabels, batchSize);
        Discriminator.Backpropagate(realGradients);
        UpdateDiscriminatorParameters();

        // Train on fake images
        var fakePredictions = Discriminator.Predict(fakeImages);
        T fakeLoss = CalculateBinaryLoss(fakePredictions, fakeLabels, batchSize);

        var fakeGradients = CalculateBinaryGradients(fakePredictions, fakeLabels, batchSize);
        Discriminator.Backpropagate(fakeGradients);
        UpdateDiscriminatorParameters();

        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));

        // ----- Train Generator and Q Network -----

        Generator.SetTrainingMode(true);
        // Keep Discriminator and QNetwork in training mode - required for backpropagation
        // We just don't call UpdateDiscriminatorParameters() during generator training
        QNetwork.SetTrainingMode(true);

        // Generate new fake images
        var newGeneratorInput = ConcatenateTensors(noise, latentCodes);
        var newFakeImages = Generator.Predict(newGeneratorInput);

        // GAN loss: fool the discriminator
        var genPredictions = Discriminator.Predict(newFakeImages);
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);
        T ganLoss = CalculateBinaryLoss(genPredictions, allRealLabels, batchSize);

        // Mutual information loss: Q predicts latent codes
        var predictedCodes = QNetwork.Predict(newFakeImages);
        T mutualInfoLoss = CalculateMutualInfoLoss(predictedCodes, latentCodes, batchSize);

        // Total generator loss
        T miCoeff = NumOps.FromDouble(_mutualInfoCoefficient);
        T generatorLoss = NumOps.Add(ganLoss, NumOps.Multiply(miCoeff, mutualInfoLoss));

        // Backpropagate through discriminator (for GAN loss) to get input gradients
        var ganGradients = CalculateBinaryGradients(genPredictions, allRealLabels, batchSize);
        var discInputGradients = Discriminator.BackwardWithInputGradient(ganGradients);

        // Backpropagate through Q network (for MI loss) to get input gradients
        var miGradients = CalculateMutualInfoGradients(predictedCodes, latentCodes, batchSize);
        var qInputGradients = QNetwork.BackwardWithInputGradient(miGradients);

        // Combine gradients - verify shapes match
        if (!discInputGradients.Shape.SequenceEqual(qInputGradients.Shape))
        {
            throw new InvalidOperationException(
                $"Gradient shape mismatch: discriminator input gradients have shape " +
                $"[{string.Join(", ", discInputGradients.Shape)}] but Q network input gradients have shape " +
                $"[{string.Join(", ", qInputGradients.Shape)}]. Both must match for gradient combining.");
        }

        var combinedGradients = new Tensor<T>(discInputGradients.Shape);
        int gradLength = discInputGradients.Shape.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < gradLength; i++)
        {
            combinedGradients.SetFlat(i, NumOps.Add(
                discInputGradients.GetFlat(i),
                NumOps.Multiply(miCoeff, qInputGradients.GetFlat(i))
            ));
        }

        // Backpropagate through generator
        Generator.Backward(combinedGradients);
        UpdateGeneratorParameters();
        UpdateQNetworkParameters();

        // Track losses
        _discriminatorLosses.Add(discriminatorLoss);
        _generatorLosses.Add(generatorLoss);

        if (_discriminatorLosses.Count > 100)
        {
            _discriminatorLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss, mutualInfoLoss);
    }

    /// <summary>
    /// Calculates mutual information loss (MSE between predicted and true codes).
    /// </summary>
    private T CalculateMutualInfoLoss(Tensor<T> predictedCodes, Tensor<T> trueCodes, int batchSize)
    {
        T totalLoss = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _latentCodeSize; c++)
            {
                T diff = NumOps.Subtract(predictedCodes[b, c], trueCodes[b, c]);
                T squaredDiff = NumOps.Multiply(diff, diff);
                totalLoss = NumOps.Add(totalLoss, squaredDiff);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble((double)batchSize * _latentCodeSize));
    }

    /// <summary>
    /// Calculates gradients for mutual information loss.
    /// </summary>
    private Tensor<T> CalculateMutualInfoGradients(Tensor<T> predictedCodes, Tensor<T> trueCodes, int batchSize)
    {
        var gradients = new Tensor<T>(predictedCodes.Shape);
        T scale = NumOps.FromDouble(2.0 / ((double)batchSize * _latentCodeSize));

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _latentCodeSize; c++)
            {
                T diff = NumOps.Subtract(predictedCodes[b, c], trueCodes[b, c]);
                gradients[b, c] = NumOps.Multiply(scale, diff);
            }
        }

        return gradients;
    }

    /// <summary>
    /// Calculates binary cross-entropy loss.
    /// </summary>
    private T CalculateBinaryLoss(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < batchSize; i++)
        {
            T prediction = predictions[i, 0];
            T target = targets[i, 0];

            T logP = NumOps.Log(NumOps.Add(prediction, epsilon));
            T logOneMinusP = NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, prediction), epsilon));

            T loss = NumOps.Negate(NumOps.Add(
                NumOps.Multiply(target, logP),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, target), logOneMinusP)
            ));

            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates gradients for binary cross-entropy.
    /// </summary>
    private Tensor<T> CalculateBinaryGradients(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        var gradients = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = NumOps.Divide(
                NumOps.Subtract(predictions[i, 0], targets[i, 0]),
                NumOps.FromDouble(batchSize)
            );
        }

        return gradients;
    }

    /// <summary>
    /// Creates a label tensor.
    /// </summary>
    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var tensor = new Tensor<T>(new int[] { batchSize, 1 });
        // === Vectorized tensor fill using IEngine (Phase B: US-GPU-015) ===
        Engine.TensorFill(tensor, value);
        return tensor;
    }

    /// <summary>
    /// Concatenates noise and latent codes.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> noise, Tensor<T> codes)
    {
        // Require exactly rank 2 since we only handle [batch, features] concatenation
        if (noise.Shape.Length != 2 || codes.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"Both noise and codes must be exactly 2D tensors [batch, features]. " +
                $"Got noise with rank {noise.Shape.Length} and codes with rank {codes.Shape.Length}.");
        }

        int noiseBatchSize = noise.Shape[0];
        int codesBatchSize = codes.Shape[0];

        if (noiseBatchSize != codesBatchSize)
        {
            throw new ArgumentException(
                $"Batch size mismatch: noise has {noiseBatchSize} samples, codes has {codesBatchSize} samples.");
        }

        int batchSize = noiseBatchSize;
        int noiseSize = noise.Shape[1];
        int codeSize = codes.Shape[1];

        var result = new Tensor<T>(new int[] { batchSize, noiseSize + codeSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i++)
            {
                result[b, i] = noise[b, i];
            }
            for (int i = 0; i < codeSize; i++)
            {
                result[b, noiseSize + i] = codes[b, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Generates images with specific latent codes.
    /// </summary>
    /// <param name="noise">Random noise.</param>
    /// <param name="latentCodes">Latent codes to control generation.</param>
    /// <returns>Generated images.</returns>
    public Tensor<T> Generate(Tensor<T> noise, Tensor<T> latentCodes)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, latentCodes);
        return Generator.Predict(input);
    }

    /// <summary>
    /// Generates random noise tensor using vectorized Gaussian noise generation with CPU/GPU acceleration.
    /// </summary>
    /// <param name="batchSize">The number of noise samples in the batch.</param>
    /// <param name="noiseSize">The size of each noise sample.</param>
    /// <returns>A tensor of shape [batchSize, noiseSize] filled with Gaussian noise.</returns>
    public Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        if (noiseSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(noiseSize), noiseSize, "Noise size must be positive.");

        // Guard against int overflow in element count calculation
        if (batchSize > int.MaxValue / noiseSize)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"Batch size ({batchSize}) * noise size ({noiseSize}) would overflow int.MaxValue.");
        }

        var totalElements = batchSize * noiseSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);
        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Generates random latent codes (continuous, uniform in [-1, 1]).
    /// </summary>
    public Tensor<T> GenerateRandomLatentCodes(int batchSize)
    {
        var random = RandomHelper.ThreadSafeRandom;
        var codes = new Tensor<T>(new int[] { batchSize, _latentCodeSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _latentCodeSize; c++)
            {
                codes[b, c] = NumOps.FromDouble(random.NextDouble() * 2.0 - 1.0);
            }
        }

        return codes;
    }

    /// <summary>
    /// Updates the parameters of the generator network using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the generator,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the generator's weights
    /// based on how well it fooled the discriminator and produced recoverable codes.
    /// </para>
    /// </remarks>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _generatorOptimizer.UpdateParameters(parameters, gradients);
        Generator.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Updates the parameters of the discriminator network using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the discriminator,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the discriminator's weights
    /// based on how well it distinguished real images from fake ones.
    /// </para>
    /// </remarks>
    private void UpdateDiscriminatorParameters()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _discriminatorOptimizer.UpdateParameters(parameters, gradients);
        Discriminator.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Updates the parameters of the Q network using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the Q network,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the Q network's weights
    /// based on how well it predicted the latent codes from generated images.
    /// </para>
    /// </remarks>
    private void UpdateQNetworkParameters()
    {
        var parameters = QNetwork.GetParameters();
        var gradients = QNetwork.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _qNetworkOptimizer.UpdateParameters(parameters, gradients);
        QNetwork.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Resets the state of all optimizers to their initial values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets all three optimizers (generator, discriminator, and Q network)
    /// to their initial state. This is useful when restarting training or when you want
    /// to clear accumulated momentum and adaptive learning rate information.
    /// </para>
    /// <para><b>For Beginners:</b> Call this method when you want to start fresh with
    /// training, as if the model had never been trained before. The network weights
    /// remain unchanged, but the optimizer's memory of past gradients is cleared.
    /// </para>
    /// </remarks>
    public void ResetOptimizerState()
    {
        _generatorOptimizer.Reset();
        _discriminatorOptimizer.Reset();
        _qNetworkOptimizer.Reset();
    }

    protected override void InitializeLayers()
    {
        // InfoGAN doesn't use layers directly
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        int batchSize = expectedOutput.Shape[0];
        var codes = GenerateRandomLatentCodes(batchSize);
        TrainStep(expectedOutput, input, codes);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.InfoGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "QNetworkParameters", QNetwork.GetParameterCount() },
                { "LatentCodeSize", _latentCodeSize },
                { "MutualInfoCoefficient", _mutualInfoCoefficient }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes InfoGAN-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the InfoGAN-specific configuration and all three networks.
    /// Optimizer state is managed by the optimizer implementations themselves.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the InfoGAN's settings and all
    /// three networks (generator, discriminator, and Q network) to a file.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize InfoGAN-specific hyperparameters
        writer.Write(_latentCodeSize);
        writer.Write(_mutualInfoCoefficient);

        // Serialize all three networks
        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);

        var qNetworkBytes = QNetwork.Serialize();
        writer.Write(qNetworkBytes.Length);
        writer.Write(qNetworkBytes);
    }

    /// <summary>
    /// Deserializes InfoGAN-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the InfoGAN-specific configuration and all three networks.
    /// After deserialization, the optimizers are reset to their initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the InfoGAN's settings and all
    /// three networks (generator, discriminator, and Q network) from a file.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        const int MaxNetworkDataLength = 100 * 1024 * 1024; // 100 MB max per network

        // Deserialize InfoGAN-specific hyperparameters
        _latentCodeSize = reader.ReadInt32();
        _mutualInfoCoefficient = reader.ReadDouble();

        // Deserialize all three networks with bounds checking
        int generatorDataLength = reader.ReadInt32();
        if (generatorDataLength < 0 || generatorDataLength > MaxNetworkDataLength)
        {
            throw new InvalidDataException(
                $"Invalid generator data length: {generatorDataLength}. " +
                $"Must be between 0 and {MaxNetworkDataLength}.");
        }
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        if (discriminatorDataLength < 0 || discriminatorDataLength > MaxNetworkDataLength)
        {
            throw new InvalidDataException(
                $"Invalid discriminator data length: {discriminatorDataLength}. " +
                $"Must be between 0 and {MaxNetworkDataLength}.");
        }
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);

        int qNetworkDataLength = reader.ReadInt32();
        if (qNetworkDataLength < 0 || qNetworkDataLength > MaxNetworkDataLength)
        {
            throw new InvalidDataException(
                $"Invalid Q network data length: {qNetworkDataLength}. " +
                $"Must be between 0 and {MaxNetworkDataLength}.");
        }
        byte[] qNetworkData = reader.ReadBytes(qNetworkDataLength);
        QNetwork.Deserialize(qNetworkData);

        // Reset optimizer state after loading network weights
        ResetOptimizerState();
    }

    /// <summary>
    /// Creates a new instance of the InfoGAN with the same configuration.
    /// </summary>
    /// <returns>A new InfoGAN instance with the same architecture and hyperparameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a fresh InfoGAN instance with the same network architectures
    /// and hyperparameters. The new instance has freshly initialized optimizers.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the InfoGAN structure
    /// but with new, untrained networks and fresh optimizers.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new InfoGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            QNetwork.Architecture,
            _latentCodeSize,
            Architecture.InputType,
            generatorOptimizer: null,
            discriminatorOptimizer: null,
            qNetworkOptimizer: null,
            _lossFunction,
            _mutualInfoCoefficient);
    }

    /// <summary>
    /// Updates the parameters of all networks in the InfoGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();
        int qNetworkCount = QNetwork.GetParameterCount();

        int totalCount = generatorCount + discriminatorCount + qNetworkCount;

        if (parameters.Length != totalCount)
        {
            throw new ArgumentException(
                $"Parameters vector length mismatch: expected {totalCount} " +
                $"(Generator: {generatorCount}, Discriminator: {discriminatorCount}, " +
                $"QNetwork: {qNetworkCount}), but received {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;

        // Update Generator parameters
        var generatorParams = new Vector<T>(generatorCount);
        for (int i = 0; i < generatorCount; i++)
            generatorParams[i] = parameters[offset + i];
        Generator.UpdateParameters(generatorParams);
        offset += generatorCount;

        // Update Discriminator parameters
        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[offset + i];
        Discriminator.UpdateParameters(discriminatorParams);
        offset += discriminatorCount;

        // Update QNetwork parameters
        var qNetworkParams = new Vector<T>(qNetworkCount);
        for (int i = 0; i < qNetworkCount; i++)
            qNetworkParams[i] = parameters[offset + i];
        QNetwork.UpdateParameters(qNetworkParams);
    }
}
