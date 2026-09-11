using AiDotNet.Tensors.Engines.Autodiff;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

using System.Linq;

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
/// <example>
/// <code>
/// var options = new InfoGANOptions { LatentSize = 62, NumCategoricalCodes = 10, NumContinuousCodes = 2 };
/// var model = new InfoGAN&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 74 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets", "https://arxiv.org/abs/1606.03657", Year = 2016, Authors = "Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel")]
public partial class InfoGAN<T> : ImageGeneratorModelLayoutBase<T>
{

    // Generator, Discriminator and QNetwork are discovered as sub-network members, in declaration
    // order, which is the order this hook used and therefore the serialization order.
    // Removed under AIDN082.
    private readonly InfoGANOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private static AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CreateStandardGanAdamOptions(double learningRate = 2e-4)
        => new()
        {
            InitialLearningRate = 0.0002,
            Beta1 = 0.5,
            Beta2 = 0.999,
        };

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
    private T _mutualInfoCoefficient;

    /// <summary>
    /// Gets the generator network.
    /// </summary>
    public NeuralNetworkBase<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network.
    /// </summary>
    public NeuralNetworkBase<T> Discriminator { get; private set; }

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
    public NeuralNetworkBase<T> QNetwork { get; private set; }

    private ILossFunction<T> _lossFunction;

    // CNN backbone for image domains (Chen et al. 2016 §4.1/§4.4),
    // MLP backbone for tabular / mixture-of-categoricals (§4.2).
    private static NeuralNetworkBase<T> CreateBackboneForArchitecture(NeuralNetworkArchitecture<T> arch)
    {
        if (arch.InputType == InputType.TwoDimensional || arch.InputType == InputType.ThreeDimensional)
            return new ConvolutionalNeuralNetwork<T>(arch);
        return new FeedForwardNeuralNetwork<T>(arch);
    }

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
    /// Creates an InfoGAN with default architectures derived from a single architecture.
    /// Per Chen et al. 2016: latent code size 10, mutual info coefficient 1.0.
    /// </summary>
    /// <param name="architecture">The shared architecture used for generator, discriminator, and Q network.</param>
    /// <param name="latentCodeSize">The size of the latent code. Default is 10.</param>
    /// <param name="mutualInfoCoefficient">The coefficient for mutual information loss. Default is 1.0.</param>
    /// <param name="options">Optional InfoGAN options.</param>
    public InfoGAN(
        NeuralNetworkArchitecture<T> architecture,
        int latentCodeSize = 10,
        double mutualInfoCoefficient = 1.0,
        InfoGANOptions? options = null)
        : this(architecture, architecture, architecture, latentCodeSize, architecture.InputType,
               mutualInfoCoefficient: mutualInfoCoefficient, options: options)
    {
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
    /// <param name="options">Optional InfoGAN options.</param>
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
        double mutualInfoCoefficient = 1.0,
        InfoGANOptions? options = null)
        : base(CreateInfoGANArchitecture(generatorArchitecture, discriminatorArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _options = options ?? new InfoGANOptions();
        Options = _options;
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
        _mutualInfoCoefficient = NumOps.FromDouble(mutualInfoCoefficient);

        Generator = CreateBackboneForArchitecture(generatorArchitecture);
        Discriminator = CreateBackboneForArchitecture(discriminatorArchitecture);
        QNetwork = CreateBackboneForArchitecture(qNetworkArchitecture);

        // Initialize optimizers - use provided optimizers or create default GAN-standard Adam optimizers.
        // Chen et al. 2016, appendix: learning rate 2e-4 for D and 1e-3 for G. Q is D's head in the
        // paper, so it trains at D's rate.
        _generatorOptimizer = generatorOptimizer
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator, CreateStandardGanAdamOptions(learningRate: 1e-3));
        _discriminatorOptimizer = discriminatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator, CreateStandardGanAdamOptions());
        _qNetworkOptimizer = qNetworkOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(QNetwork, CreateStandardGanAdamOptions());

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

        // ONE generator forward, recorded on the generator's tape (#1390); D trains on a detached copy.
        var generatorInput = ConcatenateTensors(noise, latentCodes);
        using var generatorTape = new GradientTape<T>();
        var fakeTracked = Generator.ForwardForTraining(generatorInput);
        var fakeImages = new Tensor<T>(fakeTracked.Shape.ToArray());
        fakeTracked.AsSpan().CopyTo(fakeImages.AsWritableSpan());

        // ----- Discriminator: maximise V(D, G) -----
        // Its two updates used to step on GetParameterGradients() after no backward pass at all.
        T discriminatorLoss;
        using (var discriminatorTape = new GradientTape<T>())
        {
            var realScores = Discriminator.ForwardForTraining(realImages);
            var fakeScores = Discriminator.ForwardForTraining(fakeImages);
            var discriminatorObjective = Engine.TensorAdd(
                Discriminator.BinaryCrossEntropyOnTape(realScores, targetIsReal: true),
                Discriminator.BinaryCrossEntropyOnTape(fakeScores, targetIsReal: false));
            discriminatorLoss = StepOnTape(discriminatorTape, discriminatorObjective, Discriminator,
                _discriminatorOptimizer);
        }

        // ----- Generator and Q: minimise V(D, G) - lambda * L_I(G, Q) (Chen et al. 2016, eq. 6) -----
        // One loss, two networks: the generator AND the Q network are stepped, each by its own
        // optimizer. The old closure trained the generator alone, so Q -- whose loss was in it -- never
        // moved. The adversarial term is the non-saturating -log D(G(z, c)). For the continuous codes
        // this model draws, Q(c|x) is a unit-variance factored Gaussian, whose negative log-likelihood is
        // the squared error up to a constant -- the form the paper uses for continuous codes.
        var generatorScores = Discriminator.ForwardFrozenOnTape(fakeTracked);
        var adversarialLoss = Discriminator.BinaryCrossEntropyOnTape(generatorScores, targetIsReal: true);
        var predictedCodes = QNetwork.ForwardForTraining(fakeTracked);
        var codeError = Engine.TensorSubtract(predictedCodes, latentCodes);
        var mutualInfoTensor = Engine.ReduceMean(
            Engine.TensorMultiply(codeError, codeError),
            Enumerable.Range(0, codeError.Shape.Length).ToArray(),
            keepDims: false);
        var generatorObjective = Engine.TensorAdd(
            adversarialLoss, Engine.TensorMultiplyScalar(mutualInfoTensor, _mutualInfoCoefficient));
        T generatorLoss = StepOnTape(generatorTape, generatorObjective, new[]
        {
            ((NeuralNetworkBase<T>)Generator, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_generatorOptimizer),
            ((NeuralNetworkBase<T>)QNetwork, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_qNetworkOptimizer),
        });
        T mutualInfoLoss = mutualInfoTensor.Length > 0 ? mutualInfoTensor[0] : NumOps.Zero;

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
        var gradients = new Tensor<T>(predictedCodes._shape);
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
        var gradients = new Tensor<T>(predictions._shape);

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
        return Engine.TensorConcatenate([noise, codes], axis: 1);
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

    /// <summary>
    /// Forwards weight-lifetime configuration to the generator, discriminator,
    /// and Q-network sub-networks. Without this override the registry-side
    /// effect would still happen (it is process-global), but the sub-networks'
    /// trainable tensors would not be registered, defeating the offload path.
    /// </summary>
    internal override void ConfigureWeightLifetime(
        GpuOffloadOptions options,
        IGpuOffloadAllocator? allocator = null)
    {
        base.ConfigureWeightLifetime(options, allocator);
        Generator.ConfigureWeightLifetime(options, allocator);
        Discriminator.ConfigureWeightLifetime(options, allocator);
        QNetwork.ConfigureWeightLifetime(options, allocator);
    }

    protected override void InitializeLayers()
    {
        // InfoGAN doesn't use layers directly
    }

    /// <summary>
    /// Surfaces named activations from each sub-network so introspection
    /// invariants (NamedLayerActivations_ShouldBeNonEmpty) see real
    /// per-network state. The base implementation walks
    /// <see cref="Layers"/>, but InfoGAN keeps its layers in three
    /// sub-networks and leaves Layers empty — so the inherited path
    /// returned an empty dictionary on every call (#1224 Cluster F).
    /// Per Chen et al. 2016 §3 the Q-network shares its early layers
    /// with the discriminator on a real implementation; here we surface
    /// all three independently so callers see the architectural surface
    /// even when share-layers wasn't configured.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var prepared = AppendCodesIfNoiseOnly(input);
        var result = new Dictionary<string, Tensor<T>>();
        foreach (var kv in Generator.GetNamedLayerActivations(prepared))
            result["Generator/" + kv.Key] = kv.Value;
        // Use the generated image as input to the discriminator + Q-network
        // for the named-activation surface (mirrors what TrainStep does).
        var fakeImage = Generator.Predict(prepared);
        foreach (var kv in Discriminator.GetNamedLayerActivations(fakeImage))
            result["Discriminator/" + kv.Key] = kv.Value;
        foreach (var kv in QNetwork.GetNamedLayerActivations(fakeImage))
            result["QNetwork/" + kv.Key] = kv.Value;
        return result;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Mirror Train's noise-only contract (Chen et al. 2016 §3 — public
        // Train(noise, realImages) treats `input` as raw z and concatenates
        // a code tensor internally before generator.Predict). If the input
        // already includes the latent-code dimensions, pass it straight
        // through; otherwise append default-zero codes so callers can
        // probe the generator with noise-only inputs (the natural cGAN /
        // InfoGAN public API). Without this, NN test invariants fed
        // <c>InputShape</c>-sized noise tensors and got
        // "Expected shape [noise+codes], but got [noise]" mismatches
        // because the test base uses one shape for both Train and Predict
        // (#1224 Cluster F).
        var prepared = AppendCodesIfNoiseOnly(input);
        return Generator.Predict(prepared);
    }

    /// <summary>
    /// Defines the InfoGAN forward graph for tape-based training. InfoGAN's
    /// generator / discriminator / Q-network live outside <c>Layers</c>
    /// (which is intentionally empty), so the default
    /// <c>ForwardForTraining</c> would walk an empty layer chain and return
    /// the raw input. Overriding to dispatch through
    /// <c>Generator.ForwardForTraining</c> matches the inference-side
    /// <c>Predict</c> contract and lets <c>tape.ComputeGradients</c> flow
    /// back into the generator's weights. Discriminator + Q-network +
    /// mutual-information losses are handled by <see cref="TrainStep"/>.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return Generator.ForwardForTraining(AppendCodesIfNoiseOnly(input));
    }

    /// <summary>
    /// Appends a default-zero latent-code block to the input when the
    /// input is shaped as raw noise (last dim = generator.InputSize -
    /// latentCodeSize). Inputs already shaped with codes pass through
    /// unchanged. Keeps the public Predict / ForwardForTraining contract
    /// noise-only consistent with Train.
    /// </summary>
    private Tensor<T> AppendCodesIfNoiseOnly(Tensor<T> input)
    {
        int genInputSize = Generator.Architecture.InputSize;
        int noiseOnlySize = genInputSize - _latentCodeSize;
        if (input.Rank == 0 || input.Shape[input.Rank - 1] != noiseOnlySize)
            return input;

        // Append zero-codes along the last axis. Treat the input as
        // [..., noiseDim] and emit [..., noiseDim + latentCodeSize].
        var newShape = new int[input.Rank];
        for (int i = 0; i < input.Rank - 1; i++) newShape[i] = input.Shape[i];
        newShape[input.Rank - 1] = genInputSize;
        var output = new Tensor<T>(newShape);

        int trailing = input.Shape[input.Rank - 1];
        int batches = 1;
        for (int i = 0; i < input.Rank - 1; i++) batches *= input.Shape[i];
        for (int b = 0; b < batches; b++)
        {
            int srcBase = b * trailing;
            int dstBase = b * genInputSize;
            for (int i = 0; i < trailing; i++)
                output.Data.Span[dstBase + i] = input.Data.Span[srcBase + i];
            // Latent-code slots are already zero from Tensor's default ctor.
        }
        return output;
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "QNetworkParameters", QNetwork.GetParameterCount() },
                { "LatentCodeSize", _latentCodeSize },
                { "MutualInfoCoefficient", NumOps.ToDouble(_mutualInfoCoefficient) }
            },
            ModelData = SerializeForMetadata()
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


    // UpdateParameters split the vector between Generator, Discriminator and QNetwork;
    // GetExtraTrainableLayers yields those three in the same order, so the base reproduces the
    // split. Removed under AIDN082.
}
