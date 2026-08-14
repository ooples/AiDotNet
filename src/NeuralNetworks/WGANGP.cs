using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
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
/// <example>
/// <code>
/// var options = new WGANGPOptions { LatentSize = 100, GradientPenaltyWeight = 10.0 };
/// var model = new WGANGP&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 100 });
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
[ResearchPaper("Improved Training of Wasserstein GANs", "https://arxiv.org/abs/1704.00028", Year = 2017, Authors = "Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville")]
[PreprocessesInput("ShapeAsGeneratorInput reshapes latent vectors to the generator architecture before Layers[0] runs.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true)]
public partial class WGANGP<T> : ImageGeneratorModelLayoutBase<T>
{

    // ParameterCount was Generator.GetParameterCount() + Critic.GetParameterCount(), which the
    // base already computes: this model puts BOTH sub-networks' layers in its own Layers, so the
    // base walk covers them (measured 5,637, matching the override and the vector). Removing it
    // also drops a dependence on GetParameterCount(), which several types in this hierarchy SHADOW
    // with `public new`, so its result depends on the static type of the reference.
    private readonly WGANGPOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    public NeuralNetworkBase<T> Generator { get; private set; }

    /// <summary>
    /// Gets the critic network that evaluates data quality.
    /// </summary>
    public NeuralNetworkBase<T> Critic { get; private set; }

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
    /// Derives the scalar critic required by WGAN-GP when callers use the convenience
    /// constructor that supplies only the generator architecture.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateDefaultCriticArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture)
    {
        if (generatorArchitecture is null)
            throw new ArgumentNullException(nameof(generatorArchitecture));

        // The convenience constructor describes z -> generated sample. For vector generators,
        // the critic therefore consumes generator.OutputSize values and emits one unrestricted
        // Wasserstein score. The paper also evaluates MLPs (including language models), so a CNN
        // is neither required nor valid for this case.
        if (generatorArchitecture.InputType is InputType.OneDimensional or InputType.TwoDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: generatorArchitecture.Complexity,
                inputSize: generatorArchitecture.OutputSize,
                outputSize: 1);
        }

        // The critic is a real-valued scalar function, never another image classifier.
        // Reusing the generator architecture here gave the critic the generic CNN helper's
        // Softmax output (64 probabilities in the generated fixture). A one-class Softmax is
        // constant and a multi-class Softmax is still not a Wasserstein score; either choice can
        // leave the entire first critic step with zero gradients. Gulrajani et al. Algorithm 1
        // requires D(x) to be an unrestricted scalar and applies the penalty to its input gradient.
        return new NeuralNetworkArchitecture<T>(
            inputType: generatorArchitecture.InputType,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: generatorArchitecture.Complexity,
            inputSize: generatorArchitecture.InputSize,
            inputHeight: generatorArchitecture.InputHeight,
            inputWidth: generatorArchitecture.InputWidth,
            inputDepth: generatorArchitecture.InputDepth,
            outputSize: 1,
            layers: LayerHelper<T>.CreateDefaultWGANGPCriticLayers(
                generatorArchitecture.InputType).ToList(),
            inputFrames: generatorArchitecture.InputFrames);
    }

    /// <summary>
    /// Creates a WGAN-GP with default generator and critic architectures derived from a single architecture.
    /// Per Gulrajani et al. 2017: gradient penalty coefficient 10, 5 critic iterations per generator step.
    /// </summary>
    /// <param name="architecture">The shared neural network architecture used for both generator and critic.</param>
    /// <param name="gradientPenaltyCoefficient">The gradient penalty coefficient (lambda). Default is 10.0.</param>
    /// <param name="criticIterations">Number of critic iterations per generator iteration. Default is 5.</param>
    /// <param name="options">Optional WGAN-GP options.</param>
    public WGANGP(
        NeuralNetworkArchitecture<T> architecture,
        double gradientPenaltyCoefficient = 10.0,
        int criticIterations = 5,
        WGANGPOptions? options = null)
        : this(architecture, CreateDefaultCriticArchitecture(architecture), architecture.InputType,
               gradientPenaltyCoefficient: gradientPenaltyCoefficient,
               criticIterations: criticIterations, options: options)
    {
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
    /// <param name="options">Optional WGAN-GP options.</param>
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
        int criticIterations = 5,
        WGANGPOptions? options = null)
        : base(CreateWGANGPArchitecture(generatorArchitecture, criticArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _options = options ?? new WGANGPOptions();
        Options = _options;

        // Input validation
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }
        if (criticArchitecture is null)
        {
            throw new ArgumentNullException(nameof(criticArchitecture), "Critic architecture cannot be null.");
        }
        if (criticArchitecture.OutputSize != 1)
        {
            throw new ArgumentException(
                $"WGAN-GP critic output size must be 1 (an unrestricted Wasserstein score), but was {criticArchitecture.OutputSize}.",
                nameof(criticArchitecture));
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

        Generator = CreateNetworkForArchitecture(generatorArchitecture);
        Critic = CreateNetworkForArchitecture(EnsureDefaultCriticLayers(criticArchitecture));

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        // Algorithm 1 defaults: Adam(alpha=1e-4, beta1=0, beta2=0.9).
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            Generator, CreatePaperAdamOptions());
        _criticOptimizer = criticOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            Critic, CreatePaperAdamOptions());

        InitializeLayers();
    }

    private static NeuralNetworkBase<T> CreateNetworkForArchitecture(
        NeuralNetworkArchitecture<T> architecture)
    {
        return architecture.InputType switch
        {
            InputType.ThreeDimensional or InputType.FourDimensional =>
                new ConvolutionalNeuralNetwork<T>(architecture),
            _ => new FeedForwardNeuralNetwork<T>(architecture),
        };
    }

    /// <summary>
    /// Supplies the paper-correct default critic when the caller did not provide custom layers.
    /// Custom critic layers remain untouched.
    /// </summary>
    private static NeuralNetworkArchitecture<T> EnsureDefaultCriticLayers(
        NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.Layers is { Count: > 0 })
            return architecture;

        return new NeuralNetworkArchitecture<T>(
            inputType: architecture.InputType,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: architecture.Complexity,
            inputSize: architecture.InputSize,
            inputHeight: architecture.InputHeight,
            inputWidth: architecture.InputWidth,
            inputDepth: architecture.InputDepth,
            outputSize: 1,
            layers: LayerHelper<T>.CreateDefaultWGANGPCriticLayers(
                architecture.InputType).ToList(),
            inputFrames: architecture.InputFrames);
    }

    private AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CreatePaperAdamOptions()
        => new()
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = _options.Beta1,
            Beta2 = _options.Beta2,
        };

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

        // Public Predict accepts a single unbatched sample. Algorithm 1 is batch-based, so
        // promote those samples to batch size one before interpolation/reduction. Treating a
        // length-N vector as N scalar samples makes the gradient-penalty norm mathematically wrong.
        realImages = EnsureBatchDimension(realImages, Critic.Architecture.InputType);
        noise = EnsureBatchDimension(noise, Generator.Architecture.InputType);

        Generator.SetTrainingMode(true);
        Critic.SetTrainingMode(true);

        T totalCriticLoss = NumOps.Zero;

        // Train critic multiple times
        for (int i = 0; i < _criticIterations; i++)
        {
            // Generate fake images
            Tensor<T> fakeImages = ShapeAsCriticInput(GenerateImages(noise));
            Tensor<T> realBatch = ShapeAsCriticInput(realImages);

            // Get batch size
            int batchSize = realBatch.Shape[0];

            // Train critic and get losses
            var (criticLoss, _) = TrainCriticBatchWithGP(realBatch, fakeImages, batchSize);

            totalCriticLoss = NumOps.Add(totalCriticLoss, criticLoss);
        }

        // Average critic loss
        T avgCriticLoss = NumOps.Divide(totalCriticLoss, NumOps.FromDouble(_criticIterations));

        // Train generator: minimize -mean(Critic(fake)) (Wasserstein objective with GP)
        Tensor<T> newNoise = GenerateRandomNoiseTensor(noise.Shape[0], Generator.Architecture.InputSize);
        var trainableGen = (NeuralNetworkBase<T>)Generator;
        T generatorLoss = trainableGen.TrainWithCustomLoss(ShapeAsGeneratorInput(newNoise), genOutput =>
        {
            // Keep the critic forward on the generator's active tape. Predict() is an
            // inference API and deliberately suppresses gradient recording; using it here
            // detached D(G(z)) from G and made every generator update a no-op. Only the
            // generator parameters are handed to the optimizer, so the critic remains fixed
            // while its input gradient carries the adversarial signal back into G.
            var criticScore = Critic.ForwardForTraining(ShapeAsCriticInput(genOutput));
            var negScore = Engine.TensorNegate(criticScore);
            var allAxes = Enumerable.Range(0, negScore.Shape.Length).ToArray();
            return Engine.ReduceMean(negScore, allAxes, keepDims: false);
        }, _generatorOptimizer);

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

    private static Tensor<T> EnsureBatchDimension(Tensor<T> tensor, InputType inputType)
    {
        int sampleRank = inputType switch
        {
            InputType.OneDimensional => 1,
            InputType.TwoDimensional => 2,
            InputType.ThreeDimensional => 3,
            InputType.FourDimensional => 4,
            _ => tensor.Rank,
        };

        if (tensor.Rank != sampleRank)
            return tensor;

        var batchedShape = new int[tensor.Rank + 1];
        batchedShape[0] = 1;
        for (int i = 0; i < tensor.Rank; i++)
            batchedShape[i + 1] = tensor.Shape[i];
        return tensor.Reshape(batchedShape);
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

        // Preferred fused path: WganGpFusedStep runs the full WGAN-GP critic
        // objective (Wasserstein + λ·GP with createGraph=true GP) in one
        // compiled plan. Bypasses the legacy flat-vector round-trip (which
        // needs to Predict the critic three times per step, extract flat
        // gradients, combine host-side, then apply). See ooples/AiDotNet#1845.
        var criticDiscParams = Training.TapeTrainingStep<T>.CollectParameters(Critic.Layers);
        if (criticDiscParams.Count > 0
            && TryMapToFusedOptimizerConfig(
                _criticOptimizer,
                out var wganOptType, out var wganLr, out var wganB1,
                out var wganB2, out var wganEps, out var wganWd,
                out _, out _))
        {
            using var wganStep = new AiDotNet.Training.WganGpFusedStep<T>();
            Tensor<T> DiscFwd(Tensor<T> inp)
            {
                Tensor<T> current = inp;
                foreach (var layer in Critic.Layers)
                    current = layer.Forward(current);
                return current;
            }
            Tensor<T> EpsilonSampler(int bs) =>
                Engine.TensorRandomUniformRange<T>(new[] { bs, 1 }, NumOps.Zero, NumOps.One);
            if (wganStep.TryStep(
                    discParameters: criticDiscParams,
                    realBatch: realImages,
                    fakeBatch: fakeImages,
                    discForward: DiscFwd,
                    epsilonSampler: EpsilonSampler,
                    gradientPenaltyWeight: _gradientPenaltyCoefficient,
                    optimizerType: wganOptType,
                    learningRate: wganLr,
                    beta1: wganB1,
                    beta2: wganB2,
                    epsilon: wganEps,
                    weightDecay: wganWd,
                    out T fusedLoss))
            {
                return (fusedLoss, NumOps.Zero);
            }
        }

        // Eager fallback: optimize the same single differentiable objective as Algorithm 1.
        // The previous path created real/fake output-gradient tensors but never backpropagated
        // either one, and returned an all-zero placeholder for the GP parameter gradients. Its
        // optimizer therefore received an all-zero vector and every critic iteration was a no-op.
        T gradientPenalty = NumOps.Zero;
        var trainableCritic = (NeuralNetworkBase<T>)Critic;
        T criticLoss = trainableCritic.TrainWithCustomLoss(realImages, realScores =>
        {
            var fakeScores = Critic.ForwardForTraining(fakeImages);
            var realAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
            var fakeAxes = Enumerable.Range(0, fakeScores.Shape.Length).ToArray();
            var wasserstein = Engine.TensorSubtract(
                Engine.ReduceMean(fakeScores, fakeAxes, keepDims: false),
                Engine.ReduceMean(realScores, realAxes, keepDims: false));

            // x-hat = epsilon * real + (1 - epsilon) * fake, with one epsilon per sample.
            var epsilonShape = new int[realImages.Shape.Length];
            epsilonShape[0] = batchSize;
            for (int d = 1; d < epsilonShape.Length; d++) epsilonShape[d] = 1;
            var epsilonBase = Engine.TensorRandomUniform<T>(epsilonShape);
            var tileFactors = new int[realImages.Shape.Length];
            tileFactors[0] = 1;
            for (int d = 1; d < tileFactors.Length; d++) tileFactors[d] = realImages.Shape[d];
            var epsilon = Engine.TensorTile(epsilonBase, tileFactors);
            var ones = new Tensor<T>(epsilon._shape);
            Engine.TensorFill(ones, NumOps.One);
            var interpolated = Engine.TensorAdd(
                Engine.TensorMultiply(epsilon, realImages),
                Engine.TensorMultiply(Engine.TensorSubtract(ones, epsilon), fakeImages));

            // createGraph=true is essential: the outer critic tape must differentiate the
            // input-gradient norm back into the critic weights, just as PyTorch's autograd.grad
            // and the paper's official TensorFlow implementation do.
            Tensor<T> inputGradients;
            using (var innerTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>())
            {
                var interpolatedScores = Critic.ForwardForTraining(interpolated);
                var scoreAxes = Enumerable.Range(0, interpolatedScores.Shape.Length).ToArray();
                var summedScores = Engine.ReduceSum(interpolatedScores, scoreAxes, keepDims: false);
                var innerGradients = innerTape.ComputeGradients(
                    summedScores, [interpolated], createGraph: true);
                inputGradients = innerGradients.TryGetValue(interpolated, out var gradient)
                    ? gradient
                    : new Tensor<T>(interpolated._shape);
            }

            int elementsPerSample = inputGradients.Length / batchSize;
            var flattenedGradients = Engine.Reshape(inputGradients, [batchSize, elementsPerSample]);
            var squaredGradients = Engine.TensorMultiply(flattenedGradients, flattenedGradients);
            var squaredNorm = Engine.ReduceSum(squaredGradients, [1], keepDims: false);
            var stabilizedSquaredNorm = Engine.TensorAddScalar(squaredNorm, NumOps.FromDouble(1e-12));
            var gradientNorm = Engine.TensorSqrt(stabilizedSquaredNorm);
            var normOnes = new Tensor<T>(gradientNorm._shape);
            Engine.TensorFill(normOnes, NumOps.One);
            var normDeviation = Engine.TensorSubtract(gradientNorm, normOnes);
            var perSamplePenalty = Engine.TensorMultiply(normDeviation, normDeviation);
            var penaltyAxes = Enumerable.Range(0, perSamplePenalty.Shape.Length).ToArray();
            var penalty = Engine.ReduceMean(perSamplePenalty, penaltyAxes, keepDims: false);
            gradientPenalty = penalty.Length > 0 ? penalty[0] : NumOps.Zero;

            var weightedPenalty = Engine.TensorMultiplyScalar(
                penalty, NumOps.FromDouble(_gradientPenaltyCoefficient));
            var alignedPenalty = weightedPenalty._shape.SequenceEqual(wasserstein._shape)
                ? weightedPenalty
                : Engine.Reshape(weightedPenalty, wasserstein._shape);
            return Engine.TensorAdd(wasserstein, alignedPenalty);
        }, _criticOptimizer);

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
        var onesTensor = new Tensor<T>(epsilon._shape);
        Engine.TensorFill(onesTensor, NumOps.One);
        var oneMinusEpsilon = Engine.TensorSubtract(onesTensor, epsilon);

        // interpolated = epsilon * real + (1 - epsilon) * fake
        var epsilonTimesReal = Engine.TensorMultiply(epsilon, realImages);
        var oneMinusEpsilonTimesFake = Engine.TensorMultiply(oneMinusEpsilon, fakeImages);
        var interpolatedImages = Engine.TensorAdd(epsilonTimesReal, oneMinusEpsilonTimesFake);

        // Forward pass through critic
        var interpolatedScores = Critic.Predict(interpolatedImages);

        // Create gradients of all ones using vectorized fill
        var ones = new Tensor<T>(interpolatedScores._shape);
        Engine.TensorFill(ones, NumOps.One);

        // Compute input gradients for gradient penalty using tape-based autodiff.
        // Use the inherited protected Engine from NeuralNetworkBase rather than
        // the static singleton.
        var eng = Engine;
        Tensor<T> inputGradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>())
        {
            var scores = Critic.Predict(interpolatedImages);
            var allAxes = Enumerable.Range(0, scores.Shape.Length).ToArray();
            var sumScores = eng.ReduceSum(scores, allAxes, keepDims: false);
            var grads = tape.ComputeGradients(sumScores, [interpolatedImages]);
            inputGradients = grads.TryGetValue(interpolatedImages, out var g) ? g : new Tensor<T>(interpolatedImages._shape);
        }

        // Compute L2 norm of gradients for each sample using vectorized operations
        int gradientSampleSize = inputGradients.Length / batchSize;
        var gradientsReshaped = inputGradients.Reshape([batchSize, gradientSampleSize]);

        // gradNormSquared[b] = sum(grad[b, i]^2) for each batch
        var gradSquared = Engine.TensorMultiply(gradientsReshaped, gradientsReshaped);
        var gradNormSquared = Engine.ReduceSum(gradSquared, [1], keepDims: false);

        // gradNorm = sqrt(gradNormSquared)
        var gradNorm = Engine.TensorSqrt(gradNormSquared);

        // deviation = gradNorm - 1
        var onesForDeviation = new Tensor<T>(gradNorm._shape);
        Engine.TensorFill(onesForDeviation, NumOps.One);
        var deviation = Engine.TensorSubtract(gradNorm, onesForDeviation);

        // penalty = deviation^2
        var penalty = Engine.TensorMultiply(deviation, deviation);

        // totalPenalty = mean(penalty)
        T totalPenalty = NumOps.Divide(Engine.TensorSum(penalty), NumOps.FromDouble(batchSize));

        // With tape-based training, parameter gradients are computed by the tape
        // during the training step. Return the penalty scalar only.
        var emptyGradients = new Vector<T>(Critic.GetParameters().Length);
        return (totalPenalty, emptyGradients);
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
        return ShapeAsCriticInput(Generator.Predict(ShapeAsGeneratorInput(noise)));
    }

    /// <summary>
    /// Presents a tensor in the [batch, depth, height, width] layout the critic consumes.
    /// </summary>
    /// <remarks>
    /// The generator is a convolutional network whose stack ends in a dense projection, so it emits a
    /// FLAT vector, while the critic is a convolutional network that requires a 2-D or 3-D image.
    /// Nothing bridged the two: the generator's output was handed straight to Critic.Predict, and the
    /// real samples were passed through untouched as well, so the critic reported "Expected input
    /// depth 1, but got 4" — it was reading the leading dimension of a flat vector as a channel count.
    /// A GAN's generator has to produce something shaped like its training images for the critic to
    /// compare the two at all. Reshaping is a recorded op, so the generator still trains through it.
    /// Tensors that already match, or whose element count cannot be divided into whole images, are
    /// returned unchanged rather than forced.
    /// </remarks>
    private Tensor<T> ShapeAsCriticInput(Tensor<T> images) => ShapeForArchitecture(images, Critic.Architecture);

    /// <summary>
    /// Presents a noise tensor in the layout the generator consumes.
    /// </summary>
    /// <remarks>
    /// The mirror of <see cref="ShapeAsCriticInput"/>, for the other end of the pipeline.
    /// <see cref="GenerateRandomNoiseTensor"/> emits a FLAT <c>[batch, noiseSize]</c> vector, but when
    /// the generator is convolutional it needs an image-shaped input just as the critic does. Every
    /// call site shaped the generator's OUTPUT for the critic and left its INPUT untouched, so a
    /// convolutional generator read the flat vector's trailing dimension as a channel count and threw
    /// "Expected input depth 1, but got 64" — the same defect the critic side already had fixed,
    /// surviving on the input side because nothing bridged it there.
    /// </remarks>
    private Tensor<T> ShapeAsGeneratorInput(Tensor<T> noise) => ShapeForArchitecture(noise, Generator.Architecture);

    /// <summary>
    /// Reshapes a tensor into the <c>[batch, depth, height, width]</c> layout an architecture declares.
    /// </summary>
    /// <remarks>
    /// Returns the tensor unchanged when the architecture does not describe an image, when it already
    /// matches, or when its element count cannot be divided into whole samples — reshaping is a
    /// recorded op, so a tensor that passes through here still trains.
    /// </remarks>
    private Tensor<T> ShapeForArchitecture(Tensor<T> images, NeuralNetworkArchitecture<T> arch)
    {
        int depth = arch.InputDepth;
        int height = arch.InputHeight;
        int width = arch.InputWidth;
        if (depth <= 0 || height <= 0 || width <= 0)
            return images;

        int perSample = depth * height * width;
        if (perSample <= 0 || images.Length % perSample != 0)
            return images;

        var target = new[] { images.Length / perSample, depth, height, width };
        if (images.Shape.Length == target.Length)
        {
            bool alreadyShaped = true;
            for (int i = 0; i < target.Length && alreadyShaped; i++)
                alreadyShaped = images.Shape[i] == target[i];
            if (alreadyShaped)
                return images;
        }

        return Engine.Reshape(images, target);
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
        // Publish the complete composite module graph. Predict still routes through Generator
        // only, but framework services (parameters, gradients, COW cloning, serialization and
        // layer inspection) must see both trainable subnetworks.
        Layers.AddRange(Generator.Layers);
        Layers.AddRange(Critic.Layers);
    }

    /// <summary>
    /// Collects activations by running each subnetwork in turn, not by walking the composite list.
    /// </summary>
    /// <remarks>
    /// The base implementation forwards <see cref="Layers"/> sequentially, which for this composite
    /// view means handing the generator's flat projection to the critic's first convolution — hence
    /// "ConvolutionalLayer expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank 2". The real
    /// data flow is z to generator to image to critic, and the two shape bridges this class already
    /// owns are what connect the stages, so both are applied here.
    /// <see cref="GenerativeAdversarialNetwork{T}"/> overrides this method for the same reason.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();

        var current = ShapeAsGeneratorInput(input);
        for (int i = 0; i < Generator.Layers.Count; i++)
        {
            current = Generator.Layers[i].Forward(current);
            activations[$"Generator_Layer_{i}_{Generator.Layers[i].GetType().Name}"] = current.Clone();
        }

        // The critic consumes an image, which is exactly what the generator's output becomes here.
        var criticInput = ShapeAsCriticInput(current);
        for (int i = 0; i < Critic.Layers.Count; i++)
        {
            criticInput = Critic.Layers[i].Forward(criticInput);
            activations[$"Critic_Layer_{i}_{Critic.Layers[i].GetType().Name}"] = criticInput.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Resolves each subnetwork against its OWN architecture instead of walking the composite list.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="Layers"/> is a composite VIEW of two PARALLEL subnetworks, not a feed-forward
    /// chain. The base walk documents itself as "the architecture's input shape feeds the first
    /// layer; each layer's resolved output shape feeds the next", which is exactly wrong here: it
    /// ran off the end of the generator and into the critic, so the critic's first convolution
    /// resolved against a generator feature map instead of an image. With the shared 8x8x1 image
    /// architecture that produced a critic stem of <c>[32, 32, 3, 3]</c> — 32 input channels for
    /// 1-channel images — and every later call threw "Input channels (1) must match kernel
    /// in_channels (32)".
    /// </para>
    /// <para>
    /// Resolving each subnetwork from its own declared input leaves nothing lazy for the composite
    /// walk to mis-resolve. This mirrors <c>GenerativeAdversarialNetwork.EnsureMaterialized</c>,
    /// which exists for the same reason; WGANGP derives from <see cref="NeuralNetworkBase{T}"/>
    /// rather than that class and so never inherited the treatment.
    /// </para>
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        EnsureSubnetworkMaterialized(Generator);
        EnsureSubnetworkMaterialized(Critic);
    }

    /// <summary>
    /// Walks one subnetwork's chain from its architecture's input shape, materializing lazy layers.
    /// </summary>
    /// <remarks>
    /// Per-layer failures are swallowed so a single layer wanting richer shape metadata does not
    /// abandon the rest of the walk — the first real forward picks those up through the normal
    /// <c>OnFirstForward</c> path.
    /// </remarks>
    private static void EnsureSubnetworkMaterialized(NeuralNetworkBase<T> subnet)
    {
        var archShape = subnet?.Architecture?.GetInputShape();
        if (subnet is null || archShape is null || archShape.Length == 0
            || !Array.TrueForAll(archShape, d => d > 0))
        {
            return;
        }

        int[] currentShape = archShape;
        foreach (var layer in subnet.Layers)
        {
            if (layer is null) continue;
            try
            {
                if (layer is Layers.LayerBase<T> lb && !lb.IsShapeResolved)
                    lb.ResolveFromShape(currentShape);

                var outShape = layer.GetOutputShape();
                if (outShape is { Length: > 0 } && Array.TrueForAll(outShape, d => d > 0))
                    currentShape = outShape;
                else
                    break;
            }
            catch
            {
                break;
            }
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // A GAN predicts by generating, so hand back an IMAGE rather than the generator's flat
        // projection — the same shape the critic and the training targets use.
        return ShapeAsCriticInput(Generator.Predict(ShapeAsGeneratorInput(input)));
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "CriticParameters", Critic.GetParameterCount() },
                { "GradientPenaltyCoefficient", _gradientPenaltyCoefficient },
                { "CriticIterations", _criticIterations }
            },
            ModelData = SerializeForMetadata()
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
            _criticIterations,
            new WGANGPOptions(_options));
    }

    // UpdateParameters split the vector between Generator and Critic. Both sub-networks' layers are
    // added to Layers in that same order (Layers.AddRange(Generator.Layers) then
    // Layers.AddRange(Critic.Layers)), so the base fold reproduces the split. Removed under AIDN082.
}
