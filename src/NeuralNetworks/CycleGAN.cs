using AiDotNet.Tensors.Engines.Autodiff;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

using System.Linq;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a CycleGAN for unpaired image-to-image translation.
/// </summary>
/// <remarks>
/// <para>
/// CycleGAN enables image-to-image translation without paired training data:
/// - Uses two generators (A→B and B→A) and two discriminators
/// - Enforces cycle consistency: A→B→A should equal A
/// - Works without paired examples (e.g., can learn horses→zebras from separate collections)
/// - Uses adversarial loss + cycle consistency loss + identity loss
/// </para>
/// <para><b>For Beginners:</b> CycleGAN translates images without matched pairs.
///
/// Key innovation:
/// - Doesn't need paired training data
/// - Learns from two separate collections of images
/// - Example: Photos of horses + Photos of zebras → can convert horses to zebras
///
/// How it works:
/// - Two generators: G (A→B) and F (B→A)
/// - Two discriminators: D_A and D_B
/// - Cycle consistency: G(F(B)) ≈ B and F(G(A)) ≈ A
/// - This prevents mode collapse and maintains content
///
/// Applications:
/// - Style transfer (Monet → Photo, Photo → Monet)
/// - Season transfer (Summer → Winter)
/// - Object transfiguration (Horse → Zebra)
/// - Domain adaptation
///
/// Reference: Zhu et al., "Unpaired Image-to-Image Translation using
/// Cycle-Consistent Adversarial Networks" (2017)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new CycleGANOptions { ImageSize = 256, NumResidualBlocks = 9 };
/// var model = new CycleGAN&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 3, 256, 256 });
/// var translated = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", "https://arxiv.org/abs/1703.10593", Year = 2017, Authors = "Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros")]
public partial class CycleGAN<T> : ImageTranslationModelLayoutBase<T>
{

    // The four sub-networks are discovered as members and their layers surfaced in declaration
    // order -- GeneratorAtoB, GeneratorBtoA, DiscriminatorA, DiscriminatorB -- which is the order
    // this hook used and therefore the serialization order. Removed under AIDN082.
    private readonly CycleGANOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private static AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CreateStandardGanAdamOptions()
        => new()
        {
            InitialLearningRate = 0.0002,
            Beta1 = 0.5,
            Beta2 = 0.999,
        };

    /// <summary>
    /// The optimizer used for training generator A→B.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the generator
    /// that transforms images from domain A to domain B. The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the A→B generator
    /// learns from its mistakes and adjusts its parameters during training.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorAtoBOptimizer;

    /// <summary>
    /// The optimizer used for training generator B→A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the generator
    /// that transforms images from domain B to domain A. The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the B→A generator
    /// learns from its mistakes and adjusts its parameters during training.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorBtoAOptimizer;

    /// <summary>
    /// The optimizer used for training discriminator A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the discriminator
    /// that evaluates images in domain A (real vs. generated). The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how discriminator A
    /// learns to better distinguish real images from fake ones in domain A.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorAOptimizer;

    /// <summary>
    /// The optimizer used for training discriminator B.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the discriminator
    /// that evaluates images in domain B (real vs. generated). The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how discriminator B
    /// learns to better distinguish real images from fake ones in domain B.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorBOptimizer;

    /// <summary>
    /// Coefficient for cycle consistency loss.
    /// </summary>
    /// <remarks>
    /// Controls the importance of cycle consistency. Typical value: 10.0.
    /// Higher values enforce stronger cycle consistency.
    /// </remarks>
    private T _cycleConsistencyLambda;

    /// <summary>
    /// Coefficient for identity loss.
    /// </summary>
    /// <remarks>
    /// Encourages G and F to preserve color composition. Typical value: 0.5 * cycleConsistencyLambda.
    /// </remarks>
    private T _identityLambda;

    /// <summary>
    /// Generator A→B.
    /// </summary>
    public NeuralNetworkBase<T> GeneratorAtoB { get; private set; }

    /// <summary>
    /// Generator B→A.
    /// </summary>
    public NeuralNetworkBase<T> GeneratorBtoA { get; private set; }

    /// <summary>
    /// Discriminator for domain A.
    /// </summary>
    public NeuralNetworkBase<T> DiscriminatorA { get; private set; }

    /// <summary>
    /// Discriminator for domain B.
    /// </summary>
    public NeuralNetworkBase<T> DiscriminatorB { get; private set; }

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined CycleGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateCycleGANArchitecture(
        NeuralNetworkArchitecture<T> generatorAtoB,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: generatorAtoB.InputHeight,
                inputWidth: generatorAtoB.InputWidth,
                inputDepth: generatorAtoB.InputDepth,
                outputSize: generatorAtoB.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: generatorAtoB.InputSize,
            outputSize: generatorAtoB.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CycleGAN{T}"/> class with the specified architecture and training parameters.
    /// </summary>
    /// <param name="generatorAtoB">The architecture for the generator that transforms images from domain A to domain B.</param>
    /// <param name="generatorBtoA">The architecture for the generator that transforms images from domain B to domain A.</param>
    /// <param name="discriminatorA">The architecture for the discriminator that evaluates images in domain A.</param>
    /// <param name="discriminatorB">The architecture for the discriminator that evaluates images in domain B.</param>
    /// <param name="inputType">The type of input data (e.g., ThreeDimensional for images).</param>
    /// <param name="generatorAtoBOptimizer">
    /// Optional optimizer for the A→B generator. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="generatorBtoAOptimizer">
    /// Optional optimizer for the B→A generator. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="discriminatorAOptimizer">
    /// Optional optimizer for discriminator A. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="discriminatorBOptimizer">
    /// Optional optimizer for discriminator B. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="lossFunction">Optional loss function. If null, the default loss function for generative tasks is used.</param>
    /// <param name="cycleConsistencyLambda">
    /// The coefficient for cycle consistency loss. Higher values enforce stronger cycle consistency. Default is 10.0.
    /// </param>
    /// <param name="identityLambda">
    /// The coefficient for identity loss. Helps preserve color composition. Default is 5.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a CycleGAN with four separate networks and optimizers:
    /// - Generator A→B: Transforms images from domain A to domain B
    /// - Generator B→A: Transforms images from domain B to domain A
    /// - Discriminator A: Evaluates whether images in domain A are real or generated
    /// - Discriminator B: Evaluates whether images in domain B are real or generated
    /// </para>
    /// <para><b>For Beginners:</b> CycleGAN needs four networks to work:
    /// - Two generators to translate images in both directions
    /// - Two discriminators to judge images in each domain
    ///
    /// The cycle consistency loss ensures that translating A→B→A gets back to the original,
    /// which helps maintain content while only changing style.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown when any of the architecture parameters is null.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when cycleConsistencyLambda or identityLambda is negative.
    /// </exception>
    public CycleGAN()
        : this(
            new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 784, outputSize: 784),
            new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 784, outputSize: 784),
            new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.BinaryClassification, inputSize: 784, outputSize: 1),
            new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.BinaryClassification, inputSize: 784, outputSize: 1),
            inputType: InputType.OneDimensional)
    {
    }

    public CycleGAN(
        NeuralNetworkArchitecture<T> generatorAtoB,
        NeuralNetworkArchitecture<T> generatorBtoA,
        NeuralNetworkArchitecture<T> discriminatorA,
        NeuralNetworkArchitecture<T> discriminatorB,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorAtoBOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorBtoAOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorAOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorBOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        double cycleConsistencyLambda = 10.0,
        double identityLambda = 5.0,
        CycleGANOptions? options = null)
        : base(CreateCycleGANArchitecture(generatorAtoB, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        _options = options ?? new CycleGANOptions();
        Options = _options;

        // Validate constructor inputs
        if (generatorAtoB is null)
        {
            throw new ArgumentNullException(nameof(generatorAtoB), "Generator A to B architecture cannot be null.");
        }
        if (generatorBtoA is null)
        {
            throw new ArgumentNullException(nameof(generatorBtoA), "Generator B to A architecture cannot be null.");
        }
        if (discriminatorA is null)
        {
            throw new ArgumentNullException(nameof(discriminatorA), "Discriminator A architecture cannot be null.");
        }
        if (discriminatorB is null)
        {
            throw new ArgumentNullException(nameof(discriminatorB), "Discriminator B architecture cannot be null.");
        }
        if (cycleConsistencyLambda < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(cycleConsistencyLambda), cycleConsistencyLambda, "Cycle consistency lambda must be non-negative.");
        }
        if (identityLambda < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(identityLambda), identityLambda, "Identity lambda must be non-negative.");
        }

        _cycleConsistencyLambda = NumOps.FromDouble(cycleConsistencyLambda);
        _identityLambda = NumOps.FromDouble(identityLambda);

        GeneratorAtoB = CreateNetworkForInputType(generatorAtoB, inputType);
        GeneratorBtoA = CreateNetworkForInputType(generatorBtoA, inputType);
        DiscriminatorA = CreateNetworkForInputType(discriminatorA, inputType);
        DiscriminatorB = CreateNetworkForInputType(discriminatorB, inputType);

        // Initialize optimizers - use provided optimizers or create default GAN-standard Adam optimizers.
        _generatorAtoBOptimizer = generatorAtoBOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(GeneratorAtoB, CreateStandardGanAdamOptions());
        _generatorBtoAOptimizer = generatorBtoAOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(GeneratorBtoA, CreateStandardGanAdamOptions());
        _discriminatorAOptimizer = discriminatorAOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(DiscriminatorA, CreateStandardGanAdamOptions());
        _discriminatorBOptimizer = discriminatorBOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(DiscriminatorB, CreateStandardGanAdamOptions());

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        InitializeLayers();
    }

    /// <summary>
    /// Creates the appropriate neural network type based on the input type.
    /// </summary>
    private static NeuralNetworkBase<T> CreateNetworkForInputType(NeuralNetworkArchitecture<T> architecture, InputType inputType)
    {
        return inputType switch
        {
            InputType.OneDimensional => new FeedForwardNeuralNetwork<T>(architecture),
            InputType.TwoDimensional => new FeedForwardNeuralNetwork<T>(architecture),
            InputType.ThreeDimensional => new ConvolutionalNeuralNetwork<T>(architecture),
            _ => new FeedForwardNeuralNetwork<T>(architecture)
        };
    }

    /// <summary>
    /// Performs one training step for CycleGAN.
    /// </summary>
    /// <param name="realA">Real images from domain A.</param>
    /// <param name="realB">Real images from domain B.</param>
    /// <returns>A tuple containing discriminator loss, generator loss, and cycle consistency loss.</returns>
    /// <exception cref="ArgumentNullException">Thrown when realA or realB is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batch dimensions don't match or batch size is zero.</exception>
    public (T discLoss, T genLoss, T cycleLoss) TrainStep(
        Tensor<T> realA,
        Tensor<T> realB)
    {
        // Validate input tensors
        if (realA is null)
        {
            throw new ArgumentNullException(nameof(realA), "Real images from domain A cannot be null.");
        }

        if (realB is null)
        {
            throw new ArgumentNullException(nameof(realB), "Real images from domain B cannot be null.");
        }

        int batchSize = realA.Shape[0];

        if (batchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive.", nameof(realA));
        }

        if (realB.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch size mismatch: realA has batch size {batchSize}, but realB has batch size {realB.Shape[0]}. " +
                "Both tensors must have the same batch dimension.",
                nameof(realB));
        }

        GeneratorAtoB.SetTrainingMode(true);
        GeneratorBtoA.SetTrainingMode(true);
        DiscriminatorA.SetTrainingMode(true);
        DiscriminatorB.SetTrainingMode(true);

        // Both generators' forwards are recorded once on the generator tape (#1390) and shared by
        // every term; the discriminators train on detached copies of the translations.
        using var generatorTape = new GradientTape<T>();
        var fakeB = GeneratorAtoB.ForwardForTraining(realA);
        var fakeA = GeneratorBtoA.ForwardForTraining(realB);
        var fakeBDetached = new Tensor<T>(fakeB.Shape.ToArray());
        fakeB.AsSpan().CopyTo(fakeBDetached.AsWritableSpan());
        var fakeADetached = new Tensor<T>(fakeA.Shape.ToArray());
        fakeA.AsSpan().CopyTo(fakeADetached.AsWritableSpan());

        // ----- Discriminators: least squares, halved (Zhu et al. 2017, section 4) -----
        // Both on one tape: the two losses share no parameters, so each network's gradient is its own
        // term's, and StepOnTape steps each with its own optimizer. They used to be "updated" from
        // gradients no backward had produced.
        T discriminatorLoss;
        using (var discriminatorTape = new GradientTape<T>())
        {
            var halfA = Engine.TensorAdd(
                LeastSquaresLoss(DiscriminatorA.ForwardForTraining(realA), target: 1.0),
                LeastSquaresLoss(DiscriminatorA.ForwardForTraining(fakeADetached), target: 0.0));
            var halfB = Engine.TensorAdd(
                LeastSquaresLoss(DiscriminatorB.ForwardForTraining(realB), target: 1.0),
                LeastSquaresLoss(DiscriminatorB.ForwardForTraining(fakeBDetached), target: 0.0));
            var discriminatorObjective = Engine.TensorMultiplyScalar(Engine.TensorAdd(halfA, halfB), NumOps.FromDouble(0.5));
            discriminatorLoss = StepOnTape(discriminatorTape, discriminatorObjective, new[]
            {
                (DiscriminatorA, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_discriminatorAOptimizer),
                (DiscriminatorB, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_discriminatorBOptimizer),
            });
        }

        // ----- Generators: adversarial + lambda * cycle + 0.5 * lambda * identity (eq. 3) -----
        // One loss trains both generators, since the cycle term runs through each of them. They used
        // to be trained by Train(realA, realB): a paired regression, which unpaired data cannot supply.
        var adversarial = Engine.TensorAdd(
            LeastSquaresLoss(DiscriminatorB.ForwardFrozenOnTape(fakeB), target: 1.0),
            LeastSquaresLoss(DiscriminatorA.ForwardFrozenOnTape(fakeA), target: 1.0));
        var cycleTensor = Engine.TensorAdd(
            MeanAbsoluteError(GeneratorBtoA.ForwardForTraining(fakeB), realA),
            MeanAbsoluteError(GeneratorAtoB.ForwardForTraining(fakeA), realB));
        var identityTensor = Engine.TensorAdd(
            MeanAbsoluteError(GeneratorBtoA.ForwardForTraining(realA), realA),
            MeanAbsoluteError(GeneratorAtoB.ForwardForTraining(realB), realB));
        var generatorObjective = Engine.TensorAdd(
            adversarial,
            Engine.TensorAdd(
                Engine.TensorMultiplyScalar(cycleTensor, _cycleConsistencyLambda),
                Engine.TensorMultiplyScalar(identityTensor, _identityLambda)));
        T generatorLoss = StepOnTape(generatorTape, generatorObjective, new[]
        {
            (GeneratorAtoB, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_generatorAtoBOptimizer),
            (GeneratorBtoA, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_generatorBtoAOptimizer),
        });
        T cycleLoss = cycleTensor.Length > 0 ? cycleTensor[0] : NumOps.Zero;

        return (discriminatorLoss, generatorLoss, cycleLoss);
    }

    /// <summary>
    /// Translates image from domain A to domain B.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method temporarily sets the generator to evaluation mode for inference,
    /// then restores the original training mode after prediction. This ensures
    /// batch normalization and dropout behave correctly during both inference
    /// and subsequent training steps.
    /// </para>
    /// </remarks>
    public Tensor<T> TranslateAtoB(Tensor<T> imageA)
    {
        bool originalTrainingMode = GeneratorAtoB.IsTrainingMode;
        GeneratorAtoB.SetTrainingMode(false);
        var result = GeneratorAtoB.Predict(imageA);
        GeneratorAtoB.SetTrainingMode(originalTrainingMode);
        return result;
    }

    /// <summary>
    /// Translates image from domain B to domain A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method temporarily sets the generator to evaluation mode for inference,
    /// then restores the original training mode after prediction. This ensures
    /// batch normalization and dropout behave correctly during both inference
    /// and subsequent training steps.
    /// </para>
    /// </remarks>
    public Tensor<T> TranslateBtoA(Tensor<T> imageB)
    {
        bool originalTrainingMode = GeneratorBtoA.IsTrainingMode;
        GeneratorBtoA.SetTrainingMode(false);
        var result = GeneratorBtoA.Predict(imageB);
        GeneratorBtoA.SetTrainingMode(originalTrainingMode);
        return result;
    }

    /// <summary>
    /// The mean squared distance of discriminator scores from a real (1) or fake (0) target, recorded on
    /// the active tape: the least-squares GAN loss CycleGAN uses in place of the log-likelihood.
    /// </summary>
    private Tensor<T> LeastSquaresLoss(Tensor<T> scores, double target)
    {
        var targets = new Tensor<T>(scores.Shape.ToArray());
        Engine.TensorFill(targets, NumOps.FromDouble(target));
        var difference = Engine.TensorSubtract(scores, targets);
        return Engine.ReduceMean(
            Engine.TensorMultiply(difference, difference),
            Enumerable.Range(0, difference.Shape.Length).ToArray(),
            keepDims: false);
    }

    /// <summary>The mean absolute error between two images, recorded on the active tape.</summary>
    private Tensor<T> MeanAbsoluteError(Tensor<T> predicted, Tensor<T> target)
        => Engine.ReduceMean(
            Engine.TensorAbs(Engine.TensorSubtract(predicted, target)),
            Enumerable.Range(0, predicted.Shape.Length).ToArray(),
            keepDims: false);

    /// <summary>
    /// Resets the state of all optimizers to their initial values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets all four optimizers (both generators and both discriminators)
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
        _generatorAtoBOptimizer.Reset();
        _generatorBtoAOptimizer.Reset();
        _discriminatorAOptimizer.Reset();
        _discriminatorBOptimizer.Reset();
    }

    protected override void InitializeLayers() { }

    /// <summary>
    /// Forwards the mode to the four sub-networks, which the base walk cannot reach.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="NeuralNetworkBase{T}.SetTrainingMode"/> propagates by iterating <c>Layers</c>, and
    /// this model deliberately leaves that empty — its trainable state is four whole
    /// <see cref="NeuralNetworkBase{T}"/> instances, not layers. So without this override
    /// <c>SetTrainingMode(false)</c> was a no-op: every dropout and batch-normalization inside the
    /// generators and discriminators stayed in TRAINING mode during inference, using batch
    /// statistics instead of running ones and sampling dropout masks on a prediction path.
    /// </para>
    /// <para>
    /// Only <see cref="TranslateAtoB"/> and <see cref="TranslateBtoA"/> were unaffected, because
    /// they each save, flip and restore the generator's mode by hand around a single call. Every
    /// other entry point — <c>Predict</c> included — was not covered.
    /// </para>
    /// <para>
    /// Note this cannot be solved by <c>GetExtraTrainableLayers</c>: that hook is not consulted by
    /// mode propagation, and its <c>LayerBase&lt;T&gt;</c> element type cannot hold a whole network.
    /// </para>
    /// </remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);

        GeneratorAtoB.SetTrainingMode(isTraining);
        GeneratorBtoA.SetTrainingMode(isTraining);
        DiscriminatorA.SetTrainingMode(isTraining);
        DiscriminatorB.SetTrainingMode(isTraining);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return GeneratorAtoB.Predict(input);
    }

    /// <summary>
    /// Defines the CycleGAN forward graph for tape-based training. CycleGAN's
    /// generators/discriminators live outside <c>Layers</c> (the base class
    /// list is intentionally empty), so the default <c>ForwardForTraining</c>
    /// would walk an empty layer chain and return the raw input. Overriding
    /// to dispatch through <c>GeneratorAtoB.ForwardForTraining</c> matches
    /// the inference-side <c>Predict</c> contract and lets
    /// <c>tape.ComputeGradients</c> flow back into the generator's weights.
    /// Discriminator/cycle losses are handled separately by
    /// <see cref="TrainStep"/>.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return GeneratorAtoB.ForwardForTraining(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(
            WithBatchAxis(input, GeneratorAtoB.Architecture.InputType),
            WithBatchAxis(expectedOutput, GeneratorBtoA.Architecture.InputType));
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        var fakeB = GeneratorAtoB.Predict(input);
        activations["GeneratorAtoB"] = fakeB.Clone();
        var fakeA = GeneratorBtoA.Predict(input);
        activations["GeneratorBtoA"] = fakeA.Clone();
        activations["DiscriminatorA"] = DiscriminatorA.Predict(input).Clone();
        activations["DiscriminatorB"] = DiscriminatorB.Predict(input).Clone();
        return activations;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorAtoB_Parameters", GeneratorAtoB.GetParameterCount() },
                { "GeneratorBtoA_Parameters", GeneratorBtoA.GetParameterCount() },
                { "DiscriminatorA_Parameters", DiscriminatorA.GetParameterCount() },
                { "DiscriminatorB_Parameters", DiscriminatorB.GetParameterCount() },
                { "CycleConsistencyLambda", NumOps.ToDouble(_cycleConsistencyLambda) },
                { "IdentityLambda", NumOps.ToDouble(_identityLambda) }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes CycleGAN-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the CycleGAN-specific configuration and all four networks.
    /// Optimizer state is managed by the optimizer implementations themselves.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the CycleGAN's settings and all
    /// four networks (two generators and two discriminators) to a file.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes CycleGAN-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the CycleGAN-specific configuration and all four networks.
    /// After deserialization, the optimizers are reset to their initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the CycleGAN's settings and all
    /// four networks (two generators and two discriminators) from a file.
    /// </para>
    /// </remarks>


    // UpdateParameters split the vector four ways -- GeneratorAtoB, GeneratorBtoA, DiscriminatorA,
    // DiscriminatorB -- and GetExtraTrainableLayers yields those four in the same order, so the base
    // reproduces the split. Removed under AIDN082.
}
