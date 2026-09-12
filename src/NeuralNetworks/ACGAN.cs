using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;

using System.Linq;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Auxiliary Classifier Generative Adversarial Network (AC-GAN), which extends
/// conditional GANs by having the discriminator also predict the class label of the input.
/// </summary>
/// <remarks>
/// <para>
/// AC-GAN improves upon conditional GANs by:
/// - Making the discriminator predict both authenticity AND class label
/// - Providing stronger gradient signals for class-conditional generation
/// - Improving image quality and class separability
/// - Enabling better control over generated samples
/// - Training more stable than basic conditional GANs
/// </para>
/// <para><b>For Beginners:</b> AC-GAN generates specific types of images with better quality.
///
/// Key improvements over cGAN:
/// - Discriminator has two tasks: "Is it real?" AND "What class is it?"
/// - This dual task helps the discriminator learn better features
/// - Generator must create images that fool both checks
/// - Results in higher quality and more class-consistent images
///
/// Example use case:
/// - Generate digit "7" that looks very realistic
/// - Discriminator checks: 1) Is it real? 2) Is it a "7"?
/// - This forces the generator to make better "7"s
///
/// Reference: Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs" (2017)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new ACGANOptions { LatentSize = 100, NumClasses = 10 };
/// var model = new ACGAN&lt;float&gt;(options);
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
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Conditional Image Synthesis with Auxiliary Classifier GANs", "https://arxiv.org/abs/1610.09585", Year = 2017, Authors = "Augustus Odena, Christopher Olah, Jonathon Shlens")]
public partial class ACGAN<T> : ImageGeneratorModelLayoutBase<T>
{

    // Generator and Discriminator are discovered as sub-network members and their layers surfaced
    // in declaration order, which is the order this hook used. Removed under AIDN082.
    private readonly ACGANOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private static AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CreateStandardGanAdamOptions()
        => new()
        {
            InitialLearningRate = 0.0002,
            Beta1 = 0.5,
            Beta2 = 0.999,
        };

    private readonly List<T> _generatorLosses = new List<T>();
    private readonly List<T> _discriminatorLosses = new List<T>();

    /// <summary>
    /// The number of classes for classification.
    /// </summary>
    private int _numClasses;

    /// <summary>
    /// The optimizer for the generator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer for the discriminator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    /// <summary>
    /// Gets the generator network that creates class-conditional synthetic data.
    /// </summary>
    public NeuralNetworkBase<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network that predicts both authenticity and class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Unlike standard GANs, the AC-GAN discriminator has two outputs:
    /// 1. Authenticity score (real vs fake) - 1 output
    /// 2. Class probability distribution - numClasses outputs
    /// </para>
    /// <para><b>For Beginners:</b> The discriminator is a multi-task network.
    ///
    /// Two outputs:
    /// - "Is this real or fake?" (1 number: 0-1)
    /// - "What class is this?" (probability for each class)
    ///
    /// This dual purpose makes it a better feature learner.
    /// </para>
    /// </remarks>
    public NeuralNetworkBase<T> Discriminator { get; private set; }

    private readonly ILossFunction<T> _lossFunction;

    // Stored under the constructor's own parameter names so the clone plan replays the constructor
    // with them. Without these it fell back to the model's outer Architecture for every sub-network.
    private readonly NeuralNetworkArchitecture<T> _generatorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _discriminatorArchitecture;

    /// <summary>
    /// Initializes a new instance of the <see cref="ACGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.
    /// Its output size must be 1 + numClasses: one source (real/fake) score followed by one score per
    /// class. The final layer may emit logits (no activation) or probabilities (sigmoid or softmax); the
    /// training losses read which from the layer itself, not from the values it produces.</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <param name="inputType">The type of input.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    public ACGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numClasses,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        ACGANOptions? options = null)
        // The model's public contract is its generator's: Predict hands the caller's input -- noise
        // followed by the class conditioning, which the generator's input size already counts -- to the
        // generator and returns what it generates. Declaring InputSize + numClasses counted the classes
        // twice, and declaring the discriminator's 1 + numClasses as the output described a score, not a
        // sample.
        : base(new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            generatorArchitecture.InputSize,
            0, 0, 1,
            generatorArchitecture.OutputSize,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _options = options ?? new ACGANOptions();
        Options = _options;

        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }

        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }

        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "Number of classes must be positive.");
        }

        if (discriminatorArchitecture.OutputSize != 1 + numClasses)
        {
            throw new ArgumentException(
                $"Discriminator output size must be 1 + numClasses ({1 + numClasses}), but was {discriminatorArchitecture.OutputSize}.",
                nameof(discriminatorArchitecture));
        }

        _numClasses = numClasses;

        _generatorArchitecture = generatorArchitecture;
        _discriminatorArchitecture = discriminatorArchitecture;
        Generator = CreateSubNetworkForInputType(generatorArchitecture, inputType);
        Discriminator = CreateSubNetworkForInputType(discriminatorArchitecture, inputType);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        // Initialize optimizers (default to GAN-standard Adam if not provided).
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator, CreateStandardGanAdamOptions());
        _discriminatorOptimizer = discriminatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator, CreateStandardGanAdamOptions());

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for the AC-GAN.
    /// </summary>
    /// <param name="realImages">Real images tensor.</param>
    /// <param name="realLabels">Real image class labels (one-hot encoded).</param>
    /// <param name="noise">Random noise for generator.</param>
    /// <param name="fakeLabels">Class labels for images to generate (one-hot encoded).</param>
    /// <returns>Tuple of (discriminator loss, generator loss).</returns>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> realLabels,
        Tensor<T> noise,
        Tensor<T> fakeLabels)
    {
        if (realImages is null)
        {
            throw new ArgumentNullException(nameof(realImages), "Real images tensor cannot be null.");
        }

        if (realLabels is null)
        {
            throw new ArgumentNullException(nameof(realLabels), "Real labels tensor cannot be null.");
        }

        if (noise is null)
        {
            throw new ArgumentNullException(nameof(noise), "Noise tensor cannot be null.");
        }

        if (fakeLabels is null)
        {
            throw new ArgumentNullException(nameof(fakeLabels), "Fake labels tensor cannot be null.");
        }

        int batchSize = realImages.Shape[0];

        if (realLabels.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Real labels batch size ({realLabels.Shape[0]}) must match real images batch size ({batchSize}).",
                nameof(realLabels));
        }

        if (noise.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Noise batch size ({noise.Shape[0]}) must match real images batch size ({batchSize}).",
                nameof(noise));
        }

        if (fakeLabels.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Fake labels batch size ({fakeLabels.Shape[0]}) must match real images batch size ({batchSize}).",
                nameof(fakeLabels));
        }

        // Validate label shape dimensions - must be 2D with correct class count
        if (realLabels.Shape.Length != 2 || realLabels.Shape[1] != _numClasses)
        {
            throw new ArgumentException(
                $"realLabels must be [batch,{_numClasses}], got [{string.Join(",", realLabels._shape)}].",
                nameof(realLabels));
        }

        if (fakeLabels.Shape.Length != 2 || fakeLabels.Shape[1] != _numClasses)
        {
            throw new ArgumentException(
                $"fakeLabels must be [batch,{_numClasses}], got [{string.Join(",", fakeLabels._shape)}].",
                nameof(fakeLabels));
        }

        return TrainStepCore(realImages, realLabels, noise, fakeLabels);
    }

    /// <summary>
    /// One AC-GAN step. <paramref name="realLabels"/> may be null when the real images' classes are
    /// unknown: the real batch then trains the source term only, instead of the classifier learning
    /// labels that were never observed.
    /// </summary>
    private (T discriminatorLoss, T generatorLoss) TrainStepCore(
        Tensor<T> realImages,
        Tensor<T>? realLabels,
        Tensor<T> noise,
        Tensor<T> fakeLabels)
    {
        // ONE generator forward, recorded on the generator's tape: the pattern the base GAN uses (#1390).
        // The discriminator step trains on a detached copy of it, so that step cannot reach the
        // generator. The generator step then scores the tracked original through the frozen
        // discriminator, so the adversarial and class gradients do reach it.
        //
        // The previous step computed both losses as detached scalars and then called
        // Update*WithOptimizer, which read GetParameterGradients() -- gradients no backward had produced,
        // because the Backward calls were commented out when manual backprop was removed. Neither
        // network ever trained (#2155).
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        var generatorInput = ConcatenateTensors(noise, fakeLabels);
        using var generatorTape = new GradientTape<T>();
        var fakeTracked = Generator.ForwardForTraining(generatorInput);
        var fakeImages = new Tensor<T>(fakeTracked.Shape.ToArray());
        fakeTracked.AsSpan().CopyTo(fakeImages.AsWritableSpan());

        bool emitsProbabilities = Discriminator.FinalLayerEmitsProbabilities();

        // ----- Discriminator: maximise L_S + L_C (Odena et al. 2017, eqs. 2 and 3) -----
        T discriminatorLoss;
        using (var discriminatorTape = new GradientTape<T>())
        {
            var realOutput = Discriminator.ForwardForTraining(realImages);
            var fakeOutput = Discriminator.ForwardForTraining(fakeImages);
            var discriminatorObjective = Engine.TensorAdd(
                SourceNegativeLogLikelihood(realOutput, isReal: true),
                Engine.TensorAdd(
                    SourceNegativeLogLikelihood(fakeOutput, isReal: false),
                    ClassNegativeLogLikelihood(fakeOutput, fakeLabels, emitsProbabilities)));
            if (realLabels is not null)
            {
                discriminatorObjective = Engine.TensorAdd(
                    discriminatorObjective,
                    ClassNegativeLogLikelihood(realOutput, realLabels, emitsProbabilities));
            }
            discriminatorLoss = StepOnTape(discriminatorTape, discriminatorObjective, Discriminator,
                _discriminatorOptimizer);
        }

        // ----- Generator: maximise L_C - L_S -----
        // The source term is taken in its non-saturating form (Goodfellow et al. 2014, section 3): the
        // generator minimises -log P(S = real | X_fake) instead of log P(S = fake | X_fake). Both have the
        // same fixed point, but the non-saturating one does not vanish while the discriminator still
        // rejects every sample. The class term is the paper's: a generated image must be classified as
        // the class it was asked for.
        var generatorOutput = Discriminator.ForwardFrozenOnTape(fakeTracked);
        var generatorObjective = Engine.TensorAdd(
            SourceNegativeLogLikelihood(generatorOutput, isReal: true),
            ClassNegativeLogLikelihood(generatorOutput, fakeLabels, emitsProbabilities));
        T generatorLoss = StepOnTape(generatorTape, generatorObjective, Generator, _generatorOptimizer);

        // Track losses
        _discriminatorLosses.Add(discriminatorLoss);
        _generatorLosses.Add(generatorLoss);

        if (_discriminatorLosses.Count > 100)
        {
            _discriminatorLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Concatenates noise and class labels for generator input.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> noise, Tensor<T> labels)
    {
        return Engine.TensorConcatenate([noise, labels], axis: 1);
    }

    /// <summary>
    /// The batch-mean negative log-likelihood of the source label, -log P(S = real) or -log P(S = fake),
    /// recorded on the active tape.
    /// </summary>
    private Tensor<T> SourceNegativeLogLikelihood(Tensor<T> discriminatorOutput, bool isReal)
    {
        int batchSize = discriminatorOutput.Shape[0];
        var source = Engine.TensorSlice(discriminatorOutput, new[] { 0, 0 }, new[] { batchSize, 1 });
        return Discriminator.BinaryCrossEntropyOnTape(source, isReal);
    }

    /// <summary>
    /// The batch-mean negative log-likelihood of the labelled class, -log P(C = c), under the
    /// discriminator's class posterior, recorded on the active tape.
    /// </summary>
    /// <remarks>
    /// The paper's L_C is the log-likelihood of the correct class under one categorical distribution. It
    /// replaces a per-class binary cross-entropy that scored every class as an independent yes-or-no
    /// question. When the head emits probabilities their logarithms serve as logits: the log-softmax
    /// renormalises over the class outputs alone, so a sigmoid head yields a proper distribution and a
    /// joint softmax over the source and class outputs yields the exact class posterior, because the
    /// shared normaliser cancels.
    /// </remarks>
    private Tensor<T> ClassNegativeLogLikelihood(
        Tensor<T> discriminatorOutput, Tensor<T> oneHotLabels, bool emitsProbabilities)
    {
        int batchSize = discriminatorOutput.Shape[0];
        var classScores = Engine.TensorSlice(discriminatorOutput, new[] { 0, 1 }, new[] { batchSize, _numClasses });
        var logits = emitsProbabilities
            ? Engine.TensorLog(Engine.TensorClamp(classScores, NumOps.FromDouble(ProbabilityFloor), NumOps.One))
            : classScores;

        var logPosterior = Engine.TensorLogSoftmax(logits, 1);
        var labelled = Engine.ReduceSum(Engine.TensorMultiply(logPosterior, oneHotLabels), new[] { 1 }, keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(labelled, new[] { 0 }, keepDims: false));
    }

    /// <summary>Lower bound applied to a probability before its logarithm is taken.</summary>
    private const double ProbabilityFloor = 1e-7;

    /// <summary>
    /// Generates class-conditional images.
    /// </summary>
    public Tensor<T> GenerateConditional(Tensor<T> noise, Tensor<T> classLabels)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, classLabels);
        return Generator.Predict(input);
    }

    /// <summary>
    /// Creates one-hot encoded class labels.
    /// </summary>
    public Tensor<T> CreateOneHotLabels(int batchSize, int classIndex)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                "Batch size must be positive.");
        }

        if (classIndex < 0 || classIndex >= _numClasses)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex), classIndex,
                $"Class index must be between 0 and {_numClasses - 1} (inclusive).");
        }

        var labels = new Tensor<T>(new int[] { batchSize, _numClasses });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                labels[b, c] = c == classIndex ? NumOps.One : NumOps.Zero;
            }
        }

        return labels;
    }

    /// <summary>
    /// Generates random noise tensor using vectorized Gaussian noise generation.
    /// </summary>
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
    /// Resets both optimizer states for a fresh training run.
    /// </summary>
    public void ResetOptimizerState()
    {
        _generatorOptimizer.Reset();
        _discriminatorOptimizer.Reset();
    }

    protected override void InitializeLayers()
    {
        // AC-GAN doesn't use layers directly
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The model's own Layers list is empty: its layers live in the generator and the discriminator. The
    /// generator is read on the Predict input and the discriminator on what the generator produced from it.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var result = new Dictionary<string, Tensor<T>>();
        foreach (var kv in Generator.GetNamedLayerActivations(input))
            result["Generator/" + kv.Key] = kv.Value;
        var generated = Generator.Predict(input);
        foreach (var kv in Discriminator.GetNamedLayerActivations(generated))
            result["Discriminator/" + kv.Key] = kv.Value;
        return result;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Generator.Predict is a NESTED Predict: when the inference arena is enabled this
        // opens a nested TensorArena and detaches the generator's output to a GC-owned tensor,
        // which is safe to return through this model's own funnel (which detaches again).
        return Generator.Predict(input);
    }

    /// <summary>
    /// Performs a single training iteration using the standard neural network interface.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adapts the AC-GAN's specialized training to the standard <see cref="NeuralNetworkBase{T}.Train"/>
    /// interface by automatically generating random class labels for both real and fake samples.
    /// </para>
    /// <para>
    /// The AC-GAN training process differs from standard neural networks because it requires:
    /// <list type="bullet">
    /// <item><description>Real images with their class labels</description></item>
    /// <item><description>Noise vectors for generating fake images</description></item>
    /// <item><description>Target class labels for the generated images</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// When using this simplified interface, random class labels are generated using
    /// <see cref="RandomHelper.ThreadSafeRandom"/> for thread-safe, cryptographically-seeded
    /// random number generation. For more control over class labels, use the
    /// <see cref="TrainStep"/> method directly.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you train an AC-GAN using the same
    /// interface as other neural networks. Just provide:
    /// <list type="bullet">
    /// <item><description><paramref name="input"/>: Random noise vectors (like random seeds for image generation)</description></item>
    /// <item><description><paramref name="expectedOutput"/>: Real images to learn from</description></item>
    /// </list>
    ///
    /// The method automatically assigns random class labels (like "digit 3", "digit 7", etc.)
    /// to both the real images and the images to generate. While this is convenient,
    /// for best results you should use <see cref="TrainStep"/> with actual class labels
    /// from your dataset.
    /// </para>
    /// </remarks>
    /// <param name="input">The noise tensor used as input to the generator network.
    /// Shape should be [batchSize, noiseSize] where noiseSize matches the generator's expected input.</param>
    /// <param name="expectedOutput">The real images tensor used for discriminator training.
    /// Shape should be [batchSize, height, width, channels] or equivalent flattened form.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> or
    /// <paramref name="expectedOutput"/> is null.</exception>
    /// <seealso cref="TrainStep"/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input), "Noise input tensor cannot be null.");
        }

        if (expectedOutput is null)
        {
            throw new ArgumentNullException(nameof(expectedOutput), "Real images tensor cannot be null.");
        }

        var generatorInput = WithBatchAxis(input, Generator.Architecture.InputType);
        var realImages = WithBatchAxis(expectedOutput, Discriminator.Architecture.InputType);
        int batchSize = realImages.Shape[0];

        Tensor<T> noise;
        Tensor<T> fakeLabels;
        int width = generatorInput.Rank == 2 ? generatorInput.Shape[1] : -1;
        if (width == Generator.Architecture.InputSize && width > _numClasses)
        {
            // The input Predict takes: noise followed by the class conditioning. The class to generate is
            // the conditioning's largest entry, which recovers a one-hot input exactly.
            noise = Engine.TensorSlice(generatorInput, new[] { 0, 0 }, new[] { batchSize, width - _numClasses });
            fakeLabels = OneHotOfLargest(
                Engine.TensorSlice(generatorInput, new[] { 0, width - _numClasses }, new[] { batchSize, _numClasses }));
        }
        else
        {
            // Bare noise: draw the classes to generate.
            var random = RandomHelper.ThreadSafeRandom;
            var fakeLabelIndices = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                fakeLabelIndices[i] = random.Next(_numClasses);
            }

            noise = generatorInput;
            fakeLabels = CreateOneHotLabelsFromIndices(batchSize, fakeLabelIndices);
        }

        // Train carries no labels for the real images, so they train the source term only. The previous
        // version labelled them at random, which trained the auxiliary classifier on noise.
        TrainStepCore(realImages, realLabels: null, noise, fakeLabels);
    }

    /// <summary>One-hot labels naming each row's largest entry.</summary>
    private Tensor<T> OneHotOfLargest(Tensor<T> scores)
    {
        int batchSize = scores.Shape[0];
        var indices = new int[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            int best = 0;
            for (int c = 1; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(scores[b, c], scores[b, best])) best = c;
            }

            indices[b] = best;
        }

        return CreateOneHotLabelsFromIndices(batchSize, indices);
    }

    /// <summary>
    /// Creates one-hot encoded label tensors from class indices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One-hot encoding converts class indices (0, 1, 2, ...) into binary vectors
    /// where only one element is 1 and all others are 0.
    /// </para>
    /// <para><b>For Beginners:</b> One-hot encoding converts a class number into a format
    /// neural networks understand better.
    ///
    /// Example with 4 classes:
    /// <list type="bullet">
    /// <item><description>Class 0 becomes [1, 0, 0, 0]</description></item>
    /// <item><description>Class 1 becomes [0, 1, 0, 0]</description></item>
    /// <item><description>Class 2 becomes [0, 0, 1, 0]</description></item>
    /// <item><description>Class 3 becomes [0, 0, 0, 1]</description></item>
    /// </list>
    ///
    /// This helps the network learn to distinguish between classes more clearly.
    /// </para>
    /// </remarks>
    /// <param name="batchSize">The number of samples in the batch.</param>
    /// <param name="classIndices">Array of class indices, one per sample in the batch.</param>
    /// <returns>A tensor of shape [batchSize, numClasses] with one-hot encoded labels.</returns>
    private Tensor<T> CreateOneHotLabelsFromIndices(int batchSize, int[] classIndices)
    {
        var labels = new Tensor<T>(new int[] { batchSize, _numClasses });
        for (int i = 0; i < batchSize; i++)
        {
            int classIndex = classIndices[i];
            for (int c = 0; c < _numClasses; c++)
            {
                labels[i, c] = (classIndex >= 0 && classIndex < _numClasses && c == classIndex)
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return labels;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "NumClasses", _numClasses }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes AC-GAN-specific data including networks and optimizer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method serializes all components needed to fully restore an AC-GAN's training state:
    /// <list type="bullet">
    /// <item><description>Number of classes</description></item>
    /// <item><description>Loss histories for monitoring training progress</description></item>
    /// <item><description>Generator and Discriminator networks with all learned weights</description></item>
    /// <item><description>Optimizer states (momentum, adaptive learning rates, timesteps)</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> When you save an AC-GAN during training, this method ensures
    /// that everything needed to resume training is saved:
    /// <list type="bullet">
    /// <item><description>The networks' learned knowledge (weights and biases)</description></item>
    /// <item><description>The optimizers' "memory" (like Adam's momentum vectors)</description></item>
    /// <item><description>Training history (loss values for monitoring)</description></item>
    /// </list>
    ///
    /// Without saving optimizer states, resuming training would be like starting with a new
    /// optimizer that has forgotten all the momentum and adaptive learning rates it built up,
    /// which can cause unstable training after loading.
    /// </para>
    /// </remarks>
    /// <param name="writer">The binary writer to serialize data to.</param>


    /// <summary>
    /// Deserializes AC-GAN-specific data including networks and optimizer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method restores all components needed to continue AC-GAN training from a saved state:
    /// <list type="bullet">
    /// <item><description>Number of classes for classification</description></item>
    /// <item><description>Loss histories for training progress visualization</description></item>
    /// <item><description>Generator and Discriminator networks with all learned weights</description></item>
    /// <item><description>Optimizer states (momentum vectors, adaptive learning rates, timesteps)</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> When you load a saved AC-GAN, this method restores everything
    /// needed to continue training exactly where you left off:
    /// <list type="bullet">
    /// <item><description>The networks remember everything they learned</description></item>
    /// <item><description>The optimizers remember their momentum and learning rate adjustments</description></item>
    /// <item><description>Training can resume smoothly without any "warm-up" period</description></item>
    /// </list>
    ///
    /// This is especially important for Adam optimizer which maintains momentum vectors (m and v)
    /// and a timestep counter - losing these would cause training instability after loading.
    /// </para>
    /// </remarks>
    /// <param name="reader">The binary reader to deserialize data from.</param>


    // UpdateParameters split the vector between Generator and Discriminator; GetExtraTrainableLayers
    // yields the same two in the same order, so the base reproduces the split. Removed under AIDN082.
}
