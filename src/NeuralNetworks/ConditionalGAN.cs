using AiDotNet.Tensors.Engines.Autodiff;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Conditional Generative Adversarial Network (cGAN), which generates data conditioned
/// on additional information such as class labels, attributes, or other contextual data.
/// </summary>
/// <remarks>
/// <para>
/// Conditional GANs extend the basic GAN framework by:
/// - Conditioning both the generator and discriminator on additional information
/// - Allowing controlled generation (e.g., "generate a digit 7")
/// - Enabling class-conditional image synthesis
/// - Providing explicit control over the generated output characteristics
/// </para>
/// <para><b>For Beginners:</b> cGAN lets you control what kind of image is generated.
///
/// Key features:
/// - You can specify what you want to generate (e.g., "cat" vs. "dog")
/// - Both the generator and discriminator see the conditioning information
/// - Generator: "Given this label, create a matching image"
/// - Discriminator: "Is this image both real AND matching the label?"
///
/// Example use cases:
/// - Generate a specific digit (0-9) in MNIST
/// - Create images of specific object classes
/// - Generate faces with specific attributes (smiling, glasses, etc.)
///
/// Reference: Mirza and Osindero, "Conditional Generative Adversarial Nets" (2014)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 100 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new ConditionalGAN&lt;float&gt;())
///     .Build(trainX, trainY);
/// var generated = result.Predict(noise);
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
[ResearchPaper("Conditional Generative Adversarial Nets", "https://arxiv.org/abs/1411.1784", Year = 2014, Authors = "Mehdi Mirza, Simon Osindero")]
public partial class ConditionalGAN<T> : GenerativeAdversarialNetwork<T>
{
    private readonly ConditionalGANOptions _options;

    private readonly List<T> _generatorLosses = new List<T>();

    /// <summary>
    /// The number of condition classes/categories.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This represents the number of distinct conditioning values (e.g., 10 for MNIST digits,
    /// 1000 for ImageNet classes). The conditioning information is typically provided as
    /// one-hot encoded vectors of this size.
    /// </para>
    /// <para><b>For Beginners:</b> The number of different types of things you can generate.
    ///
    /// Examples:
    /// - MNIST digits: 10 classes (0-9)
    /// - CIFAR-10: 10 object classes
    /// - ImageNet: 1000 object classes
    /// - Custom dataset: however many categories you have
    /// </para>
    /// </remarks>
    private int _numConditionClasses;

    // Store the generator's full [noise | condition] architecture and the
    // discriminator's pre-conditioning architecture for CreateNewInstance().
    private NeuralNetworkArchitecture<T>? _originalGeneratorArchitecture;
    private NeuralNetworkArchitecture<T>? _originalDiscriminatorArchitecture;

    /// <summary>
    /// Creates the combined ConditionalGAN architecture with correct dimension handling.
    /// </summary>
    /// <remarks>
    /// The supplied generator architecture already describes the full
    /// <c>[noise | condition]</c> input width. The public Predict/Train APIs
    /// accept noise-only tensors and append the condition vector at runtime,
    /// so expanding the architecture here would count the condition features
    /// twice.
    /// </remarks>
    private static NeuralNetworkArchitecture<T> CreateConditionalGeneratorArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        int numConditionClasses)
    {
        if (generatorArchitecture.InputSize > 0
            && generatorArchitecture.InputHeight <= 0
            && generatorArchitecture.InputWidth <= 0
            && generatorArchitecture.InputSize <= numConditionClasses)
        {
            throw new ArgumentException(
                "ConditionalGAN generatorArchitecture.InputSize must be the full noise+condition width " +
                "and therefore exceed numConditionClasses for flat generators.",
                nameof(generatorArchitecture));
        }

        return generatorArchitecture;
    }

    /// <summary>
    /// Creates the discriminator architecture for conditional GAN.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateConditionalDiscriminatorArchitecture(
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numConditionClasses)
    {
        // Check if discriminator expects spatial input (3D/4D tensor with H, W, D > 0)
        bool isSpatialInput = discriminatorArchitecture.InputHeight > 0
            && discriminatorArchitecture.InputWidth > 0
            && discriminatorArchitecture.InputDepth > 0;

        int inputSize;
        int inputDepth;

        if (isSpatialInput)
        {
            // Spatial discriminator: conditions are added as extra channels
            // inputDepth increases by numConditionClasses
            inputDepth = discriminatorArchitecture.InputDepth + numConditionClasses;
            // inputSize may need to account for the extra channels across spatial dims
            inputSize = discriminatorArchitecture.InputSize > 0
                ? discriminatorArchitecture.InputSize + (numConditionClasses * discriminatorArchitecture.InputHeight * discriminatorArchitecture.InputWidth)
                : 0;
        }
        else
        {
            // Flat discriminator: conditions are concatenated to the input vector
            inputSize = discriminatorArchitecture.InputSize > 0
                ? discriminatorArchitecture.InputSize + numConditionClasses
                : 0;
            inputDepth = discriminatorArchitecture.InputDepth;
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: discriminatorArchitecture.InputType,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: discriminatorArchitecture.Complexity,
            inputSize: inputSize,
            inputHeight: discriminatorArchitecture.InputHeight,
            inputWidth: discriminatorArchitecture.InputWidth,
            inputDepth: inputDepth,
            outputSize: 1,
            layers: discriminatorArchitecture.Layers);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalGAN{T}"/> class with default configuration.
    /// </summary>
    public ConditionalGAN()
        : this(
            new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 110, outputSize: 784),
            new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.BinaryClassification, inputSize: 784, outputSize: 1),
            numConditionClasses: 10,
            inputType: InputType.OneDimensional)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator. Its input width is the full noise+condition width.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.</param>
    /// <param name="numConditionClasses">The number of conditioning classes/categories.</param>
    /// <param name="inputType">The type of input the cGAN will process.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a conditional GAN where both the generator and discriminator
    /// receive conditioning information. The generator takes noise concatenated with a
    /// condition vector, and the discriminator takes an image concatenated with the same
    /// condition vector.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a GAN that can generate specific types of images.
    ///
    /// Parameters:
    /// - generatorArchitecture: How the generator network is structured
    /// - discriminatorArchitecture: How the discriminator network is structured
    /// - numConditionClasses: How many different types/classes you have
    /// - inputType: What kind of data (usually images)
    /// - generatorOptimizer/discriminatorOptimizer: Custom learning algorithms (optional)
    /// </para>
    /// </remarks>
    public ConditionalGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numConditionClasses,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        ConditionalGANOptions? options = null)
        : base(
            CreateConditionalGeneratorArchitecture(
                generatorArchitecture ?? throw new ArgumentNullException(nameof(generatorArchitecture)),
                numConditionClasses > 0 ? numConditionClasses : throw new ArgumentOutOfRangeException(nameof(numConditionClasses), numConditionClasses, "Number of condition classes must be positive.")),
            CreateConditionalDiscriminatorArchitecture(
                discriminatorArchitecture ?? throw new ArgumentNullException(nameof(discriminatorArchitecture)),
                numConditionClasses),
            inputType,
            generatorOptimizer,
            discriminatorOptimizer,
            lossFunction,
            options: null,
            defaultGeneratorOptimizerOptions: CreateAdamOptimizerOptions(),
            defaultDiscriminatorOptimizerOptions: CreateAdamOptimizerOptions())
    {
        _options = options ?? new ConditionalGANOptions();
        Options = _options;

        // The generator is already full-width; the discriminator is stored
        // before conditioning is applied so clone/deserialize does not expand
        // it a second time.
        _originalGeneratorArchitecture = generatorArchitecture;
        _originalDiscriminatorArchitecture = discriminatorArchitecture;

        _numConditionClasses = numConditionClasses;
    }

    /// <summary>
    /// Runs Predict on a network, handling batched [B, N] input by processing per-sample.
    /// FeedForwardNeuralNetwork expects 1D [N] input, so batch tensors are sliced and re-stacked.
    /// </summary>
    private static Tensor<T> PredictBatched(NeuralNetworkBase<T> network, Tensor<T> batchedInput)
    {
        // Only split 2D [B, N] feed-forward inputs; leave higher-rank spatial tensors intact
        if (batchedInput.Rank != 2)
            return network.Predict(batchedInput);

        int batchSize = batchedInput.Shape[0];
        var results = new List<Tensor<T>>(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            results.Add(network.Predict(batchedInput.GetSlice(b)));
        }

        return Tensor<T>.Stack([.. results]);
    }

    /// <summary>
    /// Predicts output by treating the input as noise and adding default conditions.
    /// </summary>
    /// <param name="input">Noise tensor of size <c>latentDim</c> (1D) or <c>[batch, latentDim]</c> (2D).
    /// This is smaller than the generator's actual input because condition features are appended internally.</param>
    /// <remarks>
    /// <para>
    /// The standard Predict interface accepts noise-only input (latentDim features).
    /// Conditions (one-hot class 0 by default) are concatenated internally, producing
    /// a generator input of size <c>latentDim + numConditionClasses</c>.
    /// To control the class, use <see cref="GenerateConditional"/> instead.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        Generator.SetTrainingMode(false);

        bool wasSingleSample = input.Rank == 1;
        Tensor<T> noise = input;

        // Ensure noise is 2D [batch, features] for concatenation
        if (wasSingleSample)
        {
            noise = new Tensor<T>(new int[] { 1, input.Length });
            for (int i = 0; i < input.Length; i++)
                noise[0, i] = input[i];
        }

        int batchSize = noise.Shape[0];
        int latentDim = Generator.Architecture.InputSize - _numConditionClasses;
        if (latentDim > 0 && noise.Shape[1] != latentDim)
        {
            throw new ArgumentException(
                $"ConditionalGAN.Predict expects NOISE of latentDim = generator inputSize - numConditionClasses " +
                $"= {Generator.Architecture.InputSize} - {_numConditionClasses} = {latentDim} features; got {noise.Shape[1]}. " +
                "Condition features are appended internally (the generator architecture's inputSize is the FULL " +
                "noise+condition width — see the parameterless ctor's 110 = 100 noise + 10 classes convention). " +
                "To control the class, use GenerateConditional.",
                nameof(input));
        }

        var conditions = CreateOneHotCondition(batchSize, 0);

        var generatorInput = ConcatenateTensors(noise, conditions);
        var output = PredictBatched(Generator, generatorInput);

        // Single sample: squeeze batch dimension [1, ...] → [...]
        // Preserves spatial shape for CNN generators (e.g., [1,H,W,C] → [H,W,C])
        if (wasSingleSample && output.Rank > 1 && output.Shape[0] == 1)
        {
            var fullShape = output._shape;
            var squeezedShape = new int[fullShape.Length - 1];
            Array.Copy(fullShape, 1, squeezedShape, 0, squeezedShape.Length);
            return output.Reshape(squeezedShape);
        }

        return output;
    }

    /// <summary>
    /// Performs one training step for the conditional GAN.
    /// </summary>
    /// <param name="realImages">A tensor containing real images.</param>
    /// <param name="conditions">A tensor containing conditioning labels (one-hot encoded).</param>
    /// <param name="noise">A tensor containing random noise for the generator.</param>
    /// <returns>A tuple containing the discriminator and generator loss values.</returns>
    /// <remarks>
    /// <para>
    /// This method trains both the generator and discriminator with conditioning information:
    /// 1. Train discriminator on real images with their true labels
    /// 2. Train discriminator on fake images with the generator's conditioning labels
    /// 3. Train generator to create images that fool the discriminator for the given conditions
    /// </para>
    /// <para><b>For Beginners:</b> One training round for conditional GAN.
    ///
    /// The training process:
    /// - Discriminator learns to verify image-label pairs are correct
    /// - Generator learns to create images matching the specified labels
    /// - Both networks use the conditioning information to guide learning
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> conditions,
        Tensor<T> noise)
    {
        Guard.NotNull(conditions);
        _pendingConditions = conditions;
        try
        {
            Train(noise, realImages);
        }
        finally
        {
            _pendingConditions = null;
        }

        return (_lastDiscriminatorLoss, _lastGeneratorLoss);
    }

    private Tensor<T>? _pendingConditions;
    private (bool Generator, bool Discriminator)? _savedModes;

    /// <inheritdoc/>
    /// <remarks>Records both networks' training modes before the step switches them on.</remarks>
    protected override void OnTrainingStepStarting(Tensor<T> input, Tensor<T> target)
    {
        base.OnTrainingStepStarting(input, target);
        _savedModes = (Generator.IsTrainingMode, Discriminator.IsTrainingMode);
    }

    /// <inheritdoc/>
    /// <remarks>Both networks get back the training modes they had before the step, whether it succeeded or threw.</remarks>
    protected override void OnTrainingStepEnded(bool succeeded)
    {
        if (_savedModes is { } modes)
        {
            Generator.SetTrainingMode(modes.Generator);
            Discriminator.SetTrainingMode(modes.Discriminator);
            _savedModes = null;
        }

        base.OnTrainingStepEnded(succeeded);
    }
    private T _lastDiscriminatorLoss = MathHelper.GetNumericOperations<T>().Zero;
    private T _lastGeneratorLoss = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// One-hot class conditions for a batch with no labels: each sample's class is read from its first value. The
    /// supervised <see cref="NeuralNetworkBase{T}.Train(Tensor{T}, Tensor{T})"/> contract carries no labels;
    /// <see cref="TrainStep"/> takes real ones.
    /// </summary>
    private Tensor<T> DeriveConditions(Tensor<T> realImages)
    {
        int batchSize = realImages.Shape[0];
        var conditions = new Tensor<T>(new int[] { batchSize, _numConditionClasses });
        for (int b = 0; b < batchSize; b++)
        {
            double sampleVal = NumOps.ToDouble(realImages.GetFlatIndexValue(b * (realImages.Length / batchSize)));
            int classIdx = ((int)(Math.Abs(sampleVal) * 1000)) % _numConditionClasses;
            for (int c = 0; c < _numConditionClasses; c++)
                conditions[b, c] = c == classIdx ? NumOps.One : NumOps.Zero;
        }

        return conditions;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The conditional GAN step (Mirza and Osindero 2014): the discriminator on real and generated samples, each paired
    /// with its condition, then the generator against the frozen discriminator. The input is the noise, the target the
    /// real batch. Each network steps on its own fused session, and both get their training modes back after.
    /// </remarks>
    protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> noise, Tensor<T> realImages)
    {
        var conditions = _pendingConditions ?? DeriveConditions(realImages);
        if (realImages.Shape[0] != conditions.Shape[0] || realImages.Shape[0] != noise.Shape[0])
            throw new ArgumentException(
                $"Batch sizes must match: realImages={realImages.Shape[0]}, conditions={conditions.Shape[0]}, noise={noise.Shape[0]}.");

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);
        Tensor<T> generatorInput = ConcatenateTensors(noise, conditions);
        Tensor<T> fakeImages = PredictBatched(Generator, generatorInput);
        return new[]
        {
            new Training.TrainingObjective<T>(
                new[] { realImages, fakeImages, conditions },
                data => Engine.TensorAdd(
                    Discriminator.BinaryCrossEntropyOnTape(
                        Discriminator.ForwardForTraining(ConcatenateImageAndCondition(data[0], data[2])), targetIsReal: true),
                    Discriminator.BinaryCrossEntropyOnTape(
                        Discriminator.ForwardForTraining(ConcatenateImageAndCondition(data[1], data[2])), targetIsReal: false)))
            {
                Network = Discriminator,
                Optimizer = DiscriminatorOptimizer,
                OnLoss = loss => _lastDiscriminatorLoss = loss,
            },
            new Training.TrainingObjective<T>(
                () => { Generator.SetTrainingMode(true); return new[] { generatorInput }; },
                data =>
                {
                    var stepConditions = Engine.TensorSlice(data[0],
                        new[] { 0, Generator.Architecture.InputSize - _numConditionClasses },
                        new[] { data[0].Shape[0], _numConditionClasses });
                    var withConditions = ConcatenateImageAndCondition(Generator.ForwardForTraining(data[0]), stepConditions);
                    return Discriminator.BinaryCrossEntropyOnTape(Discriminator.ForwardFrozenOnTape(withConditions), targetIsReal: true);
                })
            {
                Network = (NeuralNetworkBase<T>)Generator,
                Optimizer = GeneratorOptimizer,
                OnLoss = loss =>
                {
                    _lastGeneratorLoss = loss;
                    LastLoss = loss;
                    _generatorLosses.Add(loss);
                    if (_generatorLosses.Count > 100)
                        _generatorLosses.RemoveAt(0);
                },
            },
        };
    }
    /// <summary>
    /// Concatenates noise and condition vectors.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> noise, Tensor<T> conditions)
    {
        return Engine.TensorConcatenate([noise, conditions], axis: 1);
    }

    /// <summary>
    /// Concatenates images with condition vectors for discriminator input.
    /// Handles both spatial (3D/4D) and flattened (2D) discriminator architectures.
    /// </summary>
    private Tensor<T> ConcatenateImageAndCondition(Tensor<T> images, Tensor<T> conditions)
    {
        int batchSize = images.Shape[0];
        int conditionSize = conditions.Shape[1];

        // Check if discriminator expects spatial input (3D/4D tensor)
        var discArch = Discriminator.Architecture;
        bool expectsSpatialInput = discArch.InputHeight > 0 && discArch.InputWidth > 0;

        if (expectsSpatialInput && images.Shape.Length >= 3)
        {
            // Spatial architecture: tile conditions across spatial dimensions
            // Input image shape: [B, H, W, C] or [B, C, H, W]
            // Output shape: [B, H, W, C + K] or [B, C + K, H, W]
            return ConcatenateSpatialImageAndCondition(images, conditions);
        }
        else
        {
            // Flattened architecture: concatenate flattened image with condition vector
            return ConcatenateFlattenedImageAndCondition(images, conditions);
        }
    }

    /// <summary>
    /// Concatenates flattened images with condition vectors for 1D discriminator input.
    /// </summary>
    private Tensor<T> ConcatenateFlattenedImageAndCondition(Tensor<T> images, Tensor<T> conditions)
    {
        // Engine ops, not an element copy: the generator step passes its tape-tracked output through
        // here, and copying into a fresh tensor severed the gradient to the generator.
        int batchSize = images.Shape[0];
        var flatImages = images.Shape.Length == 2
            ? images
            : Engine.Reshape(images, new[] { batchSize, images.Length / batchSize });
        return Engine.TensorConcatenate(new[] { flatImages, conditions }, axis: 1);
    }

    /// <summary>
    /// Concatenates spatial images with condition vectors by tiling conditions
    /// across spatial dimensions and appending as extra channels.
    /// </summary>
    /// <remarks>
    /// For a [B, H, W, C] image and [B, K] condition:
    /// - Expands condition to [B, H, W, K] by replicating across H and W
    /// - Concatenates along channel axis to produce [B, H, W, C + K]
    /// </remarks>
    private Tensor<T> ConcatenateSpatialImageAndCondition(Tensor<T> images, Tensor<T> conditions)
    {
        int batchSize = images.Shape[0];
        int conditionSize = conditions.Shape[1];

        // Determine image dimensions based on shape
        int height, width, channels;
        bool isChannelsFirst;

        if (images.Shape.Length == 4)
        {
            // 4D tensor: [B, C, H, W] (channels-first) or [B, H, W, C] (channels-last)
            // Detect layout using discriminator architecture, accounting for condition channels
            var discArch = Discriminator.Architecture;

            // The discriminator's InputDepth includes the appended condition channels (C+K).
            // Raw images only have C channels, so we subtract the condition size to get
            // the original image channel count for proper layout detection.
            int conditionChannelsAdded = conditionSize;
            int originalChannelCount = Math.Max(0, discArch.InputDepth - conditionChannelsAdded);

            // Check if shape[1] matches original image channels (channels-first: [B, C, H, W])
            // or shape[3] matches original image channels (channels-last: [B, H, W, C])
            if (originalChannelCount > 0 && images.Shape[1] == originalChannelCount)
            {
                // Channels-first format: [B, C, H, W]
                channels = images.Shape[1];
                height = images.Shape[2];
                width = images.Shape[3];
                isChannelsFirst = true;
            }
            else
            {
                // Channels-last format: [B, H, W, C] (default)
                height = images.Shape[1];
                width = images.Shape[2];
                channels = images.Shape[3];
                isChannelsFirst = false;
            }
        }
        else if (images.Shape.Length == 3)
        {
            // 3D tensor: [B, H*W, C] or similar - treat as spatial
            height = Discriminator.Architecture.InputHeight;
            width = Discriminator.Architecture.InputWidth;
            channels = images.Length / (batchSize * height * width);
            isChannelsFirst = false;
        }
        else
        {
            // Fall back to flattened for unexpected shapes
            return ConcatenateFlattenedImageAndCondition(images, conditions);
        }

        // Engine ops, not an element copy (see ConcatenateFlattenedImageAndCondition): the condition is
        // tiled across every spatial position and appended as extra channels, in the image's own layout.
        // A 3-D [B, H*W, C] image is read as [B, H, W, C] and returns [B, H, W, C + K], as before.
        var spatialImages = images.Shape.Length == 4
            ? images
            : Engine.Reshape(images, new[] { batchSize, height, width, channels });
        Tensor<T> tiledConditions = isChannelsFirst
            ? Engine.TensorTile(Engine.Reshape(conditions, new[] { batchSize, conditionSize, 1, 1 }), new[] { 1, 1, height, width })
            : Engine.TensorTile(Engine.Reshape(conditions, new[] { batchSize, 1, 1, conditionSize }), new[] { 1, height, width, 1 });
        return Engine.TensorConcatenate(new[] { spatialImages, tiledConditions }, axis: isChannelsFirst ? 1 : 3);
    }

    /// <summary>
    /// Generates images conditioned on specific labels.
    /// </summary>
    public Tensor<T> GenerateConditional(Tensor<T> noise, Tensor<T> conditions)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, conditions);
        return PredictBatched(Generator, input);
    }

    /// <summary>
    /// Generates random noise tensor using vectorized Gaussian noise generation.
    /// </summary>
    public new Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
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

        // Use vectorized Gaussian noise generation
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Creates a one-hot encoded condition tensor.
    /// </summary>
    public Tensor<T> CreateOneHotCondition(int batchSize, int classIndex)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                "Batch size must be positive.");
        }

        if (classIndex < 0 || classIndex >= _numConditionClasses)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex), classIndex,
                $"Class index must be between 0 and {_numConditionClasses - 1} (inclusive).");
        }

        var conditions = new Tensor<T>(new int[] { batchSize, _numConditionClasses });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numConditionClasses; c++)
            {
                conditions[b, c] = c == classIndex ? NumOps.One : NumOps.Zero;
            }
        }

        return conditions;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "NumConditionClasses", _numConditionClasses }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }




}
