using AiDotNet.Tensors.Engines.Autodiff;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;

using System.Linq;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Pix2Pix GAN for paired image-to-image translation tasks.
/// </summary>
/// <remarks>
/// <para>
/// Pix2Pix is a conditional GAN for paired image-to-image translation:
/// - Uses a U-Net generator with skip connections
/// - Uses a PatchGAN discriminator that classifies image patches
/// - Combines adversarial loss with L1 reconstruction loss
/// - Requires paired training data (input-output pairs)
/// - Works for various tasks: edges to photo, day to night, sketch to image, etc.
/// </para>
/// <para><b>For Beginners:</b> Pix2Pix transforms one type of image to another.
///
/// Key features:
/// - Learns from paired examples (input A becomes output B)
/// - Generator: U-Net architecture preserves spatial information
/// - Discriminator: PatchGAN focuses on local image patches
/// - Loss: Both "looks real" and "matches input"
///
/// Example use cases:
/// - Convert sketches to realistic photos
/// - Colorize black-and-white images
/// - Transform day scenes to night
/// - Semantic labels to photorealistic images
/// - Map to satellite image
///
/// Reference: Isola et al., "Image-to-Image Translation with Conditional
/// Adversarial Networks" (2017)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new Pix2PixOptions { ImageSize = 256, InputChannels = 3, OutputChannels = 3 };
/// var model = new Pix2Pix&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 3, 256, 256 });
/// var translated = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Image-to-Image Translation with Conditional Adversarial Networks", "https://arxiv.org/abs/1611.07004", Year = 2017, Authors = "Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros")]
public partial class Pix2Pix<T> : ImageTranslationModelLayoutBase<T>
{

    // Generator then Discriminator are discovered as sub-network members, in declaration order,
    // which is the order this hook used and therefore the serialization order. Removed under AIDN082.
    private readonly Pix2PixOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly List<T> _discriminatorLosses = new List<T>();
    private readonly List<T> _generatorLosses = new List<T>();

    /// <summary>
    /// The optimizer for the generator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer for the discriminator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    /// <summary>
    /// The coefficient for the L1 reconstruction loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the trade-off between adversarial loss and L1 loss. Typical value is 100.
    /// Higher values encourage outputs to be closer to ground truth.
    /// </para>
    /// <para><b>For Beginners:</b> How important is matching the target exactly.
    ///
    /// - Higher (e.g., 100): output closely matches target
    /// - Lower (e.g., 10): more creative but less accurate
    /// - Paper uses 100 as default
    /// </para>
    /// </remarks>
    private readonly double _l1Lambda;

    /// <summary>
    /// Gets the U-Net generator network.
    /// </summary>
    public NeuralNetworkBase<T> Generator { get; private set; }

    /// <summary>
    /// Gets the PatchGAN discriminator network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PatchGAN classifies whether each N x N patch in an image is real or fake,
    /// rather than classifying the entire image. This encourages sharp high-frequency
    /// details and works well for image-to-image translation.
    /// </para>
    /// <para><b>For Beginners:</b> Discriminator checks local image quality.
    ///
    /// Instead of:
    /// - "Is the whole image real?" (standard discriminator)
    ///
    /// PatchGAN asks:
    /// - "Is this patch real? Is that patch real?" (many local checks)
    /// - This catches more detailed mistakes
    /// - Results in sharper, more realistic outputs
    /// </para>
    /// </remarks>
    public NeuralNetworkBase<T> Discriminator { get; private set; }

    private readonly ILossFunction<T> _lossFunction;

    // Stored under the constructor's own parameter names so the clone plan replays the constructor
    // with them. Without these it fell back to the model's outer Architecture for every sub-network.
    private readonly NeuralNetworkArchitecture<T> _generatorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _discriminatorArchitecture;

    /// <summary>
    /// Creates the combined Pix2Pix architecture with correct dimension handling.
    /// </summary>
    /// <param name="generatorArchitecture">The generator architecture.</param>
    /// <param name="inputType">The type of input.</param>
    /// <returns>The combined architecture for Pix2Pix.</returns>
    private static NeuralNetworkArchitecture<T> CreatePix2PixArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: generatorArchitecture.InputHeight,
                inputWidth: generatorArchitecture.InputWidth,
                inputDepth: generatorArchitecture.InputDepth,
                outputSize: generatorArchitecture.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: generatorArchitecture.InputSize,
            outputSize: generatorArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Pix2Pix{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">U-Net generator architecture.</param>
    /// <param name="discriminatorArchitecture">PatchGAN discriminator architecture.</param>
    /// <param name="inputType">Input type.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="l1Lambda">L1 loss coefficient. Default is 100.0.</param>
    /// <remarks>
    /// <para>
    /// The Pix2Pix constructor initializes both the generator and discriminator networks along with their
    /// respective optimizers. The L1 lambda coefficient controls how strongly the output should match
    /// the target image.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up Pix2Pix with sensible defaults.
    ///
    /// Key parameters:
    /// - Generator/discriminator architectures define the network structures
    /// - Optimizers control how the networks learn
    /// - L1 lambda (100.0) controls how closely output matches target
    /// </para>
    /// </remarks>
    public Pix2Pix(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        double l1Lambda = 100.0,
        Pix2PixOptions? options = null)
        : base(CreatePix2PixArchitecture(generatorArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        _options = options ?? new Pix2PixOptions();
        Options = _options;
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }
        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }
        if (l1Lambda < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(l1Lambda), l1Lambda, "L1 lambda must be non-negative.");
        }

        _l1Lambda = l1Lambda;

        _generatorArchitecture = generatorArchitecture;
        _discriminatorArchitecture = discriminatorArchitecture;
        Generator = CreateSubNetworkForInputType(generatorArchitecture, inputType);
        Discriminator = CreateSubNetworkForInputType(discriminatorArchitecture, inputType);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        // Initialize optimizers (default to Adam if not provided)
        // Isola et al. 2017, section 3.3: Adam with learning rate 0.0002 and momentum parameters
        // beta1 = 0.5, beta2 = 0.999 for both networks. Callers can pass their own optimizers.
        _generatorOptimizer = generatorOptimizer
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator, CreatePaperAdamOptions());
        _discriminatorOptimizer = discriminatorOptimizer
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator, CreatePaperAdamOptions());

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for Pix2Pix.
    /// </summary>
    /// <param name="inputImages">Input images (e.g., sketches, semantic maps).</param>
    /// <param name="targetImages">Target output images (e.g., photos).</param>
    /// <returns>Tuple of (discriminator loss, generator loss, L1 loss).</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Pix2Pix training algorithm:
    /// 1. Train discriminator on real and fake image pairs
    /// 2. Train generator with combined adversarial and L1 loss
    /// 3. The discriminator learns to distinguish real from fake
    /// 4. The generator learns to both fool the discriminator and match the target
    /// </para>
    /// <para><b>For Beginners:</b> One training round for Pix2Pix.
    ///
    /// The training process:
    /// - Discriminator learns to spot fake images
    /// - Generator learns to create realistic images that match target
    /// - L1 loss ensures output closely matches expected result
    /// - Returns loss values for monitoring progress
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss, T l1Loss) TrainStep(
        Tensor<T> inputImages,
        Tensor<T> targetImages)
    {
        if (inputImages is null)
        {
            throw new ArgumentNullException(nameof(inputImages), "Input images tensor cannot be null.");
        }

        if (targetImages is null)
        {
            throw new ArgumentNullException(nameof(targetImages), "Target images tensor cannot be null.");
        }

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        // ONE generator forward, recorded on the generator's tape: the pattern the base GAN uses (#1390).
        // The discriminator trains on a detached copy, so its step cannot reach the generator. The
        // generator step then scores the tracked original through the frozen discriminator.
        using var generatorTape = new GradientTape<T>();
        var fakeTracked = Generator.ForwardForTraining(inputImages);
        var fakeImages = new Tensor<T>(fakeTracked.Shape.ToArray());
        fakeTracked.AsSpan().CopyTo(fakeImages.AsWritableSpan());

        // ----- Discriminator: maximise L_cGAN(G, D) (Isola et al. 2017, eq. 1) -----
        // "We divide the objective by 2 while optimizing D, which slows down the rate at which D learns
        // relative to G" (section 3.3). The loss is averaged over every output, which is the PatchGAN
        // average over patches when the discriminator emits a map.
        T discriminatorLoss;
        using (var discriminatorTape = new GradientTape<T>())
        {
            var realScores = Discriminator.ForwardForTraining(ConcatenateImages(inputImages, targetImages));
            var fakeScores = Discriminator.ForwardForTraining(ConcatenateImages(inputImages, fakeImages));
            var discriminatorObjective = Engine.TensorMultiplyScalar(
                Engine.TensorAdd(
                    Discriminator.BinaryCrossEntropyOnTape(realScores, targetIsReal: true),
                    Discriminator.BinaryCrossEntropyOnTape(fakeScores, targetIsReal: false)),
                NumOps.FromDouble(0.5));
            discriminatorLoss = StepOnTape(discriminatorTape, discriminatorObjective, Discriminator,
                _discriminatorOptimizer);
        }

        // ----- Generator: L_cGAN + lambda * L_L1 (eq. 4) -----
        // The adversarial term is the one the paper trains: maximise log D(x, G(x)) rather than minimise
        // log(1 - D(x, G(x))). The L1 term is the mean absolute error to the target.
        var generatorScores = Discriminator.ForwardFrozenOnTape(ConcatenateImages(inputImages, fakeTracked));
        var adversarialLoss = Discriminator.BinaryCrossEntropyOnTape(generatorScores, targetIsReal: true);
        var l1Tensor = Engine.ReduceMean(
            Engine.TensorAbs(Engine.TensorSubtract(fakeTracked, targetImages)),
            Enumerable.Range(0, fakeTracked.Shape.Length).ToArray(),
            keepDims: false);
        var generatorObjective = Engine.TensorAdd(adversarialLoss, Engine.TensorMultiplyScalar(l1Tensor, NumOps.FromDouble(_l1Lambda)));
        T generatorLoss = StepOnTape(generatorTape, generatorObjective, Generator, _generatorOptimizer);
        T l1Loss = l1Tensor.Length > 0 ? l1Tensor[0] : NumOps.Zero;

        // Track losses
        _discriminatorLosses.Add(discriminatorLoss);
        _generatorLosses.Add(generatorLoss);

        if (_discriminatorLosses.Count > 100)
        {
            _discriminatorLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss, l1Loss);
    }

    /// <summary>
    /// Translates input images to output images.
    /// </summary>
    /// <param name="inputImages">The input images to translate.</param>
    /// <returns>The translated output images.</returns>
    public Tensor<T> Translate(Tensor<T> inputImages)
    {
        Generator.SetTrainingMode(false);
        return Generator.Predict(inputImages);
    }

    /// <summary>
    /// Pairs each input image with the image being judged, along the channel axis for spatial tensors
    /// and the feature axis otherwise: the conditional discriminator's input (Isola et al. 2017).
    /// </summary>
    /// <remarks>
    /// An Engine concatenation, not an element copy. The generator step passes its tape-tracked output
    /// through here, and copying elements one by one into a fresh tensor severed the gradient from the
    /// discriminator's input back to the generator.
    /// </remarks>
    private Tensor<T> ConcatenateImages(Tensor<T> images1, Tensor<T> images2)
    {
        if (images1.Shape.Length < 1 || images2.Shape.Length < 1)
        {
            throw new ArgumentException("Both image tensors must have at least one dimension.");
        }

        int batchSize = images1.Shape[0];
        if (images2.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch size mismatch: images1 has {batchSize} samples, images2 has {images2.Shape[0]} samples.");
        }

        if (Architecture.InputType == InputType.ThreeDimensional)
        {
            // [B, C, H, W]: this model's declared layout, so the pair stacks along channels.
            if (images1.Shape.Length == 4 && images2.Shape.Length == 4)
                return Engine.TensorConcatenate(new[] { images1, images2 }, axis: 1);

            // [B, H*W, C]: channels are the last axis.
            if (images1.Shape.Length == 3 && images2.Shape.Length == 3)
                return Engine.TensorConcatenate(new[] { images1, images2 }, axis: 2);
        }

        var flat1 = images1.Shape.Length == 2 ? images1 : Engine.Reshape(images1, new[] { batchSize, images1.Length / batchSize });
        var flat2 = images2.Shape.Length == 2 ? images2 : Engine.Reshape(images2, new[] { batchSize, images2.Length / batchSize });
        return Engine.TensorConcatenate(new[] { flat1, flat2 }, axis: 1);
    }

    /// <summary>The optimizer settings of Isola et al. 2017, section 3.3.</summary>
    private static AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CreatePaperAdamOptions()
        => new()
        {
            InitialLearningRate = 0.0002,
            Beta1 = 0.5,
            Beta2 = 0.999,
        };

    /// <summary>
    /// Resets both optimizer states for a fresh training run.
    /// </summary>
    public void ResetOptimizerState()
    {
        _generatorOptimizer.Reset();
        _discriminatorOptimizer.Reset();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Pix2Pix doesn't use layers directly
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    /// <remarks>
    /// The model's own Layers list is empty: its layers live in the generator and the PatchGAN discriminator.
    /// The discriminator is read the way it is trained, on the source image paired with the translation.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var result = new Dictionary<string, Tensor<T>>();
        foreach (var kv in Generator.GetNamedLayerActivations(input))
            result["Generator/" + kv.Key] = kv.Value;
        var source = WithBatchAxis(input, Generator.Architecture.InputType);
        var translated = WithBatchAxis(Generator.Predict(input), Generator.Architecture.InputType);
        foreach (var kv in Discriminator.GetNamedLayerActivations(ConcatenateImages(source, translated)))
            result["Discriminator/" + kv.Key] = kv.Value;
        return result;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return Generator.Predict(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Source and target share the generator's image layout.
        TrainStep(
            WithBatchAxis(input, Generator.Architecture.InputType),
            WithBatchAxis(expectedOutput, Generator.Architecture.InputType));
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
                { "L1Lambda", _l1Lambda }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    // UpdateParameters split the vector between Generator and Discriminator; GetExtraTrainableLayers
    // yields the same two in the same order, so the base reproduces the split. Removed under AIDN082.
}
