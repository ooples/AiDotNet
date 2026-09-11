using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Models.Options;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

using System.Linq;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a StyleGAN (Style-Based Generator Architecture for GANs) that generates
/// high-quality images with fine-grained control over image style at different levels.
/// </summary>
/// <remarks>
/// <para>
/// StyleGAN introduces several key innovations:
/// - Style-based generator with mapping network and synthesis network
/// - Adaptive Instance Normalization (AdaIN) for style injection
/// - Stochastic variation through noise injection
/// - Style mixing for disentangled control
/// - Progressive growing for high-resolution images
/// - State-of-the-art image quality
/// </para>
/// <para><b>For Beginners:</b> StyleGAN generates incredibly realistic images with fine control.
///
/// Key innovations:
/// - **Mapping Network**: Transforms random noise into style codes
/// - **Style Injection**: Injects style at each layer via AdaIN
/// - **Noise Injection**: Adds stochastic variation (hair, pores, etc.)
/// - **Style Mixing**: Combines styles from different sources
/// - **Progressive Growing**: Starts small, gradually adds detail
///
/// Architecture:
/// 1. Mapping Network (Z → W): Transforms latent code to intermediate space
/// 2. Synthesis Network: Generates image with style injection at each layer
/// 3. Each layer: Upsample → Conv → AdaIN → Noise → Conv → AdaIN → Noise
///
/// Why it's better:
/// - Exceptional image quality
/// - Disentangled style control (separate coarse/fine features)
/// - Style mixing (combine different sources)
/// - Perceptual path length is shorter
///
/// Applications:
/// - High-quality face generation
/// - Style transfer and manipulation
/// - Image editing and synthesis
/// - Creative AI applications
///
/// Reference: Karras et al., "A Style-Based Generator Architecture for
/// Generative Adversarial Networks" (2019)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new StyleGANOptions { LatentSize = 512, MaxResolution = 1024 };
/// var model = new StyleGAN&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 512 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Style-Based Generator Architecture for Generative Adversarial Networks", "https://arxiv.org/abs/1812.04948", Year = 2019, Authors = "Tero Karras, Samuli Laine, Timo Aila")]
public partial class StyleGAN<T> : ImageGeneratorModelLayoutBase<T>
{

    // MappingNetwork, SynthesisNetwork and Discriminator are discovered as sub-network members, in
    // declaration order, which is the order this hook used and therefore the serialization order.
    // Removed under AIDN082.
    private readonly StyleGANOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    // One Adam per network, with the hyperparameters StyleGAN reuses from Progressive GAN (Karras et al.
    // 2018, appendix A.1). The mapping network's runs at a learning
    // rate two orders of magnitude lower. These replace three hand-written Adam loops whose moment
    // vectors were stepped with gradients no backward had produced.
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _mappingOptimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _synthesisOptimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    private readonly double _initialLearningRate;

    /// <summary>
    /// The size of the latent code Z.
    /// </summary>
    private int _latentSize;

    /// <summary>
    /// The size of the intermediate latent code W.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mapping network transforms Z to W, which typically has the same dimensionality.
    /// The W space is more disentangled than Z space.
    /// </para>
    /// <para><b>For Beginners:</b> W is a "better organized" version of random noise.
    ///
    /// - Z: Random input (entangled features)
    /// - W: Organized style codes (disentangled features)
    /// - W makes it easier to control specific aspects of the image
    /// </para>
    /// </remarks>
    private int _intermediateLatentSize;

    /// <summary>
    /// Gets the mapping network that transforms Z to W.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mapping network is typically an 8-layer MLP that learns to map the input
    /// latent space Z to an intermediate latent space W with better disentanglement properties.
    /// </para>
    /// <para><b>For Beginners:</b> The "style organizer" network.
    ///
    /// Takes: Random noise (Z)
    /// Returns: Organized style codes (W)
    /// Why: W space has better separated features
    /// Result: Easier to control individual aspects
    /// </para>
    /// </remarks>
    public NeuralNetworkBase<T> MappingNetwork { get; private set; }

    /// <summary>
    /// Gets the synthesis network that generates images from styles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The synthesis network applies styles at multiple resolutions through AdaIN.
    /// It starts from a learned constant and progressively generates higher resolutions.
    /// </para>
    /// <para><b>For Beginners:</b> The "image painter" network.
    ///
    /// Process:
    /// 1. Starts from a learned constant (4x4)
    /// 2. Applies style via AdaIN at each layer
    /// 3. Adds random noise for details
    /// 4. Upsamples to next resolution
    /// 5. Repeats until final resolution
    /// </para>
    /// </remarks>
    public NeuralNetworkBase<T> SynthesisNetwork { get; private set; }

    /// <summary>
    /// Gets the discriminator network.
    /// </summary>
    public NeuralNetworkBase<T> Discriminator { get; private set; }

    /// <summary>
    /// Enables style mixing during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Style mixing uses two random latent codes and switches between them
    /// at a random layer. This encourages the network to learn localized styles.
    /// </para>
    /// <para><b>For Beginners:</b> Style mixing combines different style sources.
    ///
    /// Example:
    /// - Use Style A for coarse features (face shape, pose)
    /// - Use Style B for fine features (hair texture, skin details)
    /// - Result: Face shape from A, details from B
    ///
    /// Benefits:
    /// - Prevents features from being tied together
    /// - Enables fine-grained control
    /// - Improves disentanglement
    /// </para>
    /// </remarks>
    private bool _enableStyleMixing;

    /// <summary>
    /// Probability of style mixing during training.
    /// </summary>
    private readonly double _styleMixingProbability;

    private ILossFunction<T> _lossFunction;

    // Stored under the constructor's own parameter names so the clone plan replays the constructor
    // with them. Without these it fell back to the model's outer Architecture for every sub-network.
    private readonly NeuralNetworkArchitecture<T> _mappingNetworkArchitecture;
    private readonly NeuralNetworkArchitecture<T> _synthesisNetworkArchitecture;
    private readonly NeuralNetworkArchitecture<T> _discriminatorArchitecture;

    /// <summary>
    /// Creates the combined StyleGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateStyleGANArchitecture(
        int latentSize,
        NeuralNetworkArchitecture<T> synthesisNetworkArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType)
    {
        // Predict returns what the synthesis network generates, so that is the declared output. It was
        // the discriminator's output size -- a real/fake score, not an image.
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.VeryDeep,
                inputSize: 0,
                inputHeight: discriminatorArchitecture.InputHeight,
                inputWidth: discriminatorArchitecture.InputWidth,
                inputDepth: discriminatorArchitecture.InputDepth,
                outputSize: synthesisNetworkArchitecture.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.VeryDeep,
            inputSize: latentSize,
            outputSize: synthesisNetworkArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="StyleGAN{T}"/> class.
    /// </summary>
    /// <param name="mappingNetworkArchitecture">Architecture for the mapping network (Z → W).</param>
    /// <param name="synthesisNetworkArchitecture">Architecture for the synthesis network.</param>
    /// <param name="discriminatorArchitecture">Architecture for the discriminator.</param>
    /// <param name="latentSize">Size of input latent code Z.</param>
    /// <param name="intermediateLatentSize">Size of intermediate latent code W.</param>
    /// <param name="inputType">Input type.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Initial learning rate. Default is 0.001.</param>
    /// <param name="enableStyleMixing">Enable style mixing. Default is true.</param>
    /// <param name="styleMixingProbability">Probability of style mixing. Default is 0.9.</param>
    public StyleGAN(
        NeuralNetworkArchitecture<T> mappingNetworkArchitecture,
        NeuralNetworkArchitecture<T> synthesisNetworkArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize,
        int intermediateLatentSize,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.001,
        bool enableStyleMixing = true,
        double styleMixingProbability = 0.9,
        StyleGANOptions? options = null)
        : base(CreateStyleGANArchitecture(latentSize, synthesisNetworkArchitecture, discriminatorArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        _options = options ?? new StyleGANOptions();
        Options = _options;

        // Input validation
        if (mappingNetworkArchitecture is null)
        {
            throw new ArgumentNullException(nameof(mappingNetworkArchitecture), "Mapping network architecture cannot be null.");
        }

        if (synthesisNetworkArchitecture is null)
        {
            throw new ArgumentNullException(nameof(synthesisNetworkArchitecture), "Synthesis network architecture cannot be null.");
        }

        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }

        if (latentSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(latentSize), latentSize, "Latent size must be positive.");
        }

        if (intermediateLatentSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(intermediateLatentSize), intermediateLatentSize, "Intermediate latent size must be positive.");
        }

        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }

        if (styleMixingProbability < 0 || styleMixingProbability > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(styleMixingProbability), styleMixingProbability, "Style mixing probability must be in range [0, 1].");
        }

        _latentSize = latentSize;
        _intermediateLatentSize = intermediateLatentSize;
        _enableStyleMixing = enableStyleMixing;
        _styleMixingProbability = styleMixingProbability;
        _initialLearningRate = initialLearningRate;

        if (_options.R1Gamma < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), _options.R1Gamma, "R1Gamma must be non-negative.");
        }

        if (_options.MappingLearningRateMultiplier <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), _options.MappingLearningRateMultiplier,
                "MappingLearningRateMultiplier must be positive.");
        }

        _mappingNetworkArchitecture = mappingNetworkArchitecture;
        _synthesisNetworkArchitecture = synthesisNetworkArchitecture;
        _discriminatorArchitecture = discriminatorArchitecture;
        MappingNetwork = CreateSubNetworkForInputType(mappingNetworkArchitecture, inputType);
        SynthesisNetwork = CreateSubNetworkForInputType(synthesisNetworkArchitecture, inputType);
        Discriminator = CreateSubNetworkForInputType(discriminatorArchitecture, inputType);

        _mappingOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(MappingNetwork,
            CreatePaperAdamOptions(initialLearningRate * _options.MappingLearningRateMultiplier));
        _synthesisOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(SynthesisNetwork,
            CreatePaperAdamOptions(initialLearningRate));
        _discriminatorOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator,
            CreatePaperAdamOptions(initialLearningRate));

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for StyleGAN.
    /// </summary>
    /// <param name="realImages">Real images.</param>
    /// <param name="latentCodes">Random latent codes Z.</param>
    /// <returns>Tuple of (discriminator loss, generator loss).</returns>
    /// <remarks>
    /// <para>
    /// StyleGAN training follows the standard GAN training procedure but with
    /// the style-based generator. Style mixing is applied during training.
    /// </para>
    /// <para><b>For Beginners:</b> One round of StyleGAN training.
    ///
    /// Steps:
    /// 1. Map latent codes Z to style codes W
    /// 2. Optionally apply style mixing
    /// 3. Generate images using styles
    /// 4. Train discriminator on real/fake images
    /// 5. Train generator to fool discriminator
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> latentCodes)
    {
        MappingNetwork.SetTrainingMode(true);
        SynthesisNetwork.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        int batchSize = realImages.Shape[0];

        // ONE generator forward -- mapping, optional style mixing, synthesis -- recorded on the
        // generator's tape and shared by both steps (#1390). The previous step computed every loss as a
        // detached scalar and then ran three hand-written Adam updates on GetParameterGradients():
        // gradients no backward had produced, because the Backward calls were commented out when manual
        // backprop was removed. No network trained (#2155).
        using var generatorTape = new GradientTape<T>();
        var styles = MapToStyles(latentCodes);
        if (_enableStyleMixing && RandomHelper.ThreadSafeRandom.NextDouble() < _styleMixingProbability)
        {
            styles = MixStyles(styles, MapToStyles(GenerateRandomLatentCodes(batchSize)));
        }

        var fakeTracked = SynthesisNetwork.ForwardForTraining(ReshapeForCNN(styles, SynthesisNetwork.Architecture));
        var fakeImages = new Tensor<T>(fakeTracked.Shape.ToArray());
        fakeTracked.AsSpan().CopyTo(fakeImages.AsWritableSpan());

        // ----- Discriminator: non-saturating logistic loss plus R1 (Karras et al. 2019) -----
        T discriminatorLoss;
        using (var discriminatorTape = new GradientTape<T>())
        {
            var realScores = Discriminator.ForwardForTraining(realImages);
            var fakeScores = Discriminator.ForwardForTraining(fakeImages);
            var discriminatorObjective = Engine.TensorAdd(
                Discriminator.BinaryCrossEntropyOnTape(realScores, targetIsReal: true),
                Discriminator.BinaryCrossEntropyOnTape(fakeScores, targetIsReal: false));
            if (_options.R1Gamma > 0)
            {
                discriminatorObjective = Engine.TensorAdd(discriminatorObjective, R1Penalty(realImages));
            }

            discriminatorLoss = StepOnTape(discriminatorTape, discriminatorObjective, Discriminator,
                _discriminatorOptimizer);
        }

        // ----- Generator: the synthesis and mapping networks, trained by one loss at their own rates -----
        var generatorScores = Discriminator.ForwardFrozenOnTape(fakeTracked);
        var generatorObjective = Discriminator.BinaryCrossEntropyOnTape(generatorScores, targetIsReal: true);
        T generatorLoss = StepOnTape(generatorTape, generatorObjective, new[]
        {
            ((NeuralNetworkBase<T>)SynthesisNetwork, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_synthesisOptimizer),
            ((NeuralNetworkBase<T>)MappingNetwork, (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)_mappingOptimizer),
        });

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Maps latent codes Z to style codes W on the active tape, flattened to [batch, features].
    /// </summary>
    /// <remarks>
    /// Goes through the same <see cref="ReshapeForCNN"/> that <see cref="Generate"/> uses, so training
    /// and inference run one forward path.
    /// </remarks>
    private Tensor<T> MapToStyles(Tensor<T> latentCodes)
    {
        var styles = MappingNetwork.ForwardForTraining(ReshapeForCNN(latentCodes, MappingNetwork.Architecture));
        if (styles.Shape.Length > 2)
        {
            int batchSize = styles.Shape[0];
            styles = Engine.Reshape(styles, new[] { batchSize, styles.Length / batchSize });
        }

        return styles;
    }

    /// <summary>
    /// The R1 regulariser, (gamma / 2) * E[ ||grad_x D(x)||^2 ] over real images (Mescheder et al. 2018),
    /// which Karras et al. 2019 pair with the non-saturating loss (gamma = 10).
    /// </summary>
    /// <remarks>
    /// The inner tape takes the discriminator's gradient with respect to its input with
    /// <c>createGraph: true</c>, so the outer discriminator tape can differentiate the penalty back
    /// into the discriminator's weights -- the same construction WGAN-GP's gradient penalty uses.
    /// </remarks>
    private Tensor<T> R1Penalty(Tensor<T> realImages)
    {
        int batchSize = realImages.Shape[0];
        var realInput = new Tensor<T>(realImages.Shape.ToArray());
        realImages.AsSpan().CopyTo(realInput.AsWritableSpan());

        Tensor<T> inputGradients;
        using (var innerTape = new GradientTape<T>())
        {
            var scores = Discriminator.ForwardForTraining(realInput);
            var summed = Engine.ReduceSum(scores, Enumerable.Range(0, scores.Shape.Length).ToArray(), keepDims: false);
            var gradients = innerTape.ComputeGradients(summed, new[] { realInput }, createGraph: true);
            inputGradients = gradients.TryGetValue(realInput, out var gradient)
                ? gradient
                : new Tensor<T>(realInput.Shape.ToArray());
        }

        var flattened = Engine.Reshape(inputGradients, new[] { batchSize, inputGradients.Length / batchSize });
        var squaredNorm = Engine.ReduceSum(Engine.TensorMultiply(flattened, flattened), new[] { 1 }, keepDims: false);
        var meanSquaredNorm = Engine.ReduceMean(squaredNorm, new[] { 0 }, keepDims: false);
        return Engine.TensorMultiplyScalar(meanSquaredNorm, NumOps.FromDouble(_options.R1Gamma / 2.0));
    }

    /// <summary>
    /// Adam with beta1 = 0, beta2 = 0.99 and epsilon = 1e-8: the Progressive GAN settings (Karras et al. 2018,
    /// appendix A.1), which StyleGAN states it reuses (Karras et al. 2019, appendix C).
    /// </summary>
    private static AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CreatePaperAdamOptions(double learningRate)
        => new()
        {
            InitialLearningRate = learningRate,
            Beta1 = 0.0,
            Beta2 = 0.99,
            Epsilon = 1e-8,
        };

    /// <summary>
    /// Mixes two sets of styles at a random layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Style mixing combines coarse and fine features.
    ///
    /// Process:
    /// - Pick a random "mixing point"
    /// - Use Style A before the mixing point (coarse features)
    /// - Use Style B after the mixing point (fine features)
    /// - Result: Face shape from A, details from B
    /// </para>
    /// </remarks>
    private Tensor<T> MixStyles(Tensor<T> styles1, Tensor<T> styles2)
    {
        // Validate inputs
        if (styles1.Shape[0] != styles2.Shape[0])
        {
            throw new ArgumentException(
                $"Batch size mismatch: styles1 has {styles1.Shape[0]} samples, styles2 has {styles2.Shape[0]} samples.");
        }

        var random = RandomHelper.ThreadSafeRandom;
        int styleSize = styles1.Shape[1];

        // Handle edge cases where style size is too small for meaningful mixing
        // Need at least 3 elements to mix (1 for coarse, 1 for mixing point, 1 for fine)
        if (styleSize < 3)
        {
            // If too small to mix, just return styles1
            return styles1;
        }

        // Mix in middle layers: pick a layer between 1 and styleSize/2
        // random.Next(minInclusive, maxExclusive) requires maxExclusive > minInclusive
        int maxMixingLayer = Math.Max(2, styleSize / 2);
        int mixingLayer = random.Next(1, maxMixingLayer);

        // Blend with masks rather than copying elements: training passes tape-tracked styles through
        // here, and an element copy into a fresh tensor severed the gradient back to the mapping network.
        var coarseMask = new Tensor<T>(styles1.Shape.ToArray());
        var fineMask = new Tensor<T>(styles1.Shape.ToArray());
        for (int b = 0; b < styles1.Shape[0]; b++)
        {
            for (int i = 0; i < styleSize; i++)
            {
                bool coarse = i < mixingLayer;
                coarseMask[b, i] = coarse ? NumOps.One : NumOps.Zero;
                fineMask[b, i] = coarse ? NumOps.Zero : NumOps.One;
            }
        }

        var mixedStyles = Engine.TensorAdd(
            Engine.TensorMultiply(styles1, coarseMask),
            Engine.TensorMultiply(styles2, fineMask));
        return mixedStyles;
    }

    /// <summary>
    /// Generates images from latent codes.
    /// </summary>
    /// <param name="latentCodes">Latent codes Z. Can be any rank - will be reshaped/padded as needed.</param>
    /// <returns>Generated images.</returns>
    public Tensor<T> Generate(Tensor<T> latentCodes)
    {
        MappingNetwork.SetTrainingMode(false);
        SynthesisNetwork.SetTrainingMode(false);

        // Reshape latent codes for CNN mapping network
        // CNN expects 4D input [N, C, H, W]
        var reshapedLatent = ReshapeForCNN(latentCodes, MappingNetwork.Architecture);

        var styles = MappingNetwork.Predict(reshapedLatent);

        // Reshape styles for CNN synthesis network if needed
        var reshapedStyles = ReshapeForCNN(styles, SynthesisNetwork.Architecture);

        return SynthesisNetwork.Predict(reshapedStyles);
    }

    /// <summary>
    /// Reshapes any-rank tensor to match CNN architecture input requirements.
    /// Handles padding/tiling if input has fewer elements than required.
    /// </summary>
    private Tensor<T> ReshapeForCNN(Tensor<T> input, NeuralNetworkArchitecture<T> architecture)
    {
        // Already 4D - return as-is
        if (input.Shape.Length == 4)
        {
            return input;
        }

        // Already 3D [C, H, W] - add batch dimension
        if (input.Shape.Length == 3)
        {
            return Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        }

        // Calculate expected input size from architecture
        int expectedDepth = architecture.InputDepth;
        int expectedHeight = architecture.InputHeight;
        int expectedWidth = architecture.InputWidth;
        int expectedElements = expectedDepth * expectedHeight * expectedWidth;
        if (expectedElements <= 0)
        {
            return input;
        }

        // Get batch size and input elements
        int batchSize = input.Shape.Length == 1 ? 1 : input.Shape[0];
        int inputElements = input.Shape.Length == 1 ? input.Shape[0] : input.Shape[1];
        var rows = input.Shape.Length == 1 ? Engine.Reshape(input, new[] { 1, inputElements }) : input;

        // Every step is an Engine op because training passes tape-tracked tensors through here; the
        // element-by-element copy this replaces severed the gradient between the two networks. Tiling
        // then trimming along the feature axis reproduces the old cycle through the input elements.
        if (inputElements != expectedElements)
        {
            int repeats = (expectedElements + inputElements - 1) / inputElements;
            var tiled = repeats > 1 ? Engine.TensorTile(rows, new[] { 1, repeats }) : rows;
            rows = Engine.TensorSlice(tiled, new[] { 0, 0 }, new[] { batchSize, expectedElements });
        }

        return Engine.Reshape(rows, new[] { batchSize, expectedDepth, expectedHeight, expectedWidth });
    }

    /// <summary>
    /// Generates images with style mixing.
    /// </summary>
    /// <param name="latentCodes1">First set of latent codes (for coarse features). Can be any rank.</param>
    /// <param name="latentCodes2">Second set of latent codes (for fine features). Can be any rank.</param>
    /// <returns>Generated images with mixed styles.</returns>
    public Tensor<T> GenerateWithStyleMixing(Tensor<T> latentCodes1, Tensor<T> latentCodes2)
    {
        MappingNetwork.SetTrainingMode(false);
        SynthesisNetwork.SetTrainingMode(false);

        // Reshape latent codes for CNN mapping network
        var reshapedLatent1 = ReshapeForCNN(latentCodes1, MappingNetwork.Architecture);
        var reshapedLatent2 = ReshapeForCNN(latentCodes2, MappingNetwork.Architecture);

        var styles1 = MappingNetwork.Predict(reshapedLatent1);
        var styles2 = MappingNetwork.Predict(reshapedLatent2);

        // Flatten styles for mixing if they were 4D
        if (styles1.Shape.Length > 2)
        {
            int batchSize = styles1.Shape[0];
            int totalSize = styles1.Length / batchSize;
            styles1 = styles1.Reshape([batchSize, totalSize]);
            styles2 = styles2.Reshape([batchSize, totalSize]);
        }

        var mixedStyles = MixStyles(styles1, styles2);

        // Reshape mixed styles for CNN synthesis network
        var reshapedMixedStyles = ReshapeForCNN(mixedStyles, SynthesisNetwork.Architecture);

        return SynthesisNetwork.Predict(reshapedMixedStyles);
    }

    /// <summary>
    /// Generates random latent codes using vectorized Gaussian noise generation.
    /// Uses Engine.GenerateGaussianNoise for SIMD/GPU acceleration.
    /// </summary>
    public Tensor<T> GenerateRandomLatentCodes(int batchSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"Batch size must be positive, got {batchSize}.");
        }

        var totalElements = batchSize * _latentSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;

        // Use Engine's vectorized Gaussian noise generation
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        // Reshape to [batchSize, latentSize]
        return Tensor<T>.FromVector(noiseVector, [batchSize, _latentSize]);
    }

    protected override void InitializeLayers() { }

    /// <inheritdoc/>
    /// <remarks>
    /// The model's own Layers list is empty: its layers live in the mapping, synthesis and discriminator
    /// networks. Each is read on what feeds it in <see cref="Generate"/>, and the discriminator on the image.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var result = new Dictionary<string, Tensor<T>>();
        MappingNetwork.SetTrainingMode(false);
        SynthesisNetwork.SetTrainingMode(false);

        var latent = ReshapeForCNN(input, MappingNetwork.Architecture);
        foreach (var kv in MappingNetwork.GetNamedLayerActivations(latent))
            result["Mapping/" + kv.Key] = kv.Value;

        var styles = ReshapeForCNN(MappingNetwork.Predict(latent), SynthesisNetwork.Architecture);
        foreach (var kv in SynthesisNetwork.GetNamedLayerActivations(styles))
            result["Synthesis/" + kv.Key] = kv.Value;

        var image = SynthesisNetwork.Predict(styles);
        foreach (var kv in Discriminator.GetNamedLayerActivations(ReshapeForCNN(image, Discriminator.Architecture)))
            result["Discriminator/" + kv.Key] = kv.Value;
        return result;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return Generate(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(
            WithBatchAxis(expectedOutput, Discriminator.Architecture.InputType),
            WithBatchAxis(input, MappingNetwork.Architecture.InputType));
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MappingNetworkParameters", MappingNetwork.GetParameterCount() },
                { "SynthesisNetworkParameters", SynthesisNetwork.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "LatentSize", _latentSize },
                { "IntermediateLatentSize", _intermediateLatentSize },
                { "StyleMixingEnabled", _enableStyleMixing }
            },
            ModelData = SerializeForMetadata()
        };
    }





    // UpdateParameters split the vector between MappingNetwork, SynthesisNetwork and Discriminator;
    // GetExtraTrainableLayers yields those three in the same order, so the base reproduces the
    // split. Removed under AIDN082.
}
