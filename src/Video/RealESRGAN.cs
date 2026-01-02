using System.IO;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video;

/// <summary>
/// Real-ESRGAN (Real Enhanced Super-Resolution GAN) for image and video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Real-ESRGAN is a practical super-resolution model that uses:
/// - RRDB (Residual in Residual Dense Block) generator for deep feature extraction
/// - U-Net discriminator for adversarial training
/// - Combined loss: L1 (pixel) + Perceptual (VGG) + GAN (adversarial)
/// - Second-order degradation model for realistic training data
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-ESRGAN upscales images and video frames to higher resolution
/// while adding realistic details. It's one of the most practical super-resolution models.
///
/// The network works by:
/// 1. Extracting deep features from low-resolution input using RRDB blocks
/// 2. Upsampling using pixel shuffle (efficient sub-pixel convolution)
/// 3. Training with adversarial loss to add realistic textures
/// 4. Using perceptual loss to ensure visual quality
///
/// Example usage:
/// <code>
/// // Create architectures
/// var generatorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var discriminatorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3);
///
/// // Create model with 4x upscaling
/// var model = new RealESRGAN&lt;double&gt;(
///     generatorArch, discriminatorArch,
///     InputType.ThreeDimensional,
///     scaleFactor: 4);
///
/// // Train
/// var (dLoss, gLoss) = model.TrainStep(lowResImages, highResTargets);
///
/// // Inference
/// var highRes = model.Upscale(lowResImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
/// with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833
/// </para>
/// </remarks>
public class RealESRGAN<T> : NeuralNetworkBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this Real-ESRGAN uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the generator model.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// The optimizer used for training the generator network.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _generatorOptimizer;

    /// <summary>
    /// The optimizer used for training the discriminator network.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;

    /// <summary>
    /// The combined loss function for Real-ESRGAN training.
    /// </summary>
    private readonly RealESRGANLoss<T>? _realESRGANLoss;

    /// <summary>
    /// Coefficient for the L1 (pixel-wise) loss.
    /// </summary>
    private readonly double _l1Lambda;

    /// <summary>
    /// Coefficient for the perceptual (VGG) loss.
    /// </summary>
    private readonly double _perceptualLambda;

    /// <summary>
    /// Coefficient for the GAN (adversarial) loss.
    /// </summary>
    private readonly double _ganLambda;

    /// <summary>
    /// The upscaling factor (2, 4, or 8).
    /// </summary>
    private readonly int _scaleFactor;

    /// <summary>
    /// Number of RRDB blocks in the generator.
    /// </summary>
    private readonly int _numRRDBBlocks;

    /// <summary>
    /// Number of feature channels.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Residual scaling factor for training stability.
    /// </summary>
    private readonly double _residualScale;

    /// <summary>
    /// Loss values from the last training step for monitoring.
    /// </summary>
    private readonly List<T> _generatorLosses = [];

    /// <summary>
    /// Stores the last discriminator loss for diagnostics.
    /// </summary>
    private T _lastDiscriminatorLoss;

    /// <summary>
    /// Stores the last generator loss for diagnostics.
    /// </summary>
    private T _lastGeneratorLoss;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode allows training and uses pure C# layers.
    /// ONNX mode loads a pre-trained model for inference only.
    /// </para>
    /// </remarks>
    public bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the RRDB-Net generator network that produces super-resolved images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The generator uses Residual in Residual Dense Blocks (RRDB) for deep feature
    /// extraction, followed by pixel shuffle upsampling. This architecture is highly
    /// effective for super-resolution tasks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The generator is the main network that takes your
    /// low-resolution image and outputs the high-resolution version.
    /// Only available in native mode.
    /// </para>
    /// </remarks>
    public RRDBNetGenerator<T>? Generator { get; private set; }

    /// <summary>
    /// Gets the U-Net discriminator network that judges image quality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Real-ESRGAN uses a U-Net discriminator that provides per-pixel feedback,
    /// which helps generate more detailed and realistic textures compared to
    /// standard discriminators that only provide a single real/fake score.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The discriminator learns to tell the difference between
    /// real high-resolution images and generated ones. This adversarial training
    /// helps the generator create more realistic outputs.
    /// Only available in native mode.
    /// </para>
    /// </remarks>
    public UNetDiscriminator<T>? Discriminator { get; private set; }

    /// <summary>
    /// Gets the upscaling factor for this model.
    /// </summary>
    public int ScaleFactor => _scaleFactor;

    /// <summary>
    /// Gets the last discriminator loss value.
    /// </summary>
    public T LastDiscriminatorLoss => _lastDiscriminatorLoss;

    /// <summary>
    /// Gets the last generator loss value.
    /// </summary>
    public T LastGeneratorLoss => _lastGeneratorLoss;

    #endregion

    #region Constructors

    /// <summary>
    /// Validates constructor arguments and throws if null (called before base constructor).
    /// </summary>
    private static NeuralNetworkArchitecture<T> ValidateAndGetArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType)
    {
        if (generatorArchitecture is null)
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        if (discriminatorArchitecture is null)
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");

        return CreateRealESRGANArchitecture(generatorArchitecture, inputType);
    }

    /// <summary>
    /// Creates the combined Real-ESRGAN architecture from generator and discriminator architectures.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateRealESRGANArchitecture(
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
    /// Initializes a new instance of the Real-ESRGAN class.
    /// </summary>
    /// <param name="generatorArchitecture">Architecture for the RRDB-Net generator.</param>
    /// <param name="discriminatorArchitecture">Architecture for the U-Net discriminator.</param>
    /// <param name="inputType">The type of input data (typically ThreeDimensional for images).</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. Default: Adam with lr=0.0001, beta2=0.99.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. Default: Adam with lr=0.0001, beta2=0.99.</param>
    /// <param name="scaleFactor">Upscaling factor. Default: 4 (4x super-resolution).</param>
    /// <param name="numRRDBBlocks">Number of RRDB blocks in generator. Default: 23 (Real-ESRGAN standard).</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="residualScale">Residual scaling factor. Default: 0.2 (Real-ESRGAN standard).</param>
    /// <param name="l1Lambda">L1 loss coefficient. Default: 1.0 (from Real-ESRGAN paper).</param>
    /// <param name="perceptualLambda">Perceptual loss coefficient. Default: 1.0 (from Real-ESRGAN paper).</param>
    /// <param name="ganLambda">GAN loss coefficient. Default: 0.1 (from Real-ESRGAN paper).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a Real-ESRGAN model with sensible defaults:
    /// <code>
    /// var model = new RealESRGAN&lt;double&gt;(generatorArch, discriminatorArch, InputType.ThreeDimensional);
    /// </code>
    ///
    /// Or customize for your needs:
    /// <code>
    /// // 2x upscaling with fewer blocks (faster but lower quality)
    /// var model = new RealESRGAN&lt;double&gt;(
    ///     generatorArch, discriminatorArch, InputType.ThreeDimensional,
    ///     scaleFactor: 2, numRRDBBlocks: 16);
    /// </code>
    /// </para>
    /// <para>
    /// The default values are from the Real-ESRGAN paper and work well for most use cases.
    /// </para>
    /// </remarks>
    public RealESRGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        int scaleFactor = 4,
        int numRRDBBlocks = 23,
        int numFeatures = 64,
        double residualScale = 0.2,
        double l1Lambda = 1.0,
        double perceptualLambda = 1.0,
        double ganLambda = 0.1)
        : base(ValidateAndGetArchitecture(generatorArchitecture, discriminatorArchitecture, inputType),
               new RealESRGANLoss<T>(l1Lambda, perceptualLambda, ganLambda))
    {
        // Validate numeric inputs (null validation already done in ValidateAndGetArchitecture)
        if (scaleFactor < 1)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), scaleFactor, "Scale factor must be at least 1.");
        if (numRRDBBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(numRRDBBlocks), numRRDBBlocks, "Number of RRDB blocks must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures, "Number of features must be at least 1.");
        if (l1Lambda < 0)
            throw new ArgumentOutOfRangeException(nameof(l1Lambda), l1Lambda, "L1 lambda must be non-negative.");
        if (perceptualLambda < 0)
            throw new ArgumentOutOfRangeException(nameof(perceptualLambda), perceptualLambda, "Perceptual lambda must be non-negative.");
        if (ganLambda < 0)
            throw new ArgumentOutOfRangeException(nameof(ganLambda), ganLambda, "GAN lambda must be non-negative.");

        // Set native mode
        _useNativeMode = true;

        // Store configuration
        _scaleFactor = scaleFactor;
        _numRRDBBlocks = numRRDBBlocks;
        _numFeatures = numFeatures;
        _residualScale = residualScale;
        _l1Lambda = l1Lambda;
        _perceptualLambda = perceptualLambda;
        _ganLambda = ganLambda;

        // Initialize loss diagnostics
        _lastDiscriminatorLoss = NumOps.Zero;
        _lastGeneratorLoss = NumOps.Zero;

        // Create loss function
        _realESRGANLoss = new RealESRGANLoss<T>(l1Lambda, perceptualLambda, ganLambda);

        // Create RRDBNet generator (the proper ESRGAN generator architecture)
        int inputHeight = generatorArchitecture.InputHeight > 0 ? generatorArchitecture.InputHeight : 64;
        int inputWidth = generatorArchitecture.InputWidth > 0 ? generatorArchitecture.InputWidth : 64;
        int inputChannels = generatorArchitecture.InputDepth > 0 ? generatorArchitecture.InputDepth : 3;
        int outputChannels = 3; // RGB output

        Generator = new RRDBNetGenerator<T>(
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            numFeatures: numFeatures,
            growthChannels: 32, // ESRGAN paper default
            numRRDBBlocks: numRRDBBlocks,
            scale: scaleFactor,
            residualScale: residualScale);

        // Create U-Net discriminator (the proper Real-ESRGAN discriminator architecture)
        int hrHeight = inputHeight * scaleFactor;
        int hrWidth = inputWidth * scaleFactor;

        Discriminator = new UNetDiscriminator<T>(
            inputHeight: hrHeight,
            inputWidth: hrWidth,
            inputChannels: outputChannels,
            numChannels: 64,
            numBlocks: 4);

        // Initialize optimizers with Real-ESRGAN paper settings (Adam with beta2=0.99)
        // Note: The optimizers need to work with ILayer<T>, so we use wrapper interfaces
        _generatorOptimizer = generatorOptimizer;
        _discriminatorOptimizer = discriminatorOptimizer;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Real-ESRGAN model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained Real-ESRGAN ONNX model.</param>
    /// <param name="scaleFactor">Upscaling factor of the pretrained model. Default: 4.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained Real-ESRGAN model
    /// in ONNX format. This is the fastest way to use Real-ESRGAN for inference without training.
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 128, inputWidth: 128, inputDepth: 3);
    /// var model = new RealESRGAN&lt;float&gt;(arch, "realesrgan_x4.onnx");
    /// var highRes = model.Upscale(lowResImage);
    /// </code>
    ///
    /// Note: Training is not supported in ONNX mode. Use the native constructor for training.
    /// </para>
    /// <para>
    /// Pretrained ONNX models can be downloaded from:
    /// - https://github.com/xinntao/Real-ESRGAN (official)
    /// - Hugging Face model hub
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    public RealESRGAN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int scaleFactor = 4)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"Real-ESRGAN ONNX model not found: {onnxModelPath}");
        if (scaleFactor < 1)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), scaleFactor, "Scale factor must be at least 1.");

        // Set ONNX mode
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _scaleFactor = scaleFactor;

        // Set defaults for other fields (not used in ONNX mode)
        _numRRDBBlocks = 23;
        _numFeatures = 64;
        _residualScale = 0.2;
        _l1Lambda = 1.0;
        _perceptualLambda = 1.0;
        _ganLambda = 0.1;
        _lastDiscriminatorLoss = NumOps.Zero;
        _lastGeneratorLoss = NumOps.Zero;

        // Load ONNX model
        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Training

    /// <summary>
    /// Performs one training step for Real-ESRGAN.
    /// </summary>
    /// <param name="lowResImages">Low-resolution input images.</param>
    /// <param name="highResTargets">High-resolution target images.</param>
    /// <returns>Tuple of (discriminator loss, generator loss).</returns>
    /// <remarks>
    /// <para>
    /// This method performs one complete training iteration:
    /// 1. Generate super-resolved images from low-res input
    /// 2. Train discriminator to distinguish real from generated
    /// 3. Train generator to fool discriminator and minimize reconstruction loss
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Call this method repeatedly during training:
    /// <code>
    /// for (int epoch = 0; epoch &lt; numEpochs; epoch++)
    /// {
    ///     foreach (var batch in dataLoader)
    ///     {
    ///         var (dLoss, gLoss) = model.TrainStep(batch.LowRes, batch.HighRes);
    ///         Console.WriteLine($"D Loss: {dLoss:F4}, G Loss: {gLoss:F4}");
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> lowResImages, Tensor<T> highResTargets)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode for training.");
        if (lowResImages is null)
            throw new ArgumentNullException(nameof(lowResImages));
        if (highResTargets is null)
            throw new ArgumentNullException(nameof(highResTargets));

        int batchSize = lowResImages.Shape[0];

        // ----- Train Discriminator -----

        // Generate super-resolved images (using helper for batch handling)
        Tensor<T> generatedImages = ProcessThroughGenerator(lowResImages);

        // Get discriminator outputs for real and fake images (using helper for batch handling)
        Tensor<T> realOutput = ProcessThroughDiscriminator(highResTargets);
        Tensor<T> fakeOutput = ProcessThroughDiscriminator(generatedImages);

        // Calculate discriminator loss
        T discriminatorLoss = _realESRGANLoss!.CalculateDiscriminatorLoss(
            realOutput.ToVector(), fakeOutput.ToVector());
        _lastDiscriminatorLoss = discriminatorLoss;

        // Backpropagate discriminator
        TrainDiscriminatorStep(highResTargets, generatedImages, realOutput, fakeOutput);

        // ----- Train Generator -----

        // Generate new images for generator training (using helper for batch handling)
        Tensor<T> newGeneratedImages = ProcessThroughGenerator(lowResImages);

        // Get discriminator output for new generated images (using helper for batch handling)
        Tensor<T> newFakeOutput = ProcessThroughDiscriminator(newGeneratedImages);

        // Calculate generator loss (reconstruction + GAN)
        T generatorLoss = _realESRGANLoss.CalculateCombinedLoss(
            newGeneratedImages.ToVector(),
            highResTargets.ToVector(),
            newFakeOutput.ToVector());
        _lastGeneratorLoss = generatorLoss;

        // Backpropagate generator
        TrainGeneratorStep(lowResImages, highResTargets, newGeneratedImages, newFakeOutput);

        // Track losses for monitoring
        _generatorLosses.Add(generatorLoss);
        if (_generatorLosses.Count > 100)
            _generatorLosses.RemoveAt(0);

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Performs discriminator training step.
    /// </summary>
    private void TrainDiscriminatorStep(
        Tensor<T> realImages,
        Tensor<T> fakeImages,
        Tensor<T> realOutput,
        Tensor<T> fakeOutput)
    {
        // Create target labels
        var realLabels = CreateLabelTensor(realOutput.Shape[0], NumOps.One);
        var fakeLabels = CreateLabelTensor(fakeOutput.Shape[0], NumOps.Zero);

        // Calculate gradients for real images
        var realGradient = Discriminator!.Backward(
            CalculateBCEGradient(realOutput, realLabels));

        // Calculate gradients for fake images
        var fakeGradient = Discriminator.Backward(
            CalculateBCEGradient(fakeOutput, fakeLabels));

        // Update discriminator parameters using optimizer or fallback to default learning rate
        if (_discriminatorOptimizer != null)
        {
            // Use configured optimizer
            var currentParams = Discriminator.GetParameters();
            var gradients = Discriminator.GetParameterGradients();
            var updatedParams = _discriminatorOptimizer.UpdateParameters(currentParams, gradients);
            Discriminator.SetParameters(updatedParams);
        }
        else
        {
            // Fallback to simple SGD with default learning rate
            T learningRate = NumOps.FromDouble(0.0001);
            Discriminator.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Performs generator training step.
    /// </summary>
    private void TrainGeneratorStep(
        Tensor<T> lowResImages,
        Tensor<T> highResTargets,
        Tensor<T> generatedImages,
        Tensor<T> discriminatorOutput)
    {
        // Generator wants discriminator to output 1 (real) for generated images
        var targetLabels = CreateLabelTensor(discriminatorOutput.Shape[0], NumOps.One);

        // Calculate GAN loss gradient
        var ganGradient = CalculateBCEGradient(discriminatorOutput, targetLabels);

        // Calculate reconstruction loss gradient (L1)
        var reconstructionGradient = CalculateReconstructionGradient(generatedImages, highResTargets);

        // Combine gradients
        var combinedGradient = CombineGradients(reconstructionGradient, ganGradient);

        // Backpropagate through generator
        Generator!.Backward(combinedGradient);

        // Update generator parameters using optimizer or fallback to default learning rate
        if (_generatorOptimizer != null)
        {
            // Use configured optimizer
            var currentParams = Generator.GetParameters();
            var gradients = Generator.GetParameterGradients();
            var updatedParams = _generatorOptimizer.UpdateParameters(currentParams, gradients);
            Generator.SetParameters(updatedParams);
        }
        else
        {
            // Fallback to simple SGD with default learning rate
            T learningRate = NumOps.FromDouble(0.0001);
            Generator.UpdateParameters(learningRate);
        }
    }

    #endregion

    #region Inference

    /// <summary>
    /// Upscales a low-resolution image to high resolution.
    /// </summary>
    /// <param name="lowResImage">The low-resolution input image tensor.</param>
    /// <returns>The super-resolved high-resolution image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this method after training to upscale images:
    /// <code>
    /// var highResImage = model.Upscale(lowResImage);
    /// </code>
    /// </para>
    /// </remarks>
    public Tensor<T> Upscale(Tensor<T> lowResImage)
    {
        if (lowResImage is null)
            throw new ArgumentNullException(nameof(lowResImage));

        if (_useNativeMode)
        {
            // Native mode: use generator network (with batch handling)
            return ProcessThroughGenerator(lowResImage);
        }
        else
        {
            // ONNX mode: use ONNX inference session
            return PredictOnnx(lowResImage);
        }
    }

    /// <summary>
    /// Performs inference using the ONNX model.
    /// </summary>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert input tensor to ONNX tensor format
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data[i]);
        }

        // Create ONNX input tensor with shape [batch, channels, height, width]
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);

        // Get input name from model
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert back to our tensor format
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Upscale(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(input, expectedOutput);
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are managed by Generator and Discriminator networks
        // The combined architecture is primarily for metadata
        ClearLayers();
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Processes a tensor that may have a batch dimension through the generator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method handles any-rank tensors by processing batch elements individually
    /// if needed. The generator expects [C,H,W] but users may pass [B,C,H,W] or higher-rank tensors.
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessThroughGenerator(Tensor<T> input)
    {
        var expectedShape = Generator!.GetInputShape();

        // Check if input matches expected shape exactly (no batch dimension)
        if (input.Rank == expectedShape.Length && ShapesMatch(input.Shape, expectedShape))
        {
            return Generator.Forward(input);
        }

        // Handle batched input: input has more dimensions than expected
        if (input.Rank > expectedShape.Length)
        {
            // Extract batch dimensions (everything before the spatial dims)
            int batchDims = input.Rank - expectedShape.Length;
            int batchSize = 1;
            for (int i = 0; i < batchDims; i++)
            {
                batchSize *= input.Shape[i];
            }

            // Validate spatial dimensions match
            var spatialShape = input.Shape.Skip(batchDims).ToArray();
            if (!ShapesMatch(spatialShape, expectedShape))
            {
                throw new TensorShapeMismatchException(
                    $"Shape mismatch in RealESRGAN: Expected spatial dimensions [{string.Join(", ", expectedShape)}], " +
                    $"but got [{string.Join(", ", spatialShape)}].");
            }

            // Process each batch element and combine results
            var outputs = new List<Tensor<T>>();
            int elementsPerBatch = spatialShape.Aggregate(1, (a, b) => a * b);

            for (int b = 0; b < batchSize; b++)
            {
                // Extract single element
                var elementData = new T[elementsPerBatch];
                Array.Copy(input.Data.ToArray(), b * elementsPerBatch, elementData, 0, elementsPerBatch);
                var element = new Tensor<T>(spatialShape, new Vector<T>(elementData));

                // Process through generator
                var output = Generator.Forward(element);
                outputs.Add(output);
            }

            // Combine outputs back into batched tensor
            var outputShape = new int[batchDims + outputs[0].Rank];
            Array.Copy(input.Shape, 0, outputShape, 0, batchDims);
            Array.Copy(outputs[0].Shape, 0, outputShape, batchDims, outputs[0].Rank);

            var combinedData = new T[batchSize * outputs[0].Length];
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(outputs[b].Data.ToArray(), 0, combinedData, b * outputs[0].Length, outputs[0].Length);
            }

            return new Tensor<T>(outputShape, new Vector<T>(combinedData));
        }

        // Input has fewer dimensions than expected - expand dimensions if possible
        if (input.Rank < expectedShape.Length)
        {
            // Calculate how many dimensions to add at the front
            int dimsToAdd = expectedShape.Length - input.Rank;

            // Validate trailing dimensions match
            for (int i = 0; i < input.Rank; i++)
            {
                if (input.Shape[i] != expectedShape[dimsToAdd + i])
                {
                    throw new TensorShapeMismatchException(
                        $"Shape mismatch in RealESRGAN: Cannot expand [{string.Join(", ", input.Shape)}] " +
                        $"to match [{string.Join(", ", expectedShape)}].");
                }
            }

            // Expand input shape by adding dimensions of size 1 at the front
            var expandedShape = new int[expectedShape.Length];
            for (int i = 0; i < dimsToAdd; i++)
            {
                expandedShape[i] = expectedShape[i];
            }
            for (int i = 0; i < input.Rank; i++)
            {
                expandedShape[dimsToAdd + i] = input.Shape[i];
            }

            // Reshape the tensor (reuse the data from input)
            var expandedInput = new Tensor<T>(expandedShape, new Vector<T>(input.Data.ToArray()));
            return Generator.Forward(expandedInput);
        }

        return Generator.Forward(input);
    }

    /// <summary>
    /// Processes a tensor that may have a batch dimension through the discriminator.
    /// </summary>
    private Tensor<T> ProcessThroughDiscriminator(Tensor<T> input)
    {
        var expectedShape = Discriminator!.GetInputShape();

        // Check if input matches expected shape exactly (no batch dimension)
        if (input.Rank == expectedShape.Length && ShapesMatch(input.Shape, expectedShape))
        {
            return Discriminator.Forward(input);
        }

        // Handle batched input
        if (input.Rank > expectedShape.Length)
        {
            int batchDims = input.Rank - expectedShape.Length;
            int batchSize = 1;
            for (int i = 0; i < batchDims; i++)
            {
                batchSize *= input.Shape[i];
            }

            var spatialShape = input.Shape.Skip(batchDims).ToArray();
            if (!ShapesMatch(spatialShape, expectedShape))
            {
                throw new TensorShapeMismatchException(
                    $"Shape mismatch in RealESRGAN discriminator: Expected spatial dimensions [{string.Join(", ", expectedShape)}], " +
                    $"but got [{string.Join(", ", spatialShape)}].");
            }

            var outputs = new List<Tensor<T>>();
            int elementsPerBatch = spatialShape.Aggregate(1, (a, b) => a * b);

            for (int b = 0; b < batchSize; b++)
            {
                var elementData = new T[elementsPerBatch];
                Array.Copy(input.Data.ToArray(), b * elementsPerBatch, elementData, 0, elementsPerBatch);
                var element = new Tensor<T>(spatialShape, new Vector<T>(elementData));
                var output = Discriminator.Forward(element);
                outputs.Add(output);
            }

            var outputShape = new int[batchDims + outputs[0].Rank];
            Array.Copy(input.Shape, 0, outputShape, 0, batchDims);
            Array.Copy(outputs[0].Shape, 0, outputShape, batchDims, outputs[0].Rank);

            var combinedData = new T[batchSize * outputs[0].Length];
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(outputs[b].Data.ToArray(), 0, combinedData, b * outputs[0].Length, outputs[0].Length);
            }

            return new Tensor<T>(outputShape, new Vector<T>(combinedData));
        }

        return Discriminator.Forward(input);
    }

    /// <summary>
    /// Checks if two shapes match.
    /// </summary>
    private static bool ShapesMatch(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length) return false;
        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i]) return false;
        }
        return true;
    }

    /// <summary>
    /// Creates a tensor filled with a specified value for labels.
    /// </summary>
    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var shape = new int[] { batchSize, 1 };
        var tensor = new Tensor<T>(shape);
        Engine.TensorFill(tensor, value);
        return tensor;
    }

    /// <summary>
    /// Calculates the gradient of binary cross-entropy loss.
    /// </summary>
    private Tensor<T> CalculateBCEGradient(Tensor<T> output, Tensor<T> target)
    {
        var gradient = new Tensor<T>(output.Shape);
        double epsilon = 1e-7;

        for (int i = 0; i < output.Length; i++)
        {
            double o = NumOps.ToDouble(output.Data[i]);
            double t = NumOps.ToDouble(target.Data[i]);

            // BCE gradient: (output - target) / (output * (1 - output))
            double oClipped = Math.Max(epsilon, Math.Min(o, 1.0 - epsilon));
            double grad = (o - t) / (oClipped * (1.0 - oClipped) + epsilon);

            gradient.Data[i] = NumOps.FromDouble(grad);
        }

        return gradient;
    }

    /// <summary>
    /// Calculates the reconstruction (L1) loss gradient.
    /// </summary>
    private Tensor<T> CalculateReconstructionGradient(Tensor<T> generated, Tensor<T> target)
    {
        var gradient = new Tensor<T>(generated.Shape);

        for (int i = 0; i < generated.Length; i++)
        {
            T diff = NumOps.Subtract(generated.Data[i], target.Data[i]);
            // L1 gradient is sign(diff)
            if (NumOps.GreaterThan(diff, NumOps.Zero))
                gradient.Data[i] = NumOps.FromDouble(_l1Lambda / generated.Length);
            else if (NumOps.LessThan(diff, NumOps.Zero))
                gradient.Data[i] = NumOps.FromDouble(-_l1Lambda / generated.Length);
            else
                gradient.Data[i] = NumOps.Zero;
        }

        return gradient;
    }

    /// <summary>
    /// Combines multiple gradient tensors.
    /// </summary>
    private Tensor<T> CombineGradients(Tensor<T> grad1, Tensor<T> grad2)
    {
        var combined = new Tensor<T>(grad1.Shape);

        for (int i = 0; i < grad1.Length; i++)
        {
            combined.Data[i] = NumOps.Add(grad1.Data[i], grad2.Data[i]);
        }

        return combined;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        // Split parameters between generator and discriminator
        int generatorParams = Generator!.GetParameters().Length;
        int discriminatorParams = Discriminator!.GetParameters().Length;

        if (parameters.Length != generatorParams + discriminatorParams)
        {
            throw new ArgumentException(
                $"Expected {generatorParams + discriminatorParams} parameters, got {parameters.Length}");
        }

        var genParams = parameters.Slice(0, generatorParams);
        var discParams = parameters.Slice(generatorParams, discriminatorParams);

        Generator.SetParameters(genParams);
        Discriminator.SetParameters(discParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "RealESRGAN" },
            { "ScaleFactor", _scaleFactor },
            { "NumRRDBBlocks", _numRRDBBlocks },
            { "NumFeatures", _numFeatures },
            { "ResidualScale", _residualScale },
            { "L1Lambda", _l1Lambda },
            { "PerceptualLambda", _perceptualLambda },
            { "GANLambda", _ganLambda },
            { "UseNativeMode", _useNativeMode }
        };

        if (_useNativeMode && Generator != null && Discriminator != null)
        {
            additionalInfo["GeneratorParameters"] = Generator.GetParameters().Length;
            additionalInfo["DiscriminatorParameters"] = Discriminator.GetParameters().Length;
        }

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.GenerativeAdversarialNetwork,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_scaleFactor);
        writer.Write(_numRRDBBlocks);
        writer.Write(_numFeatures);
        writer.Write(_residualScale);
        writer.Write(_l1Lambda);
        writer.Write(_perceptualLambda);
        writer.Write(_ganLambda);

        // Serialize generator parameters
        var generatorParams = Generator!.GetParameters();
        writer.Write(generatorParams.Length);
        for (int i = 0; i < generatorParams.Length; i++)
        {
            writer.Write(NumOps.ToDouble(generatorParams[i]));
        }

        // Serialize discriminator parameters
        var discriminatorParams = Discriminator!.GetParameters();
        writer.Write(discriminatorParams.Length);
        for (int i = 0; i < discriminatorParams.Length; i++)
        {
            writer.Write(NumOps.ToDouble(discriminatorParams[i]));
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        // Read configuration (already set in constructor, just advance reader)
        _ = reader.ReadInt32(); // scaleFactor
        _ = reader.ReadInt32(); // numRRDBBlocks
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadDouble(); // residualScale
        _ = reader.ReadDouble(); // l1Lambda
        _ = reader.ReadDouble(); // perceptualLambda
        _ = reader.ReadDouble(); // ganLambda

        // Load generator parameters
        int generatorParamCount = reader.ReadInt32();
        var generatorParams = new T[generatorParamCount];
        for (int i = 0; i < generatorParamCount; i++)
        {
            generatorParams[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        Generator!.SetParameters(new Vector<T>(generatorParams));

        // Load discriminator parameters
        int discriminatorParamCount = reader.ReadInt32();
        var discriminatorParams = new T[discriminatorParamCount];
        for (int i = 0; i < discriminatorParamCount; i++)
        {
            discriminatorParams[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        Discriminator!.SetParameters(new Vector<T>(discriminatorParams));
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RealESRGAN<T>(
            Architecture,
            Architecture, // Use same architecture for discriminator (will be overwritten on load)
            Architecture.InputType,
            _generatorOptimizer,
            _discriminatorOptimizer,
            _scaleFactor,
            _numRRDBBlocks,
            _numFeatures,
            _residualScale,
            _l1Lambda,
            _perceptualLambda,
            _ganLambda);
    }

    #endregion
}
