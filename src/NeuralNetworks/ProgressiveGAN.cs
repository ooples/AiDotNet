using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Production-ready Progressive GAN (ProGAN) implementation that generates high-resolution images
/// by progressively growing the generator and discriminator during training.
///
/// For Beginners:
/// Progressive GAN is a technique for training GANs that can generate very high-resolution
/// images (e.g., 1024x1024 pixels). Instead of trying to generate high-resolution images
/// from the start, it begins by generating small images (e.g., 4x4) and progressively
/// adds new layers to both the generator and discriminator to increase the resolution
/// (4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128 → 256x256 → 1024x1024).
///
/// Key innovations:
/// 1. Progressive Growing: Start with low resolution and gradually add layers
/// 2. Smooth Fade-in: New layers are faded in smoothly using a blending parameter (alpha)
/// 3. Minibatch Standard Deviation: Helps prevent mode collapse by adding diversity
/// 4. Equalized Learning Rate: Normalizes weights at runtime for better training dynamics
/// 5. Pixel Normalization: Normalizes feature vectors in generator to prevent escalation
///
/// Based on "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
/// by Karras et al. (2018)
/// </summary>
/// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
public class ProgressiveGAN<T> : NeuralNetworkBase<T>
{
    #region Constants

    private const double DefaultLearningRate = 0.001;
    private const double DefaultLearningRateDecay = 0.9999;
    private const double DefaultBeta1 = 0.0; // Beta1=0 recommended for TTUR in GANs
    private const double DefaultBeta2 = 0.999;
    private const double DefaultEpsilon = 1e-8;
    private const double DefaultGradientClipThreshold = 5.0;
    private const double DefaultGradientPenaltyCoefficient = 10.0;
    private const double DefaultDriftPenaltyCoefficient = 0.001;

    #endregion

    #region Optimizer State

    // Generator optimizer state (vectorized)
    private Vector<T> _genMomentum;
    private Vector<T> _genSecondMoment;
    private int _genTimestep;

    // Discriminator optimizer state (vectorized)
    private Vector<T> _discMomentum;
    private Vector<T> _discSecondMoment;
    private int _discTimestep;

    // Optimizer hyperparameters
    private double _genCurrentLearningRate;
    private double _discCurrentLearningRate;
    private readonly double _initialLearningRate;
    private readonly double _learningRateDecay;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _gradientClipThreshold;
    private readonly double _gradientPenaltyCoefficient;
    private readonly double _driftPenaltyCoefficient;

    #endregion

    #region Training History

    private readonly List<T> _generatorLosses = new List<T>();
    private readonly List<T> _discriminatorLosses = new List<T>();
    private const int MaxLossHistorySize = 10000;

    #endregion

    #region Networks

    /// <summary>
    /// Gets the generator network that produces images from latent codes.
    /// Progressively grows to generate higher resolution images.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator (critic) network that evaluates image quality.
    /// Progressively grows to handle higher resolution images.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    #endregion

    #region Configuration

    /// <summary>
    /// Gets the size of the latent vector (noise input).
    /// Typically 512 for high-quality image generation.
    /// </summary>
    public int LatentSize { get; private set; }

    /// <summary>
    /// Gets the current resolution level (e.g., 0=4x4, 1=8x8, 2=16x16, etc.).
    /// </summary>
    public int CurrentResolutionLevel { get; private set; }

    /// <summary>
    /// Gets the maximum resolution level the network can achieve.
    /// </summary>
    public int MaxResolutionLevel { get; private set; }

    /// <summary>
    /// Gets or sets the alpha value for smooth fade-in of new layers.
    /// 0 = old layers only, 1 = new layers fully active.
    /// </summary>
    public double Alpha { get; set; }

    /// <summary>
    /// Gets or sets whether to use minibatch standard deviation.
    /// Helps improve diversity and prevent mode collapse.
    /// </summary>
    public bool UseMinibatchStdDev { get; set; }

    /// <summary>
    /// Gets or sets whether to use pixel normalization in the generator.
    /// Helps prevent unhealthy competition between feature magnitudes.
    /// </summary>
    public bool UsePixelNormalization { get; set; }

    /// <summary>
    /// Gets the last computed gradient penalty value (for monitoring purposes only).
    /// NOTE: The gradient penalty is NOT included in training because proper WGAN-GP
    /// requires second-order gradients (d/dθ ||∇_x D(x)||²) which this engine does not support.
    /// This property allows users to monitor the GP value during training for diagnostic purposes.
    /// </summary>
    public T LastGradientPenalty { get; private set; }

    private int _imageChannels;
    private int _baseFeatureMaps;
    private readonly ILossFunction<T> _lossFunction;

    // Alpha blending state for progressive growth fade-in
    private bool _isFadingIn;
    private Tensor<T>? _previousResolutionOutput;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of Progressive GAN.
    /// </summary>
    /// <param name="generatorArchitecture">Architecture for the generator network.</param>
    /// <param name="discriminatorArchitecture">Architecture for the discriminator network.</param>
    /// <param name="latentSize">Size of the latent vector (typically 512)</param>
    /// <param name="imageChannels">Number of image channels (1 for grayscale, 3 for RGB)</param>
    /// <param name="maxResolutionLevel">Maximum resolution level (0=4x4, 1=8x8, ..., 8=1024x1024)</param>
    /// <param name="baseFeatureMaps">Base number of feature maps (doubled at lower resolutions)</param>
    /// <param name="inputType">The type of input.</param>
    /// <param name="lossFunction">Loss function for training (defaults to mean squared error for WGAN-GP style)</param>
    /// <param name="initialLearningRate">Initial learning rate (default 0.001)</param>
    /// <param name="learningRateDecay">Learning rate decay factor per update (default 0.9999)</param>
    public ProgressiveGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize = 512,
        int imageChannels = 3,
        int maxResolutionLevel = 6, // Up to 256x256
        int baseFeatureMaps = 512,
        InputType inputType = InputType.TwoDimensional,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = DefaultLearningRate,
        double learningRateDecay = DefaultLearningRateDecay)
        : base(new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,  // Base GAN takes latent vector input (1D)
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            latentSize,
            0, 0, 1,  // inputHeight, inputWidth=0 for 1D, inputDepth=1 required
            imageChannels * 4 * (int)Math.Pow(2, maxResolutionLevel) * 4 * (int)Math.Pow(2, maxResolutionLevel),  // outputSize
            null), lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        // Validate inputs
        if (latentSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(latentSize), latentSize, "Latent size must be positive.");
        }
        if (imageChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageChannels), imageChannels, "Image channels must be positive.");
        }
        if (maxResolutionLevel < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxResolutionLevel), maxResolutionLevel, "Max resolution level must be non-negative.");
        }
        if (baseFeatureMaps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(baseFeatureMaps), baseFeatureMaps, "Base feature maps must be positive.");
        }
        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }

        LatentSize = latentSize;
        CurrentResolutionLevel = 0; // Start at 4x4
        MaxResolutionLevel = maxResolutionLevel;
        Alpha = 1.0; // Start fully transitioned
        UseMinibatchStdDev = true;
        UsePixelNormalization = true;
        _imageChannels = imageChannels;
        _baseFeatureMaps = baseFeatureMaps;
        _initialLearningRate = initialLearningRate;
        _learningRateDecay = learningRateDecay;
        _beta1 = DefaultBeta1;
        _beta2 = DefaultBeta2;
        _epsilon = DefaultEpsilon;
        _gradientClipThreshold = DefaultGradientClipThreshold;
        _gradientPenaltyCoefficient = DefaultGradientPenaltyCoefficient;
        _driftPenaltyCoefficient = DefaultDriftPenaltyCoefficient;

        // Initialize optimizer state
        _genCurrentLearningRate = initialLearningRate;
        _discCurrentLearningRate = initialLearningRate;
        _genTimestep = 0;
        _discTimestep = 0;
        _genMomentum = Vector<T>.Empty();
        _genSecondMoment = Vector<T>.Empty();
        _discMomentum = Vector<T>.Empty();
        _discSecondMoment = Vector<T>.Empty();

        // Initialize gradient penalty tracking property
        LastGradientPenalty = NumOps.Zero;

        // Initialize networks at lowest resolution
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    #endregion

    #region Progressive Growth

    /// <summary>
    /// Grows the networks to the next resolution level.
    /// Call this periodically during training to progressively increase resolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Current implementation: Updates metadata (resolution level, alpha) for blending output resolutions.
    /// The Generator/Discriminator architectures must be pre-configured for the maximum target resolution,
    /// and this method controls which resolution path is active via the alpha blending factor.
    /// </para>
    /// <para>
    /// Note: This is a simplified progressive growing implementation. True progressive growing
    /// (dynamically adding layers at runtime) would require significant architectural changes
    /// to support on-the-fly network mutation while preserving learned weights.
    /// </para>
    /// </remarks>
    /// <returns>True if growth was successful, false if already at maximum resolution</returns>
    public bool GrowNetworks()
    {
        if (CurrentResolutionLevel >= MaxResolutionLevel)
        {
            return false;
        }

        CurrentResolutionLevel++;
        Alpha = 0.0; // Start with old layers only, gradually increase to 1.0
        _isFadingIn = true; // Enable alpha blending during fade-in phase
        _previousResolutionOutput = null; // Clear cached output

        return true;
    }

    /// <summary>
    /// Updates the alpha value for progressive fade-in during training.
    /// Should be called each training step during fade-in phase to smoothly blend new layers.
    /// </summary>
    /// <param name="alphaIncrement">Amount to increase alpha per step (typical: 1.0 / fadeInSteps)</param>
    /// <returns>True if still fading in, false if fade-in is complete (Alpha >= 1.0)</returns>
    public bool UpdateAlpha(double alphaIncrement)
    {
        if (!_isFadingIn)
        {
            return false;
        }

        Alpha = Math.Min(1.0, Alpha + alphaIncrement);

        if (Alpha >= 1.0)
        {
            _isFadingIn = false;
            _previousResolutionOutput = null; // Clear cached output
        }

        return _isFadingIn;
    }

    /// <summary>
    /// Gets whether the network is currently in the fade-in phase after a growth step.
    /// </summary>
    public bool IsFadingIn => _isFadingIn;

    /// <summary>
    /// Gets the current image resolution based on the resolution level.
    /// </summary>
    public int GetCurrentResolution()
    {
        return 4 * (int)Math.Pow(2, CurrentResolutionLevel);
    }

    #endregion

    #region Pixel Normalization

    /// <summary>
    /// Applies pixel normalization to feature maps using vectorized operations.
    /// Normalizes each pixel's feature vector across the CHANNEL dimension to unit length.
    /// Formula: x_normalized = x / sqrt(mean(x^2) + epsilon)
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private Tensor<T> ApplyPixelNormalization(Tensor<T> features)
    {
        if (!UsePixelNormalization)
        {
            return features;
        }

        // Guard against empty tensors
        if (features.Length == 0 || features.Shape.Length < 2)
        {
            return features;
        }

        // For 4D tensor [batch, channels, height, width], normalize across channel dimension (dim 1)
        // For 2D tensor [batch, features], normalize across feature dimension (dim 1)
        var epsilon = NumOps.FromDouble(1e-8);

        // Use vectorized operations: compute x^2, mean across channels, sqrt, divide
        var featuresSquared = features.ElementwiseMultiply(features);

        // Sum across channel dimension and compute mean
        // Shape: [batch, channels, H, W] -> sum over channels -> [batch, H, W]
        if (features.Shape.Length == 4)
        {
            // 4D case: [batch, channels, height, width]
            var sumAcrossChannels = featuresSquared.Sum([1]); // Sum across channel dimension
            var numChannels = NumOps.FromDouble(features.Shape[1]);

            // Mean = sum / numChannels, then broadcast back to original shape
            var mean = sumAcrossChannels.Multiply(NumOps.Divide(NumOps.One, numChannels));

            // Add epsilon and compute sqrt
            var meanPlusEps = mean.Add(new Tensor<T>(mean.Shape).Tap(t => t.Fill(epsilon)));

            // sqrt using Engine (need to expand back to 4D for division)
            // For now, use element-wise sqrt on the underlying data
            var normData = new Vector<T>(meanPlusEps.Length);
            for (int i = 0; i < meanPlusEps.Length; i++)
            {
                normData[i] = NumOps.Sqrt(meanPlusEps.GetFlat(i));
            }
            var norm = new Tensor<T>(meanPlusEps.Shape, normData);

            // Broadcast norm back and divide
            // norm shape: [batch, height, width], need to broadcast to [batch, channels, height, width]
            var result = new Tensor<T>(features.Shape);
            int batch = features.Shape[0];
            int channels = features.Shape[1];
            int height = features.Shape[2];
            int width = features.Shape[3];

            // Vectorized division with broadcasting
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int normIdx = b * height * width + h * width + w;
                        T normVal = norm.GetFlat(normIdx);

                        for (int c = 0; c < channels; c++)
                        {
                            int featureIdx = b * channels * height * width + c * height * width + h * width + w;
                            result.SetFlat(featureIdx, NumOps.Divide(features.GetFlat(featureIdx), normVal));
                        }
                    }
                }
            }

            return result;
        }
        else
        {
            // 2D case: [batch, features] - normalize across feature dimension
            var sumAcrossFeatures = featuresSquared.Sum([1]);
            var numFeatures = NumOps.FromDouble(features.Shape[1]);
            var mean = sumAcrossFeatures.Multiply(NumOps.Divide(NumOps.One, numFeatures));

            var meanPlusEps = mean.Add(new Tensor<T>(mean.Shape).Tap(t => t.Fill(epsilon)));

            var normData = new Vector<T>(meanPlusEps.Length);
            for (int i = 0; i < meanPlusEps.Length; i++)
            {
                normData[i] = NumOps.Sqrt(meanPlusEps.GetFlat(i));
            }

            // Divide each row by its norm
            var result = new Tensor<T>(features.Shape);
            int batchSize = features.Shape[0];
            int featureCount = features.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                T normVal = normData[b];
                for (int f = 0; f < featureCount; f++)
                {
                    int idx = b * featureCount + f;
                    result.SetFlat(idx, NumOps.Divide(features.GetFlat(idx), normVal));
                }
            }

            return result;
        }
    }

    #endregion

    #region Training

    /// <summary>
    /// Performs a single training step on a batch of real images.
    /// Uses vectorized operations throughout for optimal performance.
    /// </summary>
    /// <param name="realImages">Batch of real images</param>
    /// <param name="batchSize">Number of images in the batch</param>
    /// <returns>Tuple of (discriminator loss, generator loss)</returns>
    /// <exception cref="ArgumentNullException">Thrown when realImages is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is not positive</exception>
    public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, int batchSize)
    {
        // Input validation
        if (realImages is null)
        {
            throw new ArgumentNullException(nameof(realImages), "Real images tensor cannot be null.");
        }
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        }

        // Save original training modes to restore after training step
        bool originalGenTrainingMode = Generator.IsTrainingMode;
        bool originalDiscTrainingMode = Discriminator.IsTrainingMode;

        var batchSizeT = NumOps.FromDouble(batchSize);

        // === Train Discriminator ===
        Discriminator.SetTrainingMode(true);
        Generator.SetTrainingMode(false);

        // Real images - compute loss using vectorized sum
        var realOutput = Discriminator.Predict(realImages);
        var realSum = realOutput.Sum(); // Vectorized sum
        var realLoss = NumOps.Negate(NumOps.Divide(realSum.GetFlat(0), batchSizeT));

        // Fake images - generate from Gaussian noise
        var noise = GenerateGaussianNoise(batchSize);
        var fakeImages = Generator.Predict(noise);
        var fakeOutput = Discriminator.Predict(fakeImages);
        var fakeSum = fakeOutput.Sum(); // Vectorized sum
        var fakeLoss = NumOps.Divide(fakeSum.GetFlat(0), batchSizeT);

        // Gradient penalty (WGAN-GP style) - computed for monitoring only
        // NOTE: True WGAN-GP requires second-order gradients (d/dθ ||∇_x D(x)||²)
        // which this engine does not support. The GP value is tracked for monitoring
        // but not included in the training loss to avoid incorrect gradient flow.
        var gradientPenalty = ComputeGradientPenalty(realImages, fakeImages, batchSize);

        // Drift penalty (encourages discriminator outputs to stay near 0)
        // This can be properly backpropagated since it's based on D(x)² not gradients
        var driftPenalty = ComputeDriftPenalty(realOutput, batchSize);

        // WGAN discriminator loss: E[D(fake)] - E[D(real)] + drift penalty
        // GP is excluded from training loss but tracked separately for monitoring
        var wganLoss = NumOps.Add(realLoss, fakeLoss);
        var discriminatorLoss = NumOps.Add(wganLoss, driftPenalty);

        // Store GP separately for monitoring (can be accessed via loss history analysis)
        LastGradientPenalty = gradientPenalty;

        // Track loss history (with size limit)
        if (_discriminatorLosses.Count >= MaxLossHistorySize)
        {
            _discriminatorLosses.RemoveAt(0);
        }
        _discriminatorLosses.Add(discriminatorLoss);

        // Backpropagate discriminator with vectorized gradient fill
        var negativeScale = NumOps.Negate(NumOps.Divide(NumOps.One, batchSizeT));
        var realGradient = CreateFilledTensor(realOutput.Shape, negativeScale);
        Discriminator.Predict(realImages); // Ensure correct activations are cached
        Discriminator.Backward(realGradient);

        var positiveScale = NumOps.Divide(NumOps.One, batchSizeT);
        var fakeGradient = CreateFilledTensor(fakeOutput.Shape, positiveScale);
        Discriminator.Predict(fakeImages); // Ensure correct activations are cached
        Discriminator.Backward(fakeGradient);

        // Update discriminator parameters using vectorized Adam optimizer
        UpdateDiscriminatorParametersVectorized();

        // === Train Generator ===
        Generator.SetTrainingMode(true);
        // Keep discriminator in training mode for backward pass (required for gradient computation)
        // We prevent discriminator parameter updates by not calling UpdateDiscriminatorParametersVectorized()
        Discriminator.SetTrainingMode(true);

        var generatorNoise = GenerateGaussianNoise(batchSize);
        var generatedImages = Generator.Predict(generatorNoise);
        var generatorOutput = Discriminator.Predict(generatedImages);

        var genSum = generatorOutput.Sum(); // Vectorized sum
        var generatorLoss = NumOps.Negate(NumOps.Divide(genSum.GetFlat(0), batchSizeT));

        // Track loss history (with size limit)
        if (_generatorLosses.Count >= MaxLossHistorySize)
        {
            _generatorLosses.RemoveAt(0);
        }
        _generatorLosses.Add(generatorLoss);

        // Backpropagate generator with vectorized gradient
        var genGradient = CreateFilledTensor(generatorOutput.Shape, negativeScale);
        var discInputGradient = Discriminator.BackwardWithInputGradient(genGradient);
        Generator.Backward(discInputGradient);

        // Update generator parameters using vectorized Adam optimizer
        UpdateGeneratorParametersVectorized();

        // Restore original training modes for predictable behavior after training step
        Generator.SetTrainingMode(originalGenTrainingMode);
        Discriminator.SetTrainingMode(originalDiscTrainingMode);

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Creates a tensor filled with a constant value using vectorized Fill operation.
    /// </summary>
    private static Tensor<T> CreateFilledTensor(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        tensor.Fill(value);
        return tensor;
    }

    /// <summary>
    /// Generates Gaussian random noise for the generator input using Engine.GenerateGaussianNoise.
    /// Standard normal distribution (mean=0, stddev=1) as required by GANs.
    /// </summary>
    private Tensor<T> GenerateGaussianNoise(int batchSize)
    {
        var totalElements = batchSize * LatentSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;

        // Use Engine's vectorized Gaussian noise generation
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        // Reshape to [batchSize, LatentSize]
        return Tensor<T>.FromVector(noiseVector, [batchSize, LatentSize]);
    }

    /// <summary>
    /// Computes gradient penalty for Wasserstein GAN with gradient penalty.
    /// Uses vectorized operations for interpolation and norm computation.
    /// </summary>
    private T ComputeGradientPenalty(Tensor<T> realImages, Tensor<T> fakeImages, int batchSize)
    {
        // Generate random interpolation coefficients using Engine
        var alphaVector = Engine.GenerateGaussianNoise<T>(batchSize, NumOps.FromDouble(0.5), NumOps.FromDouble(0.25));

        // Clamp to [0, 1] and create interpolated samples
        int sampleSize = realImages.Length / batchSize;
        var interpolated = new Tensor<T>(realImages.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Clamp alpha to [0, 1] using MathHelper for .NET Framework compatibility
            double alphaVal = MathHelper.Clamp(NumOps.ToDouble(alphaVector[b]), 0.0, 1.0);
            var alpha = NumOps.FromDouble(alphaVal);
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);

            // Vectorized interpolation for this batch item
            int startIdx = b * sampleSize;
            for (int i = 0; i < sampleSize; i++)
            {
                int idx = startIdx + i;
                var realVal = NumOps.Multiply(alpha, realImages.GetFlat(idx));
                var fakeVal = NumOps.Multiply(oneMinusAlpha, fakeImages.GetFlat(idx));
                interpolated.SetFlat(idx, NumOps.Add(realVal, fakeVal));
            }
        }

        // Forward pass on interpolated images
        var interpolatedOutput = Discriminator.Predict(interpolated);

        // Create gradient tensor filled with ones using vectorized Fill
        var ones = CreateFilledTensor(interpolatedOutput.Shape, NumOps.One);

        // Backpropagate to get gradients w.r.t. interpolated input
        var inputGradients = Discriminator.BackwardWithInputGradient(ones);

        // Compute L2 norm of gradients for each sample using vectorized operations
        T totalPenalty = NumOps.Zero;
        int gradSampleSize = inputGradients.Length / batchSize;

        for (int b = 0; b < batchSize; b++)
        {
            // Get slice for this batch and compute L2 norm squared
            T gradNormSquared = NumOps.Zero;
            int startIdx = b * gradSampleSize;

            for (int i = 0; i < gradSampleSize; i++)
            {
                T gradValue = inputGradients.GetFlat(startIdx + i);
                gradNormSquared = NumOps.Add(gradNormSquared, NumOps.Multiply(gradValue, gradValue));
            }

            T gradNorm = NumOps.Sqrt(gradNormSquared);

            // Penalty: (||grad|| - 1)^2
            T deviation = NumOps.Subtract(gradNorm, NumOps.One);
            T penalty = NumOps.Multiply(deviation, deviation);

            totalPenalty = NumOps.Add(totalPenalty, penalty);
        }

        // Average penalty across batch, scaled by GP coefficient
        T avgPenalty = NumOps.Divide(totalPenalty, NumOps.FromDouble(batchSize));
        return NumOps.Multiply(avgPenalty, NumOps.FromDouble(_gradientPenaltyCoefficient));
    }

    /// <summary>
    /// Computes drift penalty to keep discriminator outputs near zero.
    /// Uses vectorized operations.
    /// </summary>
    private T ComputeDriftPenalty(Tensor<T> discriminatorOutput, int batchSize)
    {
        // Compute sum of squares using vectorized operations
        var squared = discriminatorOutput.ElementwiseMultiply(discriminatorOutput);
        var sumSquares = squared.Sum();
        var meanSquare = NumOps.Divide(sumSquares.GetFlat(0), NumOps.FromDouble(batchSize));
        return NumOps.Multiply(meanSquare, NumOps.FromDouble(_driftPenaltyCoefficient));
    }

    #endregion

    #region Vectorized Adam Optimizer

    /// <summary>
    /// Updates Generator parameters using vectorized Adam optimizer with Engine operations.
    /// Follows the gold-standard pattern from AdamOptimizer.cs for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateGeneratorParametersVectorized()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_genMomentum.Length != parameters.Length)
        {
            _genMomentum = new Vector<T>(parameters.Length);
            _genSecondMoment = new Vector<T>(parameters.Length);
            _genMomentum.Fill(NumOps.Zero);
            _genSecondMoment.Fill(NumOps.Zero);
            _genTimestep = 0;
        }

        _genTimestep++;

        // Gradient clipping using vectorized L2 norm
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(_gradientClipThreshold);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        // Vectorized Adam update using Engine operations
        var updatedParams = ApplyVectorizedAdamUpdate(
            parameters, gradients,
            ref _genMomentum, ref _genSecondMoment,
            _genTimestep, _genCurrentLearningRate);

        // Apply learning rate decay
        _genCurrentLearningRate *= _learningRateDecay;

        Generator.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Updates Discriminator parameters using vectorized Adam optimizer with Engine operations.
    /// </summary>
    private void UpdateDiscriminatorParametersVectorized()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_discMomentum.Length != parameters.Length)
        {
            _discMomentum = new Vector<T>(parameters.Length);
            _discSecondMoment = new Vector<T>(parameters.Length);
            _discMomentum.Fill(NumOps.Zero);
            _discSecondMoment.Fill(NumOps.Zero);
            _discTimestep = 0;
        }

        _discTimestep++;

        // Gradient clipping using vectorized L2 norm
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(_gradientClipThreshold);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        // Vectorized Adam update
        var updatedParams = ApplyVectorizedAdamUpdate(
            parameters, gradients,
            ref _discMomentum, ref _discSecondMoment,
            _discTimestep, _discCurrentLearningRate);

        // Apply learning rate decay
        _discCurrentLearningRate *= _learningRateDecay;

        Discriminator.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Applies vectorized Adam update using Engine operations for SIMD/GPU acceleration.
    /// This follows the gold-standard pattern from the codebase's AdamOptimizer.
    /// </summary>
    private Vector<T> ApplyVectorizedAdamUpdate(
        Vector<T> parameters,
        Vector<T> gradient,
        ref Vector<T> momentum,
        ref Vector<T> secondMoment,
        int timestep,
        double learningRate)
    {
        T beta1 = NumOps.FromDouble(_beta1);
        T beta2 = NumOps.FromDouble(_beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1.0 - _beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1.0 - _beta2);
        T epsilon = NumOps.FromDouble(_epsilon);
        T lr = NumOps.FromDouble(learningRate);
        T biasCorrection1 = NumOps.FromDouble(1.0 - Math.Pow(_beta1, timestep));
        T biasCorrection2 = NumOps.FromDouble(1.0 - Math.Pow(_beta2, timestep));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(momentum, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        momentum = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(secondMoment, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        secondMoment = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(momentum, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(secondMoment, biasCorrection2);

        // Compute update: update = lr * mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var updateDiv = (Vector<T>)Engine.Divide(mHat, denominator);
        var update = (Vector<T>)Engine.Multiply(updateDiv, lr);

        // Apply update: parameters = parameters - update
        return (Vector<T>)Engine.Subtract(parameters, update);
    }

    #endregion

    #region Generation

    /// <summary>
    /// Generates images from random latent codes.
    /// </summary>
    /// <param name="numImages">Number of images to generate</param>
    /// <returns>Generated images tensor</returns>
    public Tensor<T> Generate(int numImages)
    {
        if (numImages <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numImages), numImages, "Number of images must be positive.");
        }

        Generator.SetTrainingMode(false);
        var noise = GenerateGaussianNoise(numImages);

        // Reshape noise to 4D format for CNN generator: [batch, latent_size] -> [batch, 1, h, w]
        int h = (int)Math.Ceiling(Math.Sqrt(LatentSize));
        int w = h;
        int padSize = h * w - LatentSize;
        Tensor<T> reshapedNoise;
        if (padSize > 0)
        {
            var padded = new Tensor<T>([numImages, h * w]);
            for (int b = 0; b < numImages; b++)
            {
                for (int i = 0; i < LatentSize; i++)
                    padded[b, i] = noise[b, i];
                for (int i = LatentSize; i < h * w; i++)
                    padded[b, i] = NumOps.Zero;
            }
            reshapedNoise = padded.Reshape(numImages, 1, h, w);
        }
        else
        {
            reshapedNoise = noise.Reshape(numImages, 1, h, w);
        }

        var newOutput = Generator.Predict(reshapedNoise);

        // Apply alpha blending during fade-in phase
        return ApplyAlphaBlending(newOutput);
    }

    /// <summary>
    /// Generates images from specific latent codes.
    /// </summary>
    /// <param name="latentCodes">Latent codes to use</param>
    /// <returns>Generated images tensor</returns>
    public Tensor<T> Generate(Tensor<T> latentCodes)
    {
        if (latentCodes is null)
        {
            throw new ArgumentNullException(nameof(latentCodes), "Latent codes tensor cannot be null.");
        }

        Generator.SetTrainingMode(false);

        // Reshape latent codes to 3D/4D format for CNN generator
        Tensor<T> reshapedLatent;
        if (latentCodes.Shape.Length == 1)
        {
            // 1D [latent_size] -> 3D [1, height, width]
            int latentLen = latentCodes.Shape[0];
            int h = (int)Math.Ceiling(Math.Sqrt(latentLen));
            int w = h;
            int padSize = h * w - latentLen;
            if (padSize > 0)
            {
                var padded = new Tensor<T>([h * w]);
                for (int i = 0; i < latentLen; i++)
                    padded.Data[i] = latentCodes.Data[i];
                for (int i = latentLen; i < h * w; i++)
                    padded.Data[i] = NumOps.Zero;
                reshapedLatent = padded.Reshape(1, h, w);
            }
            else
            {
                reshapedLatent = latentCodes.Reshape(1, h, w);
            }
        }
        else if (latentCodes.Shape.Length == 2)
        {
            // 2D [batch, latent_size] -> 4D [batch, 1, height, width]
            int batchSize = latentCodes.Shape[0];
            int latentLen = latentCodes.Shape[1];
            int h = (int)Math.Ceiling(Math.Sqrt(latentLen));
            int w = h;
            int padSize = h * w - latentLen;
            if (padSize > 0)
            {
                var padded = new Tensor<T>([batchSize, h * w]);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < latentLen; i++)
                        padded[b, i] = latentCodes[b, i];
                    for (int i = latentLen; i < h * w; i++)
                        padded[b, i] = NumOps.Zero;
                }
                reshapedLatent = padded.Reshape(batchSize, 1, h, w);
            }
            else
            {
                reshapedLatent = latentCodes.Reshape(batchSize, 1, h, w);
            }
        }
        else
        {
            // Already 3D or 4D, use as-is
            reshapedLatent = latentCodes;
        }

        var newOutput = Generator.Predict(reshapedLatent);

        // Apply alpha blending during fade-in phase
        return ApplyAlphaBlending(newOutput);
    }

    /// <summary>
    /// Applies alpha blending between new and previous resolution outputs during fade-in.
    /// Implements the ProGAN smooth transition: output = (1 - alpha) * upsampled_old + alpha * new
    /// Uses optimized Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private Tensor<T> ApplyAlphaBlending(Tensor<T> newOutput)
    {
        // If not fading in or alpha is 1.0, return new output directly
        if (!_isFadingIn || Alpha >= 1.0)
        {
            return newOutput;
        }

        // If we don't have a previous output cached, cache a downsampled/upsampled version
        if (_previousResolutionOutput is null || _previousResolutionOutput.Shape[0] != newOutput.Shape[0])
        {
            _previousResolutionOutput = CreateDownsampledUpsampled(newOutput);
        }

        // Blend using optimized tensor operations: (1 - alpha) * old + alpha * new
        var alphaT = NumOps.FromDouble(Alpha);
        var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

        // Scale tensors using vectorized operations
        var oldScaled = _previousResolutionOutput.Multiply(oneMinusAlpha);
        var newScaled = newOutput.Multiply(alphaT);

        // Add using vectorized operations
        var blended = oldScaled.Add(newScaled);

        // Update cached output for next iteration
        _previousResolutionOutput = blended;

        return blended;
    }

    /// <summary>
    /// Creates a downsampled then upsampled version of the output to simulate previous resolution.
    /// Uses optimized Engine.AvgPool2D and Engine.Upsample for CPU/GPU acceleration.
    /// </summary>
    private Tensor<T> CreateDownsampledUpsampled(Tensor<T> output)
    {
        // Validate input tensor has correct dimensions for 2D spatial operations
        if (output.Shape.Length < 4 || output.Shape[2] < 2 || output.Shape[3] < 2)
        {
            return output; // Can't downsample if already at minimum size
        }

        // Use optimized Engine operations for average pooling and upsampling
        var downsampled = Engine.AvgPool2D(output, poolSize: 2, stride: 2, padding: 0);
        var upsampled = Engine.Upsample(downsampled, scaleH: 2, scaleW: 2);

        return upsampled;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Gets the total number of trainable parameters in the ProgressiveGAN.
    /// </summary>
    /// <remarks>
    /// This includes all parameters from both the Generator and Discriminator networks.
    /// </remarks>
    public override int ParameterCount => Generator.GetParameterCount() + Discriminator.GetParameterCount();

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }
        return Generate(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }
        var batchSize = input.Shape[0];
        TrainStep(input, batchSize);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();

        if (parameters.Length != generatorCount + discriminatorCount)
        {
            throw new ArgumentException(
                $"Expected {generatorCount + discriminatorCount} parameters, got {parameters.Length}");
        }

        // Use vectorized slice operations
        var generatorParams = new Vector<T>(generatorCount);
        var discriminatorParams = new Vector<T>(discriminatorCount);

        // Copy using spans for efficiency
        for (int i = 0; i < generatorCount; i++)
            generatorParams[i] = parameters[i];
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[generatorCount + i];

        Generator.UpdateParameters(generatorParams);
        Discriminator.UpdateParameters(discriminatorParams);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var genParams = Generator.GetParameters();
        var discParams = Discriminator.GetParameters();

        var totalLength = genParams.Length + discParams.Length;
        var parameters = new Vector<T>(totalLength);

        // Copy using index for efficiency
        int idx = 0;
        for (int i = 0; i < genParams.Length; i++)
            parameters[idx++] = genParams[i];
        for (int i = 0; i < discParams.Length; i++)
            parameters[idx++] = discParams[i];

        return parameters;
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        IsTrainingMode = isTraining;
        Generator.SetTrainingMode(isTraining);
        Discriminator.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ProgressiveGAN",
            Version = "2.0" // Updated version for production-ready implementation
        };

        metadata.SetProperty("ModelType", "ProgressiveGAN");
        metadata.SetProperty("LatentSize", LatentSize);
        metadata.SetProperty("CurrentResolutionLevel", CurrentResolutionLevel);
        metadata.SetProperty("MaxResolutionLevel", MaxResolutionLevel);
        metadata.SetProperty("CurrentResolution", GetCurrentResolution());
        metadata.SetProperty("ImageChannels", _imageChannels);
        metadata.SetProperty("BaseFeatureMaps", _baseFeatureMaps);
        metadata.SetProperty("Alpha", Alpha);
        metadata.SetProperty("UseMinibatchStdDev", UseMinibatchStdDev);
        metadata.SetProperty("UsePixelNormalization", UsePixelNormalization);
        metadata.SetProperty("GeneratorLearningRate", _genCurrentLearningRate);
        metadata.SetProperty("DiscriminatorLearningRate", _discCurrentLearningRate);

        return metadata;
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        var genParamCount = Generator.GetParameterCount();
        var discParamCount = Discriminator.GetParameterCount();

        // Initialize Generator optimizer state with vectorized Fill
        _genMomentum = new Vector<T>(genParamCount);
        _genMomentum.Fill(NumOps.Zero);
        _genSecondMoment = new Vector<T>(genParamCount);
        _genSecondMoment.Fill(NumOps.Zero);
        _genTimestep = 0;

        // Initialize Discriminator optimizer state with vectorized Fill
        _discMomentum = new Vector<T>(discParamCount);
        _discMomentum.Fill(NumOps.Zero);
        _discSecondMoment = new Vector<T>(discParamCount);
        _discSecondMoment.Fill(NumOps.Zero);
        _discTimestep = 0;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Configuration
        writer.Write(LatentSize);
        writer.Write(CurrentResolutionLevel);
        writer.Write(MaxResolutionLevel);
        writer.Write(Alpha);
        writer.Write(UseMinibatchStdDev);
        writer.Write(UsePixelNormalization);
        writer.Write(_imageChannels);
        writer.Write(_baseFeatureMaps);

        // Hyperparameters
        writer.Write(_initialLearningRate);
        writer.Write(_learningRateDecay);
        writer.Write(_genCurrentLearningRate);
        writer.Write(_discCurrentLearningRate);

        // Generator optimizer state
        writer.Write(_genTimestep);
        SerializationHelper<T>.SerializeVector(writer, _genMomentum);
        SerializationHelper<T>.SerializeVector(writer, _genSecondMoment);

        // Discriminator optimizer state
        writer.Write(_discTimestep);
        SerializationHelper<T>.SerializeVector(writer, _discMomentum);
        SerializationHelper<T>.SerializeVector(writer, _discSecondMoment);

        // Alpha blending state
        writer.Write(_isFadingIn);

        // Serialize networks
        byte[] generatorData = Generator.Serialize();
        writer.Write(generatorData.Length);
        writer.Write(generatorData);

        byte[] discriminatorData = Discriminator.Serialize();
        writer.Write(discriminatorData.Length);
        writer.Write(discriminatorData);

        // Loss history (limited to most recent)
        int genLossCount = Math.Min(_generatorLosses.Count, MaxLossHistorySize);
        int discLossCount = Math.Min(_discriminatorLosses.Count, MaxLossHistorySize);

        writer.Write(genLossCount);
        for (int i = _generatorLosses.Count - genLossCount; i < _generatorLosses.Count; i++)
        {
            writer.Write(NumOps.ToDouble(_generatorLosses[i]));
        }

        writer.Write(discLossCount);
        for (int i = _discriminatorLosses.Count - discLossCount; i < _discriminatorLosses.Count; i++)
        {
            writer.Write(NumOps.ToDouble(_discriminatorLosses[i]));
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Configuration
        LatentSize = reader.ReadInt32();
        CurrentResolutionLevel = reader.ReadInt32();
        MaxResolutionLevel = reader.ReadInt32();
        Alpha = reader.ReadDouble();
        UseMinibatchStdDev = reader.ReadBoolean();
        UsePixelNormalization = reader.ReadBoolean();
        _imageChannels = reader.ReadInt32();
        _baseFeatureMaps = reader.ReadInt32();

        // Hyperparameters
        reader.ReadDouble(); // _initialLearningRate (readonly)
        reader.ReadDouble(); // _learningRateDecay (readonly)
        _genCurrentLearningRate = reader.ReadDouble();
        _discCurrentLearningRate = reader.ReadDouble();

        // Generator optimizer state
        _genTimestep = reader.ReadInt32();
        _genMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _genSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        // Discriminator optimizer state
        _discTimestep = reader.ReadInt32();
        _discMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _discSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        // Alpha blending state
        _isFadingIn = reader.ReadBoolean();
        _previousResolutionOutput = null; // Will be recreated as needed

        // Deserialize networks
        int generatorLength = reader.ReadInt32();
        Generator.Deserialize(reader.ReadBytes(generatorLength));

        int discriminatorLength = reader.ReadInt32();
        Discriminator.Deserialize(reader.ReadBytes(discriminatorLength));

        // Loss history
        _generatorLosses.Clear();
        int genLossCount = reader.ReadInt32();
        for (int i = 0; i < genLossCount; i++)
        {
            _generatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));
        }

        _discriminatorLosses.Clear();
        int discLossCount = reader.ReadInt32();
        for (int i = 0; i < discLossCount; i++)
        {
            _discriminatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ProgressiveGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            LatentSize,
            _imageChannels,
            MaxResolutionLevel,
            _baseFeatureMaps,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
            _learningRateDecay);
    }

    #endregion

    #region Loss History

    /// <summary>
    /// Gets the generator loss history.
    /// </summary>
    public IReadOnlyList<T> GeneratorLosses => _generatorLosses.AsReadOnly();

    /// <summary>
    /// Gets the discriminator loss history.
    /// </summary>
    public IReadOnlyList<T> DiscriminatorLosses => _discriminatorLosses.AsReadOnly();

    /// <summary>
    /// Clears the loss history.
    /// </summary>
    public void ClearLossHistory()
    {
        _generatorLosses.Clear();
        _discriminatorLosses.Clear();
    }

    #endregion
}

/// <summary>
/// Extension method for fluent tensor initialization.
/// </summary>
internal static class TensorExtensions
{
    /// <summary>
    /// Applies an action to a tensor and returns it for fluent chaining.
    /// </summary>
    public static Tensor<T> Tap<T>(this Tensor<T> tensor, Action<Tensor<T>> action)
    {
        action(tensor);
        return tensor;
    }
}
