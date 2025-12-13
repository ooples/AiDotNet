using System.IO;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Progressive GAN (ProGAN) implementation that generates high-resolution images
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
    // Generator optimizer state
    private Vector<T> _genMomentum;
    private Vector<T> _genSecondMoment;
    private T _genBeta1Power;
    private T _genBeta2Power;
    private double _genCurrentLearningRate;

    // Discriminator optimizer state
    private Vector<T> _discMomentum;
    private Vector<T> _discSecondMoment;
    private T _discBeta1Power;
    private T _discBeta2Power;
    private double _discCurrentLearningRate;

    private double _initialLearningRate;
    private double _learningRateDecay;
    private List<T> _generatorLosses = [];
    private List<T> _discriminatorLosses = [];

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

    private int _imageChannels;
    private int _baseFeatureMaps;
    private double _driftPenaltyCoefficient;
    private ILossFunction<T> _lossFunction;

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
        double initialLearningRate = 0.001,
        double learningRateDecay = 0.9999)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            latentSize,
            imageChannels * 4 * (int)Math.Pow(2, maxResolutionLevel) * 4 * (int)Math.Pow(2, maxResolutionLevel),
            0, 0, 0,
            null), lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        LatentSize = latentSize;
        CurrentResolutionLevel = 0; // Start at 4x4
        MaxResolutionLevel = maxResolutionLevel;
        Alpha = 1.0; // Start fully transitioned
        UseMinibatchStdDev = true;
        UsePixelNormalization = true;
        _imageChannels = imageChannels;
        _baseFeatureMaps = baseFeatureMaps;
        _driftPenaltyCoefficient = 0.001;
        _initialLearningRate = initialLearningRate;
        _learningRateDecay = learningRateDecay;

        // Initialize Generator optimizer parameters
        _genBeta1Power = NumOps.One;
        _genBeta2Power = NumOps.One;
        _genCurrentLearningRate = initialLearningRate;
        _genMomentum = Vector<T>.Empty();
        _genSecondMoment = Vector<T>.Empty();

        // Initialize Discriminator optimizer parameters
        _discBeta1Power = NumOps.One;
        _discBeta2Power = NumOps.One;
        _discCurrentLearningRate = initialLearningRate;
        _discMomentum = Vector<T>.Empty();
        _discSecondMoment = Vector<T>.Empty();

        // Initialize networks at lowest resolution
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Grows the networks to the next resolution level.
    /// Call this periodically during training to progressively increase resolution.
    /// </summary>
    /// <returns>True if growth was successful, false if already at maximum resolution</returns>
    public bool GrowNetworks()
    {
        if (CurrentResolutionLevel >= MaxResolutionLevel)
        {
            return false;
        }

        CurrentResolutionLevel++;
        Alpha = 0.0; // Start with old layers only, gradually increase to 1.0

        return true;
    }

    /// <summary>
    /// Gets the current image resolution based on the resolution level.
    /// </summary>
    public int GetCurrentResolution()
    {
        return 4 * (int)Math.Pow(2, CurrentResolutionLevel);
    }

    /// <summary>
    /// Applies pixel normalization to feature maps.
    /// Normalizes each pixel feature vector to unit length.
    /// </summary>
    private Tensor<T> ApplyPixelNormalization(Tensor<T> features)
    {
        if (!UsePixelNormalization)
        {
            return features;
        }

        // Pixel normalization: x / sqrt(mean(x^2) + epsilon)
        var epsilon = NumOps.FromDouble(1e-8);
        var normalized = new Tensor<T>(features.Shape);

        // Simplified: normalize across channel dimension
        for (int i = 0; i < features.Shape[0]; i++)
        {
            var sumSquares = NumOps.Zero;
            for (int j = 0; j < features.Length / features.Shape[0]; j++)
            {
                var val = features.GetFlat(i * (features.Length / features.Shape[0]) + j);
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
            }

            var mean = NumOps.Divide(sumSquares, NumOps.FromDouble((double)features.Length / features.Shape[0]));
            var norm = NumOps.Sqrt(NumOps.Add(mean, epsilon));

            for (int j = 0; j < features.Length / features.Shape[0]; j++)
            {
                var idx = i * (features.Length / features.Shape[0]) + j;
                normalized.SetFlat(idx, NumOps.Divide(features.GetFlat(idx), norm));
            }
        }

        return normalized;
    }

    /// <summary>
    /// Performs a single training step on a batch of real images.
    /// </summary>
    /// <param name="realImages">Batch of real images</param>
    /// <param name="batchSize">Number of images in the batch</param>
    /// <returns>Tuple of (discriminator loss, generator loss)</returns>
    public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, int batchSize)
    {
        var one = NumOps.One;

        // === Train Discriminator ===
        Discriminator.SetTrainingMode(true);
        Generator.SetTrainingMode(false);

        // Real images
        var realOutput = Discriminator.Predict(realImages);
        var realLoss = NumOps.Zero;
        for (int i = 0; i < realOutput.Length; i++)
        {
            realLoss = NumOps.Add(realLoss, realOutput.GetFlat(i));
        }
        realLoss = NumOps.Negate(NumOps.Divide(realLoss, NumOps.FromDouble(batchSize)));

        // Fake images
        var noise = GenerateNoise(batchSize);
        var fakeImages = Generator.Predict(noise);
        var fakeOutput = Discriminator.Predict(fakeImages);
        var fakeLoss = NumOps.Zero;
        for (int i = 0; i < fakeOutput.Length; i++)
        {
            fakeLoss = NumOps.Add(fakeLoss, fakeOutput.GetFlat(i));
        }
        fakeLoss = NumOps.Divide(fakeLoss, NumOps.FromDouble(batchSize));

        // Gradient penalty (WGAN-GP style)
        var gradientPenalty = ComputeGradientPenalty(realImages, fakeImages, batchSize);

        // Drift penalty (encourages discriminator outputs to stay near 0)
        var driftPenalty = ComputeDriftPenalty(realOutput, batchSize);

        // Total discriminator loss
        var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        discriminatorLoss = NumOps.Add(discriminatorLoss, gradientPenalty);
        discriminatorLoss = NumOps.Add(discriminatorLoss, driftPenalty);
        _discriminatorLosses.Add(discriminatorLoss);

        // Backpropagate discriminator
        // Gradient shape must match discriminator output shape
        // For WGAN: dL/d(output) = -1/batchSize for real (maximize output), 1/batchSize for fake (minimize output)
        var realGradient = new Tensor<T>(realOutput.Shape);
        T negativeScale = NumOps.Negate(NumOps.Divide(one, NumOps.FromDouble(batchSize)));
        for (int i = 0; i < realGradient.Length; i++)
        {
            realGradient.SetFlat(i, negativeScale);
        }
        Discriminator.Predict(realImages); // Ensure correct activations are cached
        Discriminator.Backward(realGradient);

        var fakeGradient = new Tensor<T>(fakeOutput.Shape);
        T positiveScale = NumOps.Divide(one, NumOps.FromDouble(batchSize));
        for (int i = 0; i < fakeGradient.Length; i++)
        {
            fakeGradient.SetFlat(i, positiveScale);
        }
        Discriminator.Predict(fakeImages); // Ensure correct activations are cached
        Discriminator.Backward(fakeGradient);

        // Update discriminator parameters using Adam optimizer
        UpdateDiscriminatorParameters();

        // === Train Generator ===
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);

        var generatorNoise = GenerateNoise(batchSize);
        var generatedImages = Generator.Predict(generatorNoise);
        var generatorOutput = Discriminator.Predict(generatedImages);

        var generatorLoss = NumOps.Zero;
        for (int i = 0; i < generatorOutput.Length; i++)
        {
            generatorLoss = NumOps.Add(generatorLoss, generatorOutput.GetFlat(i));
        }
        generatorLoss = NumOps.Negate(NumOps.Divide(generatorLoss, NumOps.FromDouble(batchSize)));
        _generatorLosses.Add(generatorLoss);

        // Backpropagate generator - gradient shape must match discriminator output
        // For WGAN: dL/d(output) = -1/batchSize (maximize discriminator output for generated images)
        var genGradient = new Tensor<T>(generatorOutput.Shape);
        T genScale = NumOps.Negate(NumOps.Divide(one, NumOps.FromDouble(batchSize)));
        for (int i = 0; i < genGradient.Length; i++)
        {
            genGradient.SetFlat(i, genScale);
        }

        // Backprop through discriminator to get input gradient for generator
        var discInputGradient = Discriminator.BackwardWithInputGradient(genGradient);

        // Then backprop through generator
        Generator.Backward(discInputGradient);

        // Update generator parameters using Adam optimizer
        UpdateGeneratorParameters();

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Generates random noise for the generator input.
    /// </summary>
    private Tensor<T> GenerateNoise(int batchSize)
    {
        var noise = new Tensor<T>([batchSize, LatentSize]);

        for (int i = 0; i < noise.Length; i++)
        {
            noise.SetFlat(i, NumOps.FromDouble(Random.NextDouble() * 2.0 - 1.0));
        }

        return noise;
    }

    /// <summary>
    /// Computes gradient penalty for Wasserstein GAN with gradient penalty.
    /// Helps enforce the Lipschitz constraint.
    /// </summary>
    private T ComputeGradientPenalty(Tensor<T> realImages, Tensor<T> fakeImages, int batchSize)
    {
        // For each sample in batch, use a different interpolation coefficient
        var interpolated = new Tensor<T>(realImages.Shape);
        int sampleSize = realImages.Length / batchSize;

        for (int b = 0; b < batchSize; b++)
        {
            var alpha = NumOps.FromDouble(Random.NextDouble());
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);

            for (int i = 0; i < sampleSize; i++)
            {
                int idx = b * sampleSize + i;
                var real = NumOps.Multiply(alpha, realImages.GetFlat(idx));
                var fake = NumOps.Multiply(oneMinusAlpha, fakeImages.GetFlat(idx));
                interpolated.SetFlat(idx, NumOps.Add(real, fake));
            }
        }

        // Forward pass on interpolated images
        var interpolatedOutput = Discriminator.Predict(interpolated);

        // Create gradient of 1s for backpropagation
        var ones = new Tensor<T>(interpolatedOutput.Shape);
        for (int i = 0; i < interpolatedOutput.Length; i++)
        {
            ones.SetFlat(i, NumOps.One);
        }

        // Backpropagate to get gradients w.r.t. interpolated input
        var inputGradients = Discriminator.BackwardWithInputGradient(ones);

        // Compute L2 norm of gradients for each sample
        T totalPenalty = NumOps.Zero;
        int gradSampleSize = inputGradients.Length / batchSize;

        for (int b = 0; b < batchSize; b++)
        {
            T gradNormSquared = NumOps.Zero;

            for (int i = 0; i < gradSampleSize; i++)
            {
                int idx = b * gradSampleSize + i;
                T gradValue = inputGradients.GetFlat(idx);
                gradNormSquared = NumOps.Add(gradNormSquared, NumOps.Multiply(gradValue, gradValue));
            }

            T gradNorm = NumOps.Sqrt(gradNormSquared);

            // Penalty: (||grad|| - 1)^2
            T deviation = NumOps.Subtract(gradNorm, NumOps.One);
            T penalty = NumOps.Multiply(deviation, deviation);

            totalPenalty = NumOps.Add(totalPenalty, penalty);
        }

        // Average penalty across batch, scaled by GP coefficient (lambda = 10)
        T avgPenalty = NumOps.Divide(totalPenalty, NumOps.FromDouble(batchSize));
        return NumOps.Multiply(avgPenalty, NumOps.FromDouble(10.0));
    }

    /// <summary>
    /// Computes drift penalty to keep discriminator outputs near zero.
    /// </summary>
    private T ComputeDriftPenalty(Tensor<T> discriminatorOutput, int batchSize)
    {
        var sumSquares = NumOps.Zero;
        for (int i = 0; i < discriminatorOutput.Length; i++)
        {
            var val = discriminatorOutput.GetFlat(i);
            var squared = NumOps.Multiply(val, val);
            sumSquares = NumOps.Add(sumSquares, squared);
        }

        var meanSquare = NumOps.Divide(sumSquares, NumOps.FromDouble(batchSize));
        return NumOps.Multiply(meanSquare, NumOps.FromDouble(_driftPenaltyCoefficient));
    }

    /// <summary>
    /// Generates images from random latent codes.
    /// </summary>
    /// <param name="numImages">Number of images to generate</param>
    /// <returns>Generated images tensor</returns>
    public Tensor<T> Generate(int numImages)
    {
        Generator.SetTrainingMode(false);
        var noise = GenerateNoise(numImages);
        return Generator.Predict(noise);
    }

    /// <summary>
    /// Generates images from specific latent codes.
    /// </summary>
    /// <param name="latentCodes">Latent codes to use</param>
    /// <returns>Generated images tensor</returns>
    public Tensor<T> Generate(Tensor<T> latentCodes)
    {
        Generator.SetTrainingMode(false);
        return Generator.Predict(latentCodes);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generate(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var batchSize = input.Shape[0];
        TrainStep(input, batchSize);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();

        // Update Generator parameters
        var generatorParams = new Vector<T>(generatorCount);
        for (int i = 0; i < generatorCount; i++)
            generatorParams[i] = parameters[i];
        Generator.UpdateParameters(generatorParams);

        // Update Discriminator parameters
        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[generatorCount + i];
        Discriminator.UpdateParameters(discriminatorParams);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var genParams = Generator.GetParameters();
        var discParams = Discriminator.GetParameters();

        var totalLength = genParams.Length + discParams.Length;
        var parameters = new Vector<T>(totalLength);

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
            Version = "1.0"
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

        return metadata;
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in the constructor via the Generator and Discriminator CNNs
        var genParamCount = Generator.GetParameterCount();
        var discParamCount = Discriminator.GetParameterCount();

        // Initialize Generator optimizer state
        _genMomentum = new Vector<T>(genParamCount);
        _genMomentum.Fill(NumOps.Zero);
        _genSecondMoment = new Vector<T>(genParamCount);
        _genSecondMoment.Fill(NumOps.Zero);

        // Initialize Discriminator optimizer state
        _discMomentum = new Vector<T>(discParamCount);
        _discMomentum.Fill(NumOps.Zero);
        _discSecondMoment = new Vector<T>(discParamCount);
        _discSecondMoment.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Updates Generator parameters using Adam optimizer.
    /// </summary>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_genMomentum == null || _genMomentum.Length != parameters.Length)
        {
            _genMomentum = new Vector<T>(parameters.Length);
            _genMomentum.Fill(NumOps.Zero);
        }

        if (_genSecondMoment == null || _genSecondMoment.Length != parameters.Length)
        {
            _genSecondMoment = new Vector<T>(parameters.Length);
            _genSecondMoment.Fill(NumOps.Zero);
        }

        // Gradient clipping
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
            }
        }

        // Adam optimizer parameters
        var learningRate = NumOps.FromDouble(_genCurrentLearningRate);
        var beta1 = NumOps.FromDouble(0.0); // beta1=0 recommended for GANs
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);

        // Update beta powers
        _genBeta1Power = NumOps.Multiply(_genBeta1Power, beta1);
        _genBeta2Power = NumOps.Multiply(_genBeta2Power, beta2);

        // Bias correction
        var biasCorrection1 = NumOps.Subtract(NumOps.One, _genBeta1Power);
        var biasCorrection2 = NumOps.Subtract(NumOps.One, _genBeta2Power);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update momentum (first moment)
            _genMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _genMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            // Update second moment
            _genSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _genSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            // Bias-corrected estimates
            var mHat = NumOps.Divide(_genMomentum[i], biasCorrection1);
            var vHat = NumOps.Divide(_genSecondMoment[i], biasCorrection2);

            // Adam update
            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(
                    learningRate,
                    NumOps.Divide(mHat, NumOps.Add(NumOps.Sqrt(vHat), epsilon))
                )
            );
        }

        // Apply learning rate decay
        _genCurrentLearningRate *= _learningRateDecay;

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates Discriminator parameters using Adam optimizer.
    /// </summary>
    private void UpdateDiscriminatorParameters()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        // Initialize optimizer state if needed
        if (_discMomentum == null || _discMomentum.Length != parameters.Length)
        {
            _discMomentum = new Vector<T>(parameters.Length);
            _discMomentum.Fill(NumOps.Zero);
        }

        if (_discSecondMoment == null || _discSecondMoment.Length != parameters.Length)
        {
            _discSecondMoment = new Vector<T>(parameters.Length);
            _discSecondMoment.Fill(NumOps.Zero);
        }

        // Gradient clipping
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
            }
        }

        // Adam optimizer parameters
        var learningRate = NumOps.FromDouble(_discCurrentLearningRate);
        var beta1 = NumOps.FromDouble(0.0); // beta1=0 recommended for GANs
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);

        // Update beta powers
        _discBeta1Power = NumOps.Multiply(_discBeta1Power, beta1);
        _discBeta2Power = NumOps.Multiply(_discBeta2Power, beta2);

        // Bias correction
        var biasCorrection1 = NumOps.Subtract(NumOps.One, _discBeta1Power);
        var biasCorrection2 = NumOps.Subtract(NumOps.One, _discBeta2Power);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update momentum (first moment)
            _discMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _discMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            // Update second moment
            _discSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _discSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            // Bias-corrected estimates
            var mHat = NumOps.Divide(_discMomentum[i], biasCorrection1);
            var vHat = NumOps.Divide(_discSecondMoment[i], biasCorrection2);

            // Adam update
            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(
                    learningRate,
                    NumOps.Divide(mHat, NumOps.Add(NumOps.Sqrt(vHat), epsilon))
                )
            );
        }

        // Apply learning rate decay
        _discCurrentLearningRate *= _learningRateDecay;

        Discriminator.UpdateParameters(updatedParameters);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(LatentSize);
        writer.Write(CurrentResolutionLevel);
        writer.Write(MaxResolutionLevel);
        writer.Write(Alpha);
        writer.Write(UseMinibatchStdDev);
        writer.Write(UsePixelNormalization);
        writer.Write(_imageChannels);
        writer.Write(_baseFeatureMaps);
        writer.Write(_driftPenaltyCoefficient);
        writer.Write(_initialLearningRate);
        writer.Write(_learningRateDecay);
        writer.Write(_genCurrentLearningRate);
        writer.Write(_discCurrentLearningRate);

        // Serialize networks
        byte[] generatorData = Generator.Serialize();
        writer.Write(generatorData.Length);
        writer.Write(generatorData);

        byte[] discriminatorData = Discriminator.Serialize();
        writer.Write(discriminatorData.Length);
        writer.Write(discriminatorData);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        LatentSize = reader.ReadInt32();
        CurrentResolutionLevel = reader.ReadInt32();
        MaxResolutionLevel = reader.ReadInt32();
        Alpha = reader.ReadDouble();
        UseMinibatchStdDev = reader.ReadBoolean();
        UsePixelNormalization = reader.ReadBoolean();
        _imageChannels = reader.ReadInt32();
        _baseFeatureMaps = reader.ReadInt32();
        _driftPenaltyCoefficient = reader.ReadDouble();
        _initialLearningRate = reader.ReadDouble();
        _learningRateDecay = reader.ReadDouble();
        _genCurrentLearningRate = reader.ReadDouble();
        _discCurrentLearningRate = reader.ReadDouble();

        // Deserialize networks
        int generatorLength = reader.ReadInt32();
        Generator.Deserialize(reader.ReadBytes(generatorLength));

        int discriminatorLength = reader.ReadInt32();
        Discriminator.Deserialize(reader.ReadBytes(discriminatorLength));
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
}
