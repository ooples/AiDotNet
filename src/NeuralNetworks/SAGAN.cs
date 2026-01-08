using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Self-Attention GAN (SAGAN) implementation that uses self-attention mechanisms
/// to model long-range dependencies in generated images.
///
/// For Beginners:
/// Traditional CNNs in GANs only look at nearby pixels (local receptive fields).
/// This works well for textures and local patterns, but struggles with global
/// structure and long-range relationships (like making sure both eyes of a face
/// look similar, or ensuring consistent geometric patterns).
///
/// Self-Attention solves this by letting each pixel "attend to" all other pixels,
/// similar to how Transformers work in NLP. Think of it as:
/// - CNN: "I can only see my immediate neighbors"
/// - Self-Attention: "I can see the entire image and decide what's important"
///
/// Example: When generating a dog's face:
/// - CNN: Might make one ear pointy and one floppy (inconsistent)
/// - SAGAN: Notices both ears and makes them match (consistent)
///
/// Key innovations:
/// 1. Self-Attention Layers: Allow modeling of long-range dependencies
/// 2. Spectral Normalization: Stabilizes training for both G and D
/// 3. Hinge Loss: More stable than standard GAN loss
/// 4. Two Time-Scale Update Rule (TTUR): Different learning rates for G and D
/// 5. Conditional Batch Normalization: For class-conditional generation
///
/// Based on "Self-Attention Generative Adversarial Networks" by Zhang et al. (2019)
/// </summary>
/// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
public class SAGAN<T> : NeuralNetworkBase<T>
{
    private Vector<T> _momentum;
    private Vector<T> _secondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate;
    private double _initialLearningRate;
    private List<T> _generatorLosses = new List<T>();
    private List<T> _discriminatorLosses = new List<T>();

    /// <summary>
    /// Gets the generator network with self-attention layers.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network with self-attention layers.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    /// <summary>
    /// Gets the size of the latent vector (noise input).
    /// </summary>
    public int LatentSize { get; private set; }

    /// <summary>
    /// Gets the number of classes for conditional generation.
    /// Set to 0 for unconditional generation.
    /// </summary>
    public int NumClasses { get; private set; }

    /// <summary>
    /// Gets or sets whether to use spectral normalization.
    /// Spectral normalization stabilizes GAN training by constraining
    /// the Lipschitz constant of the discriminator.
    /// </summary>
    public bool UseSpectralNormalization { get; set; }

    /// <summary>
    /// Gets the positions where self-attention layers are inserted.
    /// Typically at mid-level feature maps (e.g., 32x32 or 64x64 resolution).
    /// </summary>
    public int[] AttentionLayers { get; private set; }

    private int _imageChannels;
    private int _imageHeight;
    private int _imageWidth;
    private int _generatorChannels;
    private int _discriminatorChannels;
    private bool _isConditional;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined SAGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateSAGANArchitecture(
        int latentSize,
        int numClasses,
        int imageChannels,
        int imageHeight,
        int imageWidth,
        InputType inputType)
    {
        int inputSize = latentSize + (numClasses > 0 ? 128 : 0);
        int outputSize = imageChannels * imageHeight * imageWidth;

        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: imageHeight,
                inputWidth: imageWidth,
                inputDepth: imageChannels,
                outputSize: outputSize,
                layers: null);
        }

        if (inputType == InputType.TwoDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: imageHeight,
                inputWidth: imageWidth,
                inputDepth: 1,
                outputSize: outputSize,
                layers: null);
        }

        // OneDimensional
        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: inputSize,
            outputSize: outputSize);
    }

    /// <summary>
    /// Initializes a new instance of Self-Attention GAN.
    /// </summary>
    /// <param name="generatorArchitecture">Architecture for the generator network.</param>
    /// <param name="discriminatorArchitecture">Architecture for the discriminator network.</param>
    /// <param name="latentSize">Size of the latent vector (typically 128)</param>
    /// <param name="imageChannels">Number of image channels (1 for grayscale, 3 for RGB)</param>
    /// <param name="imageHeight">Height of generated images</param>
    /// <param name="imageWidth">Width of generated images</param>
    /// <param name="numClasses">Number of classes (0 for unconditional)</param>
    /// <param name="generatorChannels">Base number of feature maps in generator (default 64)</param>
    /// <param name="discriminatorChannels">Base number of feature maps in discriminator (default 64)</param>
    /// <param name="attentionLayers">Indices of layers where self-attention is applied</param>
    /// <param name="inputType">The type of input.</param>
    /// <param name="lossFunction">Loss function for training (defaults to hinge loss)</param>
    /// <param name="initialLearningRate">Initial learning rate (default 0.0001)</param>
    public SAGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize = 128,
        int imageChannels = 3,
        int imageHeight = 64,
        int imageWidth = 64,
        int numClasses = 0,
        int generatorChannels = 64,
        int discriminatorChannels = 64,
        int[]? attentionLayers = null,
        InputType inputType = InputType.TwoDimensional,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0001)
        : base(CreateSAGANArchitecture(latentSize, numClasses, imageChannels, imageHeight, imageWidth, inputType),
               lossFunction ?? new HingeLoss<T>())
    {
        LatentSize = latentSize;
        NumClasses = numClasses;
        _isConditional = numClasses > 0;
        UseSpectralNormalization = true;
        _imageChannels = imageChannels;
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _generatorChannels = generatorChannels;
        _discriminatorChannels = discriminatorChannels;
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;

        // Default: apply self-attention at middle layers
        AttentionLayers = attentionLayers ?? [2, 3];

        // Initialize optimizer parameters
        // Beta powers start at beta^1 (the actual beta values) so first iteration's bias correction
        // computes (1 - beta) which is non-zero. SAGAN uses Adam with beta1=0.0, beta2=0.9
        _beta1Power = NumOps.Zero; // SAGAN uses beta1=0.0 for TTUR
        _beta2Power = NumOps.FromDouble(0.9);

        // Create generator and discriminator with self-attention
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
        // Note: _lossFunction is set via base class constructor (line 173)
        // Store reference for CreateNewInstance
        _lossFunction = LossFunction;

        InitializeLayers();
    }

    /// <summary>
    /// Generates images from random latent codes.
    /// </summary>
    /// <param name="numImages">Number of images to generate</param>
    /// <param name="classIndices">Optional class indices for conditional generation</param>
    /// <returns>Generated images tensor</returns>
    public Tensor<T> Generate(int numImages, int[]? classIndices = null)
    {
        if (_isConditional && classIndices == null)
        {
            throw new ArgumentException("Class indices required for conditional generation");
        }

        if (classIndices != null && classIndices.Length != numImages)
        {
            throw new ArgumentException("Number of class indices must match number of images");
        }

        Generator.SetTrainingMode(false);
        var noise = GenerateNoise(numImages);

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

        if (_isConditional && classIndices != null)
        {
            // Concatenate class information (simplified)
            var classEmbeddings = CreateClassEmbeddings(classIndices);
            var input = ConcatenateTensors(reshapedNoise, classEmbeddings);
            return Generator.Predict(input);
        }

        return Generator.Predict(reshapedNoise);
    }

    /// <summary>
    /// Generates images from specific latent codes.
    /// </summary>
    /// <param name="latentCodes">Latent codes to use</param>
    /// <param name="classIndices">Optional class indices for conditional generation</param>
    /// <returns>Generated images tensor</returns>
    public Tensor<T> Generate(Tensor<T> latentCodes, int[]? classIndices = null)
    {
        Generator.SetTrainingMode(false);

        // Reshape latent codes to 3D/4D format for CNN generator
        Tensor<T> reshapedLatent;
        if (latentCodes.Shape.Length == 1)
        {
            // 1D [latent_size] -> 3D [1, height, width] where height*width >= latent_size
            int latentLen = latentCodes.Shape[0];
            int h = (int)Math.Ceiling(Math.Sqrt(latentLen));
            int w = h;
            int padSize = h * w - latentLen;
            if (padSize > 0)
            {
                // Pad latent code to fit h*w
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

        if (_isConditional && classIndices != null)
        {
            var classEmbeddings = CreateClassEmbeddings(classIndices);
            var input = ConcatenateTensors(reshapedLatent, classEmbeddings);
            return Generator.Predict(input);
        }

        return Generator.Predict(reshapedLatent);
    }

    /// <summary>
    /// Generates random noise from a standard normal distribution.
    /// </summary>
    /// <remarks>
    /// Uses vectorized Engine.GenerateGaussianNoise for CPU/GPU accelerated generation.
    /// </remarks>
    private Tensor<T> GenerateNoise(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");

        var totalElements = batchSize * LatentSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;

        // Vectorized Gaussian noise generation using Engine (SIMD/GPU accelerated)
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        return Tensor<T>.FromVector(noiseVector, [batchSize, LatentSize]);
    }

    /// <summary>
    /// Creates class embeddings for conditional generation.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when called in unconditional mode (NumClasses=0)</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a class index is out of valid range</exception>
    private Tensor<T> CreateClassEmbeddings(int[] classIndices)
    {
        // Validate that we're in conditional mode
        if (NumClasses <= 0)
        {
            throw new InvalidOperationException(
                "Cannot create class embeddings in unconditional mode (NumClasses=0). " +
                "Either set NumClasses > 0 in constructor or don't pass class indices.");
        }

        // One-hot embedding dimension must equal NumClasses to avoid overflow
        // when classIdx >= previous hardcoded value (128)
        var embeddingDim = NumClasses;
        var embeddings = new Tensor<T>([classIndices.Length, embeddingDim]);

        // Validate and create one-hot encoding
        for (int i = 0; i < classIndices.Length; i++)
        {
            var classIdx = classIndices[i];

            // Validate class index is within valid range
            if (classIdx < 0 || classIdx >= NumClasses)
            {
                throw new ArgumentOutOfRangeException(nameof(classIndices),
                    $"Class index {classIdx} at position {i} is out of range. Valid range: 0 to {NumClasses - 1}.");
            }

            // Set one-hot encoding at the class index position within the embedding
            embeddings.SetFlat(i * embeddingDim + classIdx, NumOps.One);
        }

        return embeddings;
    }

    /// <summary>
    /// Concatenates two tensors along the feature dimension.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        if (a.Shape[0] != b.Shape[0])
        {
            throw new ArgumentException("Batch sizes must match");
        }

        var batchSize = a.Shape[0];
        var aFeatures = a.Length / batchSize;
        var bFeatures = b.Length / batchSize;
        var totalFeatures = aFeatures + bFeatures;

        var result = new Tensor<T>([batchSize, totalFeatures]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < aFeatures; j++)
            {
                result.SetFlat(i * totalFeatures + j, a.GetFlat(i * aFeatures + j));
            }
            for (int j = 0; j < bFeatures; j++)
            {
                result.SetFlat(i * totalFeatures + aFeatures + j, b.GetFlat(i * bFeatures + j));
            }
        }

        return result;
    }

    /// <summary>
    /// Performs a single training step on a batch of real images.
    /// Uses hinge loss for improved stability.
    /// </summary>
    /// <param name="realImages">Batch of real images</param>
    /// <param name="batchSize">Number of images in the batch</param>
    /// <param name="realLabels">Optional class labels for conditional training</param>
    /// <returns>Tuple of (discriminator loss, generator loss)</returns>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        int batchSize,
        int[]? realLabels = null)
    {
        var one = NumOps.One;

        // === Train Discriminator ===
        Discriminator.SetTrainingMode(true);
        Generator.SetTrainingMode(false);

        // Real images - hinge loss: max(0, 1 - D(x_real))
        var realOutput = Discriminator.Predict(realImages);
        var realLoss = CalculateHingeLoss(realOutput, true, batchSize);

        // Fake images - hinge loss: max(0, 1 + D(G(z)))
        var noise = GenerateNoise(batchSize);
        Tensor<T> fakeImages = (_isConditional && realLabels != null)
            ? Generate(noise, realLabels)
            : Generate(noise);

        var fakeOutput = Discriminator.Predict(fakeImages);
        var fakeLoss = CalculateHingeLoss(fakeOutput, false, batchSize);

        var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        _discriminatorLosses.Add(discriminatorLoss);

        // Backpropagate discriminator
        var discGradient = new Tensor<T>([1]);
        discGradient.SetFlat(0, one);
        Discriminator.Backward(discGradient);

        // Update discriminator parameters using Adam with TTUR
        // SAGAN uses higher LR for discriminator (4x generator LR)
        ApplyAdamUpdate(Discriminator, _currentLearningRate * 4.0, isGenerator: false);

        // === Train Generator ===
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);

        var generatorNoise = GenerateNoise(batchSize);
        Tensor<T> generatedImages = (_isConditional && realLabels != null)
            ? Generate(generatorNoise, realLabels)
            : Generate(generatorNoise);

        var generatorOutput = Discriminator.Predict(generatedImages);

        // Generator loss: -D(G(z)) (hinge loss for generator)
        var generatorLoss = NumOps.Zero;
        for (int i = 0; i < generatorOutput.Length; i++)
        {
            generatorLoss = NumOps.Subtract(generatorLoss, generatorOutput.GetFlat(i));
        }
        generatorLoss = NumOps.Divide(generatorLoss, NumOps.FromDouble(batchSize));
        _generatorLosses.Add(generatorLoss);

        // Backpropagate generator
        var genGradient = new Tensor<T>([1]);
        genGradient.SetFlat(0, one);
        Generator.Backward(genGradient);

        // Update generator parameters using Adam with TTUR
        // SAGAN uses lower LR for generator (base LR)
        ApplyAdamUpdate(Generator, _currentLearningRate, isGenerator: true);

        // Update beta powers for next iteration (bias correction)
        // SAGAN uses beta1=0.0, beta2=0.9
        _beta2Power = NumOps.Multiply(_beta2Power, NumOps.FromDouble(0.9));

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Calculates hinge loss for discriminator training.
    /// Real: max(0, 1 - output)
    /// Fake: max(0, 1 + output)
    /// </summary>
    private T CalculateHingeLoss(Tensor<T> output, bool isReal, int batchSize)
    {
        var loss = NumOps.Zero;
        var one = NumOps.One;

        for (int i = 0; i < output.Length; i++)
        {
            T hingeLoss;
            if (isReal)
            {
                var margin = NumOps.Subtract(one, output.GetFlat(i));
                hingeLoss = NumOps.GreaterThan(margin, NumOps.Zero) ? margin : NumOps.Zero;
            }
            else
            {
                var margin = NumOps.Add(one, output.GetFlat(i));
                hingeLoss = NumOps.GreaterThan(margin, NumOps.Zero) ? margin : NumOps.Zero;
            }

            loss = NumOps.Add(loss, hingeLoss);
        }

        return NumOps.Divide(loss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Applies Adam optimizer update to network parameters using TTUR (Two Time-Scale Update Rule).
    /// </summary>
    /// <remarks>
    /// <para>
    /// SAGAN uses Adam with specific hyperparameters for stable GAN training:
    /// - beta1 = 0.0 (no momentum on gradients)
    /// - beta2 = 0.9 (momentum on squared gradients)
    /// - Different learning rates for G and D (TTUR)
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the network weights based on computed gradients.
    /// Adam adapts the learning rate for each parameter individually based on the history
    /// of gradients, which helps training converge faster and more stably.
    /// </para>
    /// </remarks>
    /// <param name="network">The network whose parameters to update</param>
    /// <param name="learningRate">Learning rate for this update</param>
    /// <param name="isGenerator">True for generator (uses first half of momentum arrays)</param>
    private void ApplyAdamUpdate(ConvolutionalNeuralNetwork<T> network, double learningRate, bool isGenerator)
    {
        var parameters = network.GetParameters();
        var gradients = network.GetGradients();

        if (parameters.Length == 0 || gradients.Length == 0)
            return;

        // Offset in momentum arrays: generator uses first half, discriminator uses second half
        int offset = isGenerator ? 0 : Generator.GetParameterCount();
        int paramCount = parameters.Length;

        // Ensure momentum arrays are properly sized
        if (_momentum.Length < offset + paramCount || _secondMoment.Length < offset + paramCount)
        {
            var newMomentum = new Vector<T>(offset + paramCount);
            var newSecondMoment = new Vector<T>(offset + paramCount);
            for (int i = 0; i < _momentum.Length && i < newMomentum.Length; i++)
            {
                newMomentum[i] = _momentum[i];
                newSecondMoment[i] = _secondMoment[i];
            }
            _momentum = newMomentum;
            _secondMoment = newSecondMoment;
        }

        // Adam hyperparameters for SAGAN
        T beta1 = NumOps.Zero;  // SAGAN uses beta1=0.0
        T beta2 = NumOps.FromDouble(0.9);
        T epsilon = NumOps.FromDouble(1e-8);
        T lr = NumOps.FromDouble(learningRate);
        T one = NumOps.One;

        // Bias correction terms
        T beta2Correction = NumOps.Subtract(one, _beta2Power);
        // Guard against zero/negative correction (use epsilon fallback)
        T correctionThreshold = NumOps.FromDouble(1e-8);
        if (NumOps.LessThan(beta2Correction, correctionThreshold))
            beta2Correction = correctionThreshold;

        var updatedParams = new Vector<T>(paramCount);

        for (int i = 0; i < paramCount; i++)
        {
            int momIdx = offset + i;
            T gradient = gradients[i];

            // Update biased first moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            // With beta1=0, this simplifies to: m_t = g_t
            T m = NumOps.Add(
                NumOps.Multiply(beta1, _momentum[momIdx]),
                NumOps.Multiply(NumOps.Subtract(one, beta1), gradient));
            _momentum[momIdx] = m;

            // Update biased second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            T gradSquared = NumOps.Multiply(gradient, gradient);
            T v = NumOps.Add(
                NumOps.Multiply(beta2, _secondMoment[momIdx]),
                NumOps.Multiply(NumOps.Subtract(one, beta2), gradSquared));
            _secondMoment[momIdx] = v;

            // Bias-corrected second moment: v_hat = v_t / (1 - beta2^t)
            T vHat = NumOps.Divide(v, beta2Correction);

            // Parameter update: theta = theta - lr * m / (sqrt(v_hat) + epsilon)
            // Note: With beta1=0, m = g_t, so no bias correction needed for m
            T sqrtV = NumOps.Sqrt(vHat);
            T denominator = NumOps.Add(sqrtV, epsilon);
            T update = NumOps.Multiply(lr, NumOps.Divide(m, denominator));
            updatedParams[i] = NumOps.Subtract(parameters[i], update);
        }

        network.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Hinge loss implementation for the loss function interface.
    /// </summary>
    private class HingeLoss<TLoss> : ILossFunction<TLoss>
    {
        private static readonly INumericOperations<TLoss> _ops = MathHelper.GetNumericOperations<TLoss>();

        public TLoss CalculateLoss(Vector<TLoss> predicted, Vector<TLoss> actual)
        {
            var loss = _ops.Zero;
            var one = _ops.One;

            for (int i = 0; i < predicted.Length; i++)
            {
                var yt = _ops.Multiply(actual[i], predicted[i]);
                var margin = _ops.Subtract(one, yt);
                if (_ops.GreaterThan(margin, _ops.Zero))
                {
                    loss = _ops.Add(loss, margin);
                }
            }

            return _ops.Divide(loss, _ops.FromDouble(predicted.Length));
        }

        public Vector<TLoss> CalculateDerivative(Vector<TLoss> predicted, Vector<TLoss> actual)
        {
            var gradient = new Vector<TLoss>(predicted.Length);
            var one = _ops.One;

            for (int i = 0; i < predicted.Length; i++)
            {
                var yt = _ops.Multiply(actual[i], predicted[i]);
                var margin = _ops.Subtract(one, yt);
                if (_ops.GreaterThan(margin, _ops.Zero))
                {
                    gradient[i] = _ops.Negate(_ops.Divide(actual[i], _ops.FromDouble(predicted.Length)));
                }
                else
                {
                    gradient[i] = _ops.Zero;
                }
            }

            return gradient;
        }

        public (TLoss Loss, IGpuTensor<TLoss> Gradient) CalculateLossAndGradientGpu(IGpuTensor<TLoss> predicted, IGpuTensor<TLoss> actual)
        {
            // Fall back to CPU for now
            var predictedCpu = predicted.ToTensor();
            var actualCpu = actual.ToTensor();
            
            var loss = CalculateLoss(predictedCpu.ToVector(), actualCpu.ToVector());
            var gradientCpu = CalculateDerivative(predictedCpu.ToVector(), actualCpu.ToVector());
            
            var gradientTensor = new Tensor<TLoss>(predictedCpu.Shape);
            Array.Copy(gradientCpu.ToArray(), gradientTensor.Data, gradientCpu.Length);
            
            var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
            var backend = engine?.Backend ?? throw new InvalidOperationException("GPU backend not available");
            var gradientGpu = new GpuTensor<TLoss>(backend, gradientTensor, GpuTensorRole.Gradient);
            
            return (loss, gradientGpu);
        }
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the SAGAN.
    /// </summary>
    /// <remarks>
    /// This includes all parameters from both the Generator and Discriminator networks.
    /// </remarks>
    public override int ParameterCount => Generator.GetParameterCount() + Discriminator.GetParameterCount();

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
            Name = "SAGAN",
            Version = "1.0"
        };

        metadata.SetProperty("ModelType", "SAGAN");
        metadata.SetProperty("LatentSize", LatentSize);
        metadata.SetProperty("NumClasses", NumClasses);
        metadata.SetProperty("IsConditional", _isConditional);
        metadata.SetProperty("ImageChannels", _imageChannels);
        metadata.SetProperty("ImageHeight", _imageHeight);
        metadata.SetProperty("ImageWidth", _imageWidth);
        metadata.SetProperty("GeneratorChannels", _generatorChannels);
        metadata.SetProperty("DiscriminatorChannels", _discriminatorChannels);
        metadata.SetProperty("UseSpectralNormalization", UseSpectralNormalization);
        metadata.SetProperty("AttentionLayers", string.Join(",", AttentionLayers));

        return metadata;
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in the constructor via the Generator and Discriminator CNNs
        var paramCount = Generator.GetParameterCount() + Discriminator.GetParameterCount();
        _momentum = new Vector<T>(paramCount);
        _secondMoment = new Vector<T>(paramCount);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(LatentSize);
        writer.Write(NumClasses);
        writer.Write(_isConditional);
        writer.Write(_imageChannels);
        writer.Write(_imageHeight);
        writer.Write(_imageWidth);
        writer.Write(_generatorChannels);
        writer.Write(_discriminatorChannels);
        writer.Write(UseSpectralNormalization);
        writer.Write(AttentionLayers.Length);
        foreach (var layer in AttentionLayers)
        {
            writer.Write(layer);
        }

        // Serialize optimizer state
        writer.Write(_initialLearningRate);
        writer.Write(_currentLearningRate);
        SerializationHelper<T>.SerializeVector(writer, _momentum);
        SerializationHelper<T>.SerializeVector(writer, _secondMoment);
        writer.Write(NumOps.ToDouble(_beta1Power));
        writer.Write(NumOps.ToDouble(_beta2Power));

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
        NumClasses = reader.ReadInt32();
        _isConditional = reader.ReadBoolean();
        _imageChannels = reader.ReadInt32();
        _imageHeight = reader.ReadInt32();
        _imageWidth = reader.ReadInt32();
        _generatorChannels = reader.ReadInt32();
        _discriminatorChannels = reader.ReadInt32();
        UseSpectralNormalization = reader.ReadBoolean();
        int attentionLayerCount = reader.ReadInt32();
        AttentionLayers = new int[attentionLayerCount];
        for (int i = 0; i < attentionLayerCount; i++)
        {
            AttentionLayers[i] = reader.ReadInt32();
        }

        // Deserialize optimizer state
        _initialLearningRate = reader.ReadDouble();
        _currentLearningRate = reader.ReadDouble();
        _momentum = SerializationHelper<T>.DeserializeVector(reader);
        _secondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _beta1Power = NumOps.FromDouble(reader.ReadDouble());
        _beta2Power = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize networks
        int generatorLength = reader.ReadInt32();
        Generator.Deserialize(reader.ReadBytes(generatorLength));

        int discriminatorLength = reader.ReadInt32();
        Discriminator.Deserialize(reader.ReadBytes(discriminatorLength));
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SAGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            LatentSize,
            _imageChannels,
            _imageHeight,
            _imageWidth,
            NumClasses,
            _generatorChannels,
            _discriminatorChannels,
            AttentionLayers,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate);
    }
}
