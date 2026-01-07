using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// BigGAN implementation for large-scale high-fidelity image generation.
///
/// For Beginners:
/// BigGAN is a state-of-the-art GAN architecture that generates extremely high-quality
/// images by scaling up training in several ways:
/// 1. Using very large batch sizes (256-2048 images at once)
/// 2. Increasing model capacity (more parameters and feature maps)
/// 3. Using class information to generate specific types of images
///
/// Think of it like training an artist:
/// - Small batch = showing the artist 1-2 examples at a time
/// - BigGAN batch = showing 256+ examples at once for better learning
/// - Class conditioning = telling the artist exactly what to draw ("draw a cat" vs "draw something")
///
/// Key innovations:
/// 1. Large Batch Training: Uses batch sizes of 256-2048 (vs typical 32-128)
/// 2. Spectral Normalization: Stabilizes training for both G and D
/// 3. Self-Attention: Helps model long-range dependencies in images
/// 4. Class Conditioning: Uses class embeddings for controlled generation
/// 5. Truncation Trick: Trade diversity for quality at generation time
/// 6. Orthogonal Initialization: Better weight initialization
/// 7. Skip Connections: Direct paths in generator architecture
///
/// Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
/// by Brock et al. (2019)
/// </summary>
/// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
public class BigGAN<T> : NeuralNetworkBase<T>
{
    // Generator optimizer state
    private Vector<T> _genMomentum;
    private Vector<T> _genSecondMoment;
    private T _genBeta1Power;
    private T _genBeta2Power;

    // Discriminator optimizer state
    private Vector<T> _discMomentum;
    private Vector<T> _discSecondMoment;
    private T _discBeta1Power;
    private T _discBeta2Power;

    private double _currentLearningRate;
    private double _initialLearningRate;
    private List<T> _generatorLosses = new List<T>();
    private List<T> _discriminatorLosses = new List<T>();

    /// <summary>
    /// Gets the generator network that produces images from noise and class labels.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network that evaluates images and predicts their class.
    /// Uses projection discriminator for efficient class conditioning.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    /// <summary>
    /// Gets the size of the latent noise vector.
    /// BigGAN typically uses 120-dimensional latent codes.
    /// </summary>
    public int LatentSize { get; private set; }

    /// <summary>
    /// Gets the number of classes for conditional generation.
    /// For example, ImageNet has 1000 classes.
    /// </summary>
    public int NumClasses { get; private set; }

    /// <summary>
    /// Gets the dimension of class embeddings.
    /// These learned embeddings represent each class.
    /// </summary>
    public int ClassEmbeddingDim { get; private set; }

    /// <summary>
    /// Gets or sets the truncation threshold for the truncation trick.
    /// Values in range [0, 2], where lower values trade diversity for quality.
    /// Typical value: 0.5 for high quality, 1.0 for balanced, 2.0 for high diversity.
    /// </summary>
    public double TruncationThreshold { get; set; }

    /// <summary>
    /// Gets or sets whether to use the truncation trick during generation.
    /// When enabled, samples are resampled if they fall outside the truncation threshold.
    /// </summary>
    public bool UseTruncation { get; set; }

    /// <summary>
    /// Gets or sets whether to use spectral normalization in both generator and discriminator.
    /// </summary>
    public bool UseSpectralNormalization { get; set; }

    /// <summary>
    /// Gets or sets whether to use self-attention layers.
    /// </summary>
    public bool UseSelfAttention { get; set; }

    private Matrix<T> _classEmbeddings;
    private Matrix<T> _discClassProjection;
    private int _imageChannels;
    private int _imageHeight;
    private int _imageWidth;
    private int _generatorChannels;
    private int _discriminatorChannels;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of BigGAN.
    /// </summary>
    /// <param name="generatorArchitecture">Architecture for the generator network.</param>
    /// <param name="discriminatorArchitecture">Architecture for the discriminator network.</param>
    /// <param name="latentSize">Size of the latent noise vector (default 120)</param>
    /// <param name="numClasses">Number of classes for conditional generation</param>
    /// <param name="classEmbeddingDim">Dimension of class embeddings (default 128)</param>
    /// <param name="imageChannels">Number of image channels (1 for grayscale, 3 for RGB)</param>
    /// <param name="imageHeight">Height of generated images</param>
    /// <param name="imageWidth">Width of generated images</param>
    /// <param name="generatorChannels">Base number of channels in generator (default 96)</param>
    /// <param name="discriminatorChannels">Base number of channels in discriminator (default 96)</param>
    /// <param name="inputType">The type of input.</param>
    /// <param name="lossFunction">Loss function for training (defaults to hinge loss)</param>
    /// <param name="initialLearningRate">Initial learning rate (default 0.0001)</param>
    public BigGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize = 120,
        int numClasses = 1000,
        int classEmbeddingDim = 128,
        int imageChannels = 3,
        int imageHeight = 128,
        int imageWidth = 128,
        int generatorChannels = 96,
        int discriminatorChannels = 96,
        InputType inputType = InputType.TwoDimensional,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0001)
        : base(new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,  // Base GAN takes latent vector input (1D)
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            latentSize + classEmbeddingDim,
            0, 0, 1,  // inputHeight, inputWidth=0 for 1D, inputDepth=1 required
            imageChannels * imageHeight * imageWidth,  // outputSize
            null), lossFunction ?? new HingeLoss<T>())
    {
        // Input validation
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }

        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }

        if (latentSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(latentSize), latentSize, "Latent size must be positive.");
        }

        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "Number of classes must be positive.");
        }

        if (classEmbeddingDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(classEmbeddingDim), classEmbeddingDim, "Class embedding dimension must be positive.");
        }

        if (imageChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageChannels), imageChannels, "Image channels must be positive.");
        }

        if (imageHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageHeight), imageHeight, "Image height must be positive.");
        }

        if (imageWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageWidth), imageWidth, "Image width must be positive.");
        }

        if (generatorChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(generatorChannels), generatorChannels, "Generator channels must be positive.");
        }

        if (discriminatorChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(discriminatorChannels), discriminatorChannels, "Discriminator channels must be positive.");
        }

        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }

        LatentSize = latentSize;
        NumClasses = numClasses;
        ClassEmbeddingDim = classEmbeddingDim;
        TruncationThreshold = 1.0;
        UseTruncation = false;
        UseSpectralNormalization = true;
        UseSelfAttention = true;
        _imageChannels = imageChannels;
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _generatorChannels = generatorChannels;
        _discriminatorChannels = discriminatorChannels;
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;

        // Initialize generator optimizer state
        _genBeta1Power = NumOps.One;
        _genBeta2Power = NumOps.One;
        _genMomentum = Vector<T>.Empty();
        _genSecondMoment = Vector<T>.Empty();

        // Initialize discriminator optimizer state
        _discBeta1Power = NumOps.One;
        _discBeta2Power = NumOps.One;
        _discMomentum = Vector<T>.Empty();
        _discSecondMoment = Vector<T>.Empty();

        // Initialize class embeddings with orthogonal initialization
        _classEmbeddings = InitializeClassEmbeddings();

        // Initialize discriminator class projection for projection discriminator
        _discClassProjection = InitializeDiscriminatorProjection();

        // Create generator and discriminator
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _lossFunction = lossFunction ?? new HingeLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes class embeddings using scaled uniform random initialization.
    /// Uses small random values centered at zero for stability.
    /// </summary>
    private Matrix<T> InitializeClassEmbeddings()
    {
        var embeddings = new Matrix<T>(NumClasses, ClassEmbeddingDim);

        // Use scaled uniform random initialization
        // Scale factor follows Xavier initialization: sqrt(6 / (fan_in + fan_out))
        double scale = Math.Sqrt(6.0 / (NumClasses + ClassEmbeddingDim));
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                embeddings[i, j] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * scale);
            }
        }

        return embeddings;
    }

    /// <summary>
    /// Initializes the discriminator class projection matrix for the projection discriminator.
    /// This matrix projects class embeddings to the discriminator feature space.
    /// </summary>
    private Matrix<T> InitializeDiscriminatorProjection()
    {
        // The projection maps from class embedding dimension to discriminator output dimension
        // For BigGAN, we project to a single scalar output dimension
        var projection = new Matrix<T>(NumClasses, ClassEmbeddingDim);

        // Use scaled uniform random initialization
        double scale = Math.Sqrt(6.0 / (NumClasses + ClassEmbeddingDim));
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                projection[i, j] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * scale);
            }
        }

        return projection;
    }

    /// <summary>
    /// Gets the discriminator output with class conditioning using projection discriminator pattern.
    /// The final score is: D(x) + y^T * V * embed(y)
    /// where V is the class projection matrix and embed(y) is the class embedding.
    /// </summary>
    /// <param name="images">Input images tensor</param>
    /// <param name="classIndices">Class labels for each image</param>
    /// <returns>Discriminator scores conditioned on class labels</returns>
    private Tensor<T> DiscriminatorPredictWithLabels(Tensor<T> images, int[] classIndices)
    {
        // Get base discriminator output (unconditional score)
        var baseOutput = Discriminator.Predict(images);
        int batchSize = classIndices.Length;

        // Apply projection discriminator: add inner product of class embedding and projection
        for (int i = 0; i < batchSize; i++)
        {
            int classIdx = classIndices[i];

            // Compute projection term: embed(y)^T * V_y
            // This adds a class-conditional bias to the discriminator output
            T projectionTerm = NumOps.Zero;
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                T embedding = _classEmbeddings[classIdx, j];
                T projection = _discClassProjection[classIdx, j];
                projectionTerm = NumOps.Add(projectionTerm, NumOps.Multiply(embedding, projection));
            }

            // Add projection term to discriminator output
            baseOutput.SetFlat(i, NumOps.Add(baseOutput.GetFlat(i), projectionTerm));
        }

        return baseOutput;
    }

    /// <summary>
    /// Gets the class embedding for a specific class index.
    /// </summary>
    private Vector<T> GetClassEmbedding(int classIndex)
    {
        // Vectorized row extraction using Engine
        return Engine.GetRow(_classEmbeddings, classIndex);
    }

    /// <summary>
    /// Applies the truncation trick to latent codes.
    /// Resamples values that fall outside the threshold.
    /// </summary>
    private Tensor<T> ApplyTruncation(Tensor<T> latentCodes)
    {
        if (!UseTruncation)
        {
            return latentCodes;
        }

        var truncated = new Tensor<T>(latentCodes.Shape);
        var threshold = NumOps.FromDouble(TruncationThreshold);

        for (int i = 0; i < latentCodes.Length; i++)
        {
            var value = latentCodes.GetFlat(i);
            var absValue = NumOps.Abs(value);

            // If absolute value exceeds threshold, resample
            if (NumOps.GreaterThan(absValue, threshold))
            {
                // Resample until within threshold
                do
                {
                    value = NumOps.FromDouble(Random.NextDouble() * 2.0 - 1.0);
                    value = NumOps.Multiply(value, NumOps.FromDouble(2.0)); // Scaled Gaussian approximation
                    absValue = NumOps.Abs(value);
                } while (NumOps.GreaterThan(absValue, threshold));
            }

            truncated[i] = value;
        }

        return truncated;
    }

    /// <summary>
    /// Generates images from latent codes and class labels.
    /// </summary>
    /// <param name="latentCodes">Latent noise vectors</param>
    /// <param name="classIndices">Class indices for each sample (must be in range [0, NumClasses))</param>
    /// <returns>Generated images</returns>
    /// <exception cref="ArgumentException">Thrown when class indices don't match batch size or are out of range.</exception>
    public Tensor<T> Generate(Tensor<T> latentCodes, int[] classIndices)
    {
        if (classIndices.Length != latentCodes.Shape[0])
        {
            throw new ArgumentException(
                $"Number of class indices ({classIndices.Length}) must match batch size ({latentCodes.Shape[0]})",
                nameof(classIndices));
        }

        // Validate class indices are in valid range
        for (int i = 0; i < classIndices.Length; i++)
        {
            if (classIndices[i] < 0 || classIndices[i] >= NumClasses)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(classIndices),
                    $"Class index {classIndices[i]} at position {i} is out of range. Must be in [0, {NumClasses}).");
            }
        }

        // Validate latent code dimensions
        int expectedLatentSize = latentCodes.Length / latentCodes.Shape[0];
        if (expectedLatentSize != LatentSize)
        {
            throw new ArgumentException(
                $"Latent code dimension ({expectedLatentSize}) must match LatentSize ({LatentSize})",
                nameof(latentCodes));
        }

        Generator.SetTrainingMode(false);

        // Apply truncation if enabled
        var truncatedCodes = ApplyTruncation(latentCodes);

        // Vectorized batch embedding lookup using Engine
        // Convert Matrix to Tensor for embedding lookup
        var embeddingsTensor = new Tensor<T>([NumClasses, ClassEmbeddingDim]);
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                embeddingsTensor[i, j] = _classEmbeddings[i, j];
            }
        }
        var indicesTensor = new Tensor<int>(classIndices, [classIndices.Length]);
        var classEmbeddings = Engine.TensorEmbeddingLookup(embeddingsTensor, indicesTensor);

        // Concatenate latent codes and class embeddings
        var input = ConcatenateTensors(truncatedCodes, classEmbeddings);

        // Reshape input to 3D/4D format for CNN generator
        Tensor<T> reshapedInput;
        if (input.Shape.Length == 1)
        {
            // 1D [total_size] -> 3D [1, height, width]
            int totalLen = input.Shape[0];
            int h = (int)Math.Ceiling(Math.Sqrt(totalLen));
            int w = h;
            int padded = h * w;
            if (padded > totalLen)
            {
                var paddedData = new T[padded];
                Array.Copy(input.Data, paddedData, totalLen);
                reshapedInput = new Tensor<T>(paddedData, [1, h, w]);
            }
            else
            {
                reshapedInput = input.Reshape([1, h, w]);
            }
        }
        else if (input.Shape.Length == 2)
        {
            // 2D [batch, total_size] -> 4D [batch, 1, height, width]
            int batch = input.Shape[0];
            int latentLen = input.Shape[1];
            int h = (int)Math.Ceiling(Math.Sqrt(latentLen));
            int w = h;
            int padded = h * w;
            if (padded > latentLen)
            {
                var paddedData = new T[batch * padded];
                for (int b = 0; b < batch; b++)
                {
                    for (int j = 0; j < latentLen; j++)
                    {
                        paddedData[b * padded + j] = input[b, j];
                    }
                }
                reshapedInput = new Tensor<T>(paddedData, [batch, 1, h, w]);
            }
            else
            {
                reshapedInput = input.Reshape([batch, 1, h, w]);
            }
        }
        else
        {
            // Already 3D or higher
            reshapedInput = input;
        }

        return Generator.Predict(reshapedInput);
    }

    /// <summary>
    /// Generates random images with random class labels.
    /// </summary>
    /// <param name="numImages">Number of images to generate</param>
    /// <returns>Generated images</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when numImages is not positive.</exception>
    public Tensor<T> Generate(int numImages)
    {
        if (numImages <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numImages), numImages, "Number of images must be positive.");
        }

        var noise = GenerateGaussianNoise(numImages);
        var classIndices = new int[numImages];

        for (int i = 0; i < numImages; i++)
        {
            classIndices[i] = Random.Next(NumClasses);
        }

        return Generate(noise, classIndices);
    }

    /// <summary>
    /// Generates random noise for the generator input using vectorized Gaussian noise generation.
    /// Uses Engine.GenerateGaussianNoise for SIMD/GPU acceleration.
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
            // Copy from tensor a
            for (int j = 0; j < aFeatures; j++)
            {
                result.SetFlat(i * totalFeatures + j, a.GetFlat(i * aFeatures + j));
            }
            // Copy from tensor b
            for (int j = 0; j < bFeatures; j++)
            {
                result.SetFlat(i * totalFeatures + aFeatures + j, b.GetFlat(i * bFeatures + j));
            }
        }

        return result;
    }

    /// <summary>
    /// Performs a single training step on a batch of real images with labels.
    /// Uses hinge loss by default for improved stability.
    /// </summary>
    /// <param name="realImages">Batch of real images</param>
    /// <param name="realLabels">Class labels for real images</param>
    /// <param name="batchSize">Number of images in the batch</param>
    /// <returns>Tuple of (discriminator loss, generator loss)</returns>
    /// <exception cref="ArgumentNullException">Thrown when realImages or realLabels is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is not positive.</exception>
    /// <exception cref="ArgumentException">Thrown when array lengths don't match or labels are out of range.</exception>
    public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, int[] realLabels, int batchSize)
    {
        // Input validation
        if (realImages is null)
        {
            throw new ArgumentNullException(nameof(realImages), "Real images tensor cannot be null.");
        }

        if (realLabels is null)
        {
            throw new ArgumentNullException(nameof(realLabels), "Real labels array cannot be null.");
        }

        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        }

        if (realLabels.Length != batchSize)
        {
            throw new ArgumentException(
                $"Number of labels ({realLabels.Length}) must match batch size ({batchSize}).",
                nameof(realLabels));
        }

        // Validate class indices are in valid range
        for (int i = 0; i < realLabels.Length; i++)
        {
            if (realLabels[i] < 0 || realLabels[i] >= NumClasses)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(realLabels),
                    $"Class index {realLabels[i]} at position {i} is out of range. Must be in [0, {NumClasses}).");
            }
        }

        // === Train Discriminator ===
        Discriminator.SetTrainingMode(true);
        Generator.SetTrainingMode(false);

        // Real images - maximize D(real) with class conditioning
        var realOutput = DiscriminatorPredictWithLabels(realImages, realLabels);
        var realLoss = CalculateHingeLoss(realOutput, true, batchSize);

        // Compute gradients for real images: dL/d(output) for hinge loss
        var realGradients = CalculateHingeLossGradients(realOutput, true, batchSize);
        Discriminator.Backward(realGradients);

        // Fake images - minimize D(fake)
        var noise = GenerateGaussianNoise(batchSize);
        var fakeLabels = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            fakeLabels[i] = Random.Next(NumClasses);
        }

        var fakeImages = Generate(noise, fakeLabels);
        var fakeOutput = DiscriminatorPredictWithLabels(fakeImages, fakeLabels);
        var fakeLoss = CalculateHingeLoss(fakeOutput, false, batchSize);

        // Compute gradients for fake images
        var fakeGradients = CalculateHingeLossGradients(fakeOutput, false, batchSize);
        Discriminator.Backward(fakeGradients);

        var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        _discriminatorLosses.Add(discriminatorLoss);

        // Update discriminator parameters
        UpdateDiscriminatorParameters();

        // === Train Generator ===
        Generator.SetTrainingMode(true);
        // Keep Discriminator in training mode - required for backpropagation
        // We just don't call UpdateDiscriminatorParameters() during generator training

        var generatorNoise = GenerateGaussianNoise(batchSize);
        var generatorLabels = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            generatorLabels[i] = Random.Next(NumClasses);
        }

        // Generate images and get discriminator output with class conditioning
        var generatedImages = GenerateWithGradients(generatorNoise, generatorLabels);
        var generatorOutput = DiscriminatorPredictWithLabels(generatedImages, generatorLabels);
        var generatorLoss = CalculateHingeLoss(generatorOutput, true, batchSize);
        _generatorLosses.Add(generatorLoss);

        // Compute gradient of generator loss w.r.t. discriminator output
        // For generator, we want to maximize D(fake), so loss = -D(fake), gradient = -1/batchSize
        var genOutputGradients = CalculateHingeLossGradients(generatorOutput, true, batchSize);

        // Backprop through discriminator to get gradients w.r.t. its input (the generated images)
        var discInputGradients = Discriminator.BackwardWithInputGradient(genOutputGradients);

        // Backprop through generator using the discriminator input gradients
        Generator.Backward(discInputGradients);

        // Update generator parameters
        UpdateGeneratorParameters();

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Generates images for training with proper gradient tracking.
    /// </summary>
    private Tensor<T> GenerateWithGradients(Tensor<T> noise, int[] classIndices)
    {
        // Vectorized batch embedding lookup using Engine
        // Convert Matrix to Tensor for embedding lookup
        var embeddingsTensor = new Tensor<T>([NumClasses, ClassEmbeddingDim]);
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                embeddingsTensor[i, j] = _classEmbeddings[i, j];
            }
        }
        var indicesTensor = new Tensor<int>(classIndices, [classIndices.Length]);
        var classEmbeddings = Engine.TensorEmbeddingLookup(embeddingsTensor, indicesTensor);

        // Concatenate latent codes and class embeddings
        var input = ConcatenateTensors(noise, classEmbeddings);

        return Generator.Predict(input);
    }

    /// <summary>
    /// Calculates hinge loss gradients for backpropagation.
    /// </summary>
    private Tensor<T> CalculateHingeLossGradients(Tensor<T> output, bool isReal, int batchSize)
    {
        var gradients = new Tensor<T>(output.Shape);

        for (int i = 0; i < output.Length; i++)
        {
            T gradient;
            if (isReal)
            {
                // For real: loss = max(0, 1 - output)
                // Gradient = -1 if (1 - output) > 0, else 0
                var margin = NumOps.Subtract(NumOps.One, output.GetFlat(i));
                gradient = NumOps.GreaterThan(margin, NumOps.Zero)
                    ? NumOps.Negate(NumOps.Divide(NumOps.One, NumOps.FromDouble(output.Length)))
                    : NumOps.Zero;
            }
            else
            {
                // For fake: loss = max(0, 1 + output)
                // Gradient = 1 if (1 + output) > 0, else 0
                var margin = NumOps.Add(NumOps.One, output.GetFlat(i));
                gradient = NumOps.GreaterThan(margin, NumOps.Zero)
                    ? NumOps.Divide(NumOps.One, NumOps.FromDouble(output.Length))
                    : NumOps.Zero;
            }

            gradients.SetFlat(i, gradient);
        }

        return gradients;
    }

    /// <summary>
    /// Updates discriminator parameters using vectorized Adam optimizer.
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateDiscriminatorParameters()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        // Initialize discriminator optimizer state if needed
        if (_discMomentum.Length != parameters.Length)
        {
            _discMomentum = new Vector<T>(parameters.Length);
            _discMomentum.Fill(NumOps.Zero);
        }

        if (_discSecondMoment.Length != parameters.Length)
        {
            _discSecondMoment = new Vector<T>(parameters.Length);
            _discSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer parameters (beta1=0 for GANs as per BigGAN paper)
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.999);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_currentLearningRate);

        // Update beta powers for bias correction
        _discBeta1Power = NumOps.Multiply(_discBeta1Power, beta1);
        _discBeta2Power = NumOps.Multiply(_discBeta2Power, beta2);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_discMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _discMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_discSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _discSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Bias correction (only needed when beta1 > 0)
        Vector<T> mCorrected;
        if (NumOps.GreaterThan(beta1, NumOps.Zero))
        {
            var biasCorrection1 = NumOps.Subtract(NumOps.One, _discBeta1Power);
            mCorrected = (Vector<T>)Engine.Divide(_discMomentum, biasCorrection1);
        }
        else
        {
            mCorrected = _discMomentum;
        }

        var biasCorrection2 = NumOps.Subtract(NumOps.One, _discBeta2Power);
        var vCorrected = (Vector<T>)Engine.Divide(_discSecondMoment, biasCorrection2);

        // Vectorized parameter update: p = p - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(vCorrected);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(mCorrected, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        Discriminator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates generator parameters using vectorized Adam optimizer.
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Initialize generator optimizer state if needed
        if (_genMomentum.Length != parameters.Length)
        {
            _genMomentum = new Vector<T>(parameters.Length);
            _genMomentum.Fill(NumOps.Zero);
        }

        if (_genSecondMoment.Length != parameters.Length)
        {
            _genSecondMoment = new Vector<T>(parameters.Length);
            _genSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer parameters (beta1=0 for GANs as per BigGAN paper)
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.999);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_currentLearningRate);

        // Update beta powers for bias correction
        _genBeta1Power = NumOps.Multiply(_genBeta1Power, beta1);
        _genBeta2Power = NumOps.Multiply(_genBeta2Power, beta2);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_genMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _genMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_genSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _genSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Bias correction (only needed when beta1 > 0)
        Vector<T> mCorrected;
        if (NumOps.GreaterThan(beta1, NumOps.Zero))
        {
            var biasCorrection1 = NumOps.Subtract(NumOps.One, _genBeta1Power);
            mCorrected = (Vector<T>)Engine.Divide(_genMomentum, biasCorrection1);
        }
        else
        {
            mCorrected = _genMomentum;
        }

        var biasCorrection2 = NumOps.Subtract(NumOps.One, _genBeta2Power);
        var vCorrected = (Vector<T>)Engine.Divide(_genSecondMoment, biasCorrection2);

        // Vectorized parameter update: p = p - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(vCorrected);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(mCorrected, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Calculates hinge loss for adversarial training.
    /// Hinge loss: max(0, 1 - t*y) where t is target, y is output
    /// For real: max(0, 1 - y)
    /// For fake: max(0, 1 + y)
    /// </summary>
    /// <param name="output">Discriminator output tensor</param>
    /// <param name="isReal">True if computing loss for real samples, false for fake</param>
    /// <param name="batchSize">Batch size (unused, kept for API compatibility)</param>
    /// <returns>Average hinge loss over all output elements</returns>
    private T CalculateHingeLoss(Tensor<T> output, bool isReal, int batchSize)
    {
        var loss = NumOps.Zero;
        var one = NumOps.One;

        for (int i = 0; i < output.Length; i++)
        {
            T hingeLoss;
            if (isReal)
            {
                // max(0, 1 - output)
                var margin = NumOps.Subtract(one, output.GetFlat(i));
                hingeLoss = NumOps.GreaterThan(margin, NumOps.Zero) ? margin : NumOps.Zero;
            }
            else
            {
                // max(0, 1 + output)
                var margin = NumOps.Add(one, output.GetFlat(i));
                hingeLoss = NumOps.GreaterThan(margin, NumOps.Zero) ? margin : NumOps.Zero;
            }

            loss = NumOps.Add(loss, hingeLoss);
        }

        // Normalize by total number of output elements, not just batch size
        // This correctly handles cases where output has multiple logits per sample
        return NumOps.Divide(loss, NumOps.FromDouble(output.Length));
    }

    /// <summary>
    /// GAN hinge loss implementation for use with the loss function interface.
    /// Uses GAN-specific formulation:
    /// - For real samples (actual > 0.5): max(0, 1 - predicted)
    /// - For fake samples (actual &lt;= 0.5): max(0, 1 + predicted)
    /// This differs from SVM hinge loss which uses y*t formulation.
    /// </summary>
    private class HingeLoss<TLoss> : ILossFunction<TLoss>
    {
        private static readonly INumericOperations<TLoss> _ops = MathHelper.GetNumericOperations<TLoss>();

        public TLoss CalculateLoss(Vector<TLoss> predicted, Vector<TLoss> actual)
        {
            var loss = _ops.Zero;
            var one = _ops.One;
            var half = _ops.FromDouble(0.5);

            for (int i = 0; i < predicted.Length; i++)
            {
                TLoss hingeLoss;
                bool isReal = _ops.GreaterThan(actual[i], half);

                if (isReal)
                {
                    // Real: max(0, 1 - predicted)
                    var margin = _ops.Subtract(one, predicted[i]);
                    hingeLoss = _ops.GreaterThan(margin, _ops.Zero) ? margin : _ops.Zero;
                }
                else
                {
                    // Fake: max(0, 1 + predicted)
                    var margin = _ops.Add(one, predicted[i]);
                    hingeLoss = _ops.GreaterThan(margin, _ops.Zero) ? margin : _ops.Zero;
                }

                loss = _ops.Add(loss, hingeLoss);
            }

            return _ops.Divide(loss, _ops.FromDouble(predicted.Length));
        }

        public Vector<TLoss> CalculateDerivative(Vector<TLoss> predicted, Vector<TLoss> actual)
        {
            var gradient = new Vector<TLoss>(predicted.Length);
            var one = _ops.One;
            var half = _ops.FromDouble(0.5);
            var scale = _ops.FromDouble(1.0 / predicted.Length);

            for (int i = 0; i < predicted.Length; i++)
            {
                bool isReal = _ops.GreaterThan(actual[i], half);

                if (isReal)
                {
                    // Real: d/dx max(0, 1 - x) = -1 if (1 - x) > 0, else 0
                    var margin = _ops.Subtract(one, predicted[i]);
                    gradient[i] = _ops.GreaterThan(margin, _ops.Zero)
                        ? _ops.Negate(scale)
                        : _ops.Zero;
                }
                else
                {
                    // Fake: d/dx max(0, 1 + x) = 1 if (1 + x) > 0, else 0
                    var margin = _ops.Add(one, predicted[i]);
                    gradient[i] = _ops.GreaterThan(margin, _ops.Zero)
                        ? scale
                        : _ops.Zero;
                }
            }

            return gradient;
        }

        public TLoss CalculateLossGpu(Tensor<TLoss> predicted, Tensor<TLoss> actual)
        {
            // Fall back to CPU for now
            return CalculateLoss(predicted.ToVector(), actual.ToVector());
        }

        public Tensor<TLoss> CalculateDerivativeGpu(Tensor<TLoss> predicted, Tensor<TLoss> actual)
        {
            // Fall back to CPU for now
            var derivative = CalculateDerivative(predicted.ToVector(), actual.ToVector());
            var result = new Tensor<TLoss>(predicted.Shape);
            Array.Copy(derivative.ToArray(), result.Data, derivative.Length);
            return result;
        }
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the BigGAN.
    /// </summary>
    /// <remarks>
    /// This includes all parameters from both the Generator and Discriminator networks.
    /// </remarks>
    public override int ParameterCount => Generator.GetParameterCount() + Discriminator.GetParameterCount();

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For general prediction, generate with random classes
        var batchSize = input.Shape[0];
        var classIndices = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            classIndices[i] = Random.Next(NumClasses);
        }
        return Generate(input, classIndices);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // For GANs, training is done through TrainStep
        var batchSize = input.Shape[0];
        var labels = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            labels[i] = Random.Next(NumClasses);
        }
        TrainStep(input, labels, batchSize);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();
        int conditioningMatrixCount = NumClasses * ClassEmbeddingDim * 2;
        int expectedCount = generatorCount + discriminatorCount + conditioningMatrixCount;

        if (parameters.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} parameters (generator: {generatorCount}, discriminator: {discriminatorCount}, conditioning: {conditioningMatrixCount}), got {parameters.Length}.",
                nameof(parameters));
        }

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

        // Update class embeddings
        int idx = generatorCount + discriminatorCount;
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                _classEmbeddings[i, j] = parameters[idx++];
            }
        }

        // Update discriminator class projection
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                _discClassProjection[i, j] = parameters[idx++];
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var genParams = Generator.GetParameters();
        var discParams = Discriminator.GetParameters();

        int conditioningMatrixCount = NumClasses * ClassEmbeddingDim * 2;
        var totalLength = genParams.Length + discParams.Length + conditioningMatrixCount;
        var parameters = new Vector<T>(totalLength);

        int idx = 0;
        for (int i = 0; i < genParams.Length; i++)
            parameters[idx++] = genParams[i];
        for (int i = 0; i < discParams.Length; i++)
            parameters[idx++] = discParams[i];

        // Add class embeddings as parameters
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                parameters[idx++] = _classEmbeddings[i, j];
            }
        }

        // Add discriminator class projection as parameters
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                parameters[idx++] = _discClassProjection[i, j];
            }
        }

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
            Name = "BigGAN",
            Version = "1.0"
        };

        metadata.SetProperty("ModelType", "BigGAN");
        metadata.SetProperty("LatentSize", LatentSize);
        metadata.SetProperty("NumClasses", NumClasses);
        metadata.SetProperty("ClassEmbeddingDim", ClassEmbeddingDim);
        metadata.SetProperty("ImageChannels", _imageChannels);
        metadata.SetProperty("ImageHeight", _imageHeight);
        metadata.SetProperty("ImageWidth", _imageWidth);
        metadata.SetProperty("GeneratorChannels", _generatorChannels);
        metadata.SetProperty("DiscriminatorChannels", _discriminatorChannels);
        metadata.SetProperty("TruncationThreshold", TruncationThreshold);
        metadata.SetProperty("UseTruncation", UseTruncation);
        metadata.SetProperty("UseSpectralNormalization", UseSpectralNormalization);
        metadata.SetProperty("UseSelfAttention", UseSelfAttention);

        return metadata;
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in the constructor via the Generator and Discriminator CNNs
        // Initialize momentum vectors for Adam optimizer - separate for generator and discriminator
        var genParamCount = Generator.GetParameterCount();
        var discParamCount = Discriminator.GetParameterCount();

        _genMomentum = new Vector<T>(genParamCount);
        _genSecondMoment = new Vector<T>(genParamCount);
        _discMomentum = new Vector<T>(discParamCount);
        _discSecondMoment = new Vector<T>(discParamCount);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(LatentSize);
        writer.Write(NumClasses);
        writer.Write(ClassEmbeddingDim);
        writer.Write(_imageChannels);
        writer.Write(_imageHeight);
        writer.Write(_imageWidth);
        writer.Write(_generatorChannels);
        writer.Write(_discriminatorChannels);
        writer.Write(TruncationThreshold);
        writer.Write(UseTruncation);
        writer.Write(UseSpectralNormalization);
        writer.Write(UseSelfAttention);
        writer.Write(_initialLearningRate);
        writer.Write(_currentLearningRate);

        // Serialize class embeddings
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                writer.Write(Convert.ToDouble(_classEmbeddings[i, j]));
            }
        }

        // Serialize discriminator class projection
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                writer.Write(Convert.ToDouble(_discClassProjection[i, j]));
            }
        }

        // Serialize networks
        byte[] generatorData = Generator.Serialize();
        writer.Write(generatorData.Length);
        writer.Write(generatorData);

        byte[] discriminatorData = Discriminator.Serialize();
        writer.Write(discriminatorData.Length);
        writer.Write(discriminatorData);

        // Serialize optimizer state for complete training state preservation
        SerializationHelper<T>.SerializeVector(writer, _genMomentum);
        SerializationHelper<T>.SerializeVector(writer, _genSecondMoment);
        writer.Write(Convert.ToDouble(_genBeta1Power));
        writer.Write(Convert.ToDouble(_genBeta2Power));

        SerializationHelper<T>.SerializeVector(writer, _discMomentum);
        SerializationHelper<T>.SerializeVector(writer, _discSecondMoment);
        writer.Write(Convert.ToDouble(_discBeta1Power));
        writer.Write(Convert.ToDouble(_discBeta2Power));

        // Serialize loss history
        writer.Write(_generatorLosses.Count);
        foreach (var loss in _generatorLosses)
        {
            writer.Write(Convert.ToDouble(loss));
        }

        writer.Write(_discriminatorLosses.Count);
        foreach (var loss in _discriminatorLosses)
        {
            writer.Write(Convert.ToDouble(loss));
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        LatentSize = reader.ReadInt32();
        NumClasses = reader.ReadInt32();
        ClassEmbeddingDim = reader.ReadInt32();
        _imageChannels = reader.ReadInt32();
        _imageHeight = reader.ReadInt32();
        _imageWidth = reader.ReadInt32();
        _generatorChannels = reader.ReadInt32();
        _discriminatorChannels = reader.ReadInt32();
        TruncationThreshold = reader.ReadDouble();
        UseTruncation = reader.ReadBoolean();
        UseSpectralNormalization = reader.ReadBoolean();
        UseSelfAttention = reader.ReadBoolean();
        _initialLearningRate = reader.ReadDouble();
        _currentLearningRate = reader.ReadDouble();

        // Deserialize class embeddings
        _classEmbeddings = new Matrix<T>(NumClasses, ClassEmbeddingDim);
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                _classEmbeddings[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize discriminator class projection
        _discClassProjection = new Matrix<T>(NumClasses, ClassEmbeddingDim);
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                _discClassProjection[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize networks
        int generatorLength = reader.ReadInt32();
        Generator.Deserialize(reader.ReadBytes(generatorLength));

        int discriminatorLength = reader.ReadInt32();
        Discriminator.Deserialize(reader.ReadBytes(discriminatorLength));

        // Deserialize optimizer state
        _genMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _genSecondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _genBeta1Power = NumOps.FromDouble(reader.ReadDouble());
        _genBeta2Power = NumOps.FromDouble(reader.ReadDouble());

        _discMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _discSecondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _discBeta1Power = NumOps.FromDouble(reader.ReadDouble());
        _discBeta2Power = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize loss history
        int genLossCount = reader.ReadInt32();
        _generatorLosses = new List<T>(genLossCount);
        for (int i = 0; i < genLossCount; i++)
        {
            _generatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));
        }

        int discLossCount = reader.ReadInt32();
        _discriminatorLosses = new List<T>(discLossCount);
        for (int i = 0; i < discLossCount; i++)
        {
            _discriminatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new BigGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            LatentSize,
            NumClasses,
            ClassEmbeddingDim,
            _imageChannels,
            _imageHeight,
            _imageWidth,
            _generatorChannels,
            _discriminatorChannels,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate);
    }
}
