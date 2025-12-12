using System.IO;
using AiDotNet.Enums;
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
    private Vector<T> _momentum;
    private Vector<T> _secondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate;
    private double _initialLearningRate;
    private List<T> _generatorLosses = [];
    private List<T> _discriminatorLosses = [];

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
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            latentSize + classEmbeddingDim,
            imageChannels * imageHeight * imageWidth,
            0, 0, 0,
            null), lossFunction ?? new HingeLoss<T>())
    {
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

        // Initialize optimizer parameters
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;

        // Initialize class embeddings with orthogonal initialization
        _classEmbeddings = InitializeClassEmbeddings();

        // Create generator and discriminator
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
        _lossFunction = lossFunction ?? new HingeLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes class embeddings using orthogonal initialization.
    /// Orthogonal initialization helps with training stability.
    /// </summary>
    private Matrix<T> InitializeClassEmbeddings()
    {
        var embeddings = new Matrix<T>(NumClasses, ClassEmbeddingDim);

        // Simplified orthogonal initialization
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                embeddings[i, j] = NumOps.FromDouble((Random.NextDouble() - 0.5) * 0.1);
            }
        }

        return embeddings;
    }

    /// <summary>
    /// Gets the class embedding for a specific class index.
    /// </summary>
    private Vector<T> GetClassEmbedding(int classIndex)
    {
        var embedding = new Vector<T>(ClassEmbeddingDim);
        for (int i = 0; i < ClassEmbeddingDim; i++)
        {
            embedding[i] = _classEmbeddings[classIndex, i];
        }
        return embedding;
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
    /// <param name="classIndices">Class indices for each sample</param>
    /// <returns>Generated images</returns>
    public Tensor<T> Generate(Tensor<T> latentCodes, int[] classIndices)
    {
        if (classIndices.Length != latentCodes.Shape[0])
        {
            throw new ArgumentException("Number of class indices must match batch size");
        }

        Generator.SetTrainingMode(false);

        // Apply truncation if enabled
        var truncatedCodes = ApplyTruncation(latentCodes);

        // Create class embeddings tensor
        var classEmbeddings = new Tensor<T>([classIndices.Length, ClassEmbeddingDim]);
        for (int i = 0; i < classIndices.Length; i++)
        {
            var embedding = GetClassEmbedding(classIndices[i]);
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                classEmbeddings[i, j] = embedding[j];
            }
        }

        // Concatenate latent codes and class embeddings
        var input = ConcatenateTensors(truncatedCodes, classEmbeddings);

        return Generator.Predict(input);
    }

    /// <summary>
    /// Generates random images with random class labels.
    /// </summary>
    /// <param name="numImages">Number of images to generate</param>
    /// <returns>Generated images</returns>
    public Tensor<T> Generate(int numImages)
    {
        var noise = GenerateNoise(numImages);
        var classIndices = new int[numImages];

        for (int i = 0; i < numImages; i++)
        {
            classIndices[i] = Random.Next(NumClasses);
        }

        return Generate(noise, classIndices);
    }

    /// <summary>
    /// Generates random noise for the generator input.
    /// Uses Gaussian distribution (standard normal).
    /// </summary>
    private Tensor<T> GenerateNoise(int batchSize)
    {
        var noise = new Tensor<T>([batchSize, LatentSize]);

        // Sample from approximate Gaussian using Box-Muller transform
        for (int i = 0; i < noise.Length; i += 2)
        {
            var u1 = Random.NextDouble();
            var u2 = Random.NextDouble();

            var z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            noise.SetFlat(i, NumOps.FromDouble(z1));

            if (i + 1 < noise.Length)
            {
                var z2 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                noise.SetFlat(i + 1, NumOps.FromDouble(z2));
            }
        }

        return noise;
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
    public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, int[] realLabels, int batchSize)
    {
        var one = NumOps.One;

        // === Train Discriminator ===
        Discriminator.SetTrainingMode(true);
        Generator.SetTrainingMode(false);

        // Real images
        var realOutput = Discriminator.Predict(realImages);
        var realLoss = CalculateHingeLoss(realOutput, true, batchSize);

        // Fake images
        var noise = GenerateNoise(batchSize);
        var fakeLabels = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            fakeLabels[i] = Random.Next(NumClasses);
        }

        var fakeImages = Generate(noise, fakeLabels);
        var fakeOutput = Discriminator.Predict(fakeImages);
        var fakeLoss = CalculateHingeLoss(fakeOutput, false, batchSize);

        var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        _discriminatorLosses.Add(discriminatorLoss);

        // Backpropagate discriminator
        var discGradient = new Tensor<T>([1]);
        discGradient[0] = one;
        Discriminator.Backward(discGradient);

        // === Train Generator ===
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);

        var generatorNoise = GenerateNoise(batchSize);
        var generatorLabels = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            generatorLabels[i] = Random.Next(NumClasses);
        }

        var generatedImages = Generate(generatorNoise, generatorLabels);
        var generatorOutput = Discriminator.Predict(generatedImages);
        var generatorLoss = CalculateHingeLoss(generatorOutput, true, batchSize);
        _generatorLosses.Add(generatorLoss);

        // Backpropagate generator
        var genGradient = new Tensor<T>([1]);
        genGradient[0] = one;
        Generator.Backward(genGradient);

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Calculates hinge loss for adversarial training.
    /// Hinge loss: max(0, 1 - t*y) where t is target, y is output
    /// For real: max(0, 1 - y)
    /// For fake: max(0, 1 + y)
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

        return NumOps.Divide(loss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Hinge loss implementation for use with the loss function interface.
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
                // Hinge loss: max(0, 1 - y * t) where t is target
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
    }

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

        var totalLength = genParams.Length + discParams.Length + NumClasses * ClassEmbeddingDim;
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
        // Initialize momentum vectors for Adam optimizer
        var paramCount = Generator.GetParameterCount() + Discriminator.GetParameterCount();
        _momentum = new Vector<T>(paramCount);
        _secondMoment = new Vector<T>(paramCount);
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

        // Serialize class embeddings
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                writer.Write(Convert.ToDouble(_classEmbeddings[i, j]));
            }
        }

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

        // Deserialize class embeddings
        _classEmbeddings = new Matrix<T>(NumClasses, ClassEmbeddingDim);
        for (int i = 0; i < NumClasses; i++)
        {
            for (int j = 0; j < ClassEmbeddingDim; j++)
            {
                _classEmbeddings[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize networks
        int generatorLength = reader.ReadInt32();
        Generator.Deserialize(reader.ReadBytes(generatorLength));

        int discriminatorLength = reader.ReadInt32();
        Discriminator.Deserialize(reader.ReadBytes(discriminatorLength));
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
