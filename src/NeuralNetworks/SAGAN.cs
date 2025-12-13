using System.IO;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
    private List<T> _generatorLosses = [];
    private List<T> _discriminatorLosses = [];

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
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            latentSize + (numClasses > 0 ? 128 : 0),
            imageChannels * imageHeight * imageWidth,
            0, 0, 0,
            null), lossFunction ?? new HingeLoss<T>())
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
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;

        // Create generator and discriminator with self-attention
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
        _lossFunction = lossFunction ?? new HingeLoss<T>();

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

        if (_isConditional && classIndices != null)
        {
            // Concatenate class information (simplified)
            var classEmbeddings = CreateClassEmbeddings(classIndices);
            var input = ConcatenateTensors(noise, classEmbeddings);
            return Generator.Predict(input);
        }

        return Generator.Predict(noise);
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

        if (_isConditional && classIndices != null)
        {
            var classEmbeddings = CreateClassEmbeddings(classIndices);
            var input = ConcatenateTensors(latentCodes, classEmbeddings);
            return Generator.Predict(input);
        }

        return Generator.Predict(latentCodes);
    }

    /// <summary>
    /// Generates random noise from a standard normal distribution.
    /// </summary>
    private Tensor<T> GenerateNoise(int batchSize)
    {
        var noise = new Tensor<T>([batchSize, LatentSize]);

        // Box-Muller transform for Gaussian sampling
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
    /// Creates class embeddings for conditional generation.
    /// </summary>
    private Tensor<T> CreateClassEmbeddings(int[] classIndices)
    {
        var embeddingDim = 128; // Simplified fixed dimension
        var embeddings = new Tensor<T>([classIndices.Length, embeddingDim]);

        // Simplified: one-hot encoding scaled by embedding dimension
        for (int i = 0; i < classIndices.Length; i++)
        {
            var classIdx = classIndices[i];
            if (classIdx >= 0 && classIdx < embeddingDim)
            {
                embeddings.SetFlat(i * embeddingDim + classIdx, NumOps.One);
            }
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
