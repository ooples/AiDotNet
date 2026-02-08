using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Fashion-MNIST clothing classification dataset (60k train / 10k test, 28x28 grayscale).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Fashion-MNIST is a drop-in replacement for MNIST with clothing images (t-shirt, trouser, pullover, etc.).
/// Same format (28x28 grayscale, 10 classes) but more challenging.
/// </para>
/// </remarks>
public class FashionMnistDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string BaseUrl = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
    private static readonly string TrainImagesFile = "train-images-idx3-ubyte";
    private static readonly string TrainLabelsFile = "train-labels-idx1-ubyte";
    private static readonly string TestImagesFile = "t10k-images-idx3-ubyte";
    private static readonly string TestLabelsFile = "t10k-labels-idx1-ubyte";

    private static readonly string[] ClassLabels = new[]
    {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    private readonly FashionMnistDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "FashionMNIST";
    /// <inheritdoc/>
    public override string Description => "Fashion-MNIST clothing classification dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => 784;
    /// <inheritdoc/>
    public override int OutputDimension => 10;

    /// <summary>Gets the human-readable class names.</summary>
    public IReadOnlyList<string> ClassNames => ClassLabels;

    /// <summary>Creates a new Fashion-MNIST data loader.</summary>
    public FashionMnistDataLoader(FashionMnistDataLoaderOptions? options = null)
    {
        _options = options ?? new FashionMnistDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("fashion-mnist");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string imagesFileName = _options.Split == DatasetSplit.Test ? TestImagesFile : TrainImagesFile;
        string labelsFileName = _options.Split == DatasetSplit.Test ? TestLabelsFile : TrainLabelsFile;

        string imagesPath = Path.Combine(_dataPath, imagesFileName);
        string labelsPath = Path.Combine(_dataPath, labelsFileName);

        if (_options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndDecompressGzipAsync(
                BaseUrl + imagesFileName + ".gz", imagesPath, cancellationToken);
            await DatasetDownloader.DownloadAndDecompressGzipAsync(
                BaseUrl + labelsFileName + ".gz", labelsPath, cancellationToken);
        }

        if (!File.Exists(imagesPath))
            throw new FileNotFoundException($"Fashion-MNIST images file not found: {imagesPath}");
        if (!File.Exists(labelsPath))
            throw new FileNotFoundException($"Fashion-MNIST labels file not found: {labelsPath}");

#if NET6_0_OR_GREATER
        byte[] imageBytes = await File.ReadAllBytesAsync(imagesPath, cancellationToken);
        byte[] labelBytes = await File.ReadAllBytesAsync(labelsPath, cancellationToken);
#else
        byte[] imageBytes = File.ReadAllBytes(imagesPath);
        byte[] labelBytes = File.ReadAllBytes(labelsPath);
#endif

        if (imageBytes.Length < 16)
            throw new InvalidDataException("Image file is truncated or corrupt (missing header).");
        if (labelBytes.Length < 8)
            throw new InvalidDataException("Label file is truncated or corrupt (missing header).");

        int imageCount = ReadBigEndianInt32(imageBytes, 4);
        int rows = ReadBigEndianInt32(imageBytes, 8);
        int cols = ReadBigEndianInt32(imageBytes, 12);

        if (imageBytes.Length < 16 + (long)imageCount * rows * cols)
            throw new InvalidDataException("Image file is truncated or corrupt.");

        int labelCount = ReadBigEndianInt32(labelBytes, 4);
        if (labelBytes.Length < 8 + labelCount)
            throw new InvalidDataException("Label file is truncated or corrupt.");
        if (labelCount != imageCount)
            throw new InvalidDataException($"Label count ({labelCount}) does not match image count ({imageCount}).");

        int samplesToLoad = imageCount;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samplesToLoad)
            samplesToLoad = _options.MaxSamples.Value;

        _sampleCount = samplesToLoad;
        int pixelsPerImage = rows * cols;

        T[] featuresData;
        int[] featureShape;

        if (_options.Flatten)
        {
            featuresData = new T[samplesToLoad * pixelsPerImage];
            featureShape = new[] { samplesToLoad, pixelsPerImage };
        }
        else
        {
            featuresData = new T[samplesToLoad * rows * cols * 1];
            featureShape = new[] { samplesToLoad, rows, cols, 1 };
        }

        var labelsData = new T[samplesToLoad * 10];

        for (int i = 0; i < samplesToLoad; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int imageOffset = 16 + i * pixelsPerImage;
            int featureOffset = i * pixelsPerImage;

            for (int p = 0; p < pixelsPerImage; p++)
            {
                double value = imageBytes[imageOffset + p];
                if (_options.Normalize) value /= 255.0;
                featuresData[featureOffset + p] = NumOps.FromDouble(value);
            }

            int label = labelBytes[8 + i];
            if (label < 0 || label >= 10)
            {
                throw new InvalidDataException($"Invalid label {label} at sample {i}. Expected 0-9.");
            }
            labelsData[i * 10 + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, featureShape);
        LoadedLabels = new Tensor<T>(labelsData, new[] { samplesToLoad, 10 });
        InitializeIndices(samplesToLoad);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (ExtractTensorBatch(features, indices), ExtractTensorBatch(labels, indices));
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7, double validationRatio = 0.15, int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);
        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded. Call LoadAsync() first.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded. Call LoadAsync() first.");

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static int ReadBigEndianInt32(byte[] data, int offset)
    {
        return (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3];
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
