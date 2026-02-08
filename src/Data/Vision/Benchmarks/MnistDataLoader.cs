using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the MNIST handwritten digit classification dataset (60k train / 10k test, 28x28 grayscale).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// MNIST is the "Hello World" of machine learning - a dataset of handwritten digits (0-9).
/// Each image is 28x28 pixels, grayscale, with a corresponding digit label.
/// </para>
/// <para><b>For Beginners:</b> This is the easiest way to get started with image classification:
/// <code>
/// var loader = DataLoaders.Mnist&lt;float&gt;();
/// await loader.LoadAsync();
/// foreach (var batch in loader.GetBatches(batchSize: 32))
/// {
///     // batch.Features: [32, 28, 28, 1] tensor
///     // batch.Labels:   [32, 10] one-hot tensor
/// }
/// </code>
/// </para>
/// </remarks>
public class MnistDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string BaseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    private static readonly string TrainImagesFile = "train-images-idx3-ubyte";
    private static readonly string TrainLabelsFile = "train-labels-idx1-ubyte";
    private static readonly string TestImagesFile = "t10k-images-idx3-ubyte";
    private static readonly string TestLabelsFile = "t10k-labels-idx1-ubyte";

    private readonly MnistDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "MNIST";

    /// <inheritdoc/>
    public override string Description => "MNIST handwritten digit classification dataset";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => 784;

    /// <inheritdoc/>
    public override int OutputDimension => 10;

    /// <summary>
    /// Number of classes (digits 0-9).
    /// </summary>
    public int NumClasses => 10;

    /// <summary>
    /// Creates a new MNIST data loader.
    /// </summary>
    public MnistDataLoader(MnistDataLoaderOptions? options = null)
    {
        _options = options ?? new MnistDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("mnist");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Determine which files to load
        string imagesFileName;
        string labelsFileName;

        if (_options.Split == DatasetSplit.Test)
        {
            imagesFileName = TestImagesFile;
            labelsFileName = TestLabelsFile;
        }
        else
        {
            imagesFileName = TrainImagesFile;
            labelsFileName = TrainLabelsFile;
        }

        string imagesPath = Path.Combine(_dataPath, imagesFileName);
        string labelsPath = Path.Combine(_dataPath, labelsFileName);

        // Download if needed
        if (_options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndDecompressGzipAsync(
                BaseUrl + imagesFileName + ".gz", imagesPath, cancellationToken);
            await DatasetDownloader.DownloadAndDecompressGzipAsync(
                BaseUrl + labelsFileName + ".gz", labelsPath, cancellationToken);
        }

        if (!File.Exists(imagesPath))
        {
            throw new FileNotFoundException($"MNIST images file not found: {imagesPath}");
        }

        if (!File.Exists(labelsPath))
        {
            throw new FileNotFoundException($"MNIST labels file not found: {labelsPath}");
        }

        // Parse IDX file format
#if NET6_0_OR_GREATER
        byte[] imageBytes = await File.ReadAllBytesAsync(imagesPath, cancellationToken);
        byte[] labelBytes = await File.ReadAllBytesAsync(labelsPath, cancellationToken);
#else
        byte[] imageBytes = File.ReadAllBytes(imagesPath);
        byte[] labelBytes = File.ReadAllBytes(labelsPath);
#endif

        // Validate minimum header sizes
        const int imageHeaderSize = 16;
        const int labelHeaderSize = 8;

        if (imageBytes.Length < imageHeaderSize)
        {
            throw new InvalidDataException(
                $"MNIST images file is corrupted or truncated: expected at least {imageHeaderSize} bytes, got {imageBytes.Length}.");
        }

        if (labelBytes.Length < labelHeaderSize)
        {
            throw new InvalidDataException(
                $"MNIST labels file is corrupted or truncated: expected at least {labelHeaderSize} bytes, got {labelBytes.Length}.");
        }

        // Images: [magic(4)] [count(4)] [rows(4)] [cols(4)] [pixel data...]
        int imageCount = ReadBigEndianInt32(imageBytes, 4);
        int rows = ReadBigEndianInt32(imageBytes, 8);
        int cols = ReadBigEndianInt32(imageBytes, 12);

        // Validate image data fits within file
        long expectedImageSize = imageHeaderSize + (long)imageCount * rows * cols;
        if (imageBytes.Length < expectedImageSize)
        {
            throw new InvalidDataException(
                $"MNIST images file is truncated: expected {expectedImageSize} bytes for {imageCount} images, got {imageBytes.Length}.");
        }

        // Labels: [magic(4)] [count(4)] [label data...]
        int labelCount = ReadBigEndianInt32(labelBytes, 4);

        // Validate label data fits within file
        long expectedLabelSize = labelHeaderSize + (long)labelCount;
        if (labelBytes.Length < expectedLabelSize)
        {
            throw new InvalidDataException(
                $"MNIST labels file is truncated: expected {expectedLabelSize} bytes for {labelCount} labels, got {labelBytes.Length}.");
        }

        if (imageCount != labelCount)
        {
            throw new InvalidDataException(
                $"MNIST image count ({imageCount}) doesn't match label count ({labelCount}).");
        }

        int samplesToLoad = imageCount;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samplesToLoad)
        {
            samplesToLoad = _options.MaxSamples.Value;
        }

        _sampleCount = samplesToLoad;
        int pixelsPerImage = rows * cols;

        // Build feature tensor
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

            // Read pixels (offset: 16 bytes header + i * pixelsPerImage)
            int imageOffset = 16 + i * pixelsPerImage;
            int featureOffset = i * pixelsPerImage;

            for (int p = 0; p < pixelsPerImage; p++)
            {
                double value = imageBytes[imageOffset + p];
                if (_options.Normalize)
                {
                    value /= 255.0;
                }

                featuresData[featureOffset + p] = NumOps.FromDouble(value);
            }

            // Read label (offset: 8 bytes header + i)
            int label = labelBytes[8 + i];
            if (label < 0 || label >= 10)
            {
                throw new InvalidDataException(
                    $"Invalid MNIST label {label} at sample {i}. Expected 0-9.");
            }

            int labelOffset = i * 10;
            labelsData[labelOffset + label] = NumOps.One;
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
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize + valSize).ToArray()))
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
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
