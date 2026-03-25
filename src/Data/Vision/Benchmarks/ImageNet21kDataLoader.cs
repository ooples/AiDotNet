using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the ImageNet-21K dataset (~14.2M images, 21,841 categories from the full WordNet hierarchy).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// ImageNet-21K uses the same directory structure as ImageNet-1K (synset folders), but with 21K+ categories.
/// The dataset must be downloaded manually from https://image-net.org/.
/// Use MaxClasses and MaxSamples to load manageable subsets.
/// </para>
/// </remarks>
public class ImageNet21kDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int DefaultMaxClasses = 21841;

    private readonly ImageNet21kDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _numClasses;
    private int _imageSize;

    /// <inheritdoc/>
    public override string Name => "ImageNet-21K";
    /// <inheritdoc/>
    public override string Description => "ImageNet-21K full hierarchy dataset (21,841 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _imageSize * _imageSize * 3;
    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Creates a new ImageNet-21K data loader.</summary>
    public ImageNet21kDataLoader(ImageNet21kDataLoaderOptions? options = null)
    {
        _options = options ?? new ImageNet21kDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("imagenet-21k");
        _imageSize = _options.ImageSize;
        _numClasses = _options.MaxClasses ?? DefaultMaxClasses;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitDir = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation
            ? Path.Combine(_dataPath, "val")
            : Path.Combine(_dataPath, "train");

        if (!Directory.Exists(splitDir))
        {
            throw new DirectoryNotFoundException(
                $"ImageNet-21K data not found at {splitDir}. " +
                "Download the dataset from https://image-net.org/ and extract to the data path.");
        }

        var synsetDirs = Directory.GetDirectories(splitDir)
            .OrderBy(d => Path.GetFileName(d), StringComparer.Ordinal)
            .Take(_numClasses)
            .ToArray();

        _numClasses = synsetDirs.Length;

        if (synsetDirs.Length == 0)
        {
            throw new InvalidOperationException($"No synset directories found in {splitDir}.");
        }

        var imagePaths = new List<(string Path, int Label)>();
        for (int classIdx = 0; classIdx < synsetDirs.Length; classIdx++)
        {
            var files = Directory.GetFiles(synsetDirs[classIdx], "*.JPEG", SearchOption.TopDirectoryOnly)
                .Concat(Directory.GetFiles(synsetDirs[classIdx], "*.jpeg", SearchOption.TopDirectoryOnly))
                .Concat(Directory.GetFiles(synsetDirs[classIdx], "*.jpg", SearchOption.TopDirectoryOnly))
                .Concat(Directory.GetFiles(synsetDirs[classIdx], "*.png", SearchOption.TopDirectoryOnly))
                .Distinct();
            foreach (var file in files)
            {
                imagePaths.Add((file, classIdx));
            }
        }

        int totalSamples = imagePaths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * _numClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (imgPath, label) = imagePaths[i];
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);

            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);

            if (label >= 0 && label < _numClasses)
                labelsData[i * _numClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, _numClasses });
        InitializeIndices(totalSamples);

        await Task.CompletedTask;
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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

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

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.ToArray().Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
