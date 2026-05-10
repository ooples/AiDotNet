using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Food-101 fine-grained classification dataset (Bossard et al. 2014).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/food-101/
///   meta/{train,test}.txt    (lines: "{class}/{image_id}")
///   meta/classes.txt          (sorted class names)
///   images/{class}/{image_id}.jpg
/// </code>
/// Auto-download fetches the ETH Zürich tarball.
/// </para>
/// </remarks>
public class Food101DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 101;
    private static readonly string DownloadUrl =
        "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";

    private readonly Food101DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    public override string Name => "Food-101";
    public override string Description => "Food-101 fine-grained image classification (101 classes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _imageSize * _imageSize * 3;
    public override int OutputDimension => NumClasses;

    public Food101DataLoader(Food101DataLoaderOptions? options = null)
    {
        _options = options ?? new Food101DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("food-101");
        _imageSize = _options.ImageSize;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"Food-101 not found at {_dataPath}.");

        string metaDir = Path.Combine(root, "meta");
        string imagesDir = Path.Combine(root, "images");
        string classesPath = Path.Combine(metaDir, "classes.txt");
        if (!File.Exists(classesPath))
            throw new FileNotFoundException($"Food-101 classes.txt not found at {classesPath}.");

        var classes = (await FilePolyfill.ReadAllLinesAsync(classesPath, cancellationToken))
            .Where(l => !string.IsNullOrWhiteSpace(l)).Select(l => l.Trim()).ToArray();
        var classIdx = new Dictionary<string, int>(classes.Length, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < classes.Length; i++) classIdx[classes[i]] = i;

        string splitFile = _options.Split == Geometry.DatasetSplit.Train
            ? Path.Combine(metaDir, "train.txt")
            : Path.Combine(metaDir, "test.txt");
        if (!File.Exists(splitFile))
            throw new FileNotFoundException($"Food-101 split file not found at {splitFile}.");

        var paths = new List<(string Path, int Label)>();
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(splitFile, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(line)) continue;
            string trimmed = line.Trim();
            int slash = trimmed.IndexOf('/');
            if (slash <= 0) continue;
            string className = trimmed.Substring(0, slash);
            if (!classIdx.TryGetValue(className, out int label)) continue;
            string imgPath = Path.Combine(imagesDir, trimmed + ".jpg");
            paths.Add((imgPath, label));
        }

        int totalSamples = paths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int pixelsPerImage = _imageSize * _imageSize * 3;
        var featuresData = new T[totalSamples * pixelsPerImage];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (imgPath, label) = paths[i];
            if (!File.Exists(imgPath))
                throw new FileNotFoundException(
                    $"Food-101 image referenced by meta/{(_options.Split == Geometry.DatasetSplit.Test ? "test" : "train")}.txt is missing: {imgPath}. " +
                    "Re-extract food-101.tar.gz or check archive integrity.");
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            if (label >= 0 && label < NumClasses) labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _imageSize, _imageSize, 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore() { LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0; }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (ExtractTensorBatchLocal(features, indices), ExtractTensorBatchLocal(labels, indices));
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
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        var features = LoadedFeatures ?? throw new InvalidOperationException("Not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Take(trainSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(ExtractTensorBatchLocal(features, shuffled.Skip(trainSize + valSize).ToArray()), ExtractTensorBatchLocal(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private string ResolveDataDir()
    {
        string sub = Path.Combine(_dataPath, "food-101");
        return Directory.Exists(sub) ? sub : _dataPath;
    }

    private static Tensor<T> ExtractTensorBatchLocal(Tensor<T> source, int[] indices)
    {
        var result = AiDotNet.Helpers.TensorCopyHelper.CreateEmptyBatchLike(source, indices.Length);
        for (int i = 0; i < indices.Length; i++)
            AiDotNet.Helpers.TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
