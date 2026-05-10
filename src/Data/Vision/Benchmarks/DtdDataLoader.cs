using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the Describable Textures Dataset (DTD; Cimpoi et al. 2014) — 47 texture classes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/dtd/
///   images/{class}/{class}_NNNN.jpg
///   labels/train{N}.txt   (lines: "{class}/{class}_NNNN.jpg")
///   labels/val{N}.txt
///   labels/test{N}.txt
/// </code>
/// where N is the split index (1..10). Auto-downloads the canonical
/// VGG tarball.
/// </para>
/// </remarks>
public class DtdDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 47;
    private static readonly string DownloadUrl =
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz";

    private readonly DtdDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _imageSize;

    public override string Name => "DTD";
    public override string Description => "Describable Textures Dataset (47 classes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _imageSize * _imageSize * 3;
    public override int OutputDimension => NumClasses;

    public DtdDataLoader(DtdDataLoaderOptions? options = null)
    {
        _options = options ?? new DtdDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("dtd");
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
            throw new DirectoryNotFoundException($"DTD not found at {_dataPath}.");

        string splitPrefix = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "val",
            _ => "train"
        };
        int n = Math.Clamp(_options.SplitIndex, 1, 10);
        string splitFile = Path.Combine(root, "labels", $"{splitPrefix}{n}.txt");
        if (!File.Exists(splitFile))
            throw new FileNotFoundException($"DTD split file not found: {splitFile}");

        // Build class index from the unique class subdirectory names (sorted).
        string imagesDir = Path.Combine(root, "images");
        var classNames = Directory.GetDirectories(imagesDir)
            .Select(d => Path.GetFileName(d)!)
            .OrderBy(s => s, StringComparer.Ordinal)
            .Take(NumClasses).ToArray();
        var classIdx = new Dictionary<string, int>(classNames.Length, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < classNames.Length; i++) classIdx[classNames[i]] = i;

        var paths = new List<(string Path, int Label)>();
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(splitFile, cancellationToken))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string trimmed = line.Trim();
            int slash = trimmed.IndexOf('/');
            if (slash <= 0) continue;
            string clsName = trimmed.Substring(0, slash);
            if (!classIdx.TryGetValue(clsName, out int label)) continue;
            string imgPath = Path.Combine(imagesDir, trimmed);
            if (!File.Exists(imgPath)) continue;
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
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imgPath, _imageSize, _imageSize, 3, _options.Normalize);
            int featureOffset = i * pixelsPerImage;
            int copyLen = Math.Min(pixels.Length, pixelsPerImage);
            Array.Copy(pixels, 0, featuresData, featureOffset, copyLen);
            labelsData[i * NumClasses + label] = NumOps.One;
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
        string sub = Path.Combine(_dataPath, "dtd");
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
