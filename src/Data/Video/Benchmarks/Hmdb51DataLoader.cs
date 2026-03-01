using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Loads the HMDB51 action recognition dataset (6,766 clips, 51 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class Hmdb51DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 51;
    private readonly Hmdb51DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _featureSize;

    /// <inheritdoc/>
    public override string Name => "HMDB51";
    /// <inheritdoc/>
    public override string Description => "HMDB51 action recognition (51 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _featureSize;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    public Hmdb51DataLoader(Hmdb51DataLoaderOptions? options = null)
    {
        _options = options ?? new Hmdb51DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("hmdb51");
        _featureSize = _options.FramesPerVideo * _options.FrameHeight * _options.FrameWidth * 3;
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string dataDir = _dataPath;
        if (!Directory.Exists(dataDir))
            throw new DirectoryNotFoundException($"HMDB51 data not found at {dataDir}.");

        var classDirs = Directory.GetDirectories(dataDir)
            .Where(d => !Path.GetFileName(d).StartsWith(".")) // skip hidden dirs
            .OrderBy(d => Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)
            .ToArray();
        // Build stable class name -> index mapping from sorted directory names
        var classNameToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int ci = 0; ci < Math.Min(classDirs.Length, NumClasses); ci++)
            classNameToIndex[Path.GetFileName(classDirs[ci])] = ci;

        var samples = new List<(string VideoDir, int ClassIndex)>();

        foreach (var classDir in classDirs)
        {
            string className = Path.GetFileName(classDir);
            if (!classNameToIndex.TryGetValue(className, out int classIdx)) continue;
            var videoDirs = Directory.GetDirectories(classDir);
            foreach (var videoDir in videoDirs)
                samples.Add((videoDir, classIdx));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _featureSize];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            VideoLoaderHelper.LoadFrames(samples[i].VideoDir, featuresData, i * _featureSize,
                _options.FramesPerVideo, _options.FrameWidth, _options.FrameHeight, _options.Normalize, NumOps);
            labelsData[i * NumClasses + samples[i].ClassIndex] = NumOps.FromDouble(1.0);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (VideoLoaderHelper.ExtractTensorBatch(features, indices), VideoLoaderHelper.ExtractTensorBatch(labels, indices));
    }

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
        var f = LoadedFeatures ?? throw new InvalidOperationException("Not loaded.");
        var l = LoadedLabels ?? throw new InvalidOperationException("Not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(VideoLoaderHelper.ExtractTensorBatch(f, shuffled.Take(trainSize).ToArray()), VideoLoaderHelper.ExtractTensorBatch(l, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(VideoLoaderHelper.ExtractTensorBatch(f, shuffled.Skip(trainSize).Take(valSize).ToArray()), VideoLoaderHelper.ExtractTensorBatch(l, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(VideoLoaderHelper.ExtractTensorBatch(f, shuffled.Skip(trainSize + valSize).ToArray()), VideoLoaderHelper.ExtractTensorBatch(l, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }
}
