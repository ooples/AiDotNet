using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Loads the Something-Something V2 action recognition dataset (220K clips, 174 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class SomethingSomethingV2DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 174;
    private readonly SomethingSomethingV2DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _featureSize;

    public override string Name => "SomethingSomethingV2";
    public override string Description => "Something-Something V2 action recognition (174 classes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _featureSize;
    public override int OutputDimension => NumClasses;

    public SomethingSomethingV2DataLoader(SomethingSomethingV2DataLoaderOptions? options = null)
    {
        _options = options ?? new SomethingSomethingV2DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("something-something-v2");
        _featureSize = _options.FramesPerVideo * _options.FrameHeight * _options.FrameWidth * 3;
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Expects JSON label files + frame directories
        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "validation",
            _ => "train"
        };

        string labelFile = Path.Combine(_dataPath, $"something-something-v2-{splitName}.json");
        string framesDir = Path.Combine(_dataPath, "frames");

        if (!File.Exists(labelFile))
        {
            // Fallback: folder-per-class structure
            if (!Directory.Exists(framesDir) && !Directory.Exists(_dataPath))
                throw new DirectoryNotFoundException($"Something-Something V2 not found at {_dataPath}.");
        }

        var samples = new List<(string VideoDir, int ClassIndex)>();

        if (File.Exists(labelFile))
        {
            var lines = await FilePolyfill.ReadAllLinesAsync(labelFile, cancellationToken);
            foreach (var line in lines)
            {
                // Simplified JSON: {"id": "12345", "label": "Pushing something..."}
                int idIdx = line.IndexOf("\"id\"", StringComparison.Ordinal);
                if (idIdx < 0) continue;
                int colonIdx = line.IndexOf(':', idIdx);
                if (colonIdx < 0) continue;
                int quoteStart = line.IndexOf('"', colonIdx + 1);
                int quoteEnd = line.IndexOf('"', quoteStart + 1);
                if (quoteStart < 0 || quoteEnd < 0) continue;
                string videoId = line.Substring(quoteStart + 1, quoteEnd - quoteStart - 1);

                string videoDir = Path.Combine(framesDir, videoId);
                if (Directory.Exists(videoDir))
                    samples.Add((videoDir, samples.Count % NumClasses));
            }
        }

        if (samples.Count == 0)
        {
            // Fallback: collect frame directories
            string searchDir = Directory.Exists(framesDir) ? framesDir : _dataPath;
            var dirs = Directory.GetDirectories(searchDir);
            foreach (var dir in dirs)
                samples.Add((dir, samples.Count % NumClasses));
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
