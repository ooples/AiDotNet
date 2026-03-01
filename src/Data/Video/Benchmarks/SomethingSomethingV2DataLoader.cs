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

        // Build label template -> class index mapping from label file
        var labelToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        if (File.Exists(labelFile))
        {
            // First pass: collect unique labels to build class mapping
            var lines = await FilePolyfill.ReadAllLinesAsync(labelFile, cancellationToken);

            // Also check for a separate labels.json with template->index mapping
            string labelsMapFile = Path.Combine(_dataPath, "something-something-v2-labels.json");
            if (File.Exists(labelsMapFile))
            {
                var labelLines = await FilePolyfill.ReadAllLinesAsync(labelsMapFile, cancellationToken);
                foreach (var ll in labelLines)
                {
                    // {"label_template": index} or "label_template": index
                    int colonIdx2 = ll.IndexOf(':');
                    if (colonIdx2 < 0) continue;
                    string key = ll.Substring(0, colonIdx2).Trim().Trim('"', '{', ' ');
                    string val = ll.Substring(colonIdx2 + 1).Trim().Trim(',', '}', ' ');
                    if (int.TryParse(val, out int idx))
                        labelToIndex[key] = idx;
                }
            }

            foreach (var line in lines)
            {
                // Parse: {"id": "12345", "label": "Pushing something..."}
                int idIdx = line.IndexOf("\"id\"", StringComparison.Ordinal);
                if (idIdx < 0) continue;
                int colonIdx = line.IndexOf(':', idIdx);
                if (colonIdx < 0) continue;
                int quoteStart = line.IndexOf('"', colonIdx + 1);
                int quoteEnd = line.IndexOf('"', quoteStart + 1);
                if (quoteStart < 0 || quoteEnd < 0) continue;
                string videoId = line.Substring(quoteStart + 1, quoteEnd - quoteStart - 1);

                // Parse actual label
                int labelIdx = line.IndexOf("\"label\"", StringComparison.Ordinal);
                int classIndex = -1;
                if (labelIdx >= 0)
                {
                    int lColon = line.IndexOf(':', labelIdx);
                    if (lColon >= 0)
                    {
                        int lqStart = line.IndexOf('"', lColon + 1);
                        int lqEnd = lqStart >= 0 ? line.IndexOf('"', lqStart + 1) : -1;
                        if (lqStart >= 0 && lqEnd > lqStart)
                        {
                            string labelText = line.Substring(lqStart + 1, lqEnd - lqStart - 1);
                            if (!labelToIndex.TryGetValue(labelText, out classIndex))
                            {
                                classIndex = labelToIndex.Count % NumClasses;
                                labelToIndex[labelText] = classIndex;
                            }
                        }
                    }
                }

                if (classIndex < 0) classIndex = 0; // fallback for test split (no labels)

                string videoDir = Path.Combine(framesDir, videoId);
                if (Directory.Exists(videoDir))
                    samples.Add((videoDir, classIndex % NumClasses));
            }
        }

        if (samples.Count == 0)
        {
            // Fallback: collect frame directories with folder-based class assignment
            string searchDir = Directory.Exists(framesDir) ? framesDir : _dataPath;
            var dirs = Directory.GetDirectories(searchDir);
            Array.Sort(dirs, StringComparer.OrdinalIgnoreCase);
            foreach (var dir in dirs)
            {
                string dirName = Path.GetFileName(dir);
                if (!labelToIndex.TryGetValue(dirName, out int classIdx))
                {
                    classIdx = labelToIndex.Count % NumClasses;
                    labelToIndex[dirName] = classIdx;
                }
                samples.Add((dir, classIdx));
            }
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
