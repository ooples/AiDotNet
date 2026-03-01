using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the FSD50K audio event dataset (51,197 clips, 200 sound event classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// FSD50K expects:
/// <code>
/// {DataPath}/FSD50K/
///   FSD50K.dev_audio/       (WAV files)
///   FSD50K.eval_audio/
///   FSD50K.ground_truth/
///     dev.csv               (fname, labels, mids, split)
///     eval.csv
///     vocabulary.csv
/// </code>
/// CSV columns: fname, labels (comma-separated class names), mids, split.
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are multi-hot Tensor[N, 200].
/// </para>
/// </remarks>
public class Fsd50kDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 200;

    private readonly Fsd50kDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => "FSD50K";
    /// <inheritdoc/>
    public override string Description => "FSD50K sound event dataset (200 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    /// <summary>Creates a new FSD50K data loader.</summary>
    public Fsd50kDataLoader(Fsd50kDataLoaderOptions? options = null)
    {
        _options = options ?? new Fsd50kDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("fsd50k");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string baseDir = Path.Combine(_dataPath, "FSD50K");
        if (!Directory.Exists(baseDir))
            baseDir = _dataPath;

        // Build vocabulary from file
        string vocabFile = Path.Combine(baseDir, "FSD50K.ground_truth", "vocabulary.csv");
        var classToIndex = new Dictionary<string, int>();
        if (File.Exists(vocabFile))
        {
            var vocabLines = await FilePolyfill.ReadAllLinesAsync(vocabFile, cancellationToken);
            int idx = 0;
            foreach (var line in vocabLines)
            {
                string trimmed = line.Trim();
                if (trimmed.Length > 0 && idx < NumClasses)
                {
                    classToIndex[trimmed] = idx;
                    idx++;
                }
            }
        }

        bool isEval = _options.Split == Geometry.DatasetSplit.Test;
        string csvFile = isEval
            ? Path.Combine(baseDir, "FSD50K.ground_truth", "eval.csv")
            : Path.Combine(baseDir, "FSD50K.ground_truth", "dev.csv");

        if (!File.Exists(csvFile))
        {
            throw new FileNotFoundException(
                $"FSD50K metadata not found at {csvFile}. " +
                "Download from https://zenodo.org/record/4060432.");
        }

        string audioDir = isEval
            ? Path.Combine(baseDir, "FSD50K.eval_audio")
            : Path.Combine(baseDir, "FSD50K.dev_audio");

        if (!Directory.Exists(audioDir))
            audioDir = baseDir;

        var lines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string AudioPath, int[] LabelIndices)>();

        // CSV: fname, labels, mids, split
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(new[] { ',' }, 4);
            if (parts.Length < 2) continue;

            string fname = parts[0].Trim();
            string labelsStr = parts[1].Trim();

            var labelNames = labelsStr.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var labelIndices = new List<int>();
            foreach (var name in labelNames)
            {
                if (classToIndex.TryGetValue(name.Trim(), out int idx))
                    labelIndices.Add(idx);
            }

            string wavPath = Path.Combine(audioDir, fname + ".wav");
            if (!File.Exists(wavPath))
                wavPath = Path.Combine(audioDir, fname);

            if (File.Exists(wavPath))
                samples.Add((wavPath, labelIndices.ToArray()));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _maxAudioSamples];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (audioPath, labelIndices) = samples[i];

            if (File.Exists(audioPath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
                AudioLoaderHelper.LoadAudioSamples(audioBytes, featuresData, i * _maxAudioSamples, _maxAudioSamples, NumOps);
            }

            // Multi-hot encode labels
            int labelOffset = i * NumClasses;
            foreach (int idx in labelIndices)
            {
                if (idx >= 0 && idx < NumClasses)
                    labelsData[labelOffset + idx] = NumOps.FromDouble(1.0);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _maxAudioSamples });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
        InitializeIndices(totalSamples);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (AudioLoaderHelper.ExtractTensorBatch(features, indices), AudioLoaderHelper.ExtractTensorBatch(labels, indices));
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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }
}
