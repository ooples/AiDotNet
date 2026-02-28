using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the AudioSet large-scale audio event dataset (2M+ 10-second clips, 527 categories).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// AudioSet expects pre-downloaded audio and CSV label files:
/// <code>
/// {DataPath}/
///   audio/                 (WAV files, pre-downloaded from YouTube)
///   balanced_train_segments.csv
///   unbalanced_train_segments.csv
///   eval_segments.csv
///   class_labels_indices.csv
/// </code>
/// CSV format: YTID, start_seconds, end_seconds, positive_labels (comma-separated label IDs).
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are multi-hot Tensor[N, 527].
/// </para>
/// </remarks>
public class AudioSetDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 527;

    private readonly AudioSetDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => "AudioSet";
    /// <inheritdoc/>
    public override string Description => "AudioSet audio event dataset (527 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    /// <summary>Creates a new AudioSet data loader.</summary>
    public AudioSetDataLoader(AudioSetDataLoaderOptions? options = null)
    {
        _options = options ?? new AudioSetDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("audioset");
        _maxAudioSamples = (int)(_options.SampleRate * _options.ClipDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Load class label indices
        string classFile = Path.Combine(_dataPath, "class_labels_indices.csv");
        var labelToIndex = new Dictionary<string, int>();
        if (File.Exists(classFile))
        {
            var classLines = await FilePolyfill.ReadAllLinesAsync(classFile, cancellationToken);
            foreach (var line in classLines)
            {
                var parts = line.Split(new[] { ',' }, 3);
                if (parts.Length >= 2 && int.TryParse(parts[0].Trim(), out int idx))
                    labelToIndex[parts[1].Trim()] = idx;
            }
        }

        // Choose CSV based on split
        string csvFile = _options.Split switch
        {
            Geometry.DatasetSplit.Test => Path.Combine(_dataPath, "eval_segments.csv"),
            _ => Path.Combine(_dataPath, "balanced_train_segments.csv")
        };

        if (!File.Exists(csvFile))
        {
            throw new FileNotFoundException(
                $"AudioSet CSV not found at {csvFile}. " +
                "Download from https://research.google.com/audioset/.");
        }

        string audioDir = Path.Combine(_dataPath, "audio");
        var lines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string AudioPath, int[] LabelIndices)>();

        // Skip comment lines (start with #)
        foreach (var line in lines)
        {
            if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line)) continue;

            // Format: YTID, start_seconds, end_seconds, positive_labels
            int firstComma = line.IndexOf(',');
            if (firstComma < 0) continue;

            string ytid = line.Substring(0, firstComma).Trim();
            // Find labels after the third comma
            int commaCount = 0;
            int labelStart = -1;
            for (int ci = 0; ci < line.Length; ci++)
            {
                if (line[ci] == ',')
                {
                    commaCount++;
                    if (commaCount == 3) { labelStart = ci + 1; break; }
                }
            }

            if (labelStart < 0) continue;
            string labelsStr = line.Substring(labelStart).Trim().Trim('"');
            var labelIds = labelsStr.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);

            var indices = new List<int>();
            foreach (var lid in labelIds)
            {
                string trimmed = lid.Trim();
                if (labelToIndex.TryGetValue(trimmed, out int idx))
                    indices.Add(idx);
            }

            string wavPath = Path.Combine(audioDir, ytid + ".wav");
            if (File.Exists(wavPath))
                samples.Add((wavPath, indices.ToArray()));
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
                AudioLoaderHelper.LoadWavSamples(audioBytes, featuresData, i * _maxAudioSamples, _maxAudioSamples, NumOps);
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
