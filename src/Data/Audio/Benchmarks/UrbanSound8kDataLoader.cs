using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the UrbanSound8K environmental sound dataset (8732 clips, 10 classes, 10 folds).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// UrbanSound8K expects:
/// <code>
/// {DataPath}/UrbanSound8K/
///   audio/
///     fold1/  through fold10/
///       {fsID}-{classID}-{occurrenceID}-{sliceID}.wav
///   metadata/
///     UrbanSound8K.csv
/// </code>
/// CSV columns: slice_file_name, fsID, start, end, salience, fold, classID, class.
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are one-hot Tensor[N, 10].
/// </para>
/// </remarks>
public class UrbanSound8kDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 10;

    private readonly UrbanSound8kDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => "UrbanSound8K";
    /// <inheritdoc/>
    public override string Description => "UrbanSound8K urban sound classification (10 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    /// <summary>Creates a new UrbanSound8K data loader.</summary>
    public UrbanSound8kDataLoader(UrbanSound8kDataLoaderOptions? options = null)
    {
        _options = options ?? new UrbanSound8kDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("urbansound8k");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string baseDir = Path.Combine(_dataPath, "UrbanSound8K");
        if (!Directory.Exists(baseDir))
            baseDir = _dataPath;

        string audioDir = Path.Combine(baseDir, "audio");
        string csvFile = Path.Combine(baseDir, "metadata", "UrbanSound8K.csv");

        if (!File.Exists(csvFile))
        {
            csvFile = Path.Combine(_dataPath, "UrbanSound8K.csv");
            if (!File.Exists(csvFile))
            {
                throw new FileNotFoundException(
                    $"UrbanSound8K metadata not found. " +
                    "Download from https://urbansounddataset.weebly.com/urbansound8k.html.");
            }
        }

        if (!Directory.Exists(audioDir))
            audioDir = _dataPath;

        var lines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string AudioPath, int ClassIndex)>();

        // CSV: slice_file_name, fsID, start, end, salience, fold, classID, class
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(new[] { ',' }, 8);
            if (parts.Length < 7) continue;

            string filename = parts[0].Trim();
            if (!int.TryParse(parts[5].Trim(), out int fold)) continue;
            if (!int.TryParse(parts[6].Trim(), out int classId)) continue;

            // Split by fold: test fold goes to test, others to train
            bool isTestFold = fold == _options.TestFold;
            bool wantTest = _options.Split == Geometry.DatasetSplit.Test;
            if (isTestFold != wantTest) continue;

            string wavPath = Path.Combine(audioDir, $"fold{fold}", filename);
            if (!File.Exists(wavPath))
                wavPath = Path.Combine(audioDir, filename);

            if (File.Exists(wavPath))
                samples.Add((wavPath, classId));
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
            var (audioPath, classIndex) = samples[i];

            if (File.Exists(audioPath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
                AudioLoaderHelper.LoadAudioSamples(audioBytes, featuresData, i * _maxAudioSamples, _maxAudioSamples, NumOps);
            }

            // One-hot encode class
            if (classIndex >= 0 && classIndex < NumClasses)
                labelsData[i * NumClasses + classIndex] = NumOps.FromDouble(1.0);
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
