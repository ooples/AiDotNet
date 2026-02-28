using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the ESC-50 environmental sound classification dataset (2000 clips, 50 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// ESC-50 expects:
/// <code>
/// {DataPath}/ESC-50-master/
///   audio/
///     {fold}-{clipid}-{take}-{target}.wav
///   meta/
///     esc50.csv
/// </code>
/// CSV columns: filename, fold, target, category, esc10, src_file, take.
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are class index Tensor[N, 1].
/// </para>
/// </remarks>
public class Esc50DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 50;
    private const double ClipDuration = 5.0;

    private readonly Esc50DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => "ESC-50";
    /// <inheritdoc/>
    public override string Description => "ESC-50 environmental sound classification (50 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    /// <summary>Creates a new ESC-50 data loader.</summary>
    public Esc50DataLoader(Esc50DataLoaderOptions? options = null)
    {
        _options = options ?? new Esc50DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("esc50");
        _maxAudioSamples = (int)(_options.SampleRate * ClipDuration);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string baseDir = Path.Combine(_dataPath, "ESC-50-master");
        if (!Directory.Exists(baseDir))
            baseDir = _dataPath;

        string audioDir = Path.Combine(baseDir, "audio");
        string csvFile = Path.Combine(baseDir, "meta", "esc50.csv");

        if (!File.Exists(csvFile))
        {
            // Fallback: try flat structure
            csvFile = Path.Combine(_dataPath, "esc50.csv");
            if (!File.Exists(csvFile))
            {
                throw new FileNotFoundException(
                    $"ESC-50 metadata not found. Expected at {Path.Combine(baseDir, "meta", "esc50.csv")}. " +
                    "Download from https://github.com/karolpiczak/ESC-50.");
            }
        }

        if (!Directory.Exists(audioDir))
            audioDir = _dataPath;

        var lines = await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken);
        var samples = new List<(string AudioPath, int ClassIndex)>();

        // CSV: filename, fold, target, category, esc10, src_file, take
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(new[] { ',' }, 7);
            if (parts.Length < 3) continue;

            string filename = parts[0].Trim();
            if (!int.TryParse(parts[1].Trim(), out int fold)) continue;
            if (!int.TryParse(parts[2].Trim(), out int target)) continue;

            // Split by fold: test fold goes to test, others to train
            bool isTestFold = fold == _options.TestFold;
            bool wantTest = _options.Split == Geometry.DatasetSplit.Test;
            if (isTestFold != wantTest) continue;

            string wavPath = Path.Combine(audioDir, filename);
            if (File.Exists(wavPath))
                samples.Add((wavPath, target));
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
                AudioLoaderHelper.LoadWavSamples(audioBytes, featuresData, i * _maxAudioSamples, _maxAudioSamples, NumOps);
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
