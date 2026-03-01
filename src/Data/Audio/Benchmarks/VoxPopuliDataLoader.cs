using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the VoxPopuli multilingual speech dataset from European Parliament recordings.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// VoxPopuli expects pre-downloaded audio and TSV transcription files:
/// <code>
/// {DataPath}/
///   transcribed_data/{Language}/
///     train.tsv
///     dev.tsv
///     test.tsv
///   {Language}/
///     2009/              (WAV files organized by year)
///     2010/
///     ...
/// </code>
/// Features are raw waveform samples as Tensor[N, MaxSamples].
/// Labels are character-encoded transcripts as Tensor[N, MaxTextLen].
/// </para>
/// </remarks>
public class VoxPopuliDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly VoxPopuliDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => $"VoxPopuli-{_options.Language}";
    /// <inheritdoc/>
    public override string Description => $"VoxPopuli speech dataset ({_options.Language})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => 256;

    /// <summary>Creates a new VoxPopuli data loader.</summary>
    public VoxPopuliDataLoader(VoxPopuliDataLoaderOptions? options = null)
    {
        _options = options ?? new VoxPopuliDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("voxpopuli");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string tsvDir = Path.Combine(_dataPath, "transcribed_data", _options.Language);
        if (!Directory.Exists(tsvDir))
            tsvDir = _dataPath;

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "dev",
            _ => "train"
        };
        string tsvFile = Path.Combine(tsvDir, $"{splitName}.tsv");
        if (!File.Exists(tsvFile))
        {
            throw new FileNotFoundException(
                $"VoxPopuli TSV not found at {tsvFile}. " +
                "Download from https://github.com/facebookresearch/voxpopuli.");
        }

        var lines = await FilePolyfill.ReadAllLinesAsync(tsvFile, cancellationToken);
        var samples = new List<(string AudioPath, string Transcript)>();

        // Skip header; columns typically: id, raw_text, normalized_text, speaker_id, ...
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split('\t');
            if (parts.Length < 2) continue;

            string audioId = parts[0].Trim();
            string transcript = parts.Length >= 3 ? parts[2].Trim() : parts[1].Trim();

            // Look for audio file in language directory
            string wavPath = FindAudioFile(audioId);
            if (wavPath.Length > 0)
                samples.Add((wavPath, transcript));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _maxAudioSamples];
        int maxTextLen = 256;
        var labelsData = new T[totalSamples * maxTextLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (audioPath, transcript) = samples[i];

            if (File.Exists(audioPath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
                AudioLoaderHelper.LoadAudioSamples(audioBytes, featuresData, i * _maxAudioSamples, _maxAudioSamples, NumOps);
            }

            int labelOffset = i * maxTextLen;
            for (int c = 0; c < Math.Min(transcript.Length, maxTextLen); c++)
                labelsData[labelOffset + c] = NumOps.FromDouble(transcript[c]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _maxAudioSamples });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, maxTextLen });
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

    private string FindAudioFile(string audioId)
    {
        // VoxPopuli organizes audio by year directories
        string langDir = Path.Combine(_dataPath, _options.Language);
        if (Directory.Exists(langDir))
        {
            // Audio ID format is typically "YYYYMMDD-XXXX_XXXXXX-XXXXXX"
            string wavPath = Path.Combine(langDir, audioId + ".wav");
            if (File.Exists(wavPath)) return wavPath;

            // Try searching in year subdirectories
            if (audioId.Length >= 4)
            {
                string year = audioId.Substring(0, 4);
                wavPath = Path.Combine(langDir, year, audioId + ".wav");
                if (File.Exists(wavPath)) return wavPath;
            }
        }

        // Fallback: search root
        string rootPath = Path.Combine(_dataPath, audioId + ".wav");
        return File.Exists(rootPath) ? rootPath : string.Empty;
    }
}
