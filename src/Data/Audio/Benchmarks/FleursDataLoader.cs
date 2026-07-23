using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the FLEURS multilingual speech benchmark (102 languages, ~12 hours per language).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// FLEURS expects:
/// <code>
/// {DataPath}/{Language}/
///   audio/
///     train/    (WAV files)
///     dev/
///     test/
///   train.tsv
///   dev.tsv
///   test.tsv
/// </code>
/// TSV columns: id, file_name, raw_transcription, transcription, num_samples, gender.
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are character-encoded transcripts Tensor[N, MaxTextLen].
/// </para>
/// </remarks>
public class FleursDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly FleursDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => $"FLEURS-{_options.Language}";
    /// <inheritdoc/>
    public override string Description => $"FLEURS multilingual speech benchmark ({_options.Language})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => 256;

    /// <summary>Creates a new FLEURS data loader.</summary>
    public FleursDataLoader(FleursDataLoaderOptions? options = null)
    {
        _options = options ?? new FleursDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("fleurs");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string langDir = Path.Combine(_dataPath, _options.Language);
        if (!Directory.Exists(langDir))
            langDir = _dataPath;

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "dev",
            _ => "train"
        };

        string tsvFile = Path.Combine(langDir, $"{splitName}.tsv");
        if (!File.Exists(tsvFile))
        {
            throw new FileNotFoundException(
                $"FLEURS TSV not found at {tsvFile}. " +
                "Download from https://huggingface.co/datasets/google/fleurs.");
        }

        string audioDir = Path.Combine(langDir, "audio", splitName);
        if (!Directory.Exists(audioDir))
            audioDir = langDir;

        var lines = await FilePolyfill.ReadAllLinesAsync(tsvFile, cancellationToken);
        var samples = new List<(string AudioPath, string Transcript)>();

        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split('\t');
            if (parts.Length < 4) continue;

            string fileName = parts[1].Trim();
            string transcript = parts[3].Trim(); // normalized transcription

            string wavPath = Path.Combine(audioDir, fileName);
            if (!File.Exists(wavPath))
                wavPath = Path.Combine(audioDir, Path.ChangeExtension(fileName, ".wav"));

            if (File.Exists(wavPath))
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
}
