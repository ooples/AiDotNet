using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the Mozilla Common Voice multilingual speech dataset (19K+ hours, 100+ languages).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Common Voice expects pre-converted WAV files (original MP3 format not supported):
/// <code>
/// {DataPath}/cv-corpus-{version}/{Language}/
///   clips/         (WAV files, pre-converted from MP3)
///   train.tsv
///   dev.tsv
///   test.tsv
/// </code>
/// The TSV files contain columns: client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment.
/// Features are raw waveform Tensor[N, MaxSamples]. Labels are character-encoded transcripts Tensor[N, MaxTextLen].
/// </para>
/// </remarks>
public class CommonVoiceDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly CommonVoiceDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => $"CommonVoice-{_options.Language}";
    /// <inheritdoc/>
    public override string Description => $"Mozilla Common Voice ({_options.Language})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => 256;

    /// <summary>Creates a new Common Voice data loader.</summary>
    public CommonVoiceDataLoader(CommonVoiceDataLoaderOptions? options = null)
    {
        _options = options ?? new CommonVoiceDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("common-voice");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string langDir = FindLanguageDirectory(_dataPath, _options.Language);
        if (langDir.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"Common Voice data not found at {_dataPath}. " +
                "Download from https://commonvoice.mozilla.org/ and pre-convert MP3 to WAV.");
        }

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "dev",
            _ => "train"
        };
        string tsvFile = Path.Combine(langDir, $"{splitName}.tsv");
        if (!File.Exists(tsvFile))
        {
            throw new FileNotFoundException($"Common Voice TSV file not found: {tsvFile}");
        }

        string clipsDir = Path.Combine(langDir, "clips");
        var lines = await FilePolyfill.ReadAllLinesAsync(tsvFile, cancellationToken);
        var samples = new List<(string AudioPath, string Transcript)>();

        // Skip header
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split('\t');
            if (parts.Length < 3) continue;

            string audioFile = parts[1].Trim();
            string transcript = parts[2].Trim();

            // Replace .mp3 with .wav for pre-converted files
            string wavFile = Path.ChangeExtension(audioFile, ".wav");
            string audioPath = Path.Combine(clipsDir, wavFile);
            if (!File.Exists(audioPath))
                audioPath = Path.Combine(clipsDir, audioFile);

            if (File.Exists(audioPath))
                samples.Add((audioPath, transcript));
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

            byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
            int featureOffset = i * _maxAudioSamples;
            AudioLoaderHelper.LoadAudioSamples(audioBytes, featuresData, featureOffset, _maxAudioSamples, NumOps);

            int labelOffset = i * maxTextLen;
            int charCount = Math.Min(transcript.Length, maxTextLen);
            for (int c = 0; c < charCount; c++)
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
        return (AudioLoaderHelper.ExtractTensorBatch<T>(features, indices), AudioLoaderHelper.ExtractTensorBatch<T>(labels, indices));
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
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()),
                AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()),
                AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static string FindLanguageDirectory(string rootPath, string lang)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        // Search for language folder in common corpus directories
        var dirs = Directory.GetDirectories(rootPath, "*", SearchOption.AllDirectories);
        foreach (var dir in dirs)
        {
            if (Path.GetFileName(dir).Equals(lang, StringComparison.OrdinalIgnoreCase) &&
                File.Exists(Path.Combine(dir, "train.tsv")))
                return dir;
        }

        if (File.Exists(Path.Combine(rootPath, "train.tsv")))
            return rootPath;

        return string.Empty;
    }
}
