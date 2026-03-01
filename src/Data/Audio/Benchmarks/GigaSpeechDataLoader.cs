using System.Text.Json;
using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the GigaSpeech multi-domain English ASR dataset (10K hours from audiobooks, podcasts, YouTube).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// GigaSpeech expects JSON manifest files and pre-converted WAV audio:
/// <code>
/// {DataPath}/
///   audio/
///     audiobook/
///     podcast/
///     youtube/
///   GigaSpeech.json  (manifest with segments)
/// </code>
/// </para>
/// </remarks>
public class GigaSpeechDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly GigaSpeechDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => $"GigaSpeech-{_options.Subset}";
    /// <inheritdoc/>
    public override string Description => $"GigaSpeech ASR dataset ({_options.Subset} subset)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => 256;

    /// <summary>Creates a new GigaSpeech data loader.</summary>
    public GigaSpeechDataLoader(GigaSpeechDataLoaderOptions? options = null)
    {
        _options = options ?? new GigaSpeechDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("gigaspeech");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string audioDir = Path.Combine(_dataPath, "audio");
        if (!Directory.Exists(audioDir))
        {
            throw new DirectoryNotFoundException(
                $"GigaSpeech data not found at {_dataPath}. " +
                "Download from https://github.com/SpeechColab/GigaSpeech.");
        }

        // Collect WAV files with transcriptions from segment manifest
        var samples = new List<(string AudioPath, string Transcript)>();
        string manifestFile = Path.Combine(_dataPath, "GigaSpeech.json");

        if (File.Exists(manifestFile))
        {
            // GigaSpeech manifest is a JSON file with "audios" array, each containing "segments"
            // Format: { "audios": [ { "aid": "...", "path": "...", "segments": [ { "sid": "...", "begin_time": 0.0, "end_time": 5.0, "text_tn": "..." } ] } ] }
            using var stream = File.OpenRead(manifestFile);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
            var root = doc.RootElement;

            string subsetFilter = _options.Subset.ToUpperInvariant();
            if (root.TryGetProperty("audios", out var audiosElem))
            {
                foreach (var audio in audiosElem.EnumerateArray())
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Check subset filter if specified (XS, S, M, L, XL)
                    if (audio.TryGetProperty("subsets", out var subsetsElem))
                    {
                        bool matchesSubset = false;
                        foreach (var subset in subsetsElem.EnumerateArray())
                        {
                            string subsetStr = subset.GetString() ?? string.Empty;
                            if (subsetStr.Contains(subsetFilter, StringComparison.OrdinalIgnoreCase))
                            {
                                matchesSubset = true;
                                break;
                            }
                        }
                        if (!matchesSubset) continue;
                    }

                    string audioPath = audio.TryGetProperty("path", out var pathElem)
                        ? pathElem.GetString() ?? string.Empty
                        : string.Empty;

                    if (audio.TryGetProperty("segments", out var segmentsElem))
                    {
                        foreach (var segment in segmentsElem.EnumerateArray())
                        {
                            string text = segment.TryGetProperty("text_tn", out var textElem)
                                ? textElem.GetString() ?? string.Empty
                                : string.Empty;

                            string sid = segment.TryGetProperty("sid", out var sidElem)
                                ? sidElem.GetString() ?? string.Empty
                                : string.Empty;

                            // Try segment-specific WAV, then audio-level path
                            string segWav = Path.Combine(audioDir, sid + ".wav");
                            string fullAudioPath = audioPath.Length > 0
                                ? Path.Combine(_dataPath, audioPath)
                                : string.Empty;

                            string resolvedPath = File.Exists(segWav) ? segWav
                                : fullAudioPath.Length > 0 && File.Exists(fullAudioPath) ? fullAudioPath
                                : string.Empty;

                            if (resolvedPath.Length > 0)
                            {
                                samples.Add((resolvedPath, text));
                            }
                        }
                    }
                }
            }
        }

        // Fallback: collect all WAV files in audio directory
        if (samples.Count == 0)
        {
            var wavFiles = Directory.GetFiles(audioDir, "*.wav", SearchOption.AllDirectories);
            foreach (var wav in wavFiles)
            {
                samples.Add((wav, string.Empty));
            }
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
