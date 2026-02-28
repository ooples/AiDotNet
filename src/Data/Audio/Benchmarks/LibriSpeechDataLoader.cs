using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the LibriSpeech automatic speech recognition dataset (~1000 hours of 16kHz English speech).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// LibriSpeech expects:
/// <code>
/// {DataPath}/LibriSpeech/
///   train-clean-100/
///     19/
///       198/
///         19-198-0000.flac (or .wav)
///         19-198-0001.flac
///         19-198.trans.txt
/// </code>
/// Features are raw waveform samples as Tensor[N, MaxSamples] (mono, 16kHz).
/// Labels are transcript text encoded as character indices in Tensor[N, MaxTextLen].
/// For pre-converted WAV files, audio is loaded directly. FLAC requires pre-conversion to WAV.
/// </para>
/// </remarks>
public class LibriSpeechDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string BaseUrl = "https://www.openslr.org/resources/12/";

    private readonly LibriSpeechDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _maxAudioSamples;

    /// <inheritdoc/>
    public override string Name => "LibriSpeech";
    /// <inheritdoc/>
    public override string Description => "LibriSpeech ASR dataset (1000 hours, 16kHz English)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _maxAudioSamples;
    /// <inheritdoc/>
    public override int OutputDimension => 256; // Character-level vocab

    /// <summary>Creates a new LibriSpeech data loader.</summary>
    public LibriSpeechDataLoader(LibriSpeechDataLoaderOptions? options = null)
    {
        _options = options ?? new LibriSpeechDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("librispeech");
        _maxAudioSamples = (int)(_options.SampleRate * _options.MaxDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string subsetDir = FindSubsetDirectory(_dataPath, _options.Subset);

        if (subsetDir.Length == 0 && _options.AutoDownload)
        {
            string url = $"{BaseUrl}{_options.Subset}.tar.gz";
            await DatasetDownloader.DownloadAndExtractTarGzAsync(url, _dataPath, cancellationToken);
            subsetDir = FindSubsetDirectory(_dataPath, _options.Subset);
        }

        if (subsetDir.Length == 0)
        {
            throw new DirectoryNotFoundException(
                $"LibriSpeech data not found at {_dataPath}. Enable AutoDownload or provide pre-converted WAV data.");
        }

        // Collect all audio files and their transcriptions
        var samples = new List<(string AudioPath, string Transcript)>();
        CollectSamples(subsetDir, samples);

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _maxAudioSamples];
        int maxTextLen = 256;
        var labelsData = new T[totalSamples * maxTextLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (audioPath, transcript) = samples[i];

            // Load WAV audio
            if (File.Exists(audioPath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(audioPath, cancellationToken);
                int featureOffset = i * _maxAudioSamples;
                AudioLoaderHelper.LoadWavSamples(audioBytes, featuresData, featureOffset, _maxAudioSamples, NumOps);
            }

            // Encode transcript as character indices
            int labelOffset = i * maxTextLen;
            int charCount = Math.Min(transcript.Length, maxTextLen);
            for (int c = 0; c < charCount; c++)
            {
                labelsData[labelOffset + c] = NumOps.FromDouble(transcript[c]);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _maxAudioSamples });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, maxTextLen });
        InitializeIndices(totalSamples);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
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
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
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

    private static void CollectSamples(string dir, List<(string AudioPath, string Transcript)> samples)
    {
        // Find all .trans.txt files and parse transcriptions
        var transFiles = Directory.GetFiles(dir, "*.trans.txt", SearchOption.AllDirectories);
        foreach (var transFile in transFiles)
        {
            string transDir = Path.GetDirectoryName(transFile) ?? dir;
            var lines = File.ReadAllLines(transFile);
            foreach (var line in lines)
            {
                int spaceIdx = line.IndexOf(' ');
                if (spaceIdx <= 0) continue;

                string uttId = line.Substring(0, spaceIdx);
                string transcript = line.Substring(spaceIdx + 1).Trim();

                // Look for WAV first, then FLAC
                string wavPath = Path.Combine(transDir, uttId + ".wav");
                string flacPath = Path.Combine(transDir, uttId + ".flac");

                string audioPath = File.Exists(wavPath) ? wavPath : flacPath;
                if (File.Exists(audioPath))
                {
                    samples.Add((audioPath, transcript));
                }
            }
        }
    }

    private static string FindSubsetDirectory(string rootPath, string subset)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;

        string[] candidates =
        {
            Path.Combine(rootPath, "LibriSpeech", subset),
            Path.Combine(rootPath, subset),
            rootPath
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate))
            {
                var transFiles = Directory.GetFiles(candidate, "*.trans.txt", SearchOption.AllDirectories);
                if (transFiles.Length > 0) return candidate;
            }
        }

        return string.Empty;
    }

}
