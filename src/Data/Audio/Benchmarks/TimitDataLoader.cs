using AiDotNet.Data;
using AiDotNet.Data.Loaders;
using AiDotNet.Data.Text.Benchmarks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the TIMIT acoustic-phonetic continuous-speech corpus (Garofolo et al. 1993).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects the canonical LDC distribution layout (manually extracted —
/// requires LDC membership; no auto-download):
/// <code>
/// {DataPath}/TIMIT/{TRAIN,TEST}/{DR1..DR8}/{speaker}/{utterance}.{wav,txt,wrd,phn}
/// </code>
/// Features Tensor[N, MaxTextLength] (encoded transcript); labels
/// Tensor[N, MaxAudioSamples] waveform. The .txt files contain a single
/// sentence per utterance; the WAV files are NIST sphere format originally
/// but most distributions ship standard PCM WAV (this loader assumes PCM).
/// </para>
/// </remarks>
public class TimitDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly TimitDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "TIMIT";
    public override string Description => "TIMIT acoustic-phonetic continuous speech corpus (LDC93S1)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxTextLength;
    public override int OutputDimension => _options.MaxAudioSamples;

    public TimitDataLoader(TimitDataLoaderOptions? options = null)
    {
        _options = options ?? new TimitDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("timit");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (_options.AutoDownload)
            throw new InvalidOperationException(
                "TIMIT requires an LDC membership (catalog LDC93S1) — auto-download is not supported. " +
                $"Manually extract the distribution under {_dataPath}/TIMIT/.");

        string root = ResolveDataDir();
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"TIMIT not found at {_dataPath}.");

        string splitDir = _options.Split == Geometry.DatasetSplit.Train
            ? Path.Combine(root, "TRAIN")
            : Path.Combine(root, "TEST");
        if (!Directory.Exists(splitDir))
            throw new DirectoryNotFoundException($"TIMIT split dir not found: {splitDir}");

        // Walk DR* / speaker / utterance structure. Each utterance contributes one .wav + one .txt.
        var pairs = new List<(string Wav, string Text)>();
        foreach (string drDir in Directory.EnumerateDirectories(splitDir, "DR*", SearchOption.TopDirectoryOnly))
        {
            foreach (string spkDir in Directory.EnumerateDirectories(drDir))
            {
                foreach (string wav in Directory.EnumerateFiles(spkDir, "*.WAV").Concat(Directory.EnumerateFiles(spkDir, "*.wav")))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string txt = Path.ChangeExtension(wav, ".TXT");
                    if (!File.Exists(txt)) txt = Path.ChangeExtension(wav, ".txt");
                    if (!File.Exists(txt)) continue;
                    string raw = (await FilePolyfill.ReadAllTextAsync(txt, cancellationToken)).Trim();
                    // .TXT format: "<start_sample> <end_sample> <transcription>" — strip the two leading numbers.
                    int firstSpace = raw.IndexOf(' ');
                    int secondSpace = firstSpace > 0 ? raw.IndexOf(' ', firstSpace + 1) : -1;
                    string transcription = secondSpace > 0 ? raw.Substring(secondSpace + 1).Trim() : raw;
                    if (string.IsNullOrEmpty(transcription)) continue;
                    pairs.Add((wav, transcription));
                }
            }
        }

        int totalSamples = pairs.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var transcripts = pairs.Take(totalSamples).Select(p => p.Text).ToList();
        var vocabulary = TextLoaderHelper.BuildVocabulary(transcripts, transcripts.Count, _options.VocabularySize);

        int textLen = _options.MaxTextLength;
        int audioLen = _options.MaxAudioSamples;
        var featuresData = new T[totalSamples * textLen];
        var labelsData = new T[totalSamples * audioLen];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (wav, text) = pairs[i];
            int[] tok = TextLoaderHelper.TokenizeAndEncode(text, vocabulary, textLen);
            int textOff = i * textLen;
            for (int j = 0; j < textLen; j++) featuresData[textOff + j] = NumOps.FromDouble(tok[j]);
            byte[] audioBytes = File.ReadAllBytes(wav);
            AudioLoaderHelper.LoadAudioSamples(audioBytes, labelsData, i * audioLen, audioLen, numOps);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, textLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, audioLen });
        InitializeIndices(totalSamples);
        await Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore() { LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0; }

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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()), AudioLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private string ResolveDataDir()
    {
        string sub = Path.Combine(_dataPath, "TIMIT");
        return Directory.Exists(sub) ? sub : _dataPath;
    }
}
