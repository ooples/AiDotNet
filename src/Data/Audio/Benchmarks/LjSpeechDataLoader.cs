using AiDotNet.Data;
using AiDotNet.Data.Loaders;
using AiDotNet.Data.Text.Benchmarks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the LJSpeech 1.1 single-speaker TTS corpus (Ito &amp; Johnson 2017).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/LJSpeech-1.1/
///   wavs/LJ001-0001.wav  ...  LJ050-0278.wav
///   metadata.csv          (pipe-separated: id|raw_text|normalized_text)
/// </code>
/// Auto-downloads the canonical keithito.com tar.bz2.
/// Features Tensor[N, MaxTextLength] (encoded text tokens); labels
/// Tensor[N, MaxAudioSamples] (audio waveform, mono, 22.05 kHz, [-1, 1]).
/// LJSpeech has no canonical train/val/test splits — this loader applies a
/// deterministic 90/5/5 by row index in the metadata file.
/// </para>
/// </remarks>
public class LjSpeechDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl =
        "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2";

    private readonly LjSpeechDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "LJSpeech 1.1";
    public override string Description => "LJSpeech single-speaker English TTS corpus (~24 hr, 22 kHz)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxTextLength;
    public override int OutputDimension => _options.MaxAudioSamples;

    public LjSpeechDataLoader(LjSpeechDataLoaderOptions? options = null)
    {
        _options = options ?? new LjSpeechDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("ljspeech");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarBz2Async(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"LJSpeech not found at {_dataPath}.");

        string metaPath = Path.Combine(root, "metadata.csv");
        string wavsDir = Path.Combine(root, "wavs");
        if (!File.Exists(metaPath) || !Directory.Exists(wavsDir))
            throw new FileNotFoundException($"LJSpeech metadata.csv or wavs/ missing in {root}.");

        // metadata.csv: pipe-separated, no header. id|raw_text|normalized_text
        var ids = new List<string>();
        var texts = new List<string>();
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(metaPath, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split('|');
            if (parts.Length < 3) continue;
            ids.Add(parts[0]);
            texts.Add(_options.UseNormalizedText ? parts[2] : parts[1]);
        }

        // Deterministic 90/5/5 split by row index.
        int total = ids.Count;
        int trainEnd = (int)(total * 0.90);
        int valEnd = (int)(total * 0.95);
        var keep = new List<int>(total);
        for (int i = 0; i < total; i++)
        {
            bool include = _options.Split switch
            {
                Geometry.DatasetSplit.Test => i >= valEnd,
                Geometry.DatasetSplit.Validation => i >= trainEnd && i < valEnd,
                _ => i < trainEnd
            };
            if (include) keep.Add(i);
        }

        int totalSamples = keep.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var vocabulary = TextLoaderHelper.BuildVocabulary(texts, texts.Count, _options.VocabularySize);

        int textLen = _options.MaxTextLength;
        int audioLen = _options.MaxAudioSamples;
        var featuresData = new T[totalSamples * textLen];
        var labelsData = new T[totalSamples * audioLen];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int rowIdx = keep[i];
            // Encode text
            int[] tok = TextLoaderHelper.TokenizeAndEncode(texts[rowIdx], vocabulary, textLen);
            int textOff = i * textLen;
            for (int j = 0; j < textLen; j++) featuresData[textOff + j] = NumOps.FromDouble(tok[j]);
            // Decode WAV → labels
            string wavPath = Path.Combine(wavsDir, ids[rowIdx] + ".wav");
            if (!File.Exists(wavPath)) continue;
            byte[] wavBytes = File.ReadAllBytes(wavPath);
            AudioLoaderHelper.LoadAudioSamples(wavBytes, labelsData, i * audioLen, audioLen, numOps);
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
        string sub = Path.Combine(_dataPath, "LJSpeech-1.1");
        return Directory.Exists(sub) ? sub : _dataPath;
    }
}
