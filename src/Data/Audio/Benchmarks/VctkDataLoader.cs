using AiDotNet.Data;
using AiDotNet.Data.Loaders;
using AiDotNet.Data.Text.Benchmarks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the VCTK Corpus 0.92 multi-speaker TTS dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/VCTK-Corpus-0.92/
///   wav48_silence_trimmed/{speakerId}/*.flac    (or *.wav)
///   txt/{speakerId}/*.txt
///   speaker-info.txt
/// </code>
/// Auto-downloads the canonical Edinburgh DataShare zip.
/// Features [N, MaxTextLength] (encoded transcript); labels [N, MaxAudioSamples] waveform.
/// Deterministic 90/5/5 row-index split.
/// </para>
/// </remarks>
public class VctkDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl =
        "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip";

    private readonly VctkDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "VCTK 0.92";
    public override string Description => "VCTK multi-speaker English TTS corpus (110 speakers, ~44 hr)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxTextLength;
    public override int OutputDimension => _options.MaxAudioSamples;

    public VctkDataLoader(VctkDataLoaderOptions? options = null)
    {
        _options = options ?? new VctkDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("vctk");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractZipAsync(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"VCTK not found at {_dataPath}.");

        string wavRoot = Path.Combine(root, "wav48_silence_trimmed");
        string txtRoot = Path.Combine(root, "txt");
        if (!Directory.Exists(wavRoot)) wavRoot = Path.Combine(root, "wav48"); // older release variant
        if (!Directory.Exists(wavRoot) || !Directory.Exists(txtRoot))
            throw new DirectoryNotFoundException($"VCTK wav48/txt subdirs missing in {root}.");

        var allowed = string.IsNullOrEmpty(_options.SpeakerFilter)
            ? null
            : new HashSet<string>(_options.SpeakerFilter!.Split(',').Select(s => s.Trim()),
                StringComparer.OrdinalIgnoreCase);

        // Canonical roots used to defend against symlink/path-escape via untrusted speaker
        // dirs or filenames returned by EnumerateFiles.
        string canonicalWavRoot = Path.GetFullPath(wavRoot);
        string canonicalTxtRoot = Path.GetFullPath(txtRoot);

        var pairs = new List<(string Wav, string Text)>();
        foreach (string spkDir in Directory.EnumerateDirectories(wavRoot).OrderBy(d => d, StringComparer.Ordinal))
        {
            string spk = Path.GetFileName(spkDir);
            if (allowed is not null && !allowed.Contains(spk)) continue;
            string spkTxt = Path.Combine(txtRoot, spk);
            foreach (string wav in Directory.EnumerateFiles(spkDir, "*.flac")
                                            .Concat(Directory.EnumerateFiles(spkDir, "*.wav"))
                                            .OrderBy(f => f, StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string baseName = Path.GetFileNameWithoutExtension(wav);
                // Filenames like p225_001_mic1.flac → strip _mic{N} suffix to find transcript.
                string transcriptName = baseName;
                int micIdx = transcriptName.LastIndexOf("_mic", StringComparison.OrdinalIgnoreCase);
                if (micIdx > 0) transcriptName = transcriptName.Substring(0, micIdx);
                string txtPath = Path.GetFullPath(Path.Combine(spkTxt, transcriptName + ".txt"));
                string canonicalWav = Path.GetFullPath(wav);
                // Defense in depth: ensure both resolved paths stay under their canonical roots
                // even after symlink/junction resolution.
                if (!canonicalWav.StartsWith(canonicalWavRoot, StringComparison.Ordinal)) continue;
                if (!txtPath.StartsWith(canonicalTxtRoot, StringComparison.Ordinal)) continue;
                if (!File.Exists(txtPath)) continue;
                string text = (await FilePolyfill.ReadAllTextAsync(txtPath, cancellationToken)).Trim();
                if (string.IsNullOrEmpty(text)) continue;
                pairs.Add((canonicalWav, text));
            }
        }

        // Deterministic 90/5/5 row-index split.
        int total = pairs.Count;
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

        var transcripts = keep.Take(totalSamples).Select(i => pairs[i].Text).ToList();
        var vocabulary = TextLoaderHelper.BuildVocabulary(transcripts, transcripts.Count, _options.VocabularySize);

        int textLen = _options.MaxTextLength;
        int audioLen = _options.MaxAudioSamples;
        var featuresData = new T[totalSamples * textLen];
        var labelsData = new T[totalSamples * audioLen];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (wav, text) = pairs[keep[i]];
            int[] tok = TextLoaderHelper.TokenizeAndEncode(text, vocabulary, textLen);
            int textOff = i * textLen;
            for (int j = 0; j < textLen; j++) featuresData[textOff + j] = NumOps.FromDouble(tok[j]);
            byte[] audioBytes = File.ReadAllBytes(wav);
            AudioLoaderHelper.LoadAudioSamples(audioBytes, labelsData, i * audioLen, audioLen, numOps);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, textLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, audioLen });
        InitializeIndices(totalSamples);
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
        var trainIndices = shuffled.Take(trainSize).ToArray();
        var valIndices = shuffled.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffled.Skip(trainSize + valSize).ToArray();
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, trainIndices), AudioLoaderHelper.ExtractTensorBatch(labels, trainIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, valIndices), AudioLoaderHelper.ExtractTensorBatch(labels, valIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(AudioLoaderHelper.ExtractTensorBatch(features, testIndices), AudioLoaderHelper.ExtractTensorBatch(labels, testIndices))
        );
    }

    private string ResolveDataDir()
    {
        string sub = Path.Combine(_dataPath, "VCTK-Corpus-0.92");
        return Directory.Exists(sub) ? sub : _dataPath;
    }
}
