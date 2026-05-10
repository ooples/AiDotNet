using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the NSynth Neural Synth dataset (Engel et al. 2017) — 305k musical notes
/// classified by instrument family.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/nsynth-{split}/
///   audio/*.wav        (4 sec, 16 kHz mono)
///   examples.json      (per-note metadata keyed by note_str)
/// </code>
/// where split = train / valid / test. Auto-downloads the canonical Magenta
/// jsonwav.tar.gz archives. Features [N, Samples] waveforms; labels [N, 11]
/// one-hot instrument-family classification.
/// </para>
/// </remarks>
public class NsynthDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    public const int NumInstrumentFamilies = 11; // bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal
    private static readonly string[] SplitUrls =
    {
        "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
        "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
        "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
    };

    private readonly NsynthDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "NSynth";
    public override string Description => "NSynth musical-note instrument-family classification (305k notes, 11 classes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.Samples;
    public override int OutputDimension => NumInstrumentFamilies;

    public NsynthDataLoader(NsynthDataLoaderOptions? options = null)
    {
        _options = options ?? new NsynthDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("nsynth");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "valid",
            _ => "train"
        };
        int splitIdx = splitName switch { "valid" => 1, "test" => 2, _ => 0 };
        string root = Path.Combine(_dataPath, $"nsynth-{splitName}");

        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(SplitUrls[splitIdx], _dataPath, cancellationToken);
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"NSynth not found at {root}.");

        string examplesPath = Path.Combine(root, "examples.json");
        string audioDir = Path.Combine(root, "audio");
        if (!File.Exists(examplesPath) || !Directory.Exists(audioDir))
            throw new FileNotFoundException($"NSynth examples.json or audio/ missing in {root}.");

        // examples.json is a single JSON object: { "<note_str>": {"instrument_family": 0..10, ...}, ... }
        var examples = JObject.Parse(await FilePolyfill.ReadAllTextAsync(examplesPath, cancellationToken));
        var entries = new List<(string Path, int Family)>();
        foreach (var prop in examples.Properties())
        {
            cancellationToken.ThrowIfCancellationRequested();
            string noteStr = prop.Name;
            int family = prop.Value["instrument_family"]?.Value<int>() ?? -1;
            if (family < 0 || family >= NumInstrumentFamilies) continue;
            string wavPath = Path.Combine(audioDir, noteStr + ".wav");
            if (File.Exists(wavPath)) entries.Add((wavPath, family));
        }

        int totalSamples = entries.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int N = _options.Samples;
        var featuresData = new T[totalSamples * N];
        var labelsData = new T[totalSamples * NumInstrumentFamilies];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (wav, fam) = entries[i];
            byte[] bytes = File.ReadAllBytes(wav);
            AudioLoaderHelper.LoadAudioSamples(bytes, featuresData, i * N, N, numOps);
            labelsData[i * NumInstrumentFamilies + fam] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, N });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumInstrumentFamilies });
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
}
