using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the GTZAN music genre classification dataset (Tzanetakis &amp; Cook 2002).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects the canonical Marsyas mirror layout:
/// <code>
/// {DataPath}/genres/
///   blues/blues.00000.wav  ...  blues.00099.wav
///   classical/classical.00000.wav  ...
///   ... (10 genres total)
/// </code>
/// Auto-download fetches the canonical Marsyas tarball.
/// Per-class deterministic split: first <c>TrainFraction</c> of the 100
/// clips per genre go to train, the rest to test/validation.
/// Features Tensor[N, Samples]; labels Tensor[N, 10] one-hot.
/// </para>
/// </remarks>
public class GtzanDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 10;
    private static readonly string[] Genres =
        ["blues", "classical", "country", "disco", "hiphop",
         "jazz", "metal", "pop", "reggae", "rock"];

    private static readonly string DownloadUrl =
        "http://opihi.cs.uvic.ca/sound/genres.tar.gz";

    private readonly GtzanDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "GTZAN";
    public override string Description => "GTZAN music genre classification (10 genres × 100 clips)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.Samples;
    public override int OutputDimension => NumClasses;

    public GtzanDataLoader(GtzanDataLoaderOptions? options = null)
    {
        _options = options ?? new GtzanDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("gtzan");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"GTZAN not found at {_dataPath}.");

        bool isTrain = _options.Split == Geometry.DatasetSplit.Train;
        var paths = new List<(string Path, int Label)>();

        for (int g = 0; g < Genres.Length; g++)
        {
            string genreDir = Path.Combine(root, Genres[g]);
            if (!Directory.Exists(genreDir)) continue;
            var files = Directory.EnumerateFiles(genreDir, "*.wav", SearchOption.TopDirectoryOnly)
                .OrderBy(f => f, StringComparer.Ordinal).ToArray();
            int trainN = (int)(files.Length * _options.TrainFraction);
            for (int i = 0; i < files.Length; i++)
            {
                bool isTrainImg = i < trainN;
                if (isTrain == isTrainImg) paths.Add((files[i], g));
            }
        }

        int totalSamples = paths.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        int N = _options.Samples;
        var featuresData = new T[totalSamples * N];
        var labelsData = new T[totalSamples * NumClasses];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (wavPath, label) = paths[i];
            byte[] bytes = File.ReadAllBytes(wavPath);
            AudioLoaderHelper.LoadAudioSamples(bytes, featuresData, i * N, N, numOps);
            if (label >= 0 && label < NumClasses) labelsData[i * NumClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, N });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
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
        // Tarball extracts to genres/{genre}/
        string sub = Path.Combine(_dataPath, "genres");
        return Directory.Exists(sub) ? sub : _dataPath;
    }
}
