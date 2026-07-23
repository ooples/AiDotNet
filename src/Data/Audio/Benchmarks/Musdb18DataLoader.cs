using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Loads the MUSDB18 music source separation dataset (150 tracks, 4 stems: vocals, drums, bass, other).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// MUSDB18 expects pre-extracted WAV stems:
/// <code>
/// {DataPath}/
///   train/                   (100 tracks)
///     Track Name/
///       mixture.wav
///       vocals.wav
///       drums.wav
///       bass.wav
///       other.wav
///   test/                    (50 tracks)
///     Track Name/
///       ...
/// </code>
/// Features are mixture waveform segments Tensor[N, SegmentSamples].
/// Labels are concatenated stem segments Tensor[N, SegmentSamples * 4] (vocals, drums, bass, other).
/// </para>
/// </remarks>
public class Musdb18DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumStems = 4; // vocals, drums, bass, other
    private static readonly string[] StemNames = { "vocals.wav", "drums.wav", "bass.wav", "other.wav" };

    private readonly Musdb18DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _segmentSamples;

    /// <inheritdoc/>
    public override string Name => "MUSDB18";
    /// <inheritdoc/>
    public override string Description => "MUSDB18 music source separation (4 stems)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _segmentSamples;
    /// <inheritdoc/>
    public override int OutputDimension => _segmentSamples * NumStems;

    /// <summary>Creates a new MUSDB18 data loader.</summary>
    public Musdb18DataLoader(Musdb18DataLoaderOptions? options = null)
    {
        _options = options ?? new Musdb18DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("musdb18");
        _segmentSamples = (int)(_options.SampleRate * _options.SegmentDurationSeconds);
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitDir = _options.Split == Geometry.DatasetSplit.Test
            ? Path.Combine(_dataPath, "test")
            : Path.Combine(_dataPath, "train");

        if (!Directory.Exists(splitDir))
            splitDir = _dataPath;

        // Each subdirectory is a track
        var trackDirs = Directory.GetDirectories(splitDir);
        var validTracks = new List<string>();

        foreach (var trackDir in trackDirs)
        {
            string mixturePath = Path.Combine(trackDir, "mixture.wav");
            if (File.Exists(mixturePath))
                validTracks.Add(trackDir);
        }

        if (validTracks.Count == 0)
        {
            throw new DirectoryNotFoundException(
                $"MUSDB18 data not found at {splitDir}. " +
                "Download from https://sigsep.github.io/datasets/musdb.html.");
        }

        int totalSamples = validTracks.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _segmentSamples];
        var labelsData = new T[totalSamples * _segmentSamples * NumStems];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string trackDir = validTracks[i];

            // Load mixture
            string mixturePath = Path.Combine(trackDir, "mixture.wav");
            if (File.Exists(mixturePath))
            {
                byte[] audioBytes = await FilePolyfill.ReadAllBytesAsync(mixturePath, cancellationToken);
                AudioLoaderHelper.LoadAudioSamples(audioBytes, featuresData, i * _segmentSamples, _segmentSamples, NumOps);
            }

            // Load stems
            for (int s = 0; s < NumStems; s++)
            {
                string stemPath = Path.Combine(trackDir, StemNames[s]);
                if (File.Exists(stemPath))
                {
                    byte[] stemBytes = await FilePolyfill.ReadAllBytesAsync(stemPath, cancellationToken);
                    int labelOffset = i * _segmentSamples * NumStems + s * _segmentSamples;
                    AudioLoaderHelper.LoadAudioSamples(stemBytes, labelsData, labelOffset, _segmentSamples, NumOps);
                }
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _segmentSamples });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, _segmentSamples * NumStems });
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
