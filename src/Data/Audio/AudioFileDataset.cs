using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio;

/// <summary>
/// Loads audio files from directories for audio classification and processing tasks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Supports WAV and raw PCM files. Audio is loaded as raw waveform samples
/// and can optionally be converted to mono and normalized.
/// </para>
/// <para><b>For Beginners:</b> Organize your audio files into class folders (like ImageFolder),
/// and this loader reads the raw waveforms into tensors for training.
/// </para>
/// </remarks>
public class AudioFileDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly AudioFileDatasetOptions _options;
    private int _sampleCount;
    private int _samplesPerFile;
    private int _numClasses;
    private List<string> _classNames = new List<string>();

    /// <inheritdoc/>
    public override string Name => "AudioFile";

    /// <inheritdoc/>
    public override string Description => $"Audio file dataset from {_options.RootDirectory}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _samplesPerFile;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Gets the class names discovered from directory names.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Creates a new AudioFileDataset with the specified options.
    /// </summary>
    public AudioFileDataset(AudioFileDatasetOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (string.IsNullOrWhiteSpace(options.RootDirectory))
        {
            throw new ArgumentException("Root directory cannot be empty.", nameof(options));
        }

        _samplesPerFile = (int)(options.SampleRate * options.DurationSeconds);
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string rootDir = _options.RootDirectory;
        if (!Directory.Exists(rootDir))
        {
            throw new DirectoryNotFoundException($"Audio folder root directory not found: {rootDir}");
        }

        var extensions = new HashSet<string>(
            _options.Extensions.Select(e => e.StartsWith(".", StringComparison.Ordinal) ? e : "." + e),
            StringComparer.OrdinalIgnoreCase);

        List<(string FilePath, int ClassIndex)> samples;

        if (_options.UseDirectoryLabels)
        {
            var classDirs = Directory.GetDirectories(rootDir)
                .OrderBy(d => Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)
                .ToList();

            _classNames = classDirs
                .Select(d => Path.GetFileName(d) ?? string.Empty)
                .Where(n => n.Length > 0)
                .ToList();
            _numClasses = _classNames.Count;

            samples = new List<(string, int)>();
            for (int classIdx = 0; classIdx < classDirs.Count; classIdx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var files = Directory.GetFiles(classDirs[classIdx], "*.*", SearchOption.TopDirectoryOnly)
                    .Where(f => extensions.Contains(Path.GetExtension(f)))
                    .ToList();

                foreach (var file in files)
                {
                    samples.Add((file, classIdx));
                }
            }
        }
        else
        {
            _classNames = new List<string> { "default" };
            _numClasses = 1;

            samples = Directory.GetFiles(rootDir, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => extensions.Contains(Path.GetExtension(f)))
                .Select(f => (f, 0))
                .ToList();
        }

        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < samples.Count)
        {
            var random = _options.RandomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
                : RandomHelper.CreateSecureRandom();
            samples = samples.OrderBy(_ => random.Next()).Take(_options.MaxSamples.Value).ToList();
        }

        _sampleCount = samples.Count;
        if (_sampleCount == 0)
        {
            throw new InvalidOperationException("No audio files found matching the specified extensions.");
        }

        // Build tensors: features [N, SamplesPerFile], labels [N, NumClasses]
        var featuresData = new T[_sampleCount * _samplesPerFile];
        var labelsData = new T[_sampleCount * _numClasses];

        for (int i = 0; i < _sampleCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (filePath, classIndex) = samples[i];
            var audioSamples = LoadAudioSamples(filePath);

            int featureOffset = i * _samplesPerFile;
            int copyLen = Math.Min(audioSamples.Length, _samplesPerFile);
            Array.Copy(audioSamples, 0, featuresData, featureOffset, copyLen);

            int labelOffset = i * _numClasses;
            labelsData[labelOffset + classIndex] = NumOps.One;
        }

        var features = new Tensor<T>(featuresData, new[] { _sampleCount, _samplesPerFile });
        var labels = new Tensor<T>(labelsData, new[] { _sampleCount, _numClasses });

        LoadedFeatures = features;
        LoadedLabels = labels;
        InitializeIndices(_sampleCount);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
        _classNames = new List<string>();
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        var batchFeatures = ExtractTensorBatch(features, indices);
        var batchLabels = ExtractTensorBatch(labels, indices);

        return (batchFeatures, batchLabels);
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        return (
            CreateSubset(shuffled.Take(trainSize).ToArray()),
            CreateSubset(shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSubset(shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSubset(int[] indices)
    {
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
            ExtractTensorBatch(LoadedFeatures!, indices),
            ExtractTensorBatch(LoadedLabels!, indices));
    }

    private T[] LoadAudioSamples(string filePath)
    {
        byte[] rawBytes = File.ReadAllBytes(filePath);
        var samples = new T[_samplesPerFile];

        // Simple WAV parsing: skip 44-byte header for standard WAV files
        string ext = Path.GetExtension(filePath);
        int headerOffset = ext.Equals(".wav", StringComparison.OrdinalIgnoreCase) ? 44 : 0;
        int dataLength = rawBytes.Length - headerOffset;

        // Assume 16-bit signed PCM
        int numRawSamples = dataLength / 2;
        int copyLen = Math.Min(numRawSamples, _samplesPerFile);

        for (int j = 0; j < copyLen; j++)
        {
            int byteIdx = headerOffset + j * 2;
            if (byteIdx + 1 < rawBytes.Length)
            {
                short sample = (short)(rawBytes[byteIdx] | (rawBytes[byteIdx + 1] << 8));
                double value = sample;
                if (_options.Normalize)
                {
                    value /= 32768.0;
                }

                samples[j] = NumOps.FromDouble(value);
            }
        }

        return samples;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
