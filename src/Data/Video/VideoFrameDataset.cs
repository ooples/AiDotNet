using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Video;

/// <summary>
/// Loads video files and extracts frames for video classification and understanding tasks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Extracts a fixed number of frames from each video file, producing tensors of shape
/// [N, FramesPerVideo, Height, Width, Channels]. Frames are sampled uniformly from the
/// raw bytes of each video file.
/// </para>
/// <para><b>For Beginners:</b> This loader reads video files and extracts frames as images.
/// Each video becomes a sequence of frames that can be fed to temporal models like 3D CNNs.
/// </para>
/// </remarks>
public class VideoFrameDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly VideoFrameDatasetOptions _options;
    private int _sampleCount;
    private int _featureCount;
    private int _numClasses;
    private List<string> _classNames = new List<string>();

    /// <inheritdoc/>
    public override string Name => "VideoFrame";

    /// <inheritdoc/>
    public override string Description => $"Video frame dataset from {_options.RootDirectory}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Gets the class names discovered from directory names.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Creates a new VideoFrameDataset with the specified options.
    /// </summary>
    public VideoFrameDataset(VideoFrameDatasetOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (string.IsNullOrWhiteSpace(options.RootDirectory))
        {
            throw new ArgumentException("Root directory cannot be empty.", nameof(options));
        }

        _featureCount = options.FramesPerVideo * options.FrameWidth * options.FrameHeight * options.Channels;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string rootDir = _options.RootDirectory;
        if (!Directory.Exists(rootDir))
        {
            throw new DirectoryNotFoundException($"Video folder root directory not found: {rootDir}");
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
            throw new InvalidOperationException("No video files found matching the specified extensions.");
        }

        int frames = _options.FramesPerVideo;
        int h = _options.FrameHeight;
        int w = _options.FrameWidth;
        int c = _options.Channels;
        int pixelsPerFrame = h * w * c;

        var featuresData = new T[_sampleCount * frames * pixelsPerFrame];
        var labelsData = new T[_sampleCount * _numClasses];

        for (int i = 0; i < _sampleCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (filePath, classIndex) = samples[i];
            byte[] rawBytes = File.ReadAllBytes(filePath);

            // Sample frames uniformly from the byte stream
            int bytesPerFrame = pixelsPerFrame;
            int totalFramesAvailable = rawBytes.Length / Math.Max(1, bytesPerFrame);
            int frameStride = totalFramesAvailable > frames
                ? totalFramesAvailable / frames
                : 1;

            int featureOffset = i * frames * pixelsPerFrame;
            for (int f = 0; f < frames; f++)
            {
                int byteStart = (f * frameStride) * bytesPerFrame;
                for (int p = 0; p < pixelsPerFrame; p++)
                {
                    int srcIdx = byteStart + p;
                    double value = srcIdx < rawBytes.Length ? rawBytes[srcIdx] : 0;
                    if (_options.NormalizePixels)
                    {
                        value /= 255.0;
                    }

                    featuresData[featureOffset + f * pixelsPerFrame + p] = NumOps.FromDouble(value);
                }
            }

            int labelOffset = i * _numClasses;
            labelsData[labelOffset + classIndex] = NumOps.One;
        }

        var features = new Tensor<T>(featuresData, new[] { _sampleCount, frames, h, w, c });
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

        return (ExtractTensorBatch(features, indices), ExtractTensorBatch(labels, indices));
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
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize + valSize).ToArray()))
        );
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
