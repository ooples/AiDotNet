using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Loads the Kinetics-400 human action recognition dataset (~300K clips, 400 classes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Kinetics-400 expects pre-extracted frames:
/// <code>
/// {DataPath}/
///   train/
///     class_name/
///       video_id/
///         frame_000000.jpg ... frame_000015.jpg
///   val/
///   test/
///   train.csv  (label, youtube_id, time_start, time_end, split)
/// </code>
/// Features are frame tensors Tensor[N, FramesPerVideo * FrameHeight * FrameWidth * 3].
/// Labels are one-hot Tensor[N, 400].
/// </para>
/// </remarks>
public class Kinetics400DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int NumClasses = 400;

    private readonly Kinetics400DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _featureSize;

    /// <inheritdoc/>
    public override string Name => "Kinetics-400";
    /// <inheritdoc/>
    public override string Description => "Kinetics-400 action recognition (400 classes)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _featureSize;
    /// <inheritdoc/>
    public override int OutputDimension => NumClasses;

    /// <summary>Creates a new Kinetics-400 data loader.</summary>
    public Kinetics400DataLoader(Kinetics400DataLoaderOptions? options = null)
    {
        _options = options ?? new Kinetics400DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("kinetics400");
        _featureSize = _options.FramesPerVideo * _options.FrameHeight * _options.FrameWidth * 3;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "val",
            _ => "train"
        };
        string splitDir = Path.Combine(_dataPath, splitName);
        if (!Directory.Exists(splitDir))
        {
            throw new DirectoryNotFoundException(
                $"Kinetics-400 data not found at {splitDir}. " +
                "Download and extract frames from https://www.deepmind.com/open-source/kinetics.");
        }

        // Collect class directories
        var classDirs = Directory.GetDirectories(splitDir);
        Array.Sort(classDirs, StringComparer.OrdinalIgnoreCase);
        var classNameToIndex = new Dictionary<string, int>();
        for (int c = 0; c < classDirs.Length && c < NumClasses; c++)
            classNameToIndex[Path.GetFileName(classDirs[c])] = c;

        var samples = new List<(string VideoDir, int ClassIndex)>();
        foreach (var classDir in classDirs)
        {
            string className = Path.GetFileName(classDir);
            if (!classNameToIndex.TryGetValue(className, out int classIdx)) continue;

            var videoDirs = Directory.GetDirectories(classDir);
            foreach (var videoDir in videoDirs)
                samples.Add((videoDir, classIdx));
        }

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * _featureSize];
        var labelsData = new T[totalSamples * NumClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (videoDir, classIndex) = samples[i];

            // Load frames
            var frameFiles = Directory.GetFiles(videoDir, "*.jpg");
            if (frameFiles.Length == 0)
                frameFiles = Directory.GetFiles(videoDir, "*.png");
            Array.Sort(frameFiles, StringComparer.OrdinalIgnoreCase);

            int framePixels = _options.FrameHeight * _options.FrameWidth * 3;
            for (int f = 0; f < _options.FramesPerVideo && f < frameFiles.Length; f++)
            {
                byte[] imageBytes = await FilePolyfill.ReadAllBytesAsync(frameFiles[f], cancellationToken);
                int featureOffset = i * _featureSize + f * framePixels;
                int pixelCount = Math.Min(imageBytes.Length, framePixels);
                for (int p = 0; p < pixelCount; p++)
                {
                    double val = _options.Normalize ? imageBytes[p] / 255.0 : imageBytes[p];
                    featuresData[featureOffset + p] = NumOps.FromDouble(val);
                }
            }

            labelsData[i * NumClasses + classIndex] = NumOps.FromDouble(1.0);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, NumClasses });
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
        var newFeatShape = (int[])features.Shape.Clone(); newFeatShape[0] = indices.Length;
        var newLabelShape = (int[])labels.Shape.Clone(); newLabelShape[0] = indices.Length;
        var batchFeatures = new Tensor<T>(newFeatShape);
        var batchLabels = new Tensor<T>(newLabelShape);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, batchFeatures, indices[i], i);
            TensorCopyHelper.CopySample(labels, batchLabels, indices[i], i);
        }
        return (batchFeatures, batchLabels);
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
            CreateSplitLoader(features, labels, shuffled.Take(trainSize).ToArray()),
            CreateSplitLoader(features, labels, shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSplitLoader(features, labels, shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private static InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSplitLoader(Tensor<T> features, Tensor<T> labels, int[] indices)
    {
        var newFeatShape = (int[])features.Shape.Clone(); newFeatShape[0] = indices.Length;
        var newLabelShape = (int[])labels.Shape.Clone(); newLabelShape[0] = indices.Length;
        var batchF = new Tensor<T>(newFeatShape);
        var batchL = new Tensor<T>(newLabelShape);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, batchF, indices[i], i);
            TensorCopyHelper.CopySample(labels, batchL, indices[i], i);
        }
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(batchF, batchL);
    }
}
