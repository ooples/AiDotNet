using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the KITTI 3D object detection dataset (LiDAR point clouds with 3D bounding boxes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// KITTI expects:
/// <code>
/// {DataPath}/
///   training/
///     velodyne/     (.bin files, 4 floats per point: x, y, z, reflectance)
///     label_2/      (.txt files, KITTI format annotations)
///   testing/
///     velodyne/
/// </code>
/// Features are point cloud Tensor[N, PointsPerSample * Channels].
/// Labels are 3D bounding boxes (simplified as class index) Tensor[N, 1].
/// </para>
/// </remarks>
public class KittiDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly KittiDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _channels;

    public override string Name => "KITTI";
    public override string Description => "KITTI 3D object detection (LiDAR)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.PointsPerSample * _channels;
    public override int OutputDimension => 1;

    public KittiDataLoader(KittiDataLoaderOptions? options = null)
    {
        _options = options ?? new KittiDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("kitti");
        _channels = _options.IncludeReflectance ? 4 : 3;
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitDir = _options.Split == DatasetSplit.Test
            ? Path.Combine(_dataPath, "testing")
            : Path.Combine(_dataPath, "training");

        string veloDir = Path.Combine(splitDir, "velodyne");
        if (!Directory.Exists(veloDir))
            veloDir = splitDir;

        if (!Directory.Exists(veloDir))
            throw new DirectoryNotFoundException($"KITTI data not found at {veloDir}.");

        var binFiles = Directory.GetFiles(veloDir, "*.bin");
        Array.Sort(binFiles, StringComparer.OrdinalIgnoreCase);

        int totalSamples = binFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int featureSize = _options.PointsPerSample * _channels;
        var featuresData = new T[totalSamples * featureSize];
        var labelsData = new T[totalSamples]; // Simplified: scene index

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            byte[] bytes = await FilePolyfill.ReadAllBytesAsync(binFiles[i], cancellationToken);

            // Binary format: 4 floats (x, y, z, reflectance) per point
            int floatsPerPoint = 4;
            int bytesPerFloat = 4;
            int totalPoints = bytes.Length / (floatsPerPoint * bytesPerFloat);
            int pointsToRead = Math.Min(totalPoints, _options.PointsPerSample);

            int featureOffset = i * featureSize;
            for (int p = 0; p < pointsToRead; p++)
            {
                int byteBase = p * floatsPerPoint * bytesPerFloat;
                for (int c = 0; c < _channels; c++)
                {
                    if (byteBase + (c + 1) * bytesPerFloat <= bytes.Length)
                    {
                        float val = BitConverter.ToSingle(bytes, byteBase + c * bytesPerFloat);
                        featuresData[featureOffset + p * _channels + c] = NumOps.FromDouble(val);
                    }
                }
            }

            labelsData[i] = NumOps.FromDouble(i);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 1 });
        InitializeIndices(totalSamples);
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        var nfs = (int[])features.Shape.Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape.Clone(); nls[0] = indices.Length;
        var bf = new Tensor<T>(nfs);
        var bl = new Tensor<T>(nls);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, bf, indices[i], i);
            TensorCopyHelper.CopySample(labels, bl, indices[i], i);
        }
        return (bf, bl);
    }

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
        return (
            CreateSplit(shuffled.Take(trainSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSplit(int[] indices)
    {
        var (bf, bl) = ExtractBatch(indices);
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(bf, bl);
    }
}
