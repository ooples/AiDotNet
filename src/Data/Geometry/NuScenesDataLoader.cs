using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the nuScenes dataset (LiDAR point clouds with 3D bounding box annotations).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// nuScenes expects pre-extracted point cloud binary files:
/// <code>
/// {DataPath}/
///   sweeps/LIDAR_TOP/    (.bin or .pcd.bin files, 5 floats: x, y, z, intensity, ring_index)
///   samples/LIDAR_TOP/   (.bin or .pcd.bin files)
/// </code>
/// Features are point cloud Tensor[N, PointsPerSample * Channels].
/// Labels are simplified as scene index Tensor[N, 1].
/// </para>
/// </remarks>
public class NuScenesDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly NuScenesDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _channels;

    public override string Name => "nuScenes";
    public override string Description => "nuScenes 3D object detection (LiDAR)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.PointsPerSample * _channels;
    public override int OutputDimension => 1;

    public NuScenesDataLoader(NuScenesDataLoaderOptions? options = null)
    {
        _options = options ?? new NuScenesDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("nuscenes");
        _channels = _options.IncludeIntensity ? 4 : 3;
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Try multiple directory layouts
        string[] searchDirs =
        {
            Path.Combine(_dataPath, "samples", "LIDAR_TOP"),
            Path.Combine(_dataPath, "sweeps", "LIDAR_TOP"),
            Path.Combine(_dataPath, "samples"),
            Path.Combine(_dataPath, "velodyne"),
            _dataPath
        };

        string? veloDir = null;
        foreach (var dir in searchDirs)
        {
            if (Directory.Exists(dir) && Directory.GetFiles(dir, "*.bin").Length > 0)
            {
                veloDir = dir;
                break;
            }
        }

        if (veloDir == null)
            throw new DirectoryNotFoundException($"nuScenes LiDAR data not found at {_dataPath}.");

        var binFiles = Directory.GetFiles(veloDir, "*.bin");
        Array.Sort(binFiles, StringComparer.OrdinalIgnoreCase);

        int totalSamples = binFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int featureSize = _options.PointsPerSample * _channels;
        var featuresData = new T[totalSamples * featureSize];
        var labelsData = new T[totalSamples];

        // nuScenes typically has 5 floats per point: x, y, z, intensity, ring_index
        int floatsPerPoint = 5;
        int bytesPerFloat = 4;

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            byte[] bytes = await FilePolyfill.ReadAllBytesAsync(binFiles[i], cancellationToken);

            int totalPoints = bytes.Length / (floatsPerPoint * bytesPerFloat);
            // If format doesn't match 5 floats, try 4 floats
            if (totalPoints == 0 || totalPoints * floatsPerPoint * bytesPerFloat != bytes.Length)
            {
                floatsPerPoint = 4;
                totalPoints = bytes.Length / (floatsPerPoint * bytesPerFloat);
            }

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
