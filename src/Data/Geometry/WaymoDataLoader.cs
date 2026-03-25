using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the Waymo Open Dataset (LiDAR point clouds with 3D bounding boxes).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Waymo expects pre-extracted point cloud binary files:
/// <code>
/// {DataPath}/
///   training/
///     velodyne/     (.bin files, 4 or 5 floats per point: x, y, z, intensity[, elongation])
///     label/        (.txt files, one label per line: class_id x y z l w h heading)
///   validation/
///     velodyne/
///     label/
/// </code>
/// Features are point cloud Tensor[N, PointsPerSample * Channels].
/// Labels are the dominant object class per frame Tensor[N, 1] (0=Vehicle, 1=Pedestrian, 2=Cyclist, 3=Sign).
/// </para>
/// </remarks>
public class WaymoDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly WaymoDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _channels;

    // Waymo object class mapping: 1=Vehicle, 2=Pedestrian, 3=Cyclist, 4=Sign
    // Also handle KITTI-style string labels for pre-processed exports
    private static readonly Dictionary<string, int> WaymoClassMap = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Vehicle"] = 0, ["Pedestrian"] = 1, ["Cyclist"] = 2, ["Sign"] = 3,
        ["Car"] = 0, ["Truck"] = 0, ["Van"] = 0 // KITTI-style aliases
    };

    public override string Name => "Waymo";
    public override string Description => "Waymo Open Dataset 3D detection (LiDAR)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.PointsPerSample * _channels;
    public override int OutputDimension => 1;

    public WaymoDataLoader(WaymoDataLoaderOptions? options = null)
    {
        _options = options ?? new WaymoDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("waymo");
        _channels = _options.IncludeIntensity ? 4 : 3;
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitDir = _options.Split == DatasetSplit.Test
            ? Path.Combine(_dataPath, "testing")
            : _options.Split == DatasetSplit.Validation
                ? Path.Combine(_dataPath, "validation")
                : Path.Combine(_dataPath, "training");

        string veloDir = Path.Combine(splitDir, "velodyne");
        if (!Directory.Exists(veloDir))
            veloDir = splitDir;

        if (!Directory.Exists(veloDir))
            throw new DirectoryNotFoundException($"Waymo data not found at {veloDir}.");

        var binFiles = Directory.GetFiles(veloDir, "*.bin");
        Array.Sort(binFiles, StringComparer.OrdinalIgnoreCase);

        int totalSamples = binFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int featureSize = _options.PointsPerSample * _channels;
        var featuresData = new T[totalSamples * featureSize];
        var labelsData = new T[totalSamples];

        // Find label directory
        string labelDir = Path.Combine(splitDir, "label");
        if (!Directory.Exists(labelDir))
            labelDir = Path.Combine(splitDir, "label_2");

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            byte[] bytes = await FilePolyfill.ReadAllBytesAsync(binFiles[i], cancellationToken);

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

            // Parse label file: extract dominant object class per frame
            labelsData[i] = NumOps.FromDouble(ParseWaymoLabelFile(labelDir, binFiles[i]));
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 1 });
        InitializeIndices(totalSamples);
    }

    /// <summary>
    /// Parses a Waymo label file and returns the dominant object class index.
    /// Handles both numeric class_id format (1=Vehicle, 2=Pedestrian, 3=Cyclist, 4=Sign)
    /// and KITTI-style string format (Car, Pedestrian, Cyclist, etc.).
    /// Returns -1 when no label file exists or no valid objects are found.
    /// Downstream consumers should filter or handle the -1 sentinel appropriately.
    /// </summary>
    private static int ParseWaymoLabelFile(string labelDir, string binFilePath)
    {
        string labelFile = Path.Combine(labelDir,
            Path.GetFileNameWithoutExtension(binFilePath) + ".txt");

        if (!File.Exists(labelFile))
            return -1; // No label file found — return sentinel value

        var classCounts = new Dictionary<int, int>();
        foreach (string line in File.ReadAllLines(labelFile))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;

            int classId;
            if (int.TryParse(parts[0], out int numericId))
            {
                // Numeric format: class_id (1-based) -> 0-based
                classId = Math.Max(0, numericId - 1);
            }
            else
            {
                // String format: type name -> class index
                if (!WaymoClassMap.TryGetValue(parts[0], out classId))
                    continue;
            }

            classCounts.TryGetValue(classId, out int count);
            classCounts[classId] = count + 1;
        }

        if (classCounts.Count == 0)
            return -1; // No labeled objects found — return sentinel value

        int bestClass = 0, bestCount = 0;
        foreach (var kvp in classCounts)
        {
            if (kvp.Value > bestCount)
            {
                bestClass = kvp.Key;
                bestCount = kvp.Value;
            }
        }

        return bestClass;
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        var nfs = (int[])features.Shape.ToArray().Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape.ToArray().Clone(); nls[0] = indices.Length;
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
