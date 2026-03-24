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
/// Labels are the dominant object class per frame Tensor[N, 1] (0=car, 1=truck, 2=bus,
/// 3=trailer, 4=construction_vehicle, 5=pedestrian, 6=motorcycle, 7=bicycle,
/// 8=traffic_cone, 9=barrier). Parsed from lidarseg or text label files.
/// </para>
/// </remarks>
public class NuScenesDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly NuScenesDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _channels;

    // nuScenes has 23 detection classes, simplified to 10 super-categories
    private static readonly Dictionary<string, int> NuScenesClassMap = new(StringComparer.OrdinalIgnoreCase)
    {
        ["car"] = 0, ["truck"] = 1, ["bus"] = 2, ["trailer"] = 3,
        ["construction_vehicle"] = 4, ["pedestrian"] = 5, ["motorcycle"] = 6,
        ["bicycle"] = 7, ["traffic_cone"] = 8, ["barrier"] = 9,
        // KITTI-style aliases for pre-processed exports
        ["Car"] = 0, ["Truck"] = 1, ["Van"] = 1, ["Pedestrian"] = 5,
        ["Cyclist"] = 7, ["Person_sitting"] = 5
    };

    public override string Name => "nuScenes";
    public override string Description => "nuScenes 3D object detection (LiDAR)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.PointsPerSample * _channels;
    public override int OutputDimension => 1;

    public NuScenesDataLoader(NuScenesDataLoaderOptions? options = null)
    {
        _options = options ?? new NuScenesDataLoaderOptions();
        _options.Validate();
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

        // Find label directory: try lidarseg (per-point uint8), then label/ or label_2/ (text)
        string? labelDir = null;
        string lidarsegDir = Path.Combine(_dataPath, "lidarseg");
        string labelTextDir = Path.Combine(_dataPath, "label");
        string label2Dir = Path.Combine(_dataPath, "label_2");
        bool useLidarseg = false;

        if (Directory.Exists(lidarsegDir) && Directory.GetFiles(lidarsegDir, "*.bin").Length > 0)
        {
            labelDir = lidarsegDir;
            useLidarseg = true;
        }
        else if (Directory.Exists(labelTextDir))
        {
            labelDir = labelTextDir;
        }
        else if (Directory.Exists(label2Dir))
        {
            labelDir = label2Dir;
        }

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            byte[] bytes = await FilePolyfill.ReadAllBytesAsync(binFiles[i], cancellationToken);

            int localFloatsPerPoint = floatsPerPoint;
            int totalPoints = bytes.Length / (localFloatsPerPoint * bytesPerFloat);
            // If format doesn't match 5 floats, try 4 floats
            if (totalPoints == 0 || totalPoints * localFloatsPerPoint * bytesPerFloat != bytes.Length)
            {
                localFloatsPerPoint = 4;
                totalPoints = bytes.Length / (localFloatsPerPoint * bytesPerFloat);
            }

            int pointsToRead = Math.Min(totalPoints, _options.PointsPerSample);

            int featureOffset = i * featureSize;
            for (int p = 0; p < pointsToRead; p++)
            {
                int byteBase = p * localFloatsPerPoint * bytesPerFloat;
                for (int c = 0; c < _channels; c++)
                {
                    if (byteBase + (c + 1) * bytesPerFloat <= bytes.Length)
                    {
                        float val = BitConverter.ToSingle(bytes, byteBase + c * bytesPerFloat);
                        featuresData[featureOffset + p * _channels + c] = NumOps.FromDouble(val);
                    }
                }
            }

            // Parse label for this frame
            labelsData[i] = NumOps.FromDouble(
                ParseNuScenesLabel(labelDir, binFiles[i], useLidarseg));
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 1 });
        InitializeIndices(totalSamples);
    }

    /// <summary>
    /// Parses a nuScenes label and returns the dominant object class.
    /// Supports: lidarseg binary files (per-point uint8 labels) and text label files.
    /// </summary>
    private static int ParseNuScenesLabel(string? labelDir, string binFilePath, bool useLidarseg)
    {
        if (labelDir == null) return -1;

        string baseName = Path.GetFileNameWithoutExtension(binFilePath);

        if (useLidarseg)
        {
            // Lidarseg: binary file with one uint8 per point (semantic class)
            string lblFile = Path.Combine(labelDir, baseName + ".bin");
            if (!File.Exists(lblFile))
                lblFile = Path.Combine(labelDir, baseName + "_lidarseg.bin");
            if (!File.Exists(lblFile))
                return -1;

            byte[] lblBytes = File.ReadAllBytes(lblFile);
            var classCounts = new Dictionary<int, int>();
            foreach (byte b in lblBytes)
            {
                if (b == 0) continue; // Skip background/unlabeled
                classCounts.TryGetValue(b, out int count);
                classCounts[b] = count + 1;
            }

            return FindDominantClass(classCounts);
        }
        else
        {
            // Text label file: "type ..." or "class_id ..." per line
            string lblFile = Path.Combine(labelDir, baseName + ".txt");
            if (!File.Exists(lblFile))
                return -1;

            var classCounts = new Dictionary<int, int>();
            foreach (string line in File.ReadAllLines(lblFile))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0) continue;

                int classId;
                if (int.TryParse(parts[0], out int numericId))
                {
                    classId = numericId;
                }
                else
                {
                    string key = parts[0].Replace("vehicle.", "").Replace("human.pedestrian.", "pedestrian")
                        .Replace("movable_object.", "");
                    if (!NuScenesClassMap.TryGetValue(key, out classId) &&
                        !NuScenesClassMap.TryGetValue(parts[0], out classId))
                        continue;
                }

                classCounts.TryGetValue(classId, out int count);
                classCounts[classId] = count + 1;
            }

            return FindDominantClass(classCounts);
        }
    }

    private static int FindDominantClass(Dictionary<int, int> classCounts)
    {
        if (classCounts.Count == 0)
            return -1;

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
