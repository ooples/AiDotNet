using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Loads the SemanticKITTI dataset (per-point semantic labels for LiDAR point clouds).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SemanticKITTI expects:
/// <code>
/// {DataPath}/
///   sequences/
///     00/ .. 21/
///       velodyne/     (.bin files, 4 floats per point: x, y, z, reflectance)
///       labels/       (.label files, uint32 per point: lower 16 bits = semantic label)
/// </code>
/// Features are point cloud Tensor[N, PointsPerSample * 3].
/// Labels are per-point semantic class Tensor[N, PointsPerSample].
/// </para>
/// </remarks>
public class SemanticKittiDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly SemanticKittiDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "SemanticKITTI";
    public override string Description => "SemanticKITTI per-point semantic segmentation";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.PointsPerSample * 3;
    public override int OutputDimension => _options.PointsPerSample;

    public SemanticKittiDataLoader(SemanticKittiDataLoaderOptions? options = null)
    {
        _options = options ?? new SemanticKittiDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("semantic_kitti");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // SemanticKITTI uses sequence directories
        string seqDir = Path.Combine(_dataPath, "sequences");
        if (!Directory.Exists(seqDir))
            seqDir = _dataPath;

        // Collect all velodyne/label pairs from sequence directories
        var binFiles = new List<string>();
        var labelFiles = new List<string>();

        if (Directory.Exists(seqDir))
        {
            foreach (var seq in Directory.GetDirectories(seqDir).OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
            {
                string veloDir = Path.Combine(seq, "velodyne");
                string lblDir = Path.Combine(seq, "labels");
                if (!Directory.Exists(veloDir) || !Directory.Exists(lblDir))
                    continue;

                var seqBins = Directory.GetFiles(veloDir, "*.bin");
                Array.Sort(seqBins, StringComparer.OrdinalIgnoreCase);
                foreach (var bin in seqBins)
                {
                    string labelFile = Path.Combine(lblDir,
                        Path.GetFileNameWithoutExtension(bin) + ".label");
                    if (File.Exists(labelFile))
                    {
                        binFiles.Add(bin);
                        labelFiles.Add(labelFile);
                    }
                }
            }
        }

        // Fallback: flat directory with .bin and .label files
        if (binFiles.Count == 0)
        {
            var flatBins = Directory.GetFiles(_dataPath, "*.bin", SearchOption.AllDirectories);
            Array.Sort(flatBins, StringComparer.OrdinalIgnoreCase);
            foreach (var bin in flatBins)
            {
                string dir = Path.GetDirectoryName(bin) ?? _dataPath;
                string labelFile = Path.Combine(dir,
                    Path.GetFileNameWithoutExtension(bin) + ".label");
                if (!File.Exists(labelFile))
                {
                    string lblDir = Path.Combine(Path.GetDirectoryName(dir) ?? dir, "labels");
                    labelFile = Path.Combine(lblDir, Path.GetFileNameWithoutExtension(bin) + ".label");
                }
                binFiles.Add(bin);
                labelFiles.Add(File.Exists(labelFile) ? labelFile : "");
            }
        }

        if (binFiles.Count == 0)
            throw new DirectoryNotFoundException($"SemanticKITTI data not found at {_dataPath}.");

        int totalSamples = binFiles.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int pps = _options.PointsPerSample;
        var featuresData = new T[totalSamples * pps * 3];
        var labelsData = new T[totalSamples * pps];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Read point cloud
            byte[] bytes = await FilePolyfill.ReadAllBytesAsync(binFiles[i], cancellationToken);
            int floatsPerPoint = 4;
            int bytesPerFloat = 4;
            int totalPoints = bytes.Length / (floatsPerPoint * bytesPerFloat);
            int pointsToRead = Math.Min(totalPoints, pps);

            int featOff = i * pps * 3;
            for (int p = 0; p < pointsToRead; p++)
            {
                int byteBase = p * floatsPerPoint * bytesPerFloat;
                for (int c = 0; c < 3; c++)
                {
                    if (byteBase + (c + 1) * bytesPerFloat <= bytes.Length)
                    {
                        float val = BitConverter.ToSingle(bytes, byteBase + c * bytesPerFloat);
                        featuresData[featOff + p * 3 + c] = NumOps.FromDouble(val);
                    }
                }
            }

            // Read labels (uint32 per point, lower 16 bits = semantic label)
            int lblOff = i * pps;
            if (!string.IsNullOrEmpty(labelFiles[i]) && File.Exists(labelFiles[i]))
            {
                byte[] lblBytes = await FilePolyfill.ReadAllBytesAsync(labelFiles[i], cancellationToken);
                int lblPoints = lblBytes.Length / 4;
                int lblToRead = Math.Min(lblPoints, pps);
                for (int p = 0; p < lblToRead; p++)
                {
                    uint raw = BitConverter.ToUInt32(lblBytes, p * 4);
                    int semanticLabel = (int)(raw & 0xFFFF);
                    labelsData[lblOff + p] = NumOps.FromDouble(semanticLabel % _options.NumClasses);
                }
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, pps * 3 });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, pps });
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
