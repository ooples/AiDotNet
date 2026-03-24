using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Loads temporal graph datasets (timestamped interactions for dynamic link prediction).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects CSV/TSV interaction files:
/// <code>
/// {DataPath}/
///   interactions.csv   (source, target, timestamp, label, [features...])
///   node_features.csv  (optional: node_id, feature_1, ...)
/// </code>
/// Features are interaction features Tensor[N, NodeFeatureDim + EdgeFeatureDim].
/// Labels are interaction label Tensor[N, 1].
/// </para>
/// </remarks>
public class TemporalGraphDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly TemporalGraphDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "TemporalGraph";
    public override string Description => "Temporal graph dynamic link prediction";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.NodeFeatureDimension + _options.EdgeFeatureDimension;
    public override int OutputDimension => 1;

    public TemporalGraphDataLoader(TemporalGraphDataLoaderOptions? options = null)
    {
        _options = options ?? new TemporalGraphDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("temporal_graph");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Find interactions file
        string? interactionsFile = FindDataFile("interactions", "csv", "tsv", "txt");
        if (interactionsFile == null)
            throw new FileNotFoundException($"Temporal graph interactions file not found at {_dataPath}.");

        string[] lines = await FilePolyfill.ReadAllLinesAsync(interactionsFile, cancellationToken);

        int nodeFeatDim = _options.NodeFeatureDimension;
        int edgeFeatDim = _options.EdgeFeatureDimension;
        int totalFeatDim = nodeFeatDim + edgeFeatDim;

        var interactions = new List<(int src, int dst, double timestamp, int label, double[] edgeFeats)>();

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(new[] { ',', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3) continue;

            // Skip header row
            if (!int.TryParse(parts[0].Trim(), out int src)) continue;
            if (!int.TryParse(parts[1].Trim(), out int dst)) continue;

            double.TryParse(parts.Length > 2 ? parts[2].Trim() : "0",
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double timestamp);

            int label = 0;
            if (parts.Length > 3)
                int.TryParse(parts[3].Trim(), out label);

            // Parse edge features
            var edgeFeats = new double[edgeFeatDim];
            for (int f = 0; f < edgeFeatDim && f + 4 < parts.Length; f++)
            {
                double.TryParse(parts[f + 4].Trim(),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out edgeFeats[f]);
            }

            interactions.Add((src, dst, timestamp, label, edgeFeats));

            if (_options.MaxSamples.HasValue && interactions.Count >= _options.MaxSamples.Value)
                break;
        }

        // Sort by timestamp for temporal ordering
        interactions.Sort((a, b) => a.timestamp.CompareTo(b.timestamp));

        // Optionally load node features
        var nodeFeatureMap = new Dictionary<int, double[]>();
        string? nodeFile = FindDataFile("node_features", "csv", "tsv", "txt");
        if (nodeFile != null)
        {
            string[] nodeLines = await FilePolyfill.ReadAllLinesAsync(nodeFile, cancellationToken);
            foreach (string line in nodeLines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(new[] { ',', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2 || !int.TryParse(parts[0].Trim(), out int nodeId)) continue;

                var feats = new double[nodeFeatDim];
                for (int f = 0; f < nodeFeatDim && f + 1 < parts.Length; f++)
                {
                    double.TryParse(parts[f + 1].Trim(),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out feats[f]);
                }
                nodeFeatureMap[nodeId] = feats;
            }
        }

        int totalSamples = interactions.Count;
        _sampleCount = totalSamples;
        var featuresData = new T[totalSamples * totalFeatDim];
        var labelsData = new T[totalSamples];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (src, dst, _, label, edgeFeats) = interactions[i];

            int featOff = i * totalFeatDim;

            // Node features: concatenate source and destination (split dim evenly)
            int halfNode = nodeFeatDim / 2;
            if (nodeFeatureMap.TryGetValue(src, out var srcFeats))
            {
                for (int f = 0; f < Math.Min(halfNode, srcFeats.Length); f++)
                    featuresData[featOff + f] = NumOps.FromDouble(srcFeats[f]);
            }
            else
            {
                // Simple hash embedding for unknown nodes
                for (int f = 0; f < halfNode; f++)
                    featuresData[featOff + f] = NumOps.FromDouble(Math.Sin((src + 1.0) * (f + 1.0)));
            }

            if (nodeFeatureMap.TryGetValue(dst, out var dstFeats))
            {
                for (int f = 0; f < Math.Min(halfNode, dstFeats.Length); f++)
                    featuresData[featOff + halfNode + f] = NumOps.FromDouble(dstFeats[f]);
            }
            else
            {
                for (int f = 0; f < halfNode; f++)
                    featuresData[featOff + halfNode + f] = NumOps.FromDouble(Math.Sin((dst + 1.0) * (f + 1.0)));
            }

            // Edge features
            for (int f = 0; f < edgeFeatDim; f++)
                featuresData[featOff + nodeFeatDim + f] = NumOps.FromDouble(edgeFeats[f]);

            labelsData[i] = NumOps.FromDouble(label);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, totalFeatDim });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 1 });
        InitializeIndices(totalSamples);
    }

    private string? FindDataFile(string baseName, params string[] extensions)
    {
        foreach (var ext in extensions)
        {
            string path = Path.Combine(_dataPath, $"{baseName}.{ext}");
            if (File.Exists(path)) return path;
        }

        if (Directory.Exists(_dataPath))
        {
            var files = Directory.GetFiles(_dataPath, $"*{baseName}*");
            if (files.Length > 0) return files[0];
        }

        return null;
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        var nfs = (int[])features.Shape._dims.Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape._dims.Clone(); nls[0] = indices.Length;
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
