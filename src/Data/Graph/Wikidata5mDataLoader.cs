using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Loads Wikidata5M knowledge graph triplets as tensor features and labels.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects TSV files with triplets (head, relation, tail):
/// <code>
/// {DataPath}/
///   wikidata5m_transductive_train.txt   (head_id \t relation_id \t tail_id)
///   wikidata5m_transductive_valid.txt
///   wikidata5m_transductive_test.txt
/// </code>
/// Features are entity pair embeddings Tensor[N, 2 * EmbeddingDimension].
/// Labels are relation index Tensor[N, 1].
/// </para>
/// </remarks>
public class Wikidata5mDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly Wikidata5mDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc />
    public override string Name => "Wikidata5M";
    /// <inheritdoc />
    public override string Description => "Wikidata5M knowledge graph link prediction";
    /// <inheritdoc />
    public override int TotalCount => _sampleCount;
    /// <inheritdoc />
    public override int FeatureCount => _options.EmbeddingDimension * 2;
    /// <inheritdoc />
    public override int OutputDimension => 1;

    public Wikidata5mDataLoader(Wikidata5mDataLoaderOptions? options = null)
    {
        _options = options ?? new Wikidata5mDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("wikidata5m");
    }

    /// <inheritdoc />
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Find triplet file for the requested split
        string splitName = _options.Split == DatasetSplit.Test ? "test"
            : _options.Split == DatasetSplit.Validation ? "valid"
            : "train";

        string? tripletFile = FindTripletFile(splitName);
        if (tripletFile == null)
            throw new FileNotFoundException($"Wikidata5M triplet file not found at {_dataPath} for split '{splitName}'.");

        string[] lines = await FilePolyfill.ReadAllLinesAsync(tripletFile, cancellationToken);

        // Build entity and relation vocabularies
        var entityToId = new Dictionary<string, int>(StringComparer.Ordinal);
        var relationToId = new Dictionary<string, int>(StringComparer.Ordinal);
        var triplets = new List<(int head, int relation, int tail)>();

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(new[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3) continue;

            string headStr = parts[0].Trim();
            string relStr = parts[1].Trim();
            string tailStr = parts[2].Trim();

            if (!entityToId.TryGetValue(headStr, out int headId))
            {
                headId = entityToId.Count;
                entityToId[headStr] = headId;
            }
            if (!relationToId.TryGetValue(relStr, out int relId))
            {
                relId = relationToId.Count;
                relationToId[relStr] = relId;
            }
            if (!entityToId.TryGetValue(tailStr, out int tailId))
            {
                tailId = entityToId.Count;
                entityToId[tailStr] = tailId;
            }

            triplets.Add((headId, relId, tailId));

            if (_options.MaxSamples.HasValue && triplets.Count >= _options.MaxSamples.Value)
                break;
        }

        int totalSamples = triplets.Count;
        _sampleCount = totalSamples;
        int embDim = _options.EmbeddingDimension;
        int featSize = embDim * 2;
        var headIds = new int[totalSamples];
        var tailIds = new int[totalSamples];
        var labelsData = new int[totalSamples];
        int totalEntities = Math.Max(entityToId.Count, 1);

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (head, relation, tail) = triplets[i];
            headIds[i] = head + 1;
            tailIds[i] = tail + 1;
            labelsData[i] = relation;
        }

        // Simple hash-based embedding: distribute entity ID across embedding dimensions,
        // sin((id + 1) * (d + 1) / totalEntities), as engine outer products in T.
        Tensor<T> features;
        if (totalSamples == 0 || embDim == 0)
        {
            features = new Tensor<T>(new[] { totalSamples, featSize });
        }
        else
        {
            var engine = AiDotNetEngine.Current;
            var dims = new Tensor<int>(Enumerable.Range(1, embDim).ToArray(), new[] { embDim }).Cast<T>();
            T entityCount = NumOps.FromDouble(totalEntities);
            var headEmb = engine.TensorSin(engine.TensorDivideScalar(
                engine.TensorOuter(new Tensor<int>(headIds, new[] { totalSamples }).Cast<T>(), dims), entityCount));
            var tailEmb = engine.TensorSin(engine.TensorDivideScalar(
                engine.TensorOuter(new Tensor<int>(tailIds, new[] { totalSamples }).Cast<T>(), dims), entityCount));
            features = engine.TensorConcatenate(new[] { headEmb, tailEmb }, 1);
        }

        LoadedFeatures = features;
        LoadedLabels = new Tensor<int>(labelsData, new[] { totalSamples, 1 }).Cast<T>();
        InitializeIndices(totalSamples);
    }

    private string? FindTripletFile(string splitName)
    {
        string[] patterns =
        {
            $"wikidata5m_transductive_{splitName}.txt",
            $"wikidata5m_{splitName}.txt",
            $"{splitName}.txt",
            $"{splitName}.tsv"
        };

        foreach (var pattern in patterns)
        {
            string path = Path.Combine(_dataPath, pattern);
            if (File.Exists(path)) return path;
        }

        // Try finding any .txt file in directory
        if (Directory.Exists(_dataPath))
        {
            var files = Directory.GetFiles(_dataPath, $"*{splitName}*");
            if (files.Length > 0) return files[0];
        }

        return null;
    }

    /// <inheritdoc />
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    /// <inheritdoc />
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        // Clone the shape arrays — see KittiDataLoader.ExtractBatch for the
        // SplitsData regression this avoids.
        var nfs = (int[])features._shape.Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels._shape.Clone(); nls[0] = indices.Length;
        var bf = new Tensor<T>(nfs);
        var bl = new Tensor<T>(nls);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, bf, indices[i], i);
            TensorCopyHelper.CopySample(labels, bl, indices[i], i);
        }
        return (bf, bl);
    }

    /// <inheritdoc />
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
