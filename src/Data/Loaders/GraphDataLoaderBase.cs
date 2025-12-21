using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Abstract base class for graph data loaders providing common graph-related functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// GraphDataLoaderBase provides shared implementation for all graph data loaders including:
/// - Node feature and adjacency matrix management
/// - Task creation (node classification, graph classification, link prediction)
/// - Train/validation/test mask generation
/// - Batch iteration for multiple graphs
/// </para>
/// <para><b>For Beginners:</b> This base class handles common graph operations:
/// - Storing node features and edge connections
/// - Creating different types of tasks (node classification, link prediction)
/// - Splitting data for training and evaluation
///
/// Concrete implementations (CitationNetworkLoader, MolecularDatasetLoader) extend this
/// to load specific graph datasets.
/// </para>
/// </remarks>
public abstract class GraphDataLoaderBase<T> : DataLoaderBase<T>, IGraphDataLoader<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Storage for loaded graph data.
    /// </summary>
    protected GraphData<T>? LoadedGraphData;

    /// <summary>
    /// Storage for multiple graphs (for graph classification datasets).
    /// </summary>
    protected List<GraphData<T>>? LoadedGraphs;

    private int _batchSize = 1;
    private int _currentGraphIndex;

    /// <inheritdoc/>
    public virtual int NumGraphs => LoadedGraphs?.Count ?? (LoadedGraphData != null ? 1 : 0);

    /// <inheritdoc/>
    public override int TotalCount => NumGraphs;

    /// <inheritdoc/>
    public Tensor<T> NodeFeatures
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.NodeFeatures ?? new Tensor<T>([0, 0]);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> AdjacencyMatrix
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.AdjacencyMatrix ?? new Tensor<T>([0, 0]);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> EdgeIndex
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.EdgeIndex ?? new Tensor<T>([0, 2]);
        }
    }

    /// <inheritdoc/>
    public Tensor<T>? NodeLabels
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.NodeLabels;
        }
    }

    /// <inheritdoc/>
    public Tensor<T>? GraphLabels { get; protected set; }

    /// <inheritdoc/>
    public int NumNodeFeatures
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.NumNodeFeatures ?? 0;
        }
    }

    /// <inheritdoc/>
    public int NumNodes
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.NumNodes ?? 0;
        }
    }

    /// <inheritdoc/>
    public int NumEdges
    {
        get
        {
            EnsureLoaded();
            return LoadedGraphData?.NumEdges ?? 0;
        }
    }

    /// <inheritdoc/>
    public abstract int NumClasses { get; }

    /// <inheritdoc/>
    public override int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    /// <inheritdoc/>
    public bool HasNext => _currentGraphIndex < NumGraphs;

    /// <inheritdoc/>
    public GraphData<T> GetNextBatch()
    {
        EnsureLoaded();

        if (!HasNext)
        {
            throw new InvalidOperationException("No more batches available. Call Reset() to start over.");
        }

        // For single-graph datasets
        if (LoadedGraphs == null || LoadedGraphs.Count == 0)
        {
            _currentGraphIndex = NumGraphs; // Mark as consumed
            AdvanceBatchIndex();
            return LoadedGraphData!;
        }

        // For multi-graph datasets, return next graph
        var graph = LoadedGraphs[_currentGraphIndex];
        _currentGraphIndex++;
        AdvanceIndex(1);
        AdvanceBatchIndex();
        return graph;
    }

    /// <inheritdoc/>
    public bool TryGetNextBatch(out GraphData<T> batch)
    {
        if (!HasNext)
        {
            batch = new GraphData<T>();
            return false;
        }

        batch = GetNextBatch();
        return true;
    }

    /// <inheritdoc/>
    protected override void OnReset()
    {
        _currentGraphIndex = 0;
    }

    /// <inheritdoc/>
    public virtual NodeClassificationTask<T> CreateNodeClassificationTask(
        double trainRatio = 0.1,
        double valRatio = 0.1,
        int? seed = null)
    {
        EnsureLoaded();

        if (LoadedGraphData == null)
        {
            throw new InvalidOperationException("Graph data not loaded.");
        }

        if (LoadedGraphData.NodeLabels == null)
        {
            throw new InvalidOperationException("Node labels not available for node classification task.");
        }

        int numNodes = LoadedGraphData.NumNodes;
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Create shuffled indices
        var indices = Enumerable.Range(0, numNodes).OrderBy(_ => random.Next()).ToArray();

        int trainSize = (int)(numNodes * trainRatio);
        int valSize = (int)(numNodes * valRatio);

        var trainIndices = indices.Take(trainSize).ToArray();
        var valIndices = indices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = indices.Skip(trainSize + valSize).ToArray();

        return new NodeClassificationTask<T>
        {
            Graph = LoadedGraphData,
            Labels = LoadedGraphData.NodeLabels,
            TrainIndices = trainIndices,
            ValIndices = valIndices,
            TestIndices = testIndices,
            NumClasses = NumClasses,
            IsMultiLabel = false
        };
    }

    /// <inheritdoc/>
    public virtual GraphClassificationTask<T> CreateGraphClassificationTask(
        double trainRatio = 0.8,
        double valRatio = 0.1,
        int? seed = null)
    {
        EnsureLoaded();

        if (LoadedGraphs == null || LoadedGraphs.Count == 0)
        {
            throw new InvalidOperationException("Multiple graphs required for graph classification task.");
        }

        int numGraphs = LoadedGraphs.Count;
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Create shuffled indices
        var indices = Enumerable.Range(0, numGraphs).OrderBy(_ => random.Next()).ToArray();

        int trainSize = (int)(numGraphs * trainRatio);
        int valSize = (int)(numGraphs * valRatio);

        var trainIndices = indices.Take(trainSize).ToArray();
        var valIndices = indices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = indices.Skip(trainSize + valSize).ToArray();

        // Split graphs into train/val/test sets
        var trainGraphs = trainIndices.Select(i => LoadedGraphs[i]).ToList();
        var valGraphs = valIndices.Select(i => LoadedGraphs[i]).ToList();
        var testGraphs = testIndices.Select(i => LoadedGraphs[i]).ToList();

        // Split labels
        var trainLabels = ExtractLabels(GraphLabels!, trainIndices);
        var valLabels = ExtractLabels(GraphLabels!, valIndices);
        var testLabels = ExtractLabels(GraphLabels!, testIndices);

        return new GraphClassificationTask<T>
        {
            TrainGraphs = trainGraphs,
            ValGraphs = valGraphs,
            TestGraphs = testGraphs,
            TrainLabels = trainLabels,
            ValLabels = valLabels,
            TestLabels = testLabels,
            NumClasses = NumClasses
        };
    }

    /// <summary>
    /// Extracts labels at specified indices from a label tensor.
    /// </summary>
    private Tensor<T> ExtractLabels(Tensor<T> labels, int[] indices)
    {
        if (labels.Shape.Length == 1)
        {
            var result = new Tensor<T>([indices.Length]);
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = labels[indices[i]];
            }
            return result;
        }
        else
        {
            int numClasses = labels.Shape[1];
            var result = new Tensor<T>([indices.Length, numClasses]);
            for (int i = 0; i < indices.Length; i++)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    result[i, c] = labels[indices[i], c];
                }
            }
            return result;
        }
    }

    /// <inheritdoc/>
    public virtual LinkPredictionTask<T> CreateLinkPredictionTask(
        double trainRatio = 0.85,
        double negativeRatio = 1.0,
        int? seed = null)
    {
        EnsureLoaded();

        if (LoadedGraphData == null)
        {
            throw new InvalidOperationException("Graph data not loaded.");
        }

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var edgeIndex = LoadedGraphData.EdgeIndex;
        int numEdges = LoadedGraphData.NumEdges;
        int numNodes = LoadedGraphData.NumNodes;

        // Shuffle edges
        var edgeIndices = Enumerable.Range(0, numEdges).OrderBy(_ => random.Next()).ToArray();

        int trainSize = (int)(numEdges * trainRatio);
        int valSize = (int)(numEdges * 0.05); // 5% for validation

        var trainEdgeIndices = edgeIndices.Take(trainSize).ToArray();
        var valEdgeIndices = edgeIndices.Skip(trainSize).Take(valSize).ToArray();
        var testEdgeIndices = edgeIndices.Skip(trainSize + valSize).ToArray();

        // Extract edge tensors
        var trainPosEdges = ExtractEdges(edgeIndex, trainEdgeIndices);
        var valPosEdges = ExtractEdges(edgeIndex, valEdgeIndices);
        var testPosEdges = ExtractEdges(edgeIndex, testEdgeIndices);

        // Generate negative samples
        int numTrainNeg = (int)(trainSize * negativeRatio);
        int numValNeg = (int)(valSize * negativeRatio);
        int numTestNeg = (int)(testEdgeIndices.Length * negativeRatio);

        var trainNegEdges = GenerateNegativeEdges(LoadedGraphData, numTrainNeg, random);
        var valNegEdges = GenerateNegativeEdges(LoadedGraphData, numValNeg, random);
        var testNegEdges = GenerateNegativeEdges(LoadedGraphData, numTestNeg, random);

        return new LinkPredictionTask<T>
        {
            Graph = LoadedGraphData,
            TrainPosEdges = trainPosEdges,
            TrainNegEdges = trainNegEdges,
            ValPosEdges = valPosEdges,
            ValNegEdges = valNegEdges,
            TestPosEdges = testPosEdges,
            TestNegEdges = testNegEdges,
            NegativeSamplingRatio = negativeRatio
        };
    }

    /// <summary>
    /// Extracts edges at specified indices from an edge index tensor.
    /// </summary>
    private Tensor<T> ExtractEdges(Tensor<T> edgeIndex, int[] indices)
    {
        var result = new Tensor<T>([indices.Length, 2]);
        for (int i = 0; i < indices.Length; i++)
        {
            result[i, 0] = edgeIndex[indices[i], 0];
            result[i, 1] = edgeIndex[indices[i], 1];
        }
        return result;
    }

    /// <summary>
    /// Generates negative edge samples (non-existing edges) as a tensor.
    /// </summary>
    private Tensor<T> GenerateNegativeEdges(GraphData<T> graph, int count, Random random)
    {
        var existingEdges = new HashSet<(int, int)>();
        var edgeIndex = graph.EdgeIndex;
        int numEdges = graph.NumEdges;

        // Build set of existing edges
        for (int i = 0; i < numEdges; i++)
        {
            int src = NumOps.ToInt32(edgeIndex[i, 0]);
            int dst = NumOps.ToInt32(edgeIndex[i, 1]);
            existingEdges.Add((src, dst));
        }

        var negatives = new List<(int src, int dst)>();
        int numNodes = graph.NumNodes;

        // Sample negative edges
        while (negatives.Count < count)
        {
            int src = random.Next(numNodes);
            int dst = random.Next(numNodes);

            if (src != dst && !existingEdges.Contains((src, dst)))
            {
                negatives.Add((src, dst));
                existingEdges.Add((src, dst)); // Prevent duplicates
            }
        }

        // Convert to tensor
        var result = new Tensor<T>([negatives.Count, 2]);
        for (int i = 0; i < negatives.Count; i++)
        {
            result[i, 0] = NumOps.FromDouble(negatives[i].src);
            result[i, 1] = NumOps.FromDouble(negatives[i].dst);
        }

        return result;
    }
}
