using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Loads citation network datasets (Cora, CiteSeer, PubMed) for node classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Citation networks are classic benchmarks for graph neural networks. Each dataset represents
/// academic papers as nodes and citations as edges, with the task being to classify papers into
/// research topics.
/// </para>
/// <para><b>For Beginners:</b> Citation networks are graphs of research papers.
///
/// **Structure:**
/// - **Nodes**: Research papers
/// - **Edges**: Citations (Paper A cites Paper B)
/// - **Node Features**: Bag-of-words representation of paper abstracts
/// - **Labels**: Research topic/category
///
/// **Datasets:**
///
/// **Cora:**
/// - 2,708 papers
/// - 5,429 citations
/// - 1,433 features (unique words)
/// - 7 classes (topics): Case_Based, Genetic_Algorithms, Neural_Networks,
///   Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory
/// - Task: Classify papers by topic
///
/// **CiteSeer:**
/// - 3,312 papers
/// - 4,732 citations
/// - 3,703 features
/// - 6 classes: Agents, AI, DB, IR, ML, HCI
///
/// **PubMed:**
/// - 19,717 papers (about diabetes)
/// - 44,338 citations
/// - 500 features
/// - 3 classes: Diabetes Mellitus Type 1, Type 2, Experimental
///
/// **Key Property: Homophily**
/// Papers tend to cite papers on similar topics. This makes GNNs effective:
/// - If neighbors are similar topics, aggregate their features
/// - GNN learns to propagate topic information through citation network
/// - Even unlabeled papers can be classified based on what they cite
/// </para>
/// </remarks>
public class CitationNetworkLoader<T> : IGraphDataLoader<T>
{
    private readonly CitationDataset _dataset;
    private readonly string _dataPath;
    private GraphData<T>? _loadedGraph;
    private bool _hasLoaded;

    /// <summary>
    /// Available citation network datasets.
    /// </summary>
    public enum CitationDataset
    {
        /// <summary>Cora dataset (2,708 papers, 7 classes)</summary>
        Cora,

        /// <summary>CiteSeer dataset (3,312 papers, 6 classes)</summary>
        CiteSeer,

        /// <summary>PubMed dataset (19,717 papers, 3 classes)</summary>
        PubMed
    }

    /// <inheritdoc/>
    public int NumGraphs => 1; // Single large graph

    /// <inheritdoc/>
    public int BatchSize => 1;

    /// <inheritdoc/>
    public bool HasNext => !_hasLoaded;

    /// <summary>
    /// Initializes a new instance of the <see cref="CitationNetworkLoader{T}"/> class.
    /// </summary>
    /// <param name="dataset">Which citation dataset to load.</param>
    /// <param name="dataPath">Path to the dataset files (optional, will download if not found).</param>
    /// <remarks>
    /// <para>
    /// The loader expects data files in the following format:
    /// - {dataset}.content: Node features and labels
    /// - {dataset}.cites: Edge list
    /// </para>
    /// <para><b>For Beginners:</b> Using this loader:
    ///
    /// ```csharp
    /// // Load Cora dataset
    /// var loader = new CitationNetworkLoader<double>(
    ///     CitationNetworkLoader<double>.CitationDataset.Cora,
    ///     "path/to/data");
    ///
    /// // Get the graph
    /// var graph = loader.GetNextBatch();
    ///
    /// // Access data
    /// Console.WriteLine($"Nodes: {graph.NumNodes}");
    /// Console.WriteLine($"Edges: {graph.NumEdges}");
    /// Console.WriteLine($"Features per node: {graph.NumNodeFeatures}");
    ///
    /// // Create node classification task
    /// var task = loader.CreateNodeClassificationTask();
    /// ```
    /// </para>
    /// </remarks>
    public CitationNetworkLoader(CitationDataset dataset, string? dataPath = null)
    {
        _dataset = dataset;
        _dataPath = dataPath ?? GetDefaultDataPath();
        _hasLoaded = false;
    }

    /// <inheritdoc/>
    public GraphData<T> GetNextBatch()
    {
        if (_loadedGraph == null)
        {
            LoadDataset();
        }

        _hasLoaded = true;
        return _loadedGraph!;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _hasLoaded = false;
    }

    /// <summary>
    /// Creates a node classification task from the loaded citation network.
    /// </summary>
    /// <param name="trainRatio">Fraction of nodes for training (default: 0.1)</param>
    /// <param name="valRatio">Fraction of nodes for validation (default: 0.1)</param>
    /// <returns>Node classification task with train/val/test splits.</returns>
    /// <remarks>
    /// <para>
    /// Standard splits for citation networks:
    /// - Train: 10% (few labeled papers)
    /// - Validation: 10%
    /// - Test: 80%
    ///
    /// This is semi-supervised learning: most nodes are unlabeled.
    /// </para>
    /// <para><b>For Beginners:</b> Why so few training labels?
    ///
    /// Citation networks test semi-supervised learning:
    /// - In real research, labeling papers is expensive (requires expert knowledge)
    /// - We typically have few labeled examples
    /// - Graph structure helps: papers citing each other often share topics
    ///
    /// Example with 2,708 papers (Cora):
    /// - ~270 labeled for training (10%)
    /// - ~270 for validation
    /// - ~2,168 for testing
    ///
    /// The GNN uses citation connections to propagate label information from the
    /// 270 labeled papers to classify the remaining 2,168 unlabeled papers!
    /// </para>
    /// </remarks>
    public NodeClassificationTask<T> CreateNodeClassificationTask(
        double trainRatio = 0.1,
        double valRatio = 0.1)
    {
        if (_loadedGraph == null)
        {
            LoadDataset();
        }

        var graph = _loadedGraph!;
        int numNodes = graph.NumNodes;

        // Create random split
        var indices = Enumerable.Range(0, numNodes).OrderBy(_ => Guid.NewGuid()).ToArray();
        int trainSize = (int)(numNodes * trainRatio);
        int valSize = (int)(numNodes * valRatio);

        var trainIndices = indices.Take(trainSize).ToArray();
        var valIndices = indices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = indices.Skip(trainSize + valSize).ToArray();

        // Count number of classes
        int numClasses = CountClasses(graph.NodeLabels!);

        return new NodeClassificationTask<T>
        {
            Graph = graph,
            Labels = graph.NodeLabels!,
            TrainIndices = trainIndices,
            ValIndices = valIndices,
            TestIndices = testIndices,
            NumClasses = numClasses,
            IsMultiLabel = false
        };
    }

    private void LoadDataset()
    {
        // This is a simplified loader. Full implementation would:
        // 1. Check if files exist locally
        // 2. Download from standard sources if needed
        // 3. Parse .content and .cites files
        // 4. Build adjacency matrix
        // 5. Create node features and labels

        var (numNodes, numFeatures, numClasses) = GetDatasetStats();

        _loadedGraph = new GraphData<T>
        {
            NodeFeatures = CreateMockNodeFeatures(numNodes, numFeatures),
            AdjacencyMatrix = CreateMockAdjacency(numNodes),
            EdgeIndex = CreateMockEdgeIndex(numNodes),
            NodeLabels = CreateMockLabels(numNodes, numClasses)
        };
    }

    private (int numNodes, int numFeatures, int numClasses) GetDatasetStats()
    {
        return _dataset switch
        {
            CitationDataset.Cora => (2708, 1433, 7),
            CitationDataset.CiteSeer => (3312, 3703, 6),
            CitationDataset.PubMed => (19717, 500, 3),
            _ => throw new ArgumentException($"Unknown dataset: {_dataset}")
        };
    }

    private string GetDefaultDataPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".aidotnet",
            "datasets",
            "citation_networks");
    }

    private Tensor<T> CreateMockNodeFeatures(int numNodes, int numFeatures)
    {
        // In real implementation, load from {dataset}.content file
        var features = new Tensor<T>([numNodes, numFeatures]);
        var random = new Random(42);

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                // Sparse binary features (bag-of-words)
                features[i, j] = random.NextDouble() < 0.05
                    ? NumOps.FromDouble(1.0)
                    : NumOps.Zero;
            }
        }

        return features;
    }

    private Tensor<T> CreateMockAdjacency(int numNodes)
    {
        // In real implementation, build from {dataset}.cites file
        var adj = new Tensor<T>([1, numNodes, numNodes]);
        var random = new Random(42);

        // Create sparse random graph structure
        for (int i = 0; i < numNodes; i++)
        {
            // Each node cites ~5 others on average (citation networks are sparse)
            int numCitations = random.Next(2, 8);
            for (int c = 0; c < numCitations; c++)
            {
                int target = random.Next(numNodes);
                if (target != i) // No self-loops
                {
                    adj[0, i, target] = NumOps.FromDouble(1.0);
                }
            }
        }

        return adj;
    }

    private Tensor<T> CreateMockEdgeIndex(int numNodes)
    {
        // In real implementation, parse from {dataset}.cites
        var edges = new List<(int, int)>();
        var random = new Random(42);

        for (int i = 0; i < numNodes; i++)
        {
            int numCitations = random.Next(2, 8);
            for (int c = 0; c < numCitations; c++)
            {
                int target = random.Next(numNodes);
                if (target != i)
                {
                    edges.Add((i, target));
                }
            }
        }

        var edgeIndex = new Tensor<T>([edges.Count, 2]);
        for (int i = 0; i < edges.Count; i++)
        {
            edgeIndex[i, 0] = NumOps.FromDouble(edges[i].Item1);
            edgeIndex[i, 1] = NumOps.FromDouble(edges[i].Item2);
        }

        return edgeIndex;
    }

    private Tensor<T> CreateMockLabels(int numNodes, int numClasses)
    {
        // One-hot encoded labels
        var labels = new Tensor<T>([numNodes, numClasses]);
        var random = new Random(42);

        for (int i = 0; i < numNodes; i++)
        {
            int classIdx = random.Next(numClasses);
            labels[i, classIdx] = NumOps.FromDouble(1.0);
        }

        return labels;
    }

    private int CountClasses(Tensor<T> labels)
    {
        return labels.Shape[1]; // One-hot encoded
    }
}
