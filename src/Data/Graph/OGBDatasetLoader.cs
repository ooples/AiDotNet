using System.IO.Compression;
using System.Net.Http;
using System.Text;
using AiDotNet.Data.Loaders;
using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Loads datasets from the Open Graph Benchmark (OGB) for standardized evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The Open Graph Benchmark (OGB) is a collection of realistic, large-scale graph datasets
/// with standardized evaluation protocols for graph machine learning research.
/// </para>
/// <para><b>For Beginners:</b> OGB provides standard benchmarks for fair comparison.
///
/// **What is OGB?**
/// - Collection of real-world graph datasets
/// - Standardized train/val/test splits
/// - Automated evaluation metrics
/// - Enables fair comparison between different GNN methods
///
/// **Why OGB matters:**
/// - **Reproducibility**: Everyone uses same data splits
/// - **Realism**: Real-world graphs, not toy datasets
/// - **Scale**: Large graphs that test scalability
/// - **Diversity**: Multiple domains and tasks
///
/// **OGB Dataset Categories:**
///
/// **1. Node Property Prediction:**
/// - ogbn-arxiv: Citation network (169K papers)
/// - ogbn-products: Amazon product co-purchasing network (2.4M products)
/// - ogbn-proteins: Protein association network (132K proteins)
///
/// **2. Link Property Prediction:**
/// - ogbl-collab: Author collaboration network
/// - ogbl-citation2: Citation network
/// - ogbl-ddi: Drug-drug interaction network
///
/// **3. Graph Property Prediction:**
/// - ogbg-molhiv: Molecular graphs for HIV activity prediction (41K molecules)
/// - ogbg-molpcba: Molecular graphs for biological assays (437K molecules)
/// - ogbg-ppa: Protein association graphs
/// </para>
/// </remarks>
public class OGBDatasetLoader<T> : GraphDataLoaderBase<T>
{
    private readonly string _datasetName;
    private readonly string _dataPath;
    private readonly OGBTask _taskType;
    private readonly bool _autoDownload;
    private int _numClasses;

    // Standard OGB download URLs
    private static readonly Dictionary<string, string> DatasetUrls = new()
    {
        // Node prediction datasets
        ["ogbn-arxiv"] = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        ["ogbn-products"] = "https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
        ["ogbn-proteins"] = "https://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip",

        // Graph prediction datasets
        ["ogbg-molhiv"] = "https://snap.stanford.edu/ogb/data/graphproppred/molhiv.zip",
        ["ogbg-molpcba"] = "https://snap.stanford.edu/ogb/data/graphproppred/molpcba.zip",
        ["ogbg-ppa"] = "https://snap.stanford.edu/ogb/data/graphproppred/ppa.zip",

        // Link prediction datasets
        ["ogbl-collab"] = "https://snap.stanford.edu/ogb/data/linkproppred/collab.zip",
        ["ogbl-ddi"] = "https://snap.stanford.edu/ogb/data/linkproppred/ddi.zip",
        ["ogbl-citation2"] = "https://snap.stanford.edu/ogb/data/linkproppred/citation2.zip"
    };

    // Dataset statistics for validation
    private static readonly Dictionary<string, (int nodes, int edges, int features, int classes)> DatasetStats = new()
    {
        ["ogbn-arxiv"] = (169343, 1166243, 128, 40),
        ["ogbn-products"] = (2449029, 61859140, 100, 47),
        ["ogbn-proteins"] = (132534, 39561252, 8, 112),
        ["ogbg-molhiv"] = (41127, 0, 9, 2),     // Multiple graphs
        ["ogbg-molpcba"] = (437929, 0, 9, 128), // Multiple graphs
        ["ogbg-ppa"] = (158100, 0, 58, 37),     // Multiple graphs
        ["ogbl-collab"] = (235868, 1285465, 128, 0),
        ["ogbl-ddi"] = (4267, 1334889, 0, 0),
        ["ogbl-citation2"] = (2927963, 30561187, 128, 0)
    };

    /// <summary>
    /// OGB task types.
    /// </summary>
    public enum OGBTask
    {
        /// <summary>Node-level prediction tasks (e.g., ogbn-*)</summary>
        NodePrediction,

        /// <summary>Link-level prediction tasks (e.g., ogbl-*)</summary>
        LinkPrediction,

        /// <summary>Graph-level prediction tasks (e.g., ogbg-*)</summary>
        GraphPrediction
    }

    /// <inheritdoc/>
    public override string Name => $"OGB({_datasetName})";

    /// <inheritdoc/>
    public override string Description => $"Open Graph Benchmark loader for {_datasetName}";

    /// <inheritdoc/>
    public override int NumClasses
    {
        get
        {
            EnsureLoaded();
            return _numClasses;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="OGBDatasetLoader{T}"/> class.
    /// </summary>
    /// <param name="datasetName">OGB dataset name (e.g., "ogbn-arxiv", "ogbg-molhiv").</param>
    /// <param name="taskType">Type of OGB task.</param>
    /// <param name="batchSize">Batch size for loading graphs (graph-level tasks only).</param>
    /// <param name="dataPath">Path to download/cache datasets (optional).</param>
    /// <param name="autoDownload">Whether to automatically download the dataset if not found locally.</param>
    /// <remarks>
    /// <para>
    /// Common OGB datasets:
    /// - Node: ogbn-arxiv, ogbn-products, ogbn-proteins, ogbn-papers100M
    /// - Link: ogbl-collab, ogbl-ddi, ogbl-citation2, ogbl-ppa
    /// - Graph: ogbg-molhiv, ogbg-molpcba, ogbg-ppa, ogbg-code2
    /// </para>
    /// <para><b>For Beginners:</b> Using OGB datasets:
    ///
    /// ```csharp
    /// // Load molecular HIV dataset
    /// var loader = new OGBDatasetLoader&lt;double&gt;(
    ///     "ogbg-molhiv",
    ///     OGBDatasetLoader&lt;double&gt;.OGBTask.GraphPrediction,
    ///     batchSize: 32,
    ///     autoDownload: true);
    ///
    /// // Load the data
    /// await loader.LoadAsync();
    ///
    /// // Get batches of graphs
    /// while (loader.HasNext)
    /// {
    ///     var batch = loader.GetNextBatch();
    ///     // Train on batch
    /// }
    ///
    /// // Or create task directly
    /// var task = loader.CreateGraphClassificationTask();
    /// ```
    /// </para>
    /// </remarks>
    public OGBDatasetLoader(
        string datasetName,
        OGBTask taskType,
        int batchSize = 32,
        string? dataPath = null,
        bool autoDownload = true)
    {
        if (string.IsNullOrWhiteSpace(datasetName))
        {
            throw new ArgumentNullException(nameof(datasetName));
        }

        _datasetName = datasetName.ToLowerInvariant();
        _taskType = taskType;
        BatchSize = batchSize;
        _dataPath = dataPath ?? GetDefaultDataPath();
        _autoDownload = autoDownload;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string datasetDir = Path.Combine(_dataPath, _datasetName);

        // Ensure data exists (download if needed)
        await EnsureDataExistsAsync(datasetDir, cancellationToken);

        // Parse the dataset files
        LoadedGraphs = await ParseDatasetAsync(datasetDir, cancellationToken);

        // Set first graph as the main LoadedGraphData for single-graph access
        if (LoadedGraphs.Count > 0)
        {
            LoadedGraphData = LoadedGraphs[0];
        }

        // Set number of classes
        if (DatasetStats.TryGetValue(_datasetName, out var stats))
        {
            _numClasses = stats.classes;
        }
        else
        {
            _numClasses = 2; // Default for binary classification
        }
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedGraphs = null;
        LoadedGraphData = null;
    }

    /// <summary>
    /// Ensures the dataset files exist locally, downloading if necessary.
    /// </summary>
    private async Task EnsureDataExistsAsync(string datasetDir, CancellationToken cancellationToken)
    {
        // Check if any data files exist
        if (Directory.Exists(datasetDir))
        {
            string[] npyFiles = Directory.GetFiles(datasetDir, "*.npy", SearchOption.AllDirectories);
            string[] csvFiles = Directory.GetFiles(datasetDir, "*.csv", SearchOption.AllDirectories);
            string[] txtFiles = Directory.GetFiles(datasetDir, "*.txt", SearchOption.AllDirectories);

            if (npyFiles.Length > 0 || csvFiles.Length > 0 || txtFiles.Length > 0)
            {
                return; // Data already exists
            }
        }

        if (!_autoDownload)
        {
            throw new FileNotFoundException(
                $"Dataset files not found at {datasetDir}. " +
                $"Either provide the data files or set autoDownload=true to download automatically.");
        }

        // Download the dataset
        await DownloadDatasetAsync(datasetDir, cancellationToken);
    }

    /// <summary>
    /// Downloads the dataset from the standard OGB source.
    /// </summary>
    private async Task DownloadDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(datasetDir);

        if (!DatasetUrls.TryGetValue(_datasetName, out string? url))
        {
            throw new NotSupportedException(
                $"Dataset '{_datasetName}' is not supported. " +
                $"Supported datasets: {string.Join(", ", DatasetUrls.Keys)}");
        }

        string tempFile = Path.Combine(Path.GetTempPath(), $"ogb_{_datasetName}_{Guid.NewGuid()}.zip");

        try
        {
            // Download the archive
            using (var httpClient = new HttpClient())
            {
                httpClient.Timeout = TimeSpan.FromMinutes(60); // Large datasets may take time

                using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                response.EnsureSuccessStatusCode();

                using var fileStream = new FileStream(tempFile, FileMode.Create, FileAccess.Write, FileShare.None);
                await response.Content.CopyToAsync(fileStream);
            }

            // Extract the archive (delete existing directory first for net471 compatibility)
            if (Directory.Exists(datasetDir))
            {
                Directory.Delete(datasetDir, recursive: true);
            }
            ZipFile.ExtractToDirectory(tempFile, datasetDir);
        }
        finally
        {
            // Clean up temp file
            if (File.Exists(tempFile))
            {
                try { File.Delete(tempFile); }
                catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Parses the OGB dataset files.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        return _taskType switch
        {
            OGBTask.NodePrediction => await ParseNodeDatasetAsync(datasetDir, cancellationToken),
            OGBTask.GraphPrediction => await ParseGraphDatasetAsync(datasetDir, cancellationToken),
            OGBTask.LinkPrediction => await ParseLinkDatasetAsync(datasetDir, cancellationToken),
            _ => throw new NotSupportedException($"Task type {_taskType} is not supported")
        };
    }

    /// <summary>
    /// Parses node-level prediction dataset.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseNodeDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        // Look for raw data files
        string rawDir = FindSubdirectory(datasetDir, "raw");
        if (string.IsNullOrEmpty(rawDir))
        {
            rawDir = datasetDir;
        }

        // Parse edge list file
        string edgeFile = FindFile(rawDir, "edge.csv", "edge_index.csv", "edges.csv");
        var edges = await ParseEdgeFileAsync(edgeFile, cancellationToken);

        // Parse node features
        string nodeFeatureFile = FindFile(rawDir, "node-feat.csv", "node_feat.csv", "features.csv");
        var nodeFeatures = await ParseNodeFeaturesAsync(nodeFeatureFile, cancellationToken);

        // Parse node labels
        string labelFile = FindFile(rawDir, "node-label.csv", "node_label.csv", "labels.csv");
        var labels = await ParseLabelsAsync(labelFile, cancellationToken);

        // Build the graph
        int numNodes = nodeFeatures.Count;
        int numFeatures = nodeFeatures.Count > 0 ? nodeFeatures[0].Length : DatasetStats.GetValueOrDefault(_datasetName).features;

        var nodeFeatureTensor = new Tensor<T>([numNodes, numFeatures]);
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numFeatures && j < nodeFeatures[i].Length; j++)
            {
                nodeFeatureTensor[i, j] = NumOps.FromDouble(nodeFeatures[i][j]);
            }
        }

        var edgeIndex = new Tensor<T>([edges.Count, 2]);
        for (int i = 0; i < edges.Count; i++)
        {
            edgeIndex[i, 0] = NumOps.FromDouble(edges[i].src);
            edgeIndex[i, 1] = NumOps.FromDouble(edges[i].dst);
        }

        var labelTensor = new Tensor<T>([numNodes, _numClasses]);
        for (int i = 0; i < labels.Count && i < numNodes; i++)
        {
            int labelIdx = MathPolyfill.Clamp(labels[i], 0, _numClasses - 1);
            labelTensor[i, labelIdx] = NumOps.One;
        }

        var graph = new GraphData<T>
        {
            NodeFeatures = nodeFeatureTensor,
            EdgeIndex = edgeIndex,
            NodeLabels = labelTensor
        };

        return new List<GraphData<T>> { graph };
    }

    /// <summary>
    /// Parses graph-level prediction dataset.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseGraphDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();

        // Look for raw data files
        string rawDir = FindSubdirectory(datasetDir, "raw");
        if (string.IsNullOrEmpty(rawDir))
        {
            rawDir = datasetDir;
        }

        // For molecular datasets, look for SMILES or SDF files
        string smilesFile = FindFile(rawDir, "smiles.csv", "smiles.txt");
        if (!string.IsNullOrEmpty(smilesFile))
        {
            graphs = await ParseSmilesFileAsync(smilesFile, cancellationToken);
        }
        else
        {
            // Try to parse individual graph files
            string edgeFile = FindFile(rawDir, "edge.csv", "edges.csv");
            string graphIdxFile = FindFile(rawDir, "graph_idx.csv", "batch.csv");

            if (!string.IsNullOrEmpty(edgeFile) && !string.IsNullOrEmpty(graphIdxFile))
            {
                graphs = await ParseBatchedGraphsAsync(edgeFile, graphIdxFile, rawDir, cancellationToken);
            }
            else
            {
                throw new FileNotFoundException(
                    $"Could not find graph data files in {rawDir}. " +
                    "Expected smiles.csv, edge.csv with graph_idx.csv, or SDF files.");
            }
        }

        return graphs;
    }

    /// <summary>
    /// Parses link prediction dataset.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseLinkDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        // Similar to node prediction but includes positive/negative edge splits
        var graphList = await ParseNodeDatasetAsync(datasetDir, cancellationToken);

        // Load split files if they exist
        string splitDir = FindSubdirectory(datasetDir, "split");
        if (!string.IsNullOrEmpty(splitDir))
        {
            // Store split information in metadata (not used directly in GraphData)
            // The splits would be used when creating LinkPredictionTask
        }

        return graphList;
    }

    /// <summary>
    /// Parses edge file in CSV format.
    /// </summary>
    private async Task<List<(int src, int dst)>> ParseEdgeFileAsync(string filePath, CancellationToken cancellationToken)
    {
        var edges = new List<(int, int)>();

        if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
        {
            return edges;
        }

        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);
        bool hasHeader = lines.Length > 0 && !int.TryParse(lines[0].Split(',')[0], out _);

        for (int i = hasHeader ? 1 : 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            if (parts.Length >= 2 &&
                int.TryParse(parts[0], out int src) &&
                int.TryParse(parts[1], out int dst))
            {
                edges.Add((src, dst));
            }
        }

        return edges;
    }

    /// <summary>
    /// Parses node features file in CSV format.
    /// </summary>
    private async Task<List<double[]>> ParseNodeFeaturesAsync(string filePath, CancellationToken cancellationToken)
    {
        var features = new List<double[]>();

        if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
        {
            return features;
        }

        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);
        bool hasHeader = lines.Length > 0 && !double.TryParse(lines[0].Split(',')[0], out _);

        for (int i = hasHeader ? 1 : 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            var featureVector = new double[parts.Length];
            for (int j = 0; j < parts.Length; j++)
            {
                double.TryParse(parts[j], out featureVector[j]);
            }
            features.Add(featureVector);
        }

        return features;
    }

    /// <summary>
    /// Parses labels file in CSV format.
    /// </summary>
    private async Task<List<int>> ParseLabelsAsync(string filePath, CancellationToken cancellationToken)
    {
        var labels = new List<int>();

        if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
        {
            return labels;
        }

        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);
        bool hasHeader = lines.Length > 0 && !int.TryParse(lines[0].Split(',')[0], out _);

        for (int i = hasHeader ? 1 : 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            if (parts.Length > 0 && int.TryParse(parts[0], out int label))
            {
                labels.Add(label);
            }
        }

        return labels;
    }

    /// <summary>
    /// Parses a SMILES file into molecular graphs.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseSmilesFileAsync(string filePath, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();
        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);

        // Find SMILES column
        int smilesColumn = 0;
        if (lines.Length > 0)
        {
            string[] headers = lines[0].Split(',');
            for (int i = 0; i < headers.Length; i++)
            {
                string header = headers[i].Trim().ToLowerInvariant();
                if (header == "smiles" || header == "mol")
                {
                    smilesColumn = i;
                    break;
                }
            }
        }

        bool hasHeader = lines.Length > 0 && lines[0].ToLowerInvariant().Contains("smiles");

        for (int i = hasHeader ? 1 : 0; i < lines.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            if (parts.Length > smilesColumn)
            {
                string smiles = parts[smilesColumn].Trim().Trim('"');
                var graph = ParseSMILES(smiles);
                if (graph is not null)
                {
                    graphs.Add(graph);
                }
            }
        }

        return graphs;
    }

    /// <summary>
    /// Parses batched graphs from edge file and graph index file.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseBatchedGraphsAsync(
        string edgeFile,
        string graphIdxFile,
        string rawDir,
        CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();

        // Parse all edges
        var allEdges = await ParseEdgeFileAsync(edgeFile, cancellationToken);

        // Parse graph indices (which graph each node belongs to)
        string[] graphIdxLines = await FilePolyfill.ReadAllLinesAsync(graphIdxFile, cancellationToken);
        var nodeToGraph = new List<int>();
        bool hasHeader = graphIdxLines.Length > 0 && !int.TryParse(graphIdxLines[0].Split(',')[0], out _);

        for (int i = hasHeader ? 1 : 0; i < graphIdxLines.Length; i++)
        {
            if (int.TryParse(graphIdxLines[i].Split(',')[0], out int graphIdx))
            {
                nodeToGraph.Add(graphIdx);
            }
        }

        if (nodeToGraph.Count == 0)
        {
            return graphs;
        }

        // Parse node features if available
        string nodeFeatureFile = FindFile(rawDir, "node-feat.csv", "node_feat.csv");
        var allNodeFeatures = await ParseNodeFeaturesAsync(nodeFeatureFile, cancellationToken);

        // Parse graph labels if available
        string labelFile = FindFile(rawDir, "graph-label.csv", "graph_label.csv");
        var graphLabels = await ParseLabelsAsync(labelFile, cancellationToken);

        // Group nodes and edges by graph
        int numGraphs = nodeToGraph.Max() + 1;
        var graphNodeRanges = new List<(int start, int count)>();
        int currentStart = 0;
        int currentGraph = 0;

        for (int i = 0; i < nodeToGraph.Count; i++)
        {
            if (nodeToGraph[i] != currentGraph)
            {
                graphNodeRanges.Add((currentStart, i - currentStart));
                currentStart = i;
                currentGraph = nodeToGraph[i];
            }
        }
        graphNodeRanges.Add((currentStart, nodeToGraph.Count - currentStart));

        // Build each graph
        for (int g = 0; g < graphNodeRanges.Count; g++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (nodeStart, nodeCount) = graphNodeRanges[g];

            // Get edges for this graph
            var graphEdges = allEdges
                .Where(e => e.src >= nodeStart && e.src < nodeStart + nodeCount &&
                           e.dst >= nodeStart && e.dst < nodeStart + nodeCount)
                .Select(e => (e.src - nodeStart, e.dst - nodeStart))
                .ToList();

            // Build node features
            int numFeatures = allNodeFeatures.Count > nodeStart ? allNodeFeatures[nodeStart].Length : 9;
            var nodeFeatures = new Tensor<T>([nodeCount, numFeatures]);

            for (int i = 0; i < nodeCount && nodeStart + i < allNodeFeatures.Count; i++)
            {
                var features = allNodeFeatures[nodeStart + i];
                for (int j = 0; j < numFeatures && j < features.Length; j++)
                {
                    nodeFeatures[i, j] = NumOps.FromDouble(features[j]);
                }
            }

            // Build edge index
            var edgeIndex = new Tensor<T>([graphEdges.Count, 2]);
            for (int i = 0; i < graphEdges.Count; i++)
            {
                edgeIndex[i, 0] = NumOps.FromDouble(graphEdges[i].Item1);
                edgeIndex[i, 1] = NumOps.FromDouble(graphEdges[i].Item2);
            }

            // Build graph label
            Tensor<T>? graphLabel = null;
            if (g < graphLabels.Count)
            {
                graphLabel = new Tensor<T>([1, _numClasses]);
                int labelIdx = MathPolyfill.Clamp(graphLabels[g], 0, _numClasses - 1);
                graphLabel[0, labelIdx] = NumOps.One;
            }

            graphs.Add(new GraphData<T>
            {
                NodeFeatures = nodeFeatures,
                EdgeIndex = edgeIndex,
                GraphLabel = graphLabel
            });
        }

        return graphs;
    }

    /// <summary>
    /// Simple SMILES parser for molecular graphs.
    /// </summary>
    private GraphData<T>? ParseSMILES(string smiles)
    {
        if (string.IsNullOrWhiteSpace(smiles))
        {
            return null;
        }

        // Atom type mapping
        var atomTypeMap = new Dictionary<string, int>
        {
            ["H"] = 0,
            ["C"] = 1,
            ["N"] = 2,
            ["O"] = 3,
            ["F"] = 4,
            ["P"] = 5,
            ["S"] = 6,
            ["Cl"] = 7,
            ["Br"] = 8,
            ["I"] = 9
        };

        var atomTypes = new List<int>();
        var bonds = new List<(int src, int dst, int bondType)>();
        var ringStack = new Dictionary<int, int>();
        var branchStack = new Stack<int>();

        int currentAtom = -1;
        int currentBondType = 0;

        int i = 0;
        while (i < smiles.Length)
        {
            char c = smiles[i];

            if (c == '/' || c == '\\' || c == '@')
            {
                i++;
                continue;
            }

            if (c == '[')
            {
                int end = smiles.IndexOf(']', i);
                if (end > i)
                {
                    string atomBlock = smiles.Substring(i + 1, end - i - 1);
                    string symbol = "";
                    foreach (char ch in atomBlock)
                    {
                        if (char.IsLetter(ch))
                        {
                            symbol += ch;
                            if (symbol.Length == 2) break;
                        }
                        else break;
                    }
                    symbol = symbol.Length > 0 ? char.ToUpper(symbol[0]) + (symbol.Length > 1 ? symbol.Substring(1).ToLower() : "") : "C";
                    int newAtom = atomTypes.Count;
                    atomTypes.Add(atomTypeMap.GetValueOrDefault(symbol, 0));

                    if (currentAtom >= 0)
                    {
                        bonds.Add((currentAtom, newAtom, currentBondType));
                    }
                    currentAtom = newAtom;
                    currentBondType = 0;
                    i = end + 1;
                    continue;
                }
            }

            if (char.IsLetter(c))
            {
                string symbol = c.ToString();
                if (i + 1 < smiles.Length && char.IsLower(smiles[i + 1]))
                {
                    symbol += smiles[i + 1];
                    i++;
                }

                int newAtom = atomTypes.Count;
                atomTypes.Add(atomTypeMap.GetValueOrDefault(symbol, 0));

                if (currentAtom >= 0)
                {
                    bonds.Add((currentAtom, newAtom, currentBondType));
                }
                currentAtom = newAtom;
                currentBondType = 0;
            }
            else if (c == '=')
            {
                currentBondType = 1;
            }
            else if (c == '#')
            {
                currentBondType = 2;
            }
            else if (c == '(')
            {
                branchStack.Push(currentAtom);
            }
            else if (c == ')')
            {
                if (branchStack.Count > 0)
                {
                    currentAtom = branchStack.Pop();
                }
            }
            else if (char.IsDigit(c))
            {
                int ringNum = c - '0';
                if (ringStack.TryGetValue(ringNum, out int ringAtom))
                {
                    bonds.Add((currentAtom, ringAtom, currentBondType));
                    ringStack.Remove(ringNum);
                }
                else
                {
                    ringStack[ringNum] = currentAtom;
                }
                currentBondType = 0;
            }

            i++;
        }

        if (atomTypes.Count == 0)
        {
            return null;
        }

        // Build node features (one-hot atom type)
        int numAtomTypes = 10;
        var nodeFeatures = new Tensor<T>([atomTypes.Count, numAtomTypes]);
        for (int j = 0; j < atomTypes.Count; j++)
        {
            int atomType = MathPolyfill.Clamp(atomTypes[j], 0, numAtomTypes - 1);
            nodeFeatures[j, atomType] = NumOps.One;
        }

        // Build edge index (undirected)
        var edgeList = new List<(int, int)>();
        foreach (var bond in bonds)
        {
            edgeList.Add((bond.src, bond.dst));
            edgeList.Add((bond.dst, bond.src));
        }

        var edgeIndex = new Tensor<T>([edgeList.Count, 2]);
        for (int j = 0; j < edgeList.Count; j++)
        {
            edgeIndex[j, 0] = NumOps.FromDouble(edgeList[j].Item1);
            edgeIndex[j, 1] = NumOps.FromDouble(edgeList[j].Item2);
        }

        return new GraphData<T>
        {
            NodeFeatures = nodeFeatures,
            EdgeIndex = edgeIndex
        };
    }

    /// <summary>
    /// Finds a subdirectory with the given name.
    /// </summary>
    private string FindSubdirectory(string parentDir, string subDirName)
    {
        if (!Directory.Exists(parentDir))
        {
            return string.Empty;
        }

        // Direct match
        string direct = Path.Combine(parentDir, subDirName);
        if (Directory.Exists(direct))
        {
            return direct;
        }

        // Search recursively
        string[] dirs = Directory.GetDirectories(parentDir, subDirName, SearchOption.AllDirectories);
        return dirs.Length > 0 ? dirs[0] : string.Empty;
    }

    /// <summary>
    /// Finds a file with one of the given names.
    /// </summary>
    private string FindFile(string directory, params string[] fileNames)
    {
        if (string.IsNullOrEmpty(directory) || !Directory.Exists(directory))
        {
            return string.Empty;
        }

        foreach (string fileName in fileNames)
        {
            string[] files = Directory.GetFiles(directory, fileName, SearchOption.AllDirectories);
            if (files.Length > 0)
            {
                return files[0];
            }
        }

        return string.Empty;
    }

    /// <summary>
    /// Gets the default data cache path.
    /// </summary>
    private string GetDefaultDataPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".aidotnet",
            "datasets",
            "ogb");
    }

    /// <inheritdoc/>
    public override NodeClassificationTask<T> CreateNodeClassificationTask(
        double trainRatio = 0.1,
        double valRatio = 0.1,
        int? seed = null)
    {
        if (_taskType != OGBTask.NodePrediction)
        {
            throw new InvalidOperationException(
                $"CreateNodeClassificationTask requires NodePrediction task type, got {_taskType}");
        }

        EnsureLoaded();

        if (LoadedGraphs is null || LoadedGraphs.Count == 0)
        {
            throw new InvalidOperationException("No graphs loaded");
        }

        var graph = LoadedGraphs[0];
        int numNodes = graph.NumNodes;

        // Create random split
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, numNodes).OrderBy(_ => random.Next()).ToArray();
        int trainSize = (int)(numNodes * trainRatio);
        int valSize = (int)(numNodes * valRatio);

        var trainIndices = indices.Take(trainSize).ToArray();
        var valIndices = indices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = indices.Skip(trainSize + valSize).ToArray();

        return new NodeClassificationTask<T>
        {
            Graph = graph,
            Labels = graph.NodeLabels ?? new Tensor<T>([numNodes, _numClasses]),
            TrainIndices = trainIndices,
            ValIndices = valIndices,
            TestIndices = testIndices,
            NumClasses = _numClasses,
            IsMultiLabel = _datasetName == "ogbn-proteins"
        };
    }

    /// <inheritdoc/>
    public override GraphClassificationTask<T> CreateGraphClassificationTask(
        double trainRatio = 0.8,
        double valRatio = 0.1,
        int? seed = null)
    {
        if (_taskType != OGBTask.GraphPrediction)
        {
            throw new InvalidOperationException(
                $"CreateGraphClassificationTask requires GraphPrediction task type, got {_taskType}");
        }

        EnsureLoaded();

        if (LoadedGraphs is null || LoadedGraphs.Count == 0)
        {
            throw new InvalidOperationException("No graphs loaded");
        }

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffledGraphs = LoadedGraphs.OrderBy(_ => random.Next()).ToList();

        int trainSize = (int)(shuffledGraphs.Count * trainRatio);
        int valSize = (int)(shuffledGraphs.Count * valRatio);

        var trainGraphs = shuffledGraphs.Take(trainSize).ToList();
        var valGraphs = shuffledGraphs.Skip(trainSize).Take(valSize).ToList();
        var testGraphs = shuffledGraphs.Skip(trainSize + valSize).ToList();

        // Collect labels from graphs
        var trainLabels = CollectGraphLabels(trainGraphs, _numClasses);
        var valLabels = CollectGraphLabels(valGraphs, _numClasses);
        var testLabels = CollectGraphLabels(testGraphs, _numClasses);

        bool isRegression = _datasetName.Contains("qm");

        return new GraphClassificationTask<T>
        {
            TrainGraphs = trainGraphs,
            ValGraphs = valGraphs,
            TestGraphs = testGraphs,
            TrainLabels = trainLabels,
            ValLabels = valLabels,
            TestLabels = testLabels,
            NumClasses = _numClasses,
            IsMultiLabel = _datasetName == "ogbg-molpcba",
            IsRegression = isRegression,
            AvgNumNodes = trainGraphs.Average(g => g.NumNodes),
            AvgNumEdges = trainGraphs.Average(g => g.NumEdges)
        };
    }

    /// <inheritdoc/>
    public override LinkPredictionTask<T> CreateLinkPredictionTask(
        double trainRatio = 0.85,
        double negativeRatio = 1.0,
        int? seed = null)
    {
        if (_taskType != OGBTask.LinkPrediction)
        {
            throw new InvalidOperationException(
                $"CreateLinkPredictionTask requires LinkPrediction task type, got {_taskType}");
        }

        EnsureLoaded();

        if (LoadedGraphs is null || LoadedGraphs.Count == 0)
        {
            throw new InvalidOperationException("No graphs loaded");
        }

        var graph = LoadedGraphs[0];
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Get all edges from edge index
        var allEdges = new List<(int src, int dst)>();
        var existingEdgeSet = new HashSet<(int, int)>();

        for (int i = 0; i < graph.EdgeIndex.Shape[0]; i++)
        {
            int src = NumOps.ToInt32(graph.EdgeIndex[i, 0]);
            int dst = NumOps.ToInt32(graph.EdgeIndex[i, 1]);

            if (src < dst)
            {
                allEdges.Add((src, dst));
                existingEdgeSet.Add((src, dst));
            }
        }

        // Shuffle and split edges
        var shuffledEdges = allEdges.OrderBy(_ => random.Next()).ToArray();
        int trainSize = (int)(shuffledEdges.Length * trainRatio);
        int valSize = (int)(shuffledEdges.Length * (1 - trainRatio) / 2);

        var trainEdges = shuffledEdges.Take(trainSize).ToArray();
        var valEdges = shuffledEdges.Skip(trainSize).Take(valSize).ToArray();
        var testEdges = shuffledEdges.Skip(trainSize + valSize).ToArray();

        // Generate negative edges
        var negativeEdges = new List<(int, int)>();
        int numNegative = (int)(allEdges.Count * negativeRatio);

        while (negativeEdges.Count < numNegative)
        {
            int src = random.Next(graph.NumNodes);
            int dst = random.Next(graph.NumNodes);

            if (src > dst)
            {
                (src, dst) = (dst, src);
            }

            if (src != dst && !existingEdgeSet.Contains((src, dst)))
            {
                negativeEdges.Add((src, dst));
                existingEdgeSet.Add((src, dst));
            }
        }

        var shuffledNegative = negativeEdges.OrderBy(_ => random.Next()).ToArray();
        int negTrainSize = (int)(shuffledNegative.Length * trainRatio);
        int negValSize = (int)(shuffledNegative.Length * (1 - trainRatio) / 2);

        return new LinkPredictionTask<T>
        {
            Graph = graph,
            TrainPosEdges = ConvertEdgesToTensor(trainEdges),
            TrainNegEdges = ConvertEdgesToTensor(shuffledNegative.Take(negTrainSize).ToArray()),
            ValPosEdges = ConvertEdgesToTensor(valEdges),
            ValNegEdges = ConvertEdgesToTensor(shuffledNegative.Skip(negTrainSize).Take(negValSize).ToArray()),
            TestPosEdges = ConvertEdgesToTensor(testEdges),
            TestNegEdges = ConvertEdgesToTensor(shuffledNegative.Skip(negTrainSize + negValSize).ToArray()),
            NegativeSamplingRatio = negativeRatio
        };
    }

    /// <summary>
    /// Converts an array of edge tuples to a tensor of shape [num_edges, 2].
    /// </summary>
    private Tensor<T> ConvertEdgesToTensor((int src, int dst)[] edges)
    {
        var tensor = new Tensor<T>([edges.Length, 2]);
        for (int i = 0; i < edges.Length; i++)
        {
            tensor[i, 0] = NumOps.FromDouble(edges[i].src);
            tensor[i, 1] = NumOps.FromDouble(edges[i].dst);
        }
        return tensor;
    }

    /// <summary>
    /// Collects graph labels into a single tensor.
    /// </summary>
    private Tensor<T> CollectGraphLabels(List<GraphData<T>> graphs, int numClasses)
    {
        var labels = new Tensor<T>([graphs.Count, numClasses]);
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < graphs.Count; i++)
        {
            if (graphs[i].GraphLabel is not null && graphs[i].GraphLabel!.Shape[1] >= numClasses)
            {
                for (int j = 0; j < numClasses; j++)
                {
                    labels[i, j] = graphs[i].GraphLabel![0, j];
                }
            }
            else
            {
                // Default label if not available
                int classIdx = random.Next(numClasses);
                labels[i, classIdx] = NumOps.One;
            }
        }

        return labels;
    }
}
