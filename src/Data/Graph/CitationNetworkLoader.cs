using System.IO.Compression;
using System.Net.Http;
using AiDotNet.Data.Loaders;
using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

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
public class CitationNetworkLoader<T> : GraphDataLoaderBase<T>
{
    private readonly CitationDataset _dataset;
    private readonly string _dataPath;
    private readonly bool _autoDownload;
    private int _numClasses;

    // Standard download URLs for citation network datasets
    private static readonly Dictionary<CitationDataset, string> DatasetUrls = new()
    {
        [CitationDataset.Cora] = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        [CitationDataset.CiteSeer] = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
        [CitationDataset.PubMed] = "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"
    };

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
    public override string Name => $"CitationNetwork({_dataset})";

    /// <inheritdoc/>
    public override string Description => $"Citation network loader for {_dataset} dataset";

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
    /// Initializes a new instance of the <see cref="CitationNetworkLoader{T}"/> class.
    /// </summary>
    /// <param name="dataset">Which citation dataset to load.</param>
    /// <param name="dataPath">Path to the dataset files. If null, uses default cache directory.</param>
    /// <param name="autoDownload">Whether to automatically download the dataset if not found locally.</param>
    /// <remarks>
    /// <para>
    /// The loader expects data files in the standard Planetoid format:
    /// - {dataset}.content: Tab-separated file with paper_id, word features, class_label
    /// - {dataset}.cites: Tab-separated file with cited_paper_id, citing_paper_id
    /// </para>
    /// <para><b>For Beginners:</b> Using this loader:
    ///
    /// ```csharp
    /// // Load Cora dataset (auto-downloads if not present)
    /// var loader = new CitationNetworkLoader&lt;double&gt;(
    ///     CitationNetworkLoader&lt;double&gt;.CitationDataset.Cora,
    ///     autoDownload: true);
    ///
    /// // Load the data
    /// await loader.LoadAsync();
    ///
    /// // Get the graph
    /// var graph = loader.GetNextBatch();
    ///
    /// // Access data
    /// Console.WriteLine($"Nodes: {loader.NumNodes}");
    /// Console.WriteLine($"Edges: {loader.NumEdges}");
    /// Console.WriteLine($"Features per node: {loader.NumNodeFeatures}");
    ///
    /// // Create node classification task
    /// var task = loader.CreateNodeClassificationTask();
    /// ```
    /// </para>
    /// </remarks>
    public CitationNetworkLoader(CitationDataset dataset, string? dataPath = null, bool autoDownload = true)
    {
        _dataset = dataset;
        _dataPath = dataPath ?? GetDefaultDataPath();
        _autoDownload = autoDownload;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string datasetDir = Path.Combine(_dataPath, _dataset.ToString().ToLowerInvariant());

        // Ensure data exists (download if needed)
        await EnsureDataExistsAsync(datasetDir, cancellationToken);

        // Parse the dataset files
        LoadedGraphData = await ParseDatasetFilesAsync(datasetDir, cancellationToken);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedGraphData = null;
    }

    /// <summary>
    /// Ensures the dataset files exist locally, downloading if necessary.
    /// </summary>
    private async Task EnsureDataExistsAsync(string datasetDir, CancellationToken cancellationToken)
    {
        string contentFile = GetContentFilePath(datasetDir);
        string citesFile = GetCitesFilePath(datasetDir);

        if (File.Exists(contentFile) && File.Exists(citesFile))
        {
            return; // Data already exists
        }

        if (!_autoDownload)
        {
            throw new FileNotFoundException(
                $"Dataset files not found at {datasetDir}. " +
                $"Either provide the data files or set autoDownload=true to download automatically.");
        }

        // Download and extract the dataset
        await DownloadAndExtractDatasetAsync(datasetDir, cancellationToken);

        // Verify files exist after extraction
        if (!File.Exists(contentFile))
        {
            throw new InvalidOperationException(
                $"Failed to find {Path.GetFileName(contentFile)} after extraction. " +
                "The dataset format may have changed.");
        }

        if (!File.Exists(citesFile))
        {
            throw new InvalidOperationException(
                $"Failed to find {Path.GetFileName(citesFile)} after extraction. " +
                "The dataset format may have changed.");
        }
    }

    /// <summary>
    /// Downloads and extracts the dataset from the standard source.
    /// </summary>
    private async Task DownloadAndExtractDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(datasetDir);

        string url = DatasetUrls[_dataset];
        string tempFile = Path.Combine(Path.GetTempPath(), $"{_dataset}_{Guid.NewGuid()}.tgz");

        try
        {
            // Download the archive
            using (var httpClient = new HttpClient())
            {
                httpClient.Timeout = TimeSpan.FromMinutes(10);

                using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                response.EnsureSuccessStatusCode();

                using var fileStream = new FileStream(tempFile, FileMode.Create, FileAccess.Write, FileShare.None);
                await response.Content.CopyToAsync(fileStream);
            }

            // Extract the archive
            await ExtractTarGzAsync(tempFile, datasetDir, cancellationToken);
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
    /// Extracts a .tar.gz archive to the specified directory.
    /// </summary>
    private async Task ExtractTarGzAsync(string archivePath, string destinationDir, CancellationToken cancellationToken)
    {
        using var fileStream = new FileStream(archivePath, FileMode.Open, FileAccess.Read);
        using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);

        // Read tar entries
        await ExtractTarAsync(gzipStream, destinationDir, cancellationToken);
    }

    /// <summary>
    /// Extracts a tar archive from a stream.
    /// </summary>
    private async Task ExtractTarAsync(Stream tarStream, string destinationDir, CancellationToken cancellationToken)
    {
        byte[] buffer = new byte[512];
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Read header
            int bytesRead = await ReadExactAsync(tarStream, buffer, 0, 512, cancellationToken);
            if (bytesRead == 0 || buffer.All(b => b == 0))
            {
                break; // End of archive
            }

            // Parse header
            string fileName = System.Text.Encoding.ASCII.GetString(buffer, 0, 100).TrimEnd('\0');
            if (string.IsNullOrWhiteSpace(fileName))
            {
                break;
            }

            // Parse file size (octal, bytes 124-135)
            string sizeStr = System.Text.Encoding.ASCII.GetString(buffer, 124, 11).TrimEnd('\0', ' ');
            long fileSize = string.IsNullOrEmpty(sizeStr) ? 0 : Convert.ToInt64(sizeStr, 8);

            // Check if directory (type flag at byte 156)
            char typeFlag = (char)buffer[156];
            bool isDirectory = typeFlag == '5' || fileName.EndsWith("/");

            // Normalize path
            string destPath = Path.Combine(destinationDir, fileName.Replace('/', Path.DirectorySeparatorChar));

            if (isDirectory)
            {
                Directory.CreateDirectory(destPath);
            }
            else if (fileSize > 0)
            {
                // Ensure parent directory exists
                string? parentDir = Path.GetDirectoryName(destPath);
                if (parentDir is not null)
                {
                    Directory.CreateDirectory(parentDir);
                }

                // Extract file
                using var outputStream = new FileStream(destPath, FileMode.Create, FileAccess.Write);
                long remaining = fileSize;
                byte[] fileBuffer = new byte[Math.Min(65536, fileSize)];

                while (remaining > 0)
                {
                    int toRead = (int)Math.Min(fileBuffer.Length, remaining);
                    bytesRead = await ReadExactAsync(tarStream, fileBuffer, 0, toRead, cancellationToken);
                    if (bytesRead == 0)
                    {
                        throw new InvalidOperationException("Unexpected end of tar stream");
                    }
                    await outputStream.WriteAsync(fileBuffer, 0, bytesRead, cancellationToken);
                    remaining -= bytesRead;
                }
            }

            // Skip padding to 512-byte boundary
            int padding = (int)((512 - (fileSize % 512)) % 512);
            if (padding > 0)
            {
                byte[] paddingBuffer = new byte[padding];
                await ReadExactAsync(tarStream, paddingBuffer, 0, padding, cancellationToken);
            }
        }
    }

    private static async Task<int> ReadExactAsync(Stream stream, byte[] buffer, int offset, int count, CancellationToken cancellationToken)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int bytesRead = await stream.ReadAsync(buffer, offset + totalRead, count - totalRead, cancellationToken);
            if (bytesRead == 0)
            {
                break;
            }
            totalRead += bytesRead;
        }
        return totalRead;
    }

    /// <summary>
    /// Parses the dataset files and builds the graph structure.
    /// </summary>
    private async Task<GraphData<T>> ParseDatasetFilesAsync(string datasetDir, CancellationToken cancellationToken)
    {
        string contentFile = GetContentFilePath(datasetDir);
        string citesFile = GetCitesFilePath(datasetDir);

        // Parse content file to get node features, labels, and paper ID mapping
        var (paperIdToIndex, features, labels, classLabels) = await ParseContentFileAsync(contentFile, cancellationToken);

        // Parse cites file to get edges
        var edges = await ParseCitesFileAsync(citesFile, paperIdToIndex, cancellationToken);

        int numNodes = features.Count;
        int numFeatures = features[0].Length;
        _numClasses = classLabels.Count;

        // Build node features tensor
        var nodeFeatures = new Tensor<T>([numNodes, numFeatures]);
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                nodeFeatures[i, j] = NumOps.FromDouble(features[i][j]);
            }
        }

        // Build edge index tensor (COO format)
        var edgeIndex = new Tensor<T>([edges.Count, 2]);
        for (int i = 0; i < edges.Count; i++)
        {
            edgeIndex[i, 0] = NumOps.FromDouble(edges[i].source);
            edgeIndex[i, 1] = NumOps.FromDouble(edges[i].target);
        }

        // Build adjacency matrix
        var adjacencyMatrix = new Tensor<T>([1, numNodes, numNodes]);
        foreach (var edge in edges)
        {
            adjacencyMatrix[0, edge.source, edge.target] = NumOps.One;
        }

        // Build one-hot encoded labels
        var nodeLabels = new Tensor<T>([numNodes, _numClasses]);
        for (int i = 0; i < numNodes; i++)
        {
            nodeLabels[i, labels[i]] = NumOps.One;
        }

        return new GraphData<T>
        {
            NodeFeatures = nodeFeatures,
            AdjacencyMatrix = adjacencyMatrix,
            EdgeIndex = edgeIndex,
            NodeLabels = nodeLabels
        };
    }

    /// <summary>
    /// Parses the content file to extract node features and labels.
    /// </summary>
    /// <returns>Tuple of (paper ID to index mapping, feature vectors, label indices, unique class labels).</returns>
    private async Task<(Dictionary<string, int> paperIdToIndex, List<double[]> features, List<int> labels, List<string> classLabels)>
        ParseContentFileAsync(string contentFile, CancellationToken cancellationToken)
    {
        var paperIdToIndex = new Dictionary<string, int>();
        var features = new List<double[]>();
        var labels = new List<int>();
        var classLabelToIndex = new Dictionary<string, int>();
        var classLabels = new List<string>();

        string[] lines = await FilePolyfill.ReadAllLinesAsync(contentFile, cancellationToken);

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split('\t');
            if (parts.Length < 3)
            {
                // Try space/comma separation
                parts = line.Split(new[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);
            }

            if (parts.Length < 3)
            {
                continue; // Skip malformed lines
            }

            string paperId = parts[0];
            string classLabel = parts[^1]; // Last element is the class label

            // Features are everything between paper ID and class label
            var featureValues = new double[parts.Length - 2];
            for (int i = 1; i < parts.Length - 1; i++)
            {
                if (double.TryParse(parts[i], out double value))
                {
                    featureValues[i - 1] = value;
                }
            }

            // Map paper ID to index
            int nodeIndex = paperIdToIndex.Count;
            paperIdToIndex[paperId] = nodeIndex;

            // Map class label to index
            if (!classLabelToIndex.TryGetValue(classLabel, out int labelIndex))
            {
                labelIndex = classLabelToIndex.Count;
                classLabelToIndex[classLabel] = labelIndex;
                classLabels.Add(classLabel);
            }

            features.Add(featureValues);
            labels.Add(labelIndex);
        }

        return (paperIdToIndex, features, labels, classLabels);
    }

    /// <summary>
    /// Parses the cites file to extract edges.
    /// </summary>
    private async Task<List<(int source, int target)>> ParseCitesFileAsync(
        string citesFile,
        Dictionary<string, int> paperIdToIndex,
        CancellationToken cancellationToken)
    {
        var edges = new List<(int source, int target)>();

        string[] lines = await FilePolyfill.ReadAllLinesAsync(citesFile, cancellationToken);

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(new[] { '\t', ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
            {
                continue; // Skip malformed lines
            }

            string citedPaperId = parts[0];
            string citingPaperId = parts[1];

            // Only add edge if both papers exist in our node set
            if (paperIdToIndex.TryGetValue(citedPaperId, out int citedIndex) &&
                paperIdToIndex.TryGetValue(citingPaperId, out int citingIndex))
            {
                // Add edge in both directions (undirected graph)
                edges.Add((citingIndex, citedIndex));
                edges.Add((citedIndex, citingIndex));
            }
        }

        return edges;
    }

    /// <summary>
    /// Gets the path to the content file for the current dataset.
    /// </summary>
    private string GetContentFilePath(string datasetDir)
    {
        string datasetName = _dataset.ToString().ToLowerInvariant();

        // Try different possible locations and naming conventions
        string[] possiblePaths = _dataset switch
        {
            CitationDataset.PubMed => new[]
            {
                Path.Combine(datasetDir, "Pubmed-Diabetes", "data", "Pubmed-Diabetes.NODE.paper.tab"),
                Path.Combine(datasetDir, "data", "Pubmed-Diabetes.NODE.paper.tab"),
                Path.Combine(datasetDir, "Pubmed-Diabetes.content"),
                Path.Combine(datasetDir, "pubmed.content")
            },
            _ => new[]
            {
                Path.Combine(datasetDir, datasetName, $"{datasetName}.content"),
                Path.Combine(datasetDir, $"{datasetName}.content")
            }
        };

        foreach (string path in possiblePaths)
        {
            if (File.Exists(path))
            {
                return path;
            }
        }

        return possiblePaths[0]; // Return first option as default for error messages
    }

    /// <summary>
    /// Gets the path to the cites file for the current dataset.
    /// </summary>
    private string GetCitesFilePath(string datasetDir)
    {
        string datasetName = _dataset.ToString().ToLowerInvariant();

        // Try different possible locations and naming conventions
        string[] possiblePaths = _dataset switch
        {
            CitationDataset.PubMed => new[]
            {
                Path.Combine(datasetDir, "Pubmed-Diabetes", "data", "Pubmed-Diabetes.DIRECTED.cites.tab"),
                Path.Combine(datasetDir, "data", "Pubmed-Diabetes.DIRECTED.cites.tab"),
                Path.Combine(datasetDir, "Pubmed-Diabetes.cites"),
                Path.Combine(datasetDir, "pubmed.cites")
            },
            _ => new[]
            {
                Path.Combine(datasetDir, datasetName, $"{datasetName}.cites"),
                Path.Combine(datasetDir, $"{datasetName}.cites")
            }
        };

        foreach (string path in possiblePaths)
        {
            if (File.Exists(path))
            {
                return path;
            }
        }

        return possiblePaths[0]; // Return first option as default for error messages
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
            "citation_networks");
    }

    /// <inheritdoc/>
    public override GraphClassificationTask<T> CreateGraphClassificationTask(
        double trainRatio = 0.8,
        double valRatio = 0.1,
        int? seed = null)
    {
        // Citation networks are single-graph datasets used for node classification
        // Graph classification doesn't make sense here
        throw new NotSupportedException(
            "Citation networks are single-graph datasets designed for node classification, " +
            "not graph classification. Use CreateNodeClassificationTask() instead.");
    }

    /// <inheritdoc/>
    public override LinkPredictionTask<T> CreateLinkPredictionTask(
        double trainRatio = 0.85,
        double negativeRatio = 1.0,
        int? seed = null)
    {
        EnsureLoaded();

        if (LoadedGraphData == null)
        {
            throw new InvalidOperationException("Graph data not loaded.");
        }

        var graph = LoadedGraphData;
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

            // Store unique edges (avoid counting both directions)
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

        // Generate negative edges (non-existing edges)
        var negativeEdges = new List<(int, int)>();
        int numNegative = (int)(allEdges.Count * negativeRatio);

        while (negativeEdges.Count < numNegative)
        {
            int src = random.Next(graph.NumNodes);
            int dst = random.Next(graph.NumNodes);

            // Ensure src < dst for consistency and avoid self-loops
            if (src > dst)
            {
                (src, dst) = (dst, src);
            }

            if (src != dst && !existingEdgeSet.Contains((src, dst)))
            {
                negativeEdges.Add((src, dst));
                existingEdgeSet.Add((src, dst)); // Prevent duplicates
            }
        }

        // Split negative edges
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
}
