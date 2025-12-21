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
/// Loads molecular graph datasets (ZINC, QM9) for graph-level property prediction and generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Molecular datasets represent molecules as graphs where atoms are nodes and chemical bonds are edges.
/// These datasets are fundamental benchmarks for graph neural networks in drug discovery and
/// materials science.
/// </para>
/// <para><b>For Beginners:</b> Molecular graphs represent chemistry as networks.
///
/// **Graph Representation of Molecules:**
/// ```
/// Water (H₂O):
/// - Nodes: 3 atoms (O, H, H)
/// - Edges: 2 bonds (O-H, O-H)
/// - Node features: Atom type, charge, hybridization
/// - Edge features: Bond type (single, double, triple)
/// ```
///
/// **Why model molecules as graphs?**
/// - **Structure matters**: Same atoms, different arrangement = different properties
///   * Example: Diamond vs Graphite (both pure carbon!)
/// - **Bonds are relationships**: Like social networks, but for atoms
/// - **GNNs excel**: Message passing mimics electron delocalization
///
/// **Major Molecular Datasets:**
///
/// **ZINC:**
/// - **Size**: 250,000 drug-like molecules (subset: 12,000)
/// - **Source**: ZINC database (commercially available compounds)
/// - **Tasks**: Graph regression on constrained solubility
/// - **Features**:
///   * Atoms: C, N, O, F, P, S, Cl, Br, I (28 atom types)
///   * Bonds: Single, double, triple, aromatic
/// - **Use case**: Drug discovery, molecular generation
///
/// **QM9:**
/// - **Size**: 134,000 small organic molecules
/// - **Source**: Quantum mechanical calculations
/// - **Tasks**: Regression on 19 quantum properties
///   * Energy, enthalpy, heat capacity
///   * HOMO/LUMO gap (electronic properties)
///   * Dipole moment, polarizability
/// - **Atoms**: C, H, N, O, F (up to 9 heavy atoms)
/// - **Use case**: Property prediction, molecular design
/// </para>
/// </remarks>
public class MolecularDatasetLoader<T> : GraphDataLoaderBase<T>
{
    private readonly MolecularDataset _dataset;
    private readonly string _dataPath;
    private readonly bool _autoDownload;
    private int _numClasses;

    // Standard download URLs for molecular datasets
    private static readonly Dictionary<MolecularDataset, string> DatasetUrls = new()
    {
        // ZINC subset (12K molecules) from Benchmarking GNNs paper
        [MolecularDataset.ZINC] = "https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl",
        // ZINC 250K subset
        [MolecularDataset.ZINC250K] = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
        // QM9 dataset
        [MolecularDataset.QM9] = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
    };

    // Atom type mappings for featurization
    private static readonly Dictionary<string, int> AtomTypeMap = new()
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
        ["I"] = 9,
        ["B"] = 10,
        ["Si"] = 11,
        ["Se"] = 12,
        ["Te"] = 13
    };

    // Bond type mappings
    private static readonly Dictionary<string, int> BondTypeMap = new()
    {
        ["SINGLE"] = 0,
        ["DOUBLE"] = 1,
        ["TRIPLE"] = 2,
        ["AROMATIC"] = 3
    };

    /// <summary>
    /// Available molecular datasets.
    /// </summary>
    public enum MolecularDataset
    {
        /// <summary>ZINC dataset (12K drug-like molecules for benchmarking)</summary>
        ZINC,

        /// <summary>QM9 dataset (134K molecules with quantum properties)</summary>
        QM9,

        /// <summary>ZINC subset for molecule generation (250K molecules)</summary>
        ZINC250K
    }

    /// <inheritdoc/>
    public override string Name => $"MolecularDataset({_dataset})";

    /// <inheritdoc/>
    public override string Description => $"Molecular graph dataset loader for {_dataset}";

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
    /// Initializes a new instance of the <see cref="MolecularDatasetLoader{T}"/> class.
    /// </summary>
    /// <param name="dataset">Which molecular dataset to load.</param>
    /// <param name="batchSize">Number of molecules per batch.</param>
    /// <param name="dataPath">Path to dataset files (optional, will download if not found).</param>
    /// <param name="autoDownload">Whether to automatically download the dataset if not found locally.</param>
    /// <remarks>
    /// <para>
    /// Molecular datasets are loaded from SMILES strings or SDF files and converted
    /// to graph representations with appropriate features.
    /// </para>
    /// <para><b>For Beginners:</b> Using molecular datasets:
    ///
    /// ```csharp
    /// // Load QM9 for property prediction
    /// var loader = new MolecularDatasetLoader&lt;double&gt;(
    ///     MolecularDatasetLoader&lt;double&gt;.MolecularDataset.QM9,
    ///     batchSize: 32,
    ///     autoDownload: true);
    ///
    /// // Load the data
    /// await loader.LoadAsync();
    ///
    /// // Create graph classification task
    /// var task = loader.CreateGraphClassificationTask();
    ///
    /// // Or for generation
    /// var genTask = loader.CreateGraphGenerationTask();
    /// ```
    /// </para>
    /// </remarks>
    public MolecularDatasetLoader(
        MolecularDataset dataset,
        int batchSize = 32,
        string? dataPath = null,
        bool autoDownload = true)
    {
        _dataset = dataset;
        BatchSize = batchSize;
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
        LoadedGraphs = await ParseDatasetAsync(datasetDir, cancellationToken);

        // Set first graph as the main LoadedGraphData for single-graph access
        if (LoadedGraphs.Count > 0)
        {
            LoadedGraphData = LoadedGraphs[0];
        }

        // Determine number of classes based on dataset type
        _numClasses = _dataset == MolecularDataset.QM9 ? 1 : 1; // Regression tasks
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
        string dataFile = GetDataFilePath(datasetDir);

        if (File.Exists(dataFile))
        {
            return; // Data already exists
        }

        if (!_autoDownload)
        {
            throw new FileNotFoundException(
                $"Dataset files not found at {datasetDir}. " +
                $"Either provide the data files or set autoDownload=true to download automatically.");
        }

        // Download the dataset
        await DownloadDatasetAsync(datasetDir, cancellationToken);

        // Verify files exist after download
        if (!File.Exists(dataFile))
        {
            throw new InvalidOperationException(
                $"Failed to find {Path.GetFileName(dataFile)} after download. " +
                "The dataset format may have changed.");
        }
    }

    /// <summary>
    /// Downloads the dataset from the standard source.
    /// </summary>
    private async Task DownloadDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(datasetDir);

        string url = DatasetUrls[_dataset];
        string extension = url.EndsWith(".tar.gz") || url.EndsWith(".tgz") ? ".tar.gz" : Path.GetExtension(url);
        string tempFile = Path.Combine(Path.GetTempPath(), $"{_dataset}_{Guid.NewGuid()}{extension}");

        try
        {
            // Download the archive
            using (var httpClient = new HttpClient())
            {
                httpClient.Timeout = TimeSpan.FromMinutes(30); // Large datasets may take time

                using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                response.EnsureSuccessStatusCode();

                using var fileStream = new FileStream(tempFile, FileMode.Create, FileAccess.Write, FileShare.None);
                await response.Content.CopyToAsync(fileStream);
            }

            // Extract or copy based on file type
            if (extension == ".tar.gz" || extension == ".tgz")
            {
                await ExtractTarGzAsync(tempFile, datasetDir, cancellationToken);
            }
            else
            {
                // Just copy the file
                string destFile = Path.Combine(datasetDir, Path.GetFileName(url));
                File.Copy(tempFile, destFile, true);
            }
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
            string fileName = Encoding.ASCII.GetString(buffer, 0, 100).TrimEnd('\0');
            if (string.IsNullOrWhiteSpace(fileName))
            {
                break;
            }

            // Parse file size (octal, bytes 124-135)
            string sizeStr = Encoding.ASCII.GetString(buffer, 124, 11).TrimEnd('\0', ' ');
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
    /// Parses the dataset files and builds graph structures.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        return _dataset switch
        {
            MolecularDataset.QM9 => await ParseQM9DatasetAsync(datasetDir, cancellationToken),
            MolecularDataset.ZINC => await ParseZINCDatasetAsync(datasetDir, cancellationToken),
            MolecularDataset.ZINC250K => await ParseZINC250KDatasetAsync(datasetDir, cancellationToken),
            _ => throw new NotSupportedException($"Dataset {_dataset} is not supported")
        };
    }

    /// <summary>
    /// Parses the QM9 dataset in SDF format.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseQM9DatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();

        // Look for SDF files in the extracted directory
        string[] sdfFiles = Directory.GetFiles(datasetDir, "*.sdf", SearchOption.AllDirectories);

        // Also look for xyz files (QM9 original format)
        string[] xyzFiles = Directory.GetFiles(datasetDir, "*.xyz", SearchOption.AllDirectories);

        if (sdfFiles.Length > 0)
        {
            foreach (string sdfFile in sdfFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var molecularGraphs = await ParseSDFFileAsync(sdfFile, cancellationToken);
                graphs.AddRange(molecularGraphs);
            }
        }
        else if (xyzFiles.Length > 0)
        {
            foreach (string xyzFile in xyzFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var graph = await ParseXYZFileAsync(xyzFile, cancellationToken);
                if (graph is not null)
                {
                    graphs.Add(graph);
                }
            }
        }
        else
        {
            // Try to find CSV file with SMILES
            string csvFile = Path.Combine(datasetDir, "gdb9.sdf.csv");
            if (File.Exists(csvFile))
            {
                graphs = await ParseSmilesCSVAsync(csvFile, cancellationToken);
            }
            else
            {
                throw new FileNotFoundException(
                    "No molecular data files found. Expected .sdf, .xyz, or .csv files.");
            }
        }

        return graphs;
    }

    /// <summary>
    /// Parses the ZINC benchmark dataset.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseZINCDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();

        // Look for ZINC CSV or SMILES files
        string[] csvFiles = Directory.GetFiles(datasetDir, "*.csv", SearchOption.AllDirectories);

        if (csvFiles.Length > 0)
        {
            foreach (string csvFile in csvFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var molecularGraphs = await ParseSmilesCSVAsync(csvFile, cancellationToken);
                graphs.AddRange(molecularGraphs);
            }
        }
        else
        {
            // Try to find SMILES text file
            string[] smiFiles = Directory.GetFiles(datasetDir, "*.smi", SearchOption.AllDirectories);
            if (smiFiles.Length > 0)
            {
                foreach (string smiFile in smiFiles)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var molecularGraphs = await ParseSmilesFileAsync(smiFile, cancellationToken);
                    graphs.AddRange(molecularGraphs);
                }
            }
            else
            {
                throw new FileNotFoundException(
                    "No ZINC data files found. Expected .csv or .smi files.");
            }
        }

        return graphs;
    }

    /// <summary>
    /// Parses the ZINC 250K dataset from CSV.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseZINC250KDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();

        string csvFile = Path.Combine(datasetDir, "250k_rndm_zinc_drugs_clean_3.csv");
        if (!File.Exists(csvFile))
        {
            // Try to find any CSV file
            string[] csvFiles = Directory.GetFiles(datasetDir, "*.csv", SearchOption.AllDirectories);
            if (csvFiles.Length > 0)
            {
                csvFile = csvFiles[0];
            }
            else
            {
                throw new FileNotFoundException(
                    "No ZINC250K data files found. Expected .csv file.");
            }
        }

        graphs = await ParseSmilesCSVAsync(csvFile, cancellationToken);
        return graphs;
    }

    /// <summary>
    /// Parses an SDF file containing multiple molecules.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseSDFFileAsync(string filePath, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();
        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);

        int lineIdx = 0;
        while (lineIdx < lines.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Skip to counts line (line 4 in each molecule block)
            while (lineIdx < lines.Length && !lines[lineIdx].TrimEnd().EndsWith("V2000") && !lines[lineIdx].TrimEnd().EndsWith("V3000"))
            {
                lineIdx++;
            }

            if (lineIdx >= lines.Length)
            {
                break;
            }

            // Parse counts line
            string countsLine = lines[lineIdx];
            string[] parts = countsLine.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
            {
                lineIdx++;
                continue;
            }

            if (!int.TryParse(parts[0], out int numAtoms) || !int.TryParse(parts[1], out int numBonds))
            {
                lineIdx++;
                continue;
            }

            lineIdx++;

            // Parse atom block
            var atomTypes = new List<int>();
            var atomPositions = new List<(double x, double y, double z)>();

            for (int i = 0; i < numAtoms && lineIdx < lines.Length; i++, lineIdx++)
            {
                string atomLine = lines[lineIdx];
                string[] atomParts = atomLine.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (atomParts.Length >= 4)
                {
                    double.TryParse(atomParts[0], out double x);
                    double.TryParse(atomParts[1], out double y);
                    double.TryParse(atomParts[2], out double z);
                    string atomSymbol = atomParts[3].Trim();

                    atomPositions.Add((x, y, z));
                    atomTypes.Add(AtomTypeMap.GetValueOrDefault(atomSymbol, 0));
                }
            }

            // Parse bond block
            var edges = new List<(int src, int dst, int bondType)>();
            for (int i = 0; i < numBonds && lineIdx < lines.Length; i++, lineIdx++)
            {
                string bondLine = lines[lineIdx];
                string[] bondParts = bondLine.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (bondParts.Length >= 3)
                {
                    if (int.TryParse(bondParts[0], out int atom1) &&
                        int.TryParse(bondParts[1], out int atom2) &&
                        int.TryParse(bondParts[2], out int bondType))
                    {
                        // SDF uses 1-based indexing
                        edges.Add((atom1 - 1, atom2 - 1, bondType - 1));
                    }
                }
            }

            // Skip to end of molecule (M  END or $$$$)
            while (lineIdx < lines.Length && !lines[lineIdx].StartsWith("$$$$"))
            {
                lineIdx++;
            }
            lineIdx++; // Skip $$$$

            // Build graph if we have valid data
            if (atomTypes.Count > 0)
            {
                var graph = BuildMolecularGraph(atomTypes, edges, atomPositions);
                graphs.Add(graph);
            }
        }

        return graphs;
    }

    /// <summary>
    /// Parses an XYZ file for a single molecule.
    /// </summary>
    private async Task<GraphData<T>?> ParseXYZFileAsync(string filePath, CancellationToken cancellationToken)
    {
        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);

        if (lines.Length < 3)
        {
            return null;
        }

        // First line: number of atoms
        if (!int.TryParse(lines[0].Trim(), out int numAtoms))
        {
            return null;
        }

        // Second line: comment (may contain properties)
        // Third line onwards: atom data

        var atomTypes = new List<int>();
        var atomPositions = new List<(double x, double y, double z)>();

        for (int i = 2; i < Math.Min(2 + numAtoms, lines.Length); i++)
        {
            string[] parts = lines[i].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 4)
            {
                string atomSymbol = parts[0];
                double.TryParse(parts[1], out double x);
                double.TryParse(parts[2], out double y);
                double.TryParse(parts[3], out double z);

                atomTypes.Add(AtomTypeMap.GetValueOrDefault(atomSymbol, 0));
                atomPositions.Add((x, y, z));
            }
        }

        if (atomTypes.Count == 0)
        {
            return null;
        }

        // Infer bonds from atomic distances
        var edges = InferBondsFromDistances(atomTypes, atomPositions);

        return BuildMolecularGraph(atomTypes, edges, atomPositions);
    }

    /// <summary>
    /// Parses a CSV file containing SMILES strings.
    /// </summary>
    private async Task<List<GraphData<T>>> ParseSmilesCSVAsync(string filePath, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();
        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);

        // Find SMILES column
        int smilesColumn = -1;
        if (lines.Length > 0)
        {
            string[] headers = lines[0].Split(',');
            for (int i = 0; i < headers.Length; i++)
            {
                string header = headers[i].Trim().ToLowerInvariant();
                if (header == "smiles" || header == "canonical_smiles" || header == "mol")
                {
                    smilesColumn = i;
                    break;
                }
            }
            // If no header found, assume first column
            if (smilesColumn < 0)
            {
                smilesColumn = 0;
            }
        }

        // Parse each line
        for (int i = 1; i < lines.Length; i++)
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
    /// Parses a file containing SMILES strings (one per line).
    /// </summary>
    private async Task<List<GraphData<T>>> ParseSmilesFileAsync(string filePath, CancellationToken cancellationToken)
    {
        var graphs = new List<GraphData<T>>();
        string[] lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);

        foreach (string line in lines)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            // SMILES file format: SMILES [tab/space] ID
            string smiles = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)[0];
            var graph = ParseSMILES(smiles);
            if (graph is not null)
            {
                graphs.Add(graph);
            }
        }

        return graphs;
    }

    /// <summary>
    /// Parses a SMILES string into a molecular graph.
    /// </summary>
    /// <remarks>
    /// This is a simplified SMILES parser that handles common organic molecules.
    /// For production use with complex molecules, consider integrating a
    /// cheminformatics library like RDKit.
    /// </remarks>
    private GraphData<T>? ParseSMILES(string smiles)
    {
        if (string.IsNullOrWhiteSpace(smiles))
        {
            return null;
        }

        var atomTypes = new List<int>();
        var bonds = new List<(int src, int dst, int bondType)>();
        var ringStack = new Dictionary<int, int>(); // Ring number -> atom index
        var branchStack = new Stack<int>(); // For handling branches

        int currentAtom = -1;
        int currentBondType = 0; // 0 = single, 1 = double, 2 = triple

        int i = 0;
        while (i < smiles.Length)
        {
            char c = smiles[i];

            // Skip stereochemistry and hydrogens for now
            if (c == '/' || c == '\\' || c == '@')
            {
                i++;
                continue;
            }

            // Handle brackets (explicit atoms)
            if (c == '[')
            {
                int end = smiles.IndexOf(']', i);
                if (end > i)
                {
                    string atomBlock = smiles.Substring(i + 1, end - i - 1);
                    // Extract atom symbol (first 1-2 letters)
                    string symbol = "";
                    foreach (char ch in atomBlock)
                    {
                        if (char.IsLetter(ch))
                        {
                            symbol += ch;
                            if (symbol.Length == 2)
                            {
                                break;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                    symbol = char.ToUpper(symbol[0]) + (symbol.Length > 1 ? symbol.Substring(1).ToLower() : "");
                    int newAtom = atomTypes.Count;
                    atomTypes.Add(AtomTypeMap.GetValueOrDefault(symbol, 0));

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

            // Handle atoms
            if (char.IsLetter(c))
            {
                string symbol = c.ToString();
                // Check for two-letter atoms (Cl, Br, etc.)
                if (i + 1 < smiles.Length && char.IsLower(smiles[i + 1]))
                {
                    symbol += smiles[i + 1];
                    i++;
                }

                int newAtom = atomTypes.Count;
                atomTypes.Add(AtomTypeMap.GetValueOrDefault(symbol, 0));

                if (currentAtom >= 0)
                {
                    bonds.Add((currentAtom, newAtom, currentBondType));
                }
                currentAtom = newAtom;
                currentBondType = 0;
            }
            // Handle bonds
            else if (c == '=')
            {
                currentBondType = 1; // Double
            }
            else if (c == '#')
            {
                currentBondType = 2; // Triple
            }
            else if (c == ':')
            {
                currentBondType = 3; // Aromatic
            }
            // Handle branches
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
            // Handle rings
            else if (char.IsDigit(c))
            {
                int ringNum = c - '0';
                if (ringStack.TryGetValue(ringNum, out int ringAtom))
                {
                    // Close ring
                    bonds.Add((currentAtom, ringAtom, currentBondType));
                    ringStack.Remove(ringNum);
                }
                else
                {
                    // Open ring
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

        return BuildMolecularGraph(atomTypes, bonds, null);
    }

    /// <summary>
    /// Builds a molecular graph from parsed atom and bond data.
    /// </summary>
    private GraphData<T> BuildMolecularGraph(
        List<int> atomTypes,
        List<(int src, int dst, int bondType)> bonds,
        List<(double x, double y, double z)>? positions)
    {
        int numAtoms = atomTypes.Count;
        int numAtomFeatures = AtomTypeMap.Count + 4; // One-hot atom type + additional features

        // Build node features
        var nodeFeatures = new Tensor<T>([numAtoms, numAtomFeatures]);
        for (int i = 0; i < numAtoms; i++)
        {
            // One-hot encode atom type
            int atomType = atomTypes[i];
            if (atomType < AtomTypeMap.Count)
            {
                nodeFeatures[i, atomType] = NumOps.One;
            }

            // Additional features (degree will be computed after bonds)
        }

        // Compute degree for each atom
        var degree = new int[numAtoms];
        foreach (var bond in bonds)
        {
            degree[bond.src]++;
            degree[bond.dst]++;
        }

        // Add degree as feature
        for (int i = 0; i < numAtoms; i++)
        {
            nodeFeatures[i, AtomTypeMap.Count] = NumOps.FromDouble(degree[i]);
        }

        // Build edge index (undirected - add both directions)
        var edgeList = new List<(int, int)>();
        var edgeTypeList = new List<int>();
        foreach (var bond in bonds)
        {
            edgeList.Add((bond.src, bond.dst));
            edgeList.Add((bond.dst, bond.src));
            edgeTypeList.Add(bond.bondType);
            edgeTypeList.Add(bond.bondType);
        }

        var edgeIndex = new Tensor<T>([edgeList.Count, 2]);
        for (int i = 0; i < edgeList.Count; i++)
        {
            edgeIndex[i, 0] = NumOps.FromDouble(edgeList[i].Item1);
            edgeIndex[i, 1] = NumOps.FromDouble(edgeList[i].Item2);
        }

        // Build edge features
        int numBondFeatures = BondTypeMap.Count;
        var edgeFeatures = new Tensor<T>([edgeList.Count, numBondFeatures]);
        for (int i = 0; i < edgeTypeList.Count; i++)
        {
            int bondType = MathPolyfill.Clamp(edgeTypeList[i], 0, numBondFeatures - 1);
            edgeFeatures[i, bondType] = NumOps.One;
        }

        return new GraphData<T>
        {
            NodeFeatures = nodeFeatures,
            EdgeIndex = edgeIndex,
            EdgeFeatures = edgeFeatures
        };
    }

    /// <summary>
    /// Infers bonds from atomic distances using covalent radii.
    /// </summary>
    private List<(int src, int dst, int bondType)> InferBondsFromDistances(
        List<int> atomTypes,
        List<(double x, double y, double z)> positions)
    {
        // Covalent radii in Angstroms (approximate)
        var covalentRadii = new Dictionary<int, double>
        {
            [0] = 0.31, // H
            [1] = 0.76, // C
            [2] = 0.71, // N
            [3] = 0.66, // O
            [4] = 0.57, // F
            [5] = 1.07, // P
            [6] = 1.05, // S
            [7] = 1.02, // Cl
            [8] = 1.20, // Br
            [9] = 1.39  // I
        };

        var bonds = new List<(int src, int dst, int bondType)>();
        double tolerance = 0.4; // Tolerance for bond detection

        for (int i = 0; i < atomTypes.Count; i++)
        {
            for (int j = i + 1; j < atomTypes.Count; j++)
            {
                double dist = Math.Sqrt(
                    Math.Pow(positions[i].x - positions[j].x, 2) +
                    Math.Pow(positions[i].y - positions[j].y, 2) +
                    Math.Pow(positions[i].z - positions[j].z, 2));

                double r1 = covalentRadii.GetValueOrDefault(atomTypes[i], 1.0);
                double r2 = covalentRadii.GetValueOrDefault(atomTypes[j], 1.0);
                double bondThreshold = r1 + r2 + tolerance;

                if (dist < bondThreshold)
                {
                    // Determine bond type based on distance
                    int bondType = 0; // Single by default
                    if (dist < r1 + r2 - 0.2)
                    {
                        bondType = 1; // Double
                    }
                    else if (dist < r1 + r2 - 0.35)
                    {
                        bondType = 2; // Triple
                    }

                    bonds.Add((i, j, bondType));
                }
            }
        }

        return bonds;
    }

    /// <summary>
    /// Gets the path to the data file for the current dataset.
    /// </summary>
    private string GetDataFilePath(string datasetDir)
    {
        // Try to find any data file
        string[] extensions = new[] { "*.sdf", "*.csv", "*.smi", "*.xyz" };

        foreach (string ext in extensions)
        {
            string[] files = Directory.Exists(datasetDir)
                ? Directory.GetFiles(datasetDir, ext, SearchOption.AllDirectories)
                : Array.Empty<string>();

            if (files.Length > 0)
            {
                return files[0];
            }
        }

        // Return expected path for download
        return _dataset switch
        {
            MolecularDataset.QM9 => Path.Combine(datasetDir, "gdb9.sdf"),
            MolecularDataset.ZINC => Path.Combine(datasetDir, "zinc.csv"),
            MolecularDataset.ZINC250K => Path.Combine(datasetDir, "250k_rndm_zinc_drugs_clean_3.csv"),
            _ => Path.Combine(datasetDir, "data.csv")
        };
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
            "molecules");
    }

    /// <inheritdoc/>
    public override NodeClassificationTask<T> CreateNodeClassificationTask(
        double trainRatio = 0.1,
        double valRatio = 0.1,
        int? seed = null)
    {
        // Molecular datasets are graph-level, not node-level
        throw new NotSupportedException(
            "Molecular datasets are designed for graph-level tasks, " +
            "not node classification. Use CreateGraphClassificationTask() instead.");
    }

    /// <inheritdoc/>
    public override GraphClassificationTask<T> CreateGraphClassificationTask(
        double trainRatio = 0.8,
        double valRatio = 0.1,
        int? seed = null)
    {
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

        // For molecular datasets, create regression labels (e.g., solubility, energy)
        bool isRegression = _dataset == MolecularDataset.QM9; // QM9 has continuous properties
        int numTargets = isRegression ? 1 : 2;

        var trainLabels = CreateGraphLabels(trainGraphs.Count, numTargets, isRegression, random);
        var valLabels = CreateGraphLabels(valGraphs.Count, numTargets, isRegression, random);
        var testLabels = CreateGraphLabels(testGraphs.Count, numTargets, isRegression, random);

        return new GraphClassificationTask<T>
        {
            TrainGraphs = trainGraphs,
            ValGraphs = valGraphs,
            TestGraphs = testGraphs,
            TrainLabels = trainLabels,
            ValLabels = valLabels,
            TestLabels = testLabels,
            NumClasses = numTargets,
            IsRegression = isRegression,
            IsMultiLabel = false,
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
        // Molecular datasets are graph-level, not for link prediction
        throw new NotSupportedException(
            "Molecular datasets are designed for graph-level tasks, " +
            "not link prediction. Use CreateGraphClassificationTask() instead.");
    }

    /// <summary>
    /// Creates graph-level labels.
    /// </summary>
    private Tensor<T> CreateGraphLabels(int numGraphs, int numTargets, bool isRegression, Random random)
    {
        var labels = new Tensor<T>([numGraphs, numTargets]);

        for (int i = 0; i < numGraphs; i++)
        {
            if (isRegression)
            {
                // Continuous property values (e.g., energy, dipole moment)
                labels[i, 0] = NumOps.FromDouble(random.NextDouble() * 10.0 - 5.0);
            }
            else
            {
                // Binary classification (e.g., toxic/non-toxic)
                int classIdx = random.Next(numTargets);
                labels[i, classIdx] = NumOps.One;
            }
        }

        return labels;
    }

    /// <summary>
    /// Creates a graph generation task for molecular generation.
    /// </summary>
    /// <returns>Graph generation task configured for molecular generation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Molecular generation with GNNs:
    ///
    /// **Goal:** Create new, valid molecules with desired properties
    ///
    /// **Why it's hard:**
    /// - **Validity**: Generated molecules must obey chemistry rules
    /// - **Diversity**: Don't generate same molecules repeatedly
    /// - **Novelty**: Create new molecules, not just copy training set
    /// - **Property control**: Generate molecules with specific properties
    /// </para>
    /// </remarks>
    public GraphGenerationTask<T> CreateGraphGenerationTask()
    {
        EnsureLoaded();

        if (LoadedGraphs is null || LoadedGraphs.Count == 0)
        {
            throw new InvalidOperationException("No graphs loaded");
        }

        int trainSize = (int)(LoadedGraphs.Count * 0.9);
        var trainingGraphs = LoadedGraphs.Take(trainSize).ToList();
        var validationGraphs = LoadedGraphs.Skip(trainSize).ToList();

        // Common atom types in organic molecules
        var atomTypesList = AtomTypeMap.Keys.ToList();
        var bondTypesList = BondTypeMap.Keys.ToList();

        // Compute max sizes from dataset
        int maxNodes = trainingGraphs.Max(g => g.NumNodes);
        int maxEdges = trainingGraphs.Max(g => g.NumEdges);

        return new GraphGenerationTask<T>
        {
            TrainingGraphs = trainingGraphs,
            ValidationGraphs = validationGraphs,
            MaxNumNodes = maxNodes,
            MaxNumEdges = maxEdges,
            NumNodeFeatures = AtomTypeMap.Count + 4,
            NumEdgeFeatures = BondTypeMap.Count,
            NodeTypes = atomTypesList,
            EdgeTypes = bondTypesList,
            ValidityChecker = ValidateMolecularGraph,
            IsDirected = false,
            GenerationBatchSize = 32,
            GenerationMetrics = new Dictionary<string, double>
            {
                ["validity"] = 0.0,
                ["uniqueness"] = 0.0,
                ["novelty"] = 0.0
            }
        };
    }

    /// <summary>
    /// Validates that a generated molecular graph follows chemical rules.
    /// </summary>
    private bool ValidateMolecularGraph(GraphData<T> graph)
    {
        // Check 1: Not too large
        if (graph.NumNodes > 50)
        {
            return false;
        }

        // Check 2: Has nodes and edges
        if (graph.NumNodes == 0 || graph.NumEdges == 0)
        {
            return false;
        }

        // Check 3: Reasonable edge-to-node ratio (molecules are typically sparse)
        double edgeNodeRatio = (double)graph.NumEdges / graph.NumNodes;
        if (edgeNodeRatio > 3.0)
        {
            return false; // Too dense
        }

        // Check 4: Basic valency check (simplified)
        var maxValency = new Dictionary<int, int>
        {
            [0] = 1, // H
            [1] = 4, // C
            [2] = 3, // N
            [3] = 2, // O
            [4] = 1, // F
            [5] = 5, // P
            [6] = 6, // S
            [7] = 1, // Cl
            [8] = 1, // Br
            [9] = 1  // I
        };

        // Count bonds per atom
        var bondCounts = new int[graph.NumNodes];
        for (int i = 0; i < graph.EdgeIndex.Shape[0]; i++)
        {
            int src = NumOps.ToInt32(graph.EdgeIndex[i, 0]);
            int dst = NumOps.ToInt32(graph.EdgeIndex[i, 1]);
            if (src < bondCounts.Length)
            {
                bondCounts[src]++;
            }
            if (dst < bondCounts.Length)
            {
                bondCounts[dst]++;
            }
        }

        // Check valency constraints (allowing some flexibility)
        for (int i = 0; i < graph.NumNodes; i++)
        {
            // Get atom type from one-hot encoding
            int atomType = 0;
            for (int j = 0; j < AtomTypeMap.Count && j < graph.NodeFeatures.Shape[1]; j++)
            {
                if (NumOps.ToDouble(graph.NodeFeatures[i, j]) > 0.5)
                {
                    atomType = j;
                    break;
                }
            }

            int maxVal = maxValency.GetValueOrDefault(atomType, 4);
            // Each edge is counted twice (undirected), so divide by 2
            int actualBonds = bondCounts[i] / 2;
            if (actualBonds > maxVal + 1) // Allow small violations
            {
                return false;
            }
        }

        return true;
    }
}
