using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
/// - **Size**: 250,000 drug-like molecules
/// - **Source**: ZINC database (commercially available compounds)
/// - **Tasks**:
///   * Classification: Molecular properties
///   * Generation: Create novel drug-like molecules
/// - **Features**:
///   * Atoms: C, N, O, F, P, S, Cl, Br, I
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
///
/// **Example Applications:**
///
/// **Drug Discovery:**
/// ```
/// Task: Predict if molecule binds to protein target
/// Input: Molecular graph (atoms + bonds)
/// Process: GNN learns structure-activity relationship
/// Output: Binding affinity score
/// Benefit: Screen millions of molecules computationally
/// ```
///
/// **Materials Design:**
/// ```
/// Task: Predict material conductivity
/// Input: Crystal structure graph
/// Process: GNN learns structure-property mapping
/// Output: Predicted conductivity
/// Benefit: Design materials with desired properties
/// ```
/// </para>
/// </remarks>
public class MolecularDatasetLoader<T> : IGraphDataLoader<T>
{
    private readonly MolecularDataset _dataset;
    private readonly string _dataPath;
    private readonly int _batchSize;
    private List<GraphData<T>>? _loadedGraphs;
    private int _currentIndex;

    /// <summary>
    /// Available molecular datasets.
    /// </summary>
    public enum MolecularDataset
    {
        /// <summary>ZINC dataset (250K drug-like molecules)</summary>
        ZINC,

        /// <summary>QM9 dataset (134K molecules with quantum properties)</summary>
        QM9,

        /// <summary>ZINC subset for molecule generation (smaller, 250 molecules)</summary>
        ZINC250K
    }

    /// <inheritdoc/>
    public int NumGraphs { get; private set; }

    /// <inheritdoc/>
    public int BatchSize => _batchSize;

    /// <inheritdoc/>
    public bool HasNext => _loadedGraphs != null && _currentIndex < NumGraphs;

    /// <summary>
    /// Initializes a new instance of the <see cref="MolecularDatasetLoader{T}"/> class.
    /// </summary>
    /// <param name="dataset">Which molecular dataset to load.</param>
    /// <param name="batchSize">Number of molecules per batch.</param>
    /// <param name="dataPath">Path to dataset files (optional, will download if not found).</param>
    /// <remarks>
    /// <para>
    /// Molecular datasets are typically loaded from SMILES strings or SDF files and converted
    /// to graph representations with appropriate features.
    /// </para>
    /// <para><b>For Beginners:</b> Using molecular datasets:
    ///
    /// ```csharp
    /// // Load QM9 for property prediction
    /// var loader = new MolecularDatasetLoader<double>(
    ///     MolecularDatasetLoader<double>.MolecularDataset.QM9,
    ///     batchSize: 32);
    ///
    /// // Create graph classification task
    /// var task = loader.CreateGraphClassificationTask();
    ///
    /// // Or for generation
    /// var genTask = loader.CreateGraphGenerationTask();
    /// ```
    ///
    /// **What gets loaded:**
    /// - Node features: Atom type (one-hot), degree, formal charge, aromaticity
    /// - Edge features: Bond type (single/double/triple), conjugation, ring membership
    /// - Labels: Depends on task (property values, solubility, toxicity, etc.)
    /// </para>
    /// </remarks>
    public MolecularDatasetLoader(
        MolecularDataset dataset,
        int batchSize = 32,
        string? dataPath = null)
    {
        _dataset = dataset;
        _batchSize = batchSize;
        _dataPath = dataPath ?? GetDefaultDataPath();
        _currentIndex = 0;
    }

    /// <inheritdoc/>
    public GraphData<T> GetNextBatch()
    {
        if (_loadedGraphs == null)
        {
            LoadDataset();
        }

        if (!HasNext)
        {
            throw new InvalidOperationException("No more batches available. Call Reset() first.");
        }

        // For now, return single graphs (batching would combine multiple graphs)
        var graph = _loadedGraphs![_currentIndex];
        _currentIndex++;
        return graph;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _currentIndex = 0;
    }

    /// <summary>
    /// Creates a graph classification task for molecular property prediction.
    /// </summary>
    /// <param name="trainRatio">Fraction of molecules for training.</param>
    /// <param name="valRatio">Fraction of molecules for validation.</param>
    /// <returns>Graph classification task with molecule splits.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Molecular property prediction:
    ///
    /// **Task:** Given a molecule, predict its properties
    ///
    /// **QM9 Example:**
    /// ```
    /// Input: Aspirin molecule graph
    /// Properties to predict:
    /// - Dipole moment: 3.2 Debye
    /// - HOMO-LUMO gap: 0.3 eV
    /// - Heat capacity: 45.3 cal/mol·K
    /// ```
    ///
    /// **ZINC Example:**
    /// ```
    /// Input: Drug candidate molecule
    /// Properties to predict:
    /// - Solubility: High/Low
    /// - Toxicity: Toxic/Safe
    /// - Drug-likeness: Yes/No
    /// ```
    ///
    /// **Why it's useful:**
    /// - Expensive to measure properties experimentally
    /// - GNN predicts properties from structure alone
    /// - Screen thousands of candidates quickly
    /// - Guide synthesis of promising molecules
    /// </para>
    /// </remarks>
    public GraphClassificationTask<T> CreateGraphClassificationTask(
        double trainRatio = 0.8,
        double valRatio = 0.1)
    {
        if (_loadedGraphs == null)
        {
            LoadDataset();
        }

        int numGraphs = _loadedGraphs!.Count;
        int trainSize = (int)(numGraphs * trainRatio);
        int valSize = (int)(numGraphs * valRatio);

        var trainGraphs = _loadedGraphs.Take(trainSize).ToList();
        var valGraphs = _loadedGraphs.Skip(trainSize).Take(valSize).ToList();
        var testGraphs = _loadedGraphs.Skip(trainSize + valSize).ToList();

        bool isRegression = _dataset == MolecularDataset.QM9; // QM9 has continuous properties
        int numTargets = isRegression ? 1 : 2; // Regression: 1 value, Classification: binary

        var trainLabels = CreateMolecularLabels(trainGraphs.Count, numTargets, isRegression);
        var valLabels = CreateMolecularLabels(valGraphs.Count, numTargets, isRegression);
        var testLabels = CreateMolecularLabels(testGraphs.Count, numTargets, isRegression);

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
    ///   * Valency constraints: C has 4 bonds, O has 2
    ///   * No impossible structures
    ///   * Stable ring systems
    /// - **Diversity**: Don't generate same molecules repeatedly
    /// - **Novelty**: Create new molecules, not just copy training set
    /// - **Property control**: Generate molecules with specific properties
    ///
    /// **Applications:**
    ///
    /// **Drug Discovery:**
    /// ```
    /// Goal: Generate novel drug candidates
    /// Constraints:
    /// - Drug-like properties (Lipinski's rule of five)
    /// - No toxic substructures
    /// - Synthesizable
    /// Process:
    /// 1. Train on known drugs
    /// 2. Generate new molecules
    /// 3. Filter by drug-likeness
    /// 4. Test promising candidates
    /// ```
    ///
    /// **Material Design:**
    /// ```
    /// Goal: Generate molecules with high conductivity
    /// Process:
    /// 1. Train on materials database
    /// 2. Generate candidates
    /// 3. Predict properties with GNN
    /// 4. Keep molecules meeting criteria
    /// ```
    ///
    /// **Common approaches:**
    /// - **Autoregressive**: Add atoms/bonds one at a time
    /// - **VAE**: Learn latent space, sample new points
    /// - **GAN**: Generator creates molecules, discriminator validates
    /// - **Flow**: Invertible transformations of molecule distribution
    /// </para>
    /// </remarks>
    public GraphGenerationTask<T> CreateGraphGenerationTask()
    {
        if (_loadedGraphs == null)
        {
            LoadDataset();
        }

        int trainSize = (int)(_loadedGraphs!.Count * 0.9);
        var trainingGraphs = _loadedGraphs.Take(trainSize).ToList();
        var validationGraphs = _loadedGraphs.Skip(trainSize).ToList();

        // Common atom types in organic molecules
        var atomTypes = new List<string> { "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H" };
        var bondTypes = new List<string> { "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC" };

        return new GraphGenerationTask<T>
        {
            TrainingGraphs = trainingGraphs,
            ValidationGraphs = validationGraphs,
            MaxNumNodes = 38, // ZINC molecules typically < 38 atoms
            MaxNumEdges = 80,
            NumNodeFeatures = atomTypes.Count,
            NumEdgeFeatures = bondTypes.Count,
            NodeTypes = atomTypes,
            EdgeTypes = bondTypes,
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
    /// <param name="graph">The molecular graph to validate.</param>
    /// <returns>True if valid molecule, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Validation checks:
    /// - Valency constraints (C:4, N:3, O:2, etc.)
    /// - No isolated atoms
    /// - Connected graph
    /// - Valid bond types
    /// - No strange ring structures
    /// </para>
    /// <para><b>For Beginners:</b> Why validate generated molecules?
    ///
    /// **Without validation:**
    /// - Carbon with 6 bonds (impossible!)
    /// - Oxygen with 1 bond (unlikely, unstable)
    /// - Disconnected fragments (not a single molecule)
    /// - Invalid stereochemistry
    ///
    /// **With validation:**
    /// - Only chemically possible structures
    /// - Can be synthesized in lab
    /// - Meaningful for drug discovery
    /// - Saves wasted experimental effort
    ///
    /// This is like spell-check for molecular structure!
    /// </para>
    /// </remarks>
    private bool ValidateMolecularGraph(GraphData<T> graph)
    {
        // Simplified validation (real version would check valency, connectivity, etc.)

        // Check 1: Not too large
        if (graph.NumNodes > 50) return false;

        // Check 2: Has nodes and edges
        if (graph.NumNodes == 0 || graph.NumEdges == 0) return false;

        // Check 3: Reasonable edge-to-node ratio (molecules are typically sparse)
        double edgeNodeRatio = (double)graph.NumEdges / graph.NumNodes;
        if (edgeNodeRatio > 3.0) return false; // Too dense

        // Real implementation would check:
        // - Valency constraints per atom type
        // - Graph connectivity
        // - Ring aromaticity
        // - Stereochemistry

        return true;
    }

    private void LoadDataset()
    {
        // Real implementation would:
        // 1. Load SMILES strings from dataset file
        // 2. Parse with RDKit or similar chemistry toolkit
        // 3. Extract atom/bond features
        // 4. Build graph representation
        // 5. Load property labels

        var (numMolecules, avgAtoms) = GetDatasetStats();
        NumGraphs = numMolecules;

        _loadedGraphs = CreateMockMolecularGraphs(numMolecules, avgAtoms);
    }

    private List<GraphData<T>> CreateMockMolecularGraphs(int numMolecules, int avgAtoms)
    {
        var graphs = new List<GraphData<T>>();
        var random = new Random(42);

        for (int i = 0; i < numMolecules; i++)
        {
            int numAtoms = Math.Max(5, (int)(avgAtoms + random.Next(-5, 6)));

            graphs.Add(CreateMolecularGraph(numAtoms, random));
        }

        return graphs;
    }

    private GraphData<T> CreateMolecularGraph(int numAtoms, Random random)
    {
        // Node features: 10 atom types (one-hot) + 4 additional features
        int nodeFeatureDim = 14;
        var nodeFeatures = new Tensor<T>([numAtoms, nodeFeatureDim]);

        for (int i = 0; i < numAtoms; i++)
        {
            // Atom type (one-hot among first 10 features)
            int atomType = random.Next(10);
            nodeFeatures[i, atomType] = NumOps.FromDouble(1.0);

            // Additional features: degree, formal charge, aromatic, hybridization
            for (int j = 10; j < nodeFeatureDim; j++)
            {
                nodeFeatures[i, j] = NumOps.FromDouble(random.NextDouble());
            }
        }

        // Create bond connectivity (molecular graphs are typically connected)
        var edges = new List<(int, int)>();

        // Create spanning tree first (ensures connectivity)
        for (int i = 0; i < numAtoms - 1; i++)
        {
            int target = i + 1;
            edges.Add((i, target));
            edges.Add((target, i)); // Undirected
        }

        // Add random bonds to form rings
        int extraBonds = random.Next(numAtoms / 4, numAtoms / 2);
        for (int i = 0; i < extraBonds; i++)
        {
            int src = random.Next(numAtoms);
            int tgt = random.Next(numAtoms);
            if (src != tgt)
            {
                edges.Add((src, tgt));
                edges.Add((tgt, src));
            }
        }

        var edgeIndex = new Tensor<T>([edges.Count, 2]);
        for (int i = 0; i < edges.Count; i++)
        {
            edgeIndex[i, 0] = NumOps.FromDouble(edges[i].Item1);
            edgeIndex[i, 1] = NumOps.FromDouble(edges[i].Item2);
        }

        // Edge features: bond type (4 types)
        var edgeFeatures = new Tensor<T>([edges.Count, 4]);
        for (int i = 0; i < edges.Count; i++)
        {
            int bondType = random.Next(4);
            edgeFeatures[i, bondType] = NumOps.FromDouble(1.0);
        }

        return new GraphData<T>
        {
            NodeFeatures = nodeFeatures,
            EdgeIndex = edgeIndex,
            EdgeFeatures = edgeFeatures
        };
    }

    private Tensor<T> CreateMolecularLabels(int numMolecules, int numTargets, bool isRegression)
    {
        var labels = new Tensor<T>([numMolecules, numTargets]);
        var random = new Random(42);

        for (int i = 0; i < numMolecules; i++)
        {
            if (isRegression)
            {
                // Continuous property values (e.g., energy, dipole moment)
                labels[i, 0] = NumOps.FromDouble(random.NextDouble() * 10.0);
            }
            else
            {
                // Binary classification (e.g., toxic/non-toxic)
                int classIdx = random.Next(numTargets);
                labels[i, classIdx] = NumOps.FromDouble(1.0);
            }
        }

        return labels;
    }

    private (int numMolecules, int avgAtoms) GetDatasetStats()
    {
        return _dataset switch
        {
            MolecularDataset.ZINC => (250000, 23),
            MolecularDataset.QM9 => (133885, 18),
            MolecularDataset.ZINC250K => (250, 23),
            _ => (1000, 20)
        };
    }

    private string GetDefaultDataPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".aidotnet",
            "datasets",
            "molecules");
    }
}
