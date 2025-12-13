using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Structures;

/// <summary>
/// Represents a graph generation task where the goal is to generate new valid graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph generation creates new graph structures that follow learned patterns from training data.
/// This is useful for generating novel molecules, designing new materials, creating synthetic
/// networks, and other generative tasks.
/// </para>
/// <para><b>For Beginners:</b> Graph generation is like creating new objects that look realistic.
///
/// **Real-world examples:**
///
/// **Drug Discovery:**
/// - Task: Generate novel drug-like molecules
/// - Input: Training set of known drugs
/// - Output: New molecular structures with desired properties
/// - Goal: Discover new drug candidates automatically
/// - Example: Generate molecules that bind to a specific protein target
///
/// **Material Design:**
/// - Task: Generate new material structures
/// - Input: Database of materials with known properties
/// - Output: Novel material configurations
/// - Goal: Design materials with specific properties (strength, conductivity, etc.)
///
/// **Synthetic Data Generation:**
/// - Task: Create realistic social network graphs
/// - Input: Real social network data
/// - Output: Synthetic networks preserving statistical properties
/// - Goal: Generate data for testing while preserving privacy
///
/// **Molecular Optimization:**
/// - Task: Modify molecules to improve properties
/// - Input: Starting molecule
/// - Output: Similar molecules with better properties
/// - Example: Improve drug efficacy while maintaining safety
///
/// **Approaches:**
/// - **Autoregressive**: Generate nodes/edges one at a time
/// - **VAE**: Learn latent space of graphs, sample new ones
/// - **GAN**: Generator creates graphs, discriminator evaluates them
/// - **Flow-based**: Learn invertible transformations of graph distributions
/// </para>
/// </remarks>
public class GraphGenerationTask<T>
{
    /// <summary>
    /// Training graphs used to learn the distribution.
    /// </summary>
    /// <remarks>
    /// The generative model learns patterns from these graphs and generates similar ones.
    /// </remarks>
    public List<GraphData<T>> TrainingGraphs { get; set; } = new List<GraphData<T>>();

    /// <summary>
    /// Validation graphs for monitoring generation quality.
    /// </summary>
    public List<GraphData<T>> ValidationGraphs { get; set; } = new List<GraphData<T>>();

    /// <summary>
    /// Maximum number of nodes allowed in generated graphs.
    /// </summary>
    /// <remarks>
    /// This constraint helps control computational cost and memory usage during generation.
    /// </remarks>
    public int MaxNumNodes { get; set; } = 100;

    /// <summary>
    /// Maximum number of edges allowed in generated graphs.
    /// </summary>
    public int MaxNumEdges { get; set; } = 200;

    /// <summary>
    /// Number of node feature dimensions.
    /// </summary>
    public int NumNodeFeatures { get; set; }

    /// <summary>
    /// Number of edge feature dimensions (0 if no edge features).
    /// </summary>
    public int NumEdgeFeatures { get; set; }

    /// <summary>
    /// Possible node types/labels (for categorical node features).
    /// </summary>
    /// <remarks>
    /// <para>
    /// In molecule generation, this could be atom types: C, N, O, F, etc.
    /// </para>
    /// <para><b>For Beginners:</b> When generating molecules:
    /// - NodeTypes might be: ["C", "N", "O", "F", "S", "Cl"]
    /// - Each generated node must be one of these atom types
    /// - This ensures generated molecules use valid atoms
    /// </para>
    /// </remarks>
    public List<string> NodeTypes { get; set; } = new List<string>();

    /// <summary>
    /// Possible edge types/labels (for categorical edge features).
    /// </summary>
    /// <remarks>
    /// In molecule generation, this could be bond types: single, double, triple, aromatic.
    /// </remarks>
    public List<string> EdgeTypes { get; set; } = new List<string>();

    /// <summary>
    /// Validity constraints for generated graphs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Custom validation function to check if a generated graph is valid.
    /// For molecules, this might check chemical valency rules.
    /// </para>
    /// <para><b>For Beginners:</b> Generated graphs must be valid/realistic:
    ///
    /// **Molecular constraints:**
    /// - Carbon can have max 4 bonds
    /// - Oxygen typically has 2 bonds
    /// - No impossible bond types
    /// - Valid ring structures
    ///
    /// **Social network constraints:**
    /// - No self-loops (people can't be friends with themselves)
    /// - Degree distribution matches real networks
    /// - Community structure makes sense
    ///
    /// Validity constraints help ensure generated graphs are meaningful.
    /// </para>
    /// </remarks>
    public Func<GraphData<T>, bool>? ValidityChecker { get; set; }

    /// <summary>
    /// Whether to generate directed graphs.
    /// </summary>
    public bool IsDirected { get; set; } = false;

    /// <summary>
    /// Number of graphs to generate per batch during training.
    /// </summary>
    public int GenerationBatchSize { get; set; } = 32;

    /// <summary>
    /// Metrics to track during generation (e.g., validity rate, uniqueness, novelty).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common metrics for graph generation:
    /// - **Validity**: Percentage of generated graphs that satisfy constraints
    /// - **Uniqueness**: Percentage of unique graphs (not duplicates)
    /// - **Novelty**: Percentage not in training set (not memorized)
    /// - **Property matching**: Do generated graphs have desired properties?
    /// </para>
    /// </remarks>
    public Dictionary<string, double> GenerationMetrics { get; set; } = new Dictionary<string, double>();
}
