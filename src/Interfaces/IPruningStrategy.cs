using AiDotNet.ModelCompression;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for pruning strategies that remove unimportant weights to create sparsity.
/// </summary>
/// <typeparam name="T">Numeric type for weights and gradients.</typeparam>
/// <remarks>
/// <para>
/// Pruning strategies determine which weights to remove from a neural network to reduce size
/// and computational requirements. This interface supports all data types (Vector, Matrix, Tensor)
/// and multiple sparsity patterns including unstructured, structured, and hardware-optimized formats.
/// </para>
/// <para><b>For Beginners:</b> Pruning removes unnecessary connections from neural networks.
///
/// Think of it like pruning a tree - you remove branches that don't contribute much:
/// - Magnitude pruning: Remove smallest weights
/// - Gradient pruning: Remove weights with smallest gradients (learning slowly)
/// - Structured pruning: Remove entire neurons/filters (cleaner architecture)
/// - Movement pruning: Remove weights that don't change during training
/// - Lottery ticket: Find sparse subnetworks that train well from scratch
///
/// Sparsity patterns:
/// - Unstructured: Random individual weights (flexible but needs sparse libraries)
/// - Structured: Entire rows/columns (actual speedup on any hardware)
/// - 2:4 Sparsity: 2 zeros per 4 elements (NVIDIA Ampere 2x speedup)
/// - N:M Sparsity: N zeros per M elements (customizable)
///
/// Pruning can remove 50-99% of weights with minimal accuracy loss!
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("PruningStrategy")]
public interface IPruningStrategy<T>
{
    #region Importance Scoring

    /// <summary>
    /// Computes importance scores for vector weights.
    /// </summary>
    /// <param name="weights">Weight vector.</param>
    /// <param name="gradients">Gradient vector (optional, can be null).</param>
    /// <returns>Importance score for each weight (higher = more important).</returns>
    Vector<T> ComputeImportanceScores(Vector<T> weights, Vector<T>? gradients = null);

    /// <summary>
    /// Computes importance scores for matrix weights.
    /// </summary>
    /// <param name="weights">Weight matrix.</param>
    /// <param name="gradients">Gradient matrix (optional, can be null).</param>
    /// <returns>Importance score for each weight (higher = more important).</returns>
    Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null);

    /// <summary>
    /// Computes importance scores for tensor weights.
    /// </summary>
    /// <param name="weights">Weight tensor.</param>
    /// <param name="gradients">Gradient tensor (optional, can be null).</param>
    /// <returns>Importance score for each weight (higher = more important).</returns>
    Tensor<T> ComputeImportanceScores(Tensor<T> weights, Tensor<T>? gradients = null);

    #endregion

    #region Mask Creation

    /// <summary>
    /// Creates a pruning mask for vector weights based on target sparsity.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores.</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1).</param>
    /// <returns>Binary mask (1 = keep, 0 = prune).</returns>
    IPruningMask<T> CreateMask(Vector<T> importanceScores, double targetSparsity);

    /// <summary>
    /// Creates a pruning mask for matrix weights based on target sparsity.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores.</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1).</param>
    /// <returns>Binary mask (1 = keep, 0 = prune).</returns>
    IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity);

    /// <summary>
    /// Creates a pruning mask for tensor weights based on target sparsity.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores.</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1).</param>
    /// <returns>Binary mask (1 = keep, 0 = prune).</returns>
    IPruningMask<T> CreateMask(Tensor<T> importanceScores, double targetSparsity);

    /// <summary>
    /// Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible).
    /// </summary>
    /// <param name="importanceScores">Importance scores.</param>
    /// <returns>2:4 structured mask (exactly 2 zeros per 4 elements).</returns>
    IPruningMask<T> Create2to4Mask(Tensor<T> importanceScores);

    /// <summary>
    /// Creates an N:M structured sparsity mask.
    /// </summary>
    /// <param name="importanceScores">Importance scores.</param>
    /// <param name="n">Number of zeros per group.</param>
    /// <param name="m">Group size.</param>
    /// <returns>N:M structured mask.</returns>
    IPruningMask<T> CreateNtoMMask(Tensor<T> importanceScores, int n, int m);

    #endregion

    #region Pruning Application

    /// <summary>
    /// Applies pruning mask to vector weights in-place.
    /// </summary>
    void ApplyPruning(Vector<T> weights, IPruningMask<T> mask);

    /// <summary>
    /// Applies pruning mask to matrix weights in-place.
    /// </summary>
    void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask);

    /// <summary>
    /// Applies pruning mask to tensor weights in-place.
    /// </summary>
    void ApplyPruning(Tensor<T> weights, IPruningMask<T> mask);

    /// <summary>
    /// Converts pruned weights to sparse format for efficient storage.
    /// </summary>
    /// <param name="weights">Pruned weights (containing zeros).</param>
    /// <param name="format">Target sparse format.</param>
    /// <returns>Sparse representation.</returns>
    SparseCompressionResult<T> ToSparseFormat(Tensor<T> weights, SparseFormat format);

    #endregion

    #region Strategy Properties

    /// <summary>
    /// Gets the name of this pruning strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether this strategy requires gradients.
    /// </summary>
    bool RequiresGradients { get; }

    /// <summary>
    /// Gets whether this is structured pruning (removes entire rows/cols/filters).
    /// </summary>
    bool IsStructured { get; }

    /// <summary>
    /// Gets supported sparsity patterns.
    /// </summary>
    IReadOnlyList<SparsityPattern> SupportedPatterns { get; }

    #endregion
}

/// <summary>
/// Types of sparsity patterns.
/// </summary>
public enum SparsityPattern
{
    /// <summary>
    /// Unstructured - individual weights pruned randomly.
    /// </summary>
    Unstructured,

    /// <summary>
    /// Row-wise structured - entire rows removed.
    /// </summary>
    RowStructured,

    /// <summary>
    /// Column-wise structured - entire columns removed.
    /// </summary>
    ColumnStructured,

    /// <summary>
    /// Filter-wise structured - entire conv filters removed.
    /// </summary>
    FilterStructured,

    /// <summary>
    /// Channel-wise structured - entire channels removed.
    /// </summary>
    ChannelStructured,

    /// <summary>
    /// Block structured - dense blocks pruned together.
    /// </summary>
    BlockStructured,

    /// <summary>
    /// 2:4 fine-grained structured (NVIDIA Ampere).
    /// </summary>
    Structured2to4,

    /// <summary>
    /// N:M fine-grained structured (generalized).
    /// </summary>
    StructuredNtoM
}

/// <summary>
/// Configuration for pruning operations.
/// </summary>
/// <remarks>
/// <para><b>Layer-aware pruning:</b> When used with models implementing <see cref="ILayeredModel{T}"/>,
/// the <see cref="CategorySparsityTargets"/> property enables per-category sparsity levels.
/// For example, attention layers can be pruned less aggressively than dense layers.</para>
/// </remarks>
public class PruningConfig
{
    /// <summary>
    /// Target sparsity level (0.0 to 1.0).
    /// </summary>
    public double TargetSparsity { get; set; } = 0.5;

    /// <summary>
    /// Sparsity pattern to use.
    /// </summary>
    public SparsityPattern Pattern { get; set; } = SparsityPattern.Unstructured;

    /// <summary>
    /// N value for N:M sparsity (zeros per group).
    /// </summary>
    public int SparsityN { get; set; } = 2;

    /// <summary>
    /// M value for N:M sparsity (group size).
    /// </summary>
    public int SparsityM { get; set; } = 4;

    /// <summary>
    /// Whether to use gradual pruning (multiple iterations).
    /// </summary>
    public bool GradualPruning { get; set; } = false;

    /// <summary>
    /// Number of pruning iterations for gradual pruning.
    /// </summary>
    public int PruningIterations { get; set; } = 10;

    /// <summary>
    /// Initial sparsity for gradual pruning.
    /// </summary>
    public double InitialSparsity { get; set; } = 0.0;

    /// <summary>
    /// Whether to apply different sparsity per layer (sensitivity-based).
    /// </summary>
    public bool LayerWiseSparsity { get; set; } = false;

    /// <summary>
    /// Per-layer sparsity targets (layer name → sparsity).
    /// </summary>
    public Dictionary<string, double>? LayerSparsityTargets { get; set; }

    /// <summary>
    /// Per-category sparsity targets (category → sparsity).
    /// </summary>
    /// <remarks>
    /// <para>When <see cref="LayerWiseSparsity"/> is true and the model implements
    /// <see cref="ILayeredModel{T}"/>, these category-level sparsity targets are applied
    /// to all layers of that category. Layer-specific targets in <see cref="LayerSparsityTargets"/>
    /// take precedence over category targets.</para>
    ///
    /// <para>Example: Set attention layers to 30% sparsity (sensitive) while dense layers
    /// use 70% sparsity (more tolerant).</para>
    /// </remarks>
    public Dictionary<LayerCategory, double>? CategorySparsityTargets { get; set; }

    /// <summary>
    /// Whether to fine-tune after pruning.
    /// </summary>
    public bool FineTuneAfterPruning { get; set; } = true;

    /// <summary>
    /// Number of fine-tuning epochs after pruning.
    /// </summary>
    public int FineTuningEpochs { get; set; } = 10;

    /// <summary>
    /// Output sparse format for storage.
    /// </summary>
    public SparseFormat OutputFormat { get; set; } = SparseFormat.CSR;
}
