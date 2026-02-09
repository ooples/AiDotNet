using AiDotNet.ActivationFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for NODE (Neural Oblivious Decision Ensembles).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// NODE combines differentiable oblivious decision trees with neural network training:
/// 1. Oblivious trees: All nodes at the same depth use the same splitting feature
/// 2. Soft splits: Differentiable split decisions using entmax for sparse attention
/// 3. Ensemble: Multiple trees aggregated for the final prediction
/// </para>
/// <para>
/// <b>For Beginners:</b> NODE brings the interpretability of decision trees to deep learning:
///
/// - **Oblivious trees**: Simpler trees that are faster to evaluate
/// - **Soft splits**: Instead of hard left/right decisions, gradual transitions
/// - **End-to-end training**: Trees are trained with gradient descent like neural networks
///
/// This makes NODE both interpretable AND trainable with standard deep learning tools.
///
/// Example:
/// <code>
/// var options = new NODEOptions&lt;double&gt;
/// {
///     NumTrees = 20,
///     TreeDepth = 6,
///     OutputDimension = 1
/// };
/// var model = new NODERegression&lt;double&gt;(numFeatures: 10, options);
/// </code>
/// </para>
/// <para>
/// Reference: "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data" (2019)
/// </para>
/// </remarks>
public class NODEOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of trees in the ensemble.
    /// </summary>
    /// <value>The number of trees, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More trees generally mean better accuracy but slower training.
    /// Start with 20-50 trees for most tasks.
    /// </para>
    /// </remarks>
    public int NumTrees { get; set; } = 20;

    /// <summary>
    /// Gets or sets the depth of each tree.
    /// </summary>
    /// <value>The tree depth, defaulting to 6.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Depth determines how many splits each tree can make:
    /// - Depth 6 = 2^6 = 64 possible leaf nodes
    /// - Deeper trees can model more complex patterns but may overfit
    /// </para>
    /// </remarks>
    public int TreeDepth { get; set; } = 6;

    /// <summary>
    /// Gets or sets the output dimension for each tree.
    /// </summary>
    /// <value>The output dimension, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each leaf stores a vector of this size.
    /// Larger values give more expressiveness but cost more memory.
    /// </para>
    /// </remarks>
    public int TreeOutputDimension { get; set; } = 3;

    /// <summary>
    /// Gets or sets the hidden dimension for feature selection.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    public int FeatureSelectionDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the entmax alpha parameter for sparse attention.
    /// </summary>
    /// <value>The alpha parameter, defaulting to 1.5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls the sparsity of feature selection:
    /// - 1.0 = softmax (dense, no sparsity)
    /// - 1.5 = sparsemax-like (some features get exactly zero weight)
    /// - 2.0 = very sparse (fewer features selected)
    /// </para>
    /// </remarks>
    public double EntmaxAlpha { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the temperature for soft tree splits.
    /// </summary>
    /// <value>The temperature, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how "hard" the split decisions are:
    /// - Lower temperature → sharper, more discrete splits
    /// - Higher temperature → softer, more gradual splits
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0.</value>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use batch normalization on inputs.
    /// </summary>
    /// <value>True to use batch normalization; false otherwise. Defaults to true.</value>
    public bool UseBatchNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization scale for tree parameters.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.01.</value>
    public double InitScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to use feature preprocessing before trees.
    /// </summary>
    /// <value>True to use preprocessing; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Preprocessing transforms raw features before the trees,
    /// which can help with features that have different scales or distributions.
    /// </para>
    /// </remarks>
    public bool UseFeaturePreprocessing { get; set; } = true;

    /// <summary>
    /// Gets or sets the hidden dimensions for the optional MLP head.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to empty (direct output from trees).</value>
    public int[] MLPHiddenDimensions { get; set; } = [];

    /// <summary>
    /// Gets the number of leaf nodes per tree.
    /// </summary>
    public int NumLeaves => 1 << TreeDepth;  // 2^depth

    /// <summary>
    /// Gets or sets the hidden layer activation function for the MLP head.
    /// </summary>
    /// <value>The activation function, defaulting to ReLU.</value>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets the hidden layer vector activation function (alternative to scalar activation).
    /// </summary>
    /// <value>The vector activation function, or null to use scalar activation.</value>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }
}
