using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.SemiSupervised;

/// <summary>
/// Implements the Label Spreading algorithm for semi-supervised classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Label Spreading is a variant of Label Propagation that uses symmetric normalization
/// and a clamping factor (alpha) to balance between original labels and propagated information.
/// </para>
/// <para>
/// <b>For Beginners:</b> Label Spreading improves on Label Propagation in two ways:
///
/// 1. <b>Symmetric Normalization:</b> Instead of just normalizing rows, it normalizes both
///    rows and columns. This makes the algorithm more stable when you have clusters of
///    very different sizes. Think of it like making sure influence flows equally in both
///    directions between connected points.
///
/// 2. <b>Alpha Parameter:</b> This controls how much the original labels are preserved.
///    With alpha = 0.2, each iteration keeps 20% of the original label and takes 80% from
///    neighbors. This prevents the algorithm from "forgetting" the original labels while
///    still allowing information to spread.
///
/// The result is often more robust predictions, especially on noisy data or when clusters
/// have unequal sizes.
/// </para>
/// </remarks>
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// // Create label spreading with symmetric normalization
///
/// // Prepare data: -1 indicates unlabeled samples
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, -1, -1, 1, -1, -1 });
///
/// // Spread labels with alpha clamping to preserve original labels
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LabelSpreading&lt;double&gt;())
///     .Build(features, labels);
///
/// // Predict labels for all samples
/// var prediction = result.Predict(features);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Learning with Local and Global Consistency", "https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html", Year = 2003, Authors = "Dengyong Zhou, Olivier Bousquet, Thomas N. Lal, Jason Weston, Bernhard Scholkopf")]
public partial class LabelSpreading<T> : SemiSupervisedClassifierBase<T>
{

    /// <inheritdoc />
    /// <remarks>The spread label distributions, as LabelPropagation. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
    #region Fields

    /// <summary>
    /// The kernel function used to compute similarity between samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel function measures how similar two data points are.
    /// A common choice is the RBF (Radial Basis Function) kernel, which gives high similarity
    /// to nearby points and low similarity to distant points.
    /// </para>
    /// </remarks>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// Maximum number of iterations for label spreading.
    /// </summary>
    private int _maxIterations;

    /// <summary>
    /// Convergence tolerance for stopping the spreading early.
    /// </summary>
    private T _tolerance;

    /// <summary>
    /// The clamping factor (alpha) that balances original labels vs. propagated information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Alpha controls how much the original labels "persist" through iterations.
    /// - alpha = 0: Completely ignores neighbors, only uses original labels (essentially memorization)
    /// - alpha = 1: Completely ignores original labels, only uses neighbors (pure propagation)
    /// - alpha = 0.2 (default): Keeps 20% of original labels, takes 80% from neighbors
    ///
    /// Lower alpha values are more conservative and stay closer to the original labeled examples.
    /// Higher values allow more influence from the graph structure.
    /// </para>
    /// </remarks>
    private T _alpha;

    /// <summary>
    /// The symmetrically normalized affinity matrix (Laplacian-style normalization).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _normalizedAffinity = new Matrix<T>(0, 0);

    /// <summary>
    /// The combined feature matrix (labeled + unlabeled) after training.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _allFeatures = new Matrix<T>(0, 0);

    /// <summary>
    /// The label distribution matrix where each row is a sample and each column is a class.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _labelDistributions = new Matrix<T>(0, 0);

    /// <summary>
    /// The initial label distributions (before propagation).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _initialDistributions = new Matrix<T>(0, 0);

    /// <summary>
    /// Number of labeled samples stored during training.
    /// </summary>
    private int _numLabeled;

    /// <summary>
    /// Random number generator for tie-breaking.
    /// </summary>
    private readonly Random _random;

    /// <summary>The seed this model was built with, kept so a clone can be built the same way.</summary>
    /// <remarks>
    /// The Random built from it cannot be read back, so without this the seed is gone the
    /// moment the constructor returns and the model cannot be rebuilt from its own state.
    /// </remarks>
    private readonly int? _seed;


    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the LabelSpreading class with default settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a Label Spreading classifier with sensible defaults:
    /// - RBF kernel for measuring similarity
    /// - Up to 30 iterations of spreading
    /// - Alpha = 0.2 (keeps 20% original labels, 80% from neighbors)
    /// </para>
    /// </remarks>
    public LabelSpreading()
    {
        _kernel = CreateDefaultKernel();
        _maxIterations = 30;
        _tolerance = NumOps.FromDouble(1e-3);
        _alpha = NumOps.FromDouble(0.2);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the LabelSpreading class with specified parameters.
    /// </summary>
    /// <param name="kernel">The kernel function for computing sample similarities. If null, uses RBF kernel.</param>
    /// <param name="maxIterations">Maximum number of spreading iterations. Default is 30.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-3.</param>
    /// <param name="alpha">Clamping factor (0 to 1). Default is 0.2.</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses cryptographically secure random.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you customize how the algorithm works:
    /// - kernel: How to measure similarity between data points
    /// - maxIterations: How many times to spread labels before stopping
    /// - tolerance: How small the changes need to be to stop early
    /// - alpha: How much to trust neighbors vs. original labels (higher = trust neighbors more)
    /// - seed: A number for reproducible results
    /// </para>
    /// </remarks>
    public LabelSpreading(
        IKernelFunction<T>? kernel,
        int maxIterations,
        T tolerance,
        T alpha,
        int? seed)
    {
        _kernel = kernel ?? CreateDefaultKernel();
        _maxIterations = maxIterations;

        // Accept provided tolerance as-is - zero is valid (means run until maxIterations)
        _tolerance = tolerance;

        // Accept provided alpha as-is - zero is valid (means keep original labels, no spreading)
        _alpha = alpha;

        _seed = seed;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Validate alpha is in [0, 1]
        if (NumOps.Compare(_alpha, NumOps.Zero) < 0 || NumOps.Compare(_alpha, NumOps.One) > 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be between 0 and 1.");
        }
    }

    #endregion

    #region Semi-Supervised Training

    /// <summary>
    /// Core implementation of semi-supervised training using label spreading.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the actual work of learning from both labeled
    /// and unlabeled data. It:
    /// 1. Combines all data into one matrix
    /// 2. Builds a similarity graph with symmetric normalization
    /// 3. Initializes labels (known for labeled samples, uniform for unlabeled)
    /// 4. Repeatedly spreads labels while keeping some of the original information
    /// </para>
    /// </remarks>
    protected override void TrainSemiSupervisedCore(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        _numLabeled = labeledX.Rows;
        int totalSamples = labeledX.Rows + unlabeledX.Rows;

        // Combine labeled and unlabeled data
        _allFeatures = CombineData(labeledX, unlabeledX);

        // Build and normalize affinity matrix using symmetric normalization
        var affinity = BuildAffinityMatrix(_allFeatures);
        _normalizedAffinity = SymmetricNormalize(affinity);

        // Initialize label distributions
        _initialDistributions = InitializeLabelDistributions(labeledY, totalSamples);
        _labelDistributions = _initialDistributions.Clone();

        // Spread labels
        SpreadLabels();

        // Extract pseudo-labels for unlabeled data
        ExtractPseudoLabels();
    }

    /// <summary>
    /// Core implementation of supervised training (using only labeled data).
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you only have labeled data, Label Spreading stores the
    /// training data for later prediction using similarity-weighted voting.
    /// </para>
    /// </remarks>
    protected override void TrainSupervisedCore(Matrix<T> x, Vector<T> y)
    {
        _numLabeled = x.Rows;
        _allFeatures = x;
        var affinity = BuildAffinityMatrix(x);
        _normalizedAffinity = SymmetricNormalize(affinity);
        _initialDistributions = InitializeLabelDistributions(y, x.Rows);
        _labelDistributions = _initialDistributions.Clone();
    }

    #endregion

    #region Graph Construction

    /// <summary>
    /// Combines labeled and unlabeled data into a single feature matrix.
    /// </summary>
    /// <param name="labeledX">The labeled feature matrix.</param>
    /// <param name="unlabeledX">The unlabeled feature matrix.</param>
    /// <returns>A combined matrix with labeled samples first, then unlabeled.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This puts all your data together into one big table.
    /// The labeled examples come first, followed by the unlabeled examples.
    /// </para>
    /// </remarks>
    private Matrix<T> CombineData(Matrix<T> labeledX, Matrix<T> unlabeledX)
    {
        if (unlabeledX.Rows == 0)
        {
            return labeledX.Clone();
        }

        if (labeledX.Rows == 0)
        {
            return unlabeledX.Clone();
        }

        return Engine.TensorConcatenate(
            new[] { Tensor<T>.FromMatrix(labeledX), Tensor<T>.FromMatrix(unlabeledX) }, 0).ToMatrix();
    }

    /// <summary>
    /// Builds the affinity (similarity) matrix using the kernel function.
    /// </summary>
    /// <param name="features">The combined feature matrix.</param>
    /// <returns>A symmetric matrix of pairwise similarities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The affinity matrix is like a table showing how similar
    /// each pair of samples is. High values mean the samples are similar.
    /// </para>
    /// </remarks>
    private Matrix<T> BuildAffinityMatrix(Matrix<T> features)
    {
        // Self-similarity is 0 (no self-loops in the graph); the matrix is symmetric
        return KernelSimilarities(_kernel, features, features, true).ToMatrix();
    }

    /// <summary>
    /// Applies symmetric normalization to the affinity matrix.
    /// </summary>
    /// <param name="affinity">The affinity matrix.</param>
    /// <returns>The symmetrically normalized affinity matrix D^(-1/2) * W * D^(-1/2).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Symmetric normalization is a key difference from Label Propagation.
    /// Instead of just making rows sum to 1, it normalizes both rows and columns equally.
    ///
    /// The formula is: S = D^(-1/2) * W * D^(-1/2)
    /// where D is the degree matrix (sum of each row) and W is the affinity matrix.
    ///
    /// This makes the influence between two points equal in both directions, which is
    /// more mathematically principled and leads to more stable results.
    /// </para>
    /// </remarks>
    private Matrix<T> SymmetricNormalize(Matrix<T> affinity)
    {
        var engine = Engine;
        int n = affinity.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        var weights = Tensor<T>.FromMatrix(affinity);

        // Degree (row sums) and D^(-1/2), with 0 for non-positive degrees
        var degrees = engine.ReduceSum(weights, new[] { 1 }, false);
        var positive = engine.TensorGreaterThan(degrees, NumOps.Zero);
        var ones = new Tensor<T>(new[] { n });
        engine.TensorFill(ones, NumOps.One);
        var zeros = new Tensor<T>(new[] { n });
        var invSqrt = engine.TensorDivide(ones, engine.TensorSqrt(engine.TensorWhere(positive, degrees, ones)));
        var invSqrtDegrees = engine.TensorWhere(positive, invSqrt, zeros);

        // D^(-1/2) * W * D^(-1/2)
        var scaledRows = engine.TensorMultiply(invSqrtDegrees.Reshape(n, 1), weights);
        return engine.TensorMultiply(scaledRows, invSqrtDegrees.Reshape(1, n)).ToMatrix();
    }

    #endregion

    #region Label Spreading

    /// <summary>
    /// Initializes the label distribution matrix.
    /// </summary>
    /// <param name="labeledY">The known labels for labeled samples.</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <returns>A matrix where each row is a sample and each column is a class probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the starting point for label spreading.
    /// Labeled samples get probability 1 for their known class and 0 for others.
    /// Unlabeled samples start with equal probability for all classes.
    /// </para>
    /// </remarks>
    private Matrix<T> InitializeLabelDistributions(Vector<T> labeledY, int totalSamples)
    {
        // Labeled samples are one-hot; unlabeled samples start uniform
        return InitialLabelDistributions(labeledY, totalSamples);
    }

    /// <summary>
    /// Spreads labels through the graph iteratively with clamping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the heart of Label Spreading. In each iteration:
    /// 1. Spread labels through the graph (like Label Propagation)
    /// 2. Mix the result with the original labels using alpha: new = alpha * original + (1-alpha) * spread
    /// 3. Normalize so probabilities sum to 1
    /// 4. Check if changes are small enough to stop
    ///
    /// The alpha mixing ensures that labeled samples don't completely forget their original
    /// labels, while still allowing information to flow through the graph.
    /// </para>
    /// </remarks>
    private void SpreadLabels()
    {
        var engine = Engine;
        T oneMinusAlpha = NumOps.Subtract(NumOps.One, _alpha);
        var affinity = Tensor<T>.FromMatrix(_normalizedAffinity);
        var clamped = engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(_initialDistributions), _alpha);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var previous = Tensor<T>.FromMatrix(_labelDistributions);

            // Spread: Y = S @ Y
            var spread = engine.TensorMatMul(affinity, previous);

            // Clamp: Y = alpha * Y_0 + (1 - alpha) * spread, then normalize rows to sum to 1
            var updated = NormalizeRowSums(engine.TensorAdd(clamped, engine.TensorMultiplyScalar(spread, oneMinusAlpha)), false);
            _labelDistributions = updated.ToMatrix();

            // Check convergence
            if (NumOps.Compare(MaxAbsoluteChange(updated, previous), _tolerance) < 0)
            {
                break;
            }
        }
    }

    /// <summary>
    /// Extracts pseudo-labels from the converged label distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After spreading finishes, each unlabeled sample has a probability
    /// distribution. This method picks the most likely class as the pseudo-label and records
    /// the confidence.
    /// </para>
    /// </remarks>
    private void ExtractPseudoLabels()
    {
        SetPseudoLabelsFromDistributions(_labelDistributions, _numLabeled, "Model has not been fitted.");
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts class labels for new samples.
    /// </summary>
    /// <param name="input">The feature matrix of samples to classify.</param>
    /// <returns>A vector of predicted class labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For new samples, we compute their similarity to training samples
    /// and use weighted voting to predict the class.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_allFeatures.Rows == 0 || _labelDistributions.Rows == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        // Weighted label distribution from normalized similarities, then the best class per row
        var distributions = PropagatedDistributions(_kernel, input, _allFeatures, _labelDistributions);
        return ClassIndicesToLabels(ArgMaxRows(distributions));
    }

    /// <summary>
    /// Predicts class probabilities for each sample.
    /// </summary>
    /// <param name="input">The feature matrix of samples.</param>
    /// <returns>A matrix where each row is a sample and each column is a class probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the full probability distribution over classes
    /// for each sample, useful when you need confidence information.
    /// </para>
    /// </remarks>
    public Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_allFeatures.Rows == 0 || _labelDistributions.Rows == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        return PropagatedDistributions(_kernel, input, _allFeatures, _labelDistributions);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates the default kernel (RBF/Gaussian).
    /// </summary>
    /// <returns>An RBF kernel with default parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The RBF kernel is a good default choice that works well
    /// for most datasets.
    /// </para>
    /// </remarks>
    private IKernelFunction<T> CreateDefaultKernel()
    {
        return new Kernels.GaussianKernel<T>(1.0);
    }

    #endregion

    #region Serialization

    #endregion

    #region ICloneable Implementation

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Gets all learnable parameters of the model as a single vector.
    /// </summary>
    /// <returns>An empty vector as Label Spreading is non-parametric.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike neural networks that have weights to learn, Label Spreading
    /// is a non-parametric method. It stores all training data and computes similarities at
    /// prediction time, so there are no "learned" parameters in the traditional sense.
    /// </para>
    /// </remarks>
    private Vector<T> PackParameters()
    {
        // Label Spreading is non-parametric - it stores training data, not learned weights
        return new Vector<T>(0);
    }

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use (ignored for non-parametric models).</param>
    /// <returns>A new instance of the classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Spreading doesn't have learnable parameters,
    /// this just creates a new instance with the same configuration.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new LabelSpreading<T>(_kernel, _maxIterations, _tolerance, _alpha, _random.Next());
    }

    /// <summary>
    /// Sets the parameters of this model.
    /// </summary>
    /// <param name="parameters">The parameters to set (ignored for non-parametric models).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Spreading doesn't have learnable parameters,
    /// this method does nothing.
    /// </para>
    /// </remarks>
    private void UnpackParameters(Vector<T> parameters)
    {
        // Non-parametric model - no parameters to set
    }

    #endregion
}
