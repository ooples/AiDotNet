using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using One-Class SVM with simplified SGD training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> One-Class SVM finds a boundary that encompasses the normal data.
/// It learns the shape of normal data in a high-dimensional kernel space and flags points
/// outside this region as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Map data to kernel space using RBF kernel
/// 2. Find hyperplane that separates data from origin with maximum margin
/// 3. Points on the negative side of the hyperplane are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have only normal examples
/// - High-dimensional data
/// - When you need a decision boundary
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Nu: 0.1 (roughly proportion of outliers)
/// - Gamma: auto (1/n_features)
/// - Kernel: RBF
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution."
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SVM)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Estimating the Support of a High-Dimensional Distribution", "https://doi.org/10.1162/089976601750264965", Year = 2001, Authors = "Bernhard Scholkopf, John C. Platt, John Shawe-Taylor, Alex J. Smola, Robert C. Williamson")]
public partial class OCSVMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _nu;
    private readonly double _gamma;
    private readonly int _maxIterations;
    [Buffer]
    private Matrix<T>? _supportVectors;
    private Vector<T>? _alphas;

    private T _rho;


    /// <summary>
    /// Gets the nu parameter.
    /// </summary>
    public double Nu => _nu;

    /// <summary>
    /// Gets the gamma parameter.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Creates a new OCSVM anomaly detector.
    /// </summary>
    /// <param name="nu">Upper bound on outlier fraction. Default is 0.1.</param>
    /// <param name="gamma">RBF kernel parameter. -1 means auto (1/n_features). Default is -1.</param>
    /// <param name="maxIterations">Maximum training iterations. Default is 1000.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public OCSVMDetector(double nu = 0.1, double gamma = -1, int maxIterations = 1000,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _rho = NumOps.Zero;
        if (nu <= 0 || nu > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nu),
                "Nu must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.1.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _nu = nu;
        _gamma = gamma;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int d = X.Columns;
        T effectiveGamma = _gamma > 0
            ? NumOps.FromDouble(_gamma)
            : NumOps.Divide(NumOps.One, NumOps.FromDouble(d));

        // Train using simplified SMO-like algorithm
        TrainOCSVM(X, effectiveGamma);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void TrainOCSVM(Matrix<T> data, T gamma)
    {
        int n = data.Rows;

        // The one-class dual (Scholkopf et al., 2001) is
        //     minimize  ½·alphaᵀK·alpha    subject to  sum(alpha) = 1,  0 <= alpha_i <= 1/(nu·n)
        // with no linear term. Treating every point as carrying label +1 makes this an instance of
        // the shared SMO solver's problem: a step that moves one multiplier up and another down by
        // the same amount leaves sum(alpha) at 1, which is the constraint this formulation needs.
        //
        // This replaces an inlined loop that described itself as "SMO-like", chose its partner
        // multiplier at RANDOM, and applied a fixed fractional step of the exact update rather than
        // the exact update itself — so it neither converged to the optimum nor respected the
        // equality constraint, since clipping each multiplier independently after the step breaks
        // the sum.
        var engine = Engine;
        var points = AnomalyTensorOps.ToTensor(data);
        var kernel = RBFKernel(points, points, gamma);
        var K = kernel.ToMatrix();

        T maxAlpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(_nu * n));

        // Spread the unit mass evenly, capped at the per-multiplier bound, so the start is feasible.
        T evenShare = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));
        T even = NumOps.LessThan(maxAlpha, evenShare) ? maxAlpha : evenShare;
        var labels = engine.Fill(n, NumOps.One);
        var linear = new Vector<T>(n);          // one-class has no linear term
        var upperBounds = engine.Fill(n, maxAlpha);
        var initialAlphas = engine.Fill(n, even);

        var solver = new AiDotNet.Solvers.QuadraticProgramming.SequentialMinimalOptimizationSolver<T>(
            new AiDotNet.Models.Options.SequentialMinimalOptimizationOptions
            {
                MaxIterations = _maxIterations * Math.Max(1, n),
            });

        var (solved, _, _) = solver.Solve(
            (a, b) => K[a, b], labels, linear, upperBounds, initialAlphas);

        var alphas = solved;
        T eps8 = NumOps.FromDouble(1e-8);

        // Extract support vectors
        var svIndices = AnomalyTensorOps.NonzeroIndices(engine.TensorGreaterThan(alphas.AsTensor(), eps8));
        var supportAlphas = AnomalyTensorOps.SelectElements(alphas, svIndices);
        _alphas = supportAlphas;
        _supportVectors = AnomalyTensorOps.SelectRows(points, svIndices).ToMatrix();

        // Compute rho (threshold): the mean decision value sum_j alpha_j K(sv_j, sv_i) over the free
        // support vectors (eps8 < alpha_i < maxAlpha - eps8)
        int svCount = svIndices.Length;
        var svKernel = AnomalyTensorOps.SelectColumns(AnomalyTensorOps.SelectRows(kernel, svIndices), svIndices);
        var decision = engine.Reshape(
            engine.TensorMatMul(svKernel, engine.Reshape(supportAlphas.AsTensor(), new[] { svCount, 1 })), new[] { svCount });
        var free = engine.TensorMultiply(
            engine.TensorGreaterThan(supportAlphas.AsTensor(), eps8),
            engine.TensorLessThan(supportAlphas.AsTensor(), NumOps.Subtract(maxAlpha, eps8)));
        T count = engine.TensorSum(free);
        _rho = NumOps.GreaterThan(count, NumOps.Zero)
            ? NumOps.Divide(engine.TensorSum(engine.TensorMultiply(free, decision)), count)
            : NumOps.Zero;
    }

    // RBF kernel exp(-gamma * ||a_i - b_j||^2) between the rows of a and b.
    private Tensor<T> RBFKernel(Tensor<T> a, Tensor<T> b, T gamma)
    {
        var engine = Engine;
        return engine.TensorExp(engine.TensorNegate(
            engine.TensorMultiplyScalar(AnomalyTensorOps.PairwiseSquaredDistances(a, b), gamma)));
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var supportVectors = _supportVectors;
        var alphas = _alphas;

        if (supportVectors == null || alphas == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        T effectiveGamma = _gamma > 0
            ? NumOps.FromDouble(_gamma)
            : NumOps.Divide(NumOps.One, NumOps.FromDouble(X.Columns));
        var engine = Engine;
        int svCount = supportVectors.Rows;

        // Decision function: sum(alpha_k * K(sv_k, x)) - rho; higher (more positive) = more normal, so
        // it is negated to make higher = more anomalous
        var kernel = RBFKernel(AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(supportVectors), effectiveGamma);
        var decision = engine.Reshape(
            engine.TensorMatMul(kernel, engine.Reshape(alphas.AsTensor(), new[] { svCount, 1 })), new[] { X.Rows });
        return engine.TensorNegate(engine.TensorSubtractScalar(decision, _rho)).ToVector();
    }
}
