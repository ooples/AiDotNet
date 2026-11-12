using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using Gaussian Mixture Models (GMM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GMM models data as a mixture of several Gaussian distributions.
/// Points with low probability under this model are anomalies - they don't fit well
/// into any of the learned Gaussian clusters.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Fit a mixture of k Gaussians using Expectation-Maximization
/// 2. For each point, compute its probability under the mixture model
/// 3. Points with low probability are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data is a mixture of Gaussian clusters
/// - Different clusters have different shapes/sizes
/// - You need probabilistic anomaly scores
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Components: 3 (typical range 2-10)
/// - Covariance type: diagonal (uses variance per feature for stability)
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Pattern Recognition and Machine Learning", "https://www.springer.com/gp/book/9780387310732")]
public partial class GMMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nComponents;
    private readonly int _maxIterations;
    // Component means and (diagonal) variances, [nComponents, nFeatures]; the densities use the
    // diagonal approximation only
    private Tensor<T> _means = Tensor<T>.CreateZeros(new[] { 0, 0 });
    private Tensor<T> _variances = Tensor<T>.CreateZeros(new[] { 0, 0 });
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _weights = new Vector<T>(0);
    private int _nFeatures;
    [Buffer]
    private Vector<T> _globalVariance = new Vector<T>(0);

    /// <summary>
    /// Gets the number of Gaussian components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Creates a new GMM anomaly detector.
    /// </summary>
    /// <param name="nComponents">Number of Gaussian components. Default is 3.</param>
    /// <param name="maxIterations">Maximum EM iterations. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GMMDetector(int nComponents = 3, int maxIterations = 100,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nComponents < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nComponents),
                "NComponents must be at least 1. Recommended is 3.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _nComponents = nComponents;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;

        if (n < _nComponents)
        {
            throw new ArgumentException(
                $"Number of samples ({n}) must be at least nComponents ({_nComponents}).",
                nameof(X));
        }

        // Initialize GMM using k-means
        InitializeParameters(X);

        // Run EM algorithm
        RunEM(X);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        // Anomaly score = negative log-likelihood under the mixture model (log-likelihood -1000 where the
        // likelihood underflows to 0)
        var engine = Engine;
        var likelihood = engine.ReduceSum(WeightedDensities(AnomalyTensorOps.ToTensor(X)), new[] { 1 }, false);
        var positive = engine.TensorGreaterThan(likelihood, NumOps.Zero);
        var logLikelihood = engine.TensorWhere(positive,
            engine.TensorLog(engine.TensorWhere(positive, likelihood, Tensor<T>.CreateOnes(new[] { X.Rows }))),
            engine.Fill(X.Rows, NumOps.FromDouble(-1000)).AsTensor());
        return engine.TensorNegate(logLikelihood).ToVector();
    }

    private void InitializeParameters(Matrix<T> X)
    {
        var engine = Engine;
        int n = X.Rows;
        int d = X.Columns;
        var random = RandomHelper.CreateSeededRandom(_randomSeed);
        var data = AnomalyTensorOps.ToTensor(X);

        // Compute global (population) variance per feature, at least 1e-6, for use as a covariance floor.
        // This prevents components with very few points (e.g., a singleton outlier)
        // from collapsing to a tiny covariance that produces astronomically high density.
        var featureMeans = engine.ReduceMean(data, new[] { 0 }, false);
        var variance = engine.ReduceMean(engine.TensorSquare(AnomalyTensorOps.SubtractRow(data, featureMeans)), new[] { 0 }, false);
        var minVariance = NumOps.FromDouble(1e-6);
        _globalVariance = engine.TensorWhere(engine.TensorGreaterThan(variance, minVariance), variance,
            engine.Fill(d, minVariance).AsTensor()).ToVector();

        // Select initial means randomly (distinct points)
        var selectedIndices = new HashSet<int>();
        var initial = new int[_nComponents];
        for (int c = 0; c < _nComponents; c++)
        {
            int idx;
            do
            {
                idx = random.Next(n);
            } while (selectedIndices.Contains(idx));

            selectedIndices.Add(idx);
            initial[c] = idx;
        }

        _means = AnomalyTensorOps.SelectRows(data, initial);

        // Initialize covariance as identity and equal weights
        _variances = Tensor<T>.CreateOnes(new[] { _nComponents, d });
        _weights = engine.Fill(_nComponents, NumOps.FromDouble(1.0 / _nComponents));
    }

    private void RunEM(Matrix<T> X)
    {
        var engine = Engine;
        int n = X.Rows;
        int k = _nComponents;
        var data = AnomalyTensorOps.ToTensor(X);
        var uniform = Tensor<T>.CreateOnes(new[] { n, k });
        T uniformResp = NumOps.Divide(NumOps.One, NumOps.FromDouble(k));
        T ncEps = NumOps.FromDouble(1e-6);
        var minVar = VarianceFloor();

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // E-step: Compute responsibilities (uniform where the total probability is not positive)
            var probs = WeightedDensities(data);
            var totalProb = engine.ReduceSum(probs, new[] { 1 }, false);
            var positive = engine.TensorGreaterThan(totalProb, NumOps.Zero);
            var safeTotal = engine.TensorWhere(positive, totalProb, Tensor<T>.CreateOnes(new[] { n }));
            var responsibilities = engine.TensorWhere(AnomalyTensorOps.BroadcastColumn(positive, k),
                AnomalyTensorOps.DivideColumn(probs, safeTotal), engine.TensorMultiplyScalar(uniform, uniformResp));

            // M-step: Update parameters. Weights Nc / n; means (where Nc > 0) the responsibility-weighted
            // averages; then, where Nc > 1e-6, the responsibility-weighted variances around the new means,
            // floored at a fraction of the global variance. This prevents components with very few points
            // from having extremely small variance, which would cause astronomically high density and
            // break anomaly scoring.
            var resT = engine.TensorTranspose(responsibilities);
            var Nc = engine.ReduceSum(responsibilities, new[] { 0 }, false);
            _weights = engine.TensorDivideScalar(Nc, NumOps.FromDouble(n)).ToVector();

            var hasMass = engine.TensorGreaterThan(Nc, NumOps.Zero);
            var safeNc = engine.TensorWhere(hasMass, Nc, Tensor<T>.CreateOnes(new[] { k }));
            var newMeans = AnomalyTensorOps.DivideColumn(engine.TensorMatMul(resT, data), safeNc);
            _means = engine.TensorWhere(AnomalyTensorOps.BroadcastColumn(hasMass, _nFeatures), newMeans, _means);

            var squaredDiff = engine.TensorSquare(Differences(data));
            var weighted = engine.TensorMultiply(squaredDiff,
                engine.TensorBroadcastTo(engine.Reshape(responsibilities, new[] { n, k, 1 }), new[] { n, k, _nFeatures }));
            var newVariances = AnomalyTensorOps.DivideColumn(engine.ReduceSum(weighted, new[] { 0 }, false), safeNc);
            newVariances = engine.TensorWhere(engine.TensorLessThan(newVariances, minVar), minVar, newVariances);
            var enoughMass = engine.TensorGreaterThan(Nc, ncEps);
            _variances = engine.TensorWhere(AnomalyTensorOps.BroadcastColumn(enoughMass, _nFeatures), newVariances, _variances);
        }
    }

    // x_i - mean_c for every point and component, [n, nComponents, nFeatures].
    private Tensor<T> Differences(Tensor<T> data)
    {
        var engine = Engine;
        int n = data.Shape[0];
        int k = _nComponents;
        int d = _nFeatures;
        return engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(data, new[] { n, 1, d }), new[] { n, k, d }),
            engine.TensorBroadcastTo(engine.Reshape(_means, new[] { 1, k, d }), new[] { n, k, d }));
    }

    // Variance floor per component and feature: max(0.01 * global variance, 1e-6), [nComponents, nFeatures].
    private Tensor<T> VarianceFloor()
    {
        var engine = Engine;
        T minFloor = NumOps.FromDouble(1e-6);
        var varFloor = engine.TensorMultiplyScalar(_globalVariance.AsTensor(), NumOps.FromDouble(0.01));
        var floor = engine.TensorWhere(engine.TensorGreaterThan(varFloor, minFloor), varFloor,
            engine.Fill(_nFeatures, minFloor).AsTensor());
        return AnomalyTensorOps.BroadcastRow(floor, _nComponents);
    }

    // weight_c * N(x_i | mean_c, diag(variance_c)) for every point and component ([n, nComponents]), with the
    // variances floored at the global-variance floor (diagonal approximation for stability).
    private Tensor<T> WeightedDensities(Tensor<T> data)
    {
        var engine = Engine;
        int n = data.Shape[0];
        int k = _nComponents;
        int d = _nFeatures;
        T half = NumOps.FromDouble(0.5);
        T constant = NumOps.Multiply(NumOps.FromDouble(d), NumOps.FromDouble(Math.Log(2 * Math.PI)));

        var minVar = VarianceFloor();
        var variances = engine.TensorWhere(engine.TensorLessThan(_variances, minVar), minVar, _variances);
        var logDet = engine.ReduceSum(engine.TensorLog(variances), new[] { 1 }, false);
        var mahalanobis = engine.ReduceSum(engine.TensorDivide(engine.TensorSquare(Differences(data)),
            engine.TensorBroadcastTo(engine.Reshape(variances, new[] { 1, k, d }), new[] { n, k, d })), new[] { 2 }, false);
        var logDensity = engine.TensorNegate(engine.TensorMultiplyScalar(
            engine.TensorAdd(AnomalyTensorOps.BroadcastRow(engine.TensorAddScalar(logDet, constant), n), mahalanobis), half));
        return AnomalyTensorOps.MultiplyRow(engine.TensorExp(logDensity), _weights.AsTensor());
    }
}
