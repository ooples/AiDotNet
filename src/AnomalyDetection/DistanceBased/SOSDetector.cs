using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using SOS (Stochastic Outlier Selection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SOS is based on the concept of affinity - how likely a point is to be
/// selected as a neighbor by other points. If a point has low affinity (other points rarely
/// select it as a neighbor), it is likely an outlier.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute pairwise affinities using Gaussian kernel
/// 2. Normalize affinities (like t-SNE probabilities)
/// 3. Compute binding probability for each point
/// 4. Points with low binding probability are outliers
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you want a probabilistic interpretation of outlierness
/// - Medium-sized datasets
/// - When local density variations exist
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Perplexity: 30 (similar to t-SNE)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Janssens, J.H.M., et al. (2012). "Stochastic Outlier Selection." Technical Report.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Stochastic Outlier Selection", "https://jmlr.org/papers/v14/janssens13a.html")]
public partial class SOSDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _perplexity;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Vector<T>? _trainingBindingProbs;

    /// <summary>
    /// Gets the perplexity parameter.
    /// </summary>
    public double Perplexity => _perplexity;

    /// <summary>
    /// Creates a new SOS anomaly detector.
    /// </summary>
    /// <param name="perplexity">
    /// Perplexity parameter (similar to t-SNE). Controls the effective number of neighbors.
    /// Default is 30.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SOSDetector(double perplexity = 30, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (perplexity < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(perplexity),
                "Perplexity must be at least 1. Recommended is 30.");
        }

        _perplexity = perplexity;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _trainingData = X;

        // Compute binding probabilities for training data
        _trainingBindingProbs = ComputeBindingProbabilities(X);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Vector<T> ComputeBindingProbabilities(Matrix<T> data)
    {
        int n = data.Rows;
        var engine = Engine;

        // Squared distances; the diagonal never takes part (affinities a_ii stay 0)
        var sqDistances = AnomalyTensorOps.PairwiseSquaredDistances(AnomalyTensorOps.ToTensor(data), AnomalyTensorOps.ToTensor(data));
        var offDiagonal = engine.ScalarMinusTensor(NumOps.One, engine.TensorEye<T>(n));

        // Binary search, for every point at once, for the sigma that gives the target perplexity
        T targetEntropy = NumOps.Log(NumOps.FromDouble(_perplexity));
        T eps10 = NumOps.FromDouble(1e-10);
        T eps5 = NumOps.FromDouble(1e-5);
        T two = NumOps.FromDouble(2);
        T sigmaMaxInit = NumOps.FromDouble(1e20);
        var zeros = Tensor<T>.CreateZeros(new[] { n });
        var ones = Tensor<T>.CreateOnes(new[] { n });
        var sigmaMin = engine.TensorAddScalar(zeros, NumOps.FromDouble(1e-20));
        var sigmaMax = engine.TensorAddScalar(zeros, sigmaMaxInit);
        var sigma = ones;
        var evaluatedSigma = ones;
        var active = ones;

        for (int iter = 0; iter < 50; iter++)
        {
            // Affinities and entropy with the current sigma (rows that converged keep their last sigma)
            evaluatedSigma = engine.TensorWhere(active, sigma, evaluatedSigma);
            var affinities = Affinities(sqDistances, offDiagonal, sigma);
            var sumAff = engine.ReduceSum(affinities, new[] { 1 }, false);
            var normalizable = engine.TensorGreaterThan(sumAff, eps10);
            var p = AnomalyTensorOps.DivideColumn(affinities, engine.TensorWhere(normalizable, sumAff, ones));
            var significant = engine.TensorGreaterThan(p, eps10);
            var pLogP = engine.TensorMultiply(significant,
                engine.TensorMultiply(p, engine.TensorLog(engine.TensorWhere(significant, p, engine.TensorAddScalar(engine.TensorMultiplyScalar(p, NumOps.Zero), NumOps.One)))));
            var entropy = engine.TensorWhere(normalizable,
                engine.TensorNegate(engine.ReduceSum(pLogP, new[] { 1 }, false)), zeros);

            // Converged rows stop; the others move their bracket
            var converged = engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtractScalar(entropy, targetEntropy)), eps5);
            active = engine.TensorMultiply(active, engine.ScalarMinusTensor(NumOps.One, converged));
            var raise = engine.TensorMultiply(active, engine.TensorLessThan(entropy, targetEntropy));
            var lower = engine.TensorSubtract(active, raise);
            var unbounded = engine.TensorEquals(sigmaMax, engine.TensorAddScalar(zeros, sigmaMaxInit));
            var raised = engine.TensorWhere(unbounded,
                engine.TensorMultiplyScalar(sigma, two),
                engine.TensorDivideScalar(engine.TensorAdd(sigma, sigmaMax), two));
            var lowered = engine.TensorDivideScalar(engine.TensorAdd(sigma, sigmaMin), two);
            sigmaMin = engine.TensorWhere(raise, sigma, sigmaMin);
            sigmaMax = engine.TensorWhere(lower, sigma, sigmaMax);
            sigma = engine.TensorWhere(raise, raised, engine.TensorWhere(lower, lowered, sigma));
        }

        // Affinities of the last evaluated sigma, normalized per row when their sum exceeds 1e-10
        var finalAffinities = Affinities(sqDistances, offDiagonal, evaluatedSigma);
        var sum = engine.ReduceSum(finalAffinities, new[] { 1 }, false);
        var normalize = engine.TensorGreaterThan(sum, eps10);
        finalAffinities = AnomalyTensorOps.DivideColumn(finalAffinities, engine.TensorWhere(normalize, sum, ones));

        // Binding probability p_j = product over i != j of (1 - a_ij): the probability that j is NOT
        // selected by ANY point (outlier score), computed in log space
        var logTerms = engine.TensorMultiply(offDiagonal,
            engine.TensorLog(engine.TensorAddScalar(engine.ScalarMinusTensor(NumOps.One, finalAffinities), eps10)));
        return engine.TensorExp(engine.ReduceSum(logTerms, new[] { 0 }, false)).ToVector();
    }

    // a_ij = exp(-d_ij^2 / (2 sigma_i^2)) off the diagonal, 0 on it.
    private Tensor<T> Affinities(Tensor<T> sqDistances, Tensor<T> offDiagonal, Tensor<T> sigma)
    {
        var engine = Engine;
        var twoSigmaSq = engine.TensorMultiplyScalar(engine.TensorMultiply(sigma, sigma), NumOps.FromDouble(2));
        var affinities = engine.TensorExp(engine.TensorNegate(AnomalyTensorOps.DivideColumn(sqDistances, twoSigmaSq)));
        return engine.TensorMultiply(affinities, offDiagonal);
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

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // For scoring, we need to include the new points in the affinity computation
        int nTrain = trainingData.Rows;
        int nTest = X.Rows;
        int d = X.Columns;

        if (trainingData.Columns != d)
        {
            throw new ArgumentException($"Training data has {trainingData.Columns} features but test data has {d}.", nameof(X));
        }

        // Combine training and test data, compute binding probabilities for all of it and keep the test points
        var engine = Engine;
        var allData = engine.TensorConcatenate(
            new[] { AnomalyTensorOps.ToTensor(trainingData), AnomalyTensorOps.ToTensor(X) }, 0);
        var allBindingProbs = ComputeBindingProbabilities(allData.ToMatrix());
        return engine.TensorSlice(allBindingProbs.AsTensor(), new[] { nTrain }, new[] { nTest }).ToVector();
    }
}
