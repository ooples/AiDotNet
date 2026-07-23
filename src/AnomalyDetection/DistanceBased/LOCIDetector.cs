using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Local Correlation Integral (LOCI).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LOCI is a density-based outlier detection method that automatically
/// determines the appropriate neighborhood size. It computes a Multi-Granularity Deviation
/// Factor (MDEF) that indicates how much a point's local density deviates from its
/// neighborhood's density.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For multiple radius values, compute local density
/// 2. Compare point's density to average density of neighbors (MDEF)
/// 3. Flag points with MDEF exceeding a threshold based on standard deviation
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you don't want to manually tune neighborhood size
/// - Data has varying local densities
/// - You want automatic threshold determination
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha: 0.5 (neighborhood multiplier)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Papadimitriou, S., et al. (2003). "LOCI: Fast Outlier Detection Using
/// the Local Correlation Integral." ICDE.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("LOCI: Fast Outlier Detection Using the Local Correlation Integral", "https://doi.org/10.1109/ICDE.2003.1260802", Year = 2003, Authors = "Spiros Papadimitriou, Hiroyuki Kitagawa, Phillip B. Gibbons, Christos Faloutsos")]
public class LOCIDetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>Number of radius steps used to sweep from zero to <c>_maxRadius</c>.</summary>
    private const int NumRadiiSteps = 20;

    /// <summary>Multiplier applied to <c>_maxRadius</c> so the sweep slightly exceeds it, avoiding floating-point edge cases.</summary>
    private const double RadiusOvershootFactor = 1.1;

    /// <summary>MDEF score assigned to points that have counting neighbors but zero sampling neighbors, indicating extreme isolation.</summary>
    private const double IsolatedPointScore = 1.0;

    /// <summary>Score for points with no neighbors at any radius. Chosen to be within Half max (~65504).</summary>
    private const double NoNeighborScore = 60000.0;

    private readonly double _alpha;
    private readonly int _kMax;
    private Matrix<T>? _trainingData;

    private T _maxRadius;


    /// <summary>
    /// Gets the alpha parameter (sampling neighborhood ratio).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the maximum number of neighbors to consider.
    /// </summary>
    public int KMax => _kMax;

    /// <summary>
    /// Creates a new LOCI anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// Neighborhood multiplier. The sampling neighborhood radius is alpha times the counting radius.
    /// Default is 0.5. Must be between 0 and 1.
    /// </param>
    /// <param name="kMax">Maximum number of neighbors to consider. Default is 20.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LOCIDetector(double alpha = 0.5, int kMax = 20, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _maxRadius = NumOps.Zero;
        if (alpha <= 0 || alpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 (exclusive) and 1 (inclusive). Recommended value is 0.5.");
        }

        if (kMax < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kMax),
                "KMax must be at least 1. Recommended value is 20.");
        }

        _alpha = alpha;
        _kMax = kMax;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < 3)
        {
            throw new ArgumentException(
                "LOCI requires at least 3 samples.",
                nameof(X));
        }

        _trainingData = X;

        // Estimate max radius from data
        _maxRadius = EstimateMaxRadius(X);

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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T maxMdef = NumOps.Zero;
            bool hadNeighbors = false;

            // Test multiple radii (use more steps for better resolution)
            for (int ri = 1; ri <= NumRadiiSteps; ri++)
            {
                // Use slightly larger than _maxRadius to handle floating-point edge cases
                T r = NumOps.Divide(
                    NumOps.Multiply(NumOps.Multiply(_maxRadius, NumOps.FromDouble(RadiusOvershootFactor)), NumOps.FromDouble(ri)),
                    NumOps.FromDouble(NumRadiiSteps));
                T alphaR = NumOps.Multiply(NumOps.FromDouble(_alpha), r);

                // Count points in r-neighborhood (counting neighborhood)
                int nR = CountPointsInRadius(X, i, _trainingData ?? throw new InvalidOperationException("Model not properly fitted."), r);

                if (nR < 2) continue;
                hadNeighbors = true;

                // Get points in alpha*r neighborhood (sampling neighborhood)
                var samplingNeighbors = GetPointsInRadius(X, i, _trainingData, alphaR);

                if (samplingNeighbors.Count == 0)
                {
                    T isolatedScore = NumOps.FromDouble(IsolatedPointScore);
                    if (NumOps.GreaterThan(isolatedScore, maxMdef)) maxMdef = isolatedScore;
                    continue;
                }

                // Compute average n(r) of neighbors in sampling neighborhood
                T avgNR = NumOps.Zero;

                foreach (int neighborIdx in samplingNeighbors)
                {
                    int neighborNR = CountPointsInRadiusFromTraining(neighborIdx, r);
                    avgNR = NumOps.Add(avgNR, NumOps.FromDouble(neighborNR));
                }
                avgNR = NumOps.Divide(avgNR, NumOps.FromDouble(samplingNeighbors.Count));

                // Compute variance
                T varNR = NumOps.Zero;
                foreach (int neighborIdx in samplingNeighbors)
                {
                    int neighborNR = CountPointsInRadiusFromTraining(neighborIdx, r);
                    T diff = NumOps.Subtract(NumOps.FromDouble(neighborNR), avgNR);
                    varNR = NumOps.Add(varNR, NumOps.Multiply(diff, diff));
                }
                varNR = NumOps.Divide(varNR, NumOps.FromDouble(samplingNeighbors.Count));

                // MDEF = 1 - n(p,r) / avg_n(r)
                T mdef = NumOps.GreaterThan(avgNR, NumOps.Zero)
                    ? NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.FromDouble(nR), avgNR))
                    : NumOps.Zero;

                // Sigma-MDEF (normalized deviation)
                T sigmaMdef = NumOps.GreaterThan(avgNR, NumOps.Zero)
                    ? NumOps.Divide(NumOps.Sqrt(varNR), avgNR)
                    : NumOps.Zero;

                // Final score: MDEF / sigma-MDEF (or just MDEF if sigma is 0)
                T absMdef = NumOps.Abs(mdef);
                T lociScore = NumOps.GreaterThan(sigmaMdef, NumOps.Zero)
                    ? NumOps.Divide(absMdef, sigmaMdef)
                    : absMdef;

                if (NumOps.GreaterThan(lociScore, maxMdef))
                {
                    maxMdef = lociScore;
                }
            }

            // If the point had no neighbors at ANY tested radius, it is an extreme isolate.
            if (!hadNeighbors)
            {
                maxMdef = NumOps.FromDouble(NoNeighborScore);
            }

            scores[i] = maxMdef;
        }

        return scores;
    }

    private T EstimateMaxRadius(Matrix<T> X)
    {
        // Estimate maximum radius as the max distance between any two points
        // For efficiency, sample a subset
        T maxDist = NumOps.Zero;
        int sampleSize = Math.Min(100, X.Rows);
        var random = RandomHelper.CreateSeededRandom(_randomSeed);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx1 = random.Next(X.Rows);
            int idx2 = random.Next(X.Rows);

            var p1 = new Vector<T>(X.GetRowReadOnlySpan(idx1).ToArray());
            var p2 = new Vector<T>(X.GetRowReadOnlySpan(idx2).ToArray());

            var diff = Engine.Subtract(p1, p2);
            T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
            if (NumOps.GreaterThan(dist, maxDist)) maxDist = dist;
        }

        return maxDist;
    }

    private int CountPointsInRadius(Matrix<T> X, int pointIdx, Matrix<T> reference, T radius)
    {
        int count = 0;
        var point = new Vector<T>(X.GetRowReadOnlySpan(pointIdx).ToArray());

        for (int i = 0; i < reference.Rows; i++)
        {
            var refPoint = new Vector<T>(reference.GetRowReadOnlySpan(i).ToArray());
            var diff = Engine.Subtract(point, refPoint);
            T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
            if (!NumOps.GreaterThan(dist, radius)) count++;
        }

        return count;
    }

    private int CountPointsInRadiusFromTraining(int trainingIdx, T radius)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int count = 0;
        var point = new Vector<T>(trainingData.GetRowReadOnlySpan(trainingIdx).ToArray());

        for (int i = 0; i < trainingData.Rows; i++)
        {
            var refPoint = new Vector<T>(trainingData.GetRowReadOnlySpan(i).ToArray());
            var diff = Engine.Subtract(point, refPoint);
            T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
            if (!NumOps.GreaterThan(dist, radius)) count++;
        }

        return count;
    }

    private List<int> GetPointsInRadius(Matrix<T> X, int pointIdx, Matrix<T> reference, T radius)
    {
        var neighbors = new List<int>();
        var point = new Vector<T>(X.GetRowReadOnlySpan(pointIdx).ToArray());

        for (int i = 0; i < reference.Rows; i++)
        {
            var refPoint = new Vector<T>(reference.GetRowReadOnlySpan(i).ToArray());
            var diff = Engine.Subtract(point, refPoint);
            T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
            if (!NumOps.GreaterThan(dist, radius) && NumOps.GreaterThan(dist, NumOps.Zero))
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }
}
