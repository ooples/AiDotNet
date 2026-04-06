using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using Median Absolute Deviation (MAD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MAD is a robust measure of spread that uses the median instead of
/// the mean. Unlike standard deviation, MAD is resistant to outliers. Points far from the
/// median (in terms of MAD units) are flagged as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute the median of each feature
/// 2. Compute MAD = median(|x - median|) for each feature
/// 3. Score points based on their deviation from median in MAD units
/// 4. High scores indicate anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data with heavy-tailed distributions
/// - When outliers may skew standard statistics
/// - As a robust alternative to Z-score
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Threshold: 3.5 MAD units
/// - Scale factor: 1.4826 (for Gaussian consistency)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Leys, C., et al. (2013). "Detecting outliers: Do not use standard deviation
/// around the mean, use absolute deviation around the median." Journal of Experimental Social Psychology.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median", "https://doi.org/10.1016/j.jesp.2013.03.013", Year = 2013, Authors = "Christophe Leys, Christophe Ley, Olivier Klein, Philippe Bernard, Laurent Licata")]
public class MADDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _madThreshold;
    private readonly double _scaleFactor;
    private Vector<T>? _medians;
    private Vector<T>? _mads;

    /// <summary>
    /// Gets the MAD threshold for anomaly detection.
    /// </summary>
    public double MADThreshold => _madThreshold;

    /// <summary>
    /// Creates a new MAD anomaly detector.
    /// </summary>
    /// <param name="madThreshold">
    /// Number of MAD units beyond which a point is considered anomalous. Default is 3.5.
    /// </param>
    /// <param name="scaleFactor">
    /// Scale factor for MAD (1.4826 for consistency with std for normal data). Default is 1.4826.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public MADDetector(double madThreshold = 3.5, double scaleFactor = 1.4826,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (madThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(madThreshold),
                "MADThreshold must be positive. Recommended is 3.5.");
        }

        if (scaleFactor <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scaleFactor),
                "ScaleFactor must be positive. Standard is 1.4826.");
        }

        _madThreshold = madThreshold;
        _scaleFactor = scaleFactor;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        _medians = new Vector<T>(d);
        _mads = new Vector<T>(d);
        T eps = NumOps.FromDouble(1e-10);
        T scaleT = NumOps.FromDouble(_scaleFactor);
        T two = NumOps.FromDouble(2);
        T half = NumOps.FromDouble(0.5);

        for (int j = 0; j < d; j++)
        {
            // Extract column values
            var values = new T[n];
            for (int i = 0; i < n; i++)
            {
                values[i] = X[i, j];
            }

            // Compute median
            Array.Sort(values, (a, b) => NumOps.Compare(a, b));
            _medians[j] = n % 2 == 0
                ? NumOps.Multiply(NumOps.Add(values[n / 2 - 1], values[n / 2]), half)
                : values[n / 2];

            // Compute MAD
            var absDeviations = new T[n];
            for (int i = 0; i < n; i++)
            {
                absDeviations[i] = NumOps.Abs(NumOps.Subtract(X[i, j], _medians[j]));
            }

            Array.Sort(absDeviations, (a, b) => NumOps.Compare(a, b));
            T medianAbsDev = n % 2 == 0
                ? NumOps.Multiply(NumOps.Add(absDeviations[n / 2 - 1], absDeviations[n / 2]), half)
                : absDeviations[n / 2];

            _mads[j] = NumOps.Multiply(scaleT, medianAbsDev);
            if (NumOps.LessThan(_mads[j], eps)) _mads[j] = eps;
        }

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

        var medians = _medians;
        var mads = _mads;

        if (medians == null || mads == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T maxScore = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                T deviation = NumOps.Abs(NumOps.Subtract(X[i, j], medians[j]));
                T madScore = NumOps.Divide(deviation, mads[j]);

                if (NumOps.GreaterThan(madScore, maxScore))
                {
                    maxScore = madScore;
                }
            }

            scores[i] = maxScore;
        }

        return scores;
    }
}
