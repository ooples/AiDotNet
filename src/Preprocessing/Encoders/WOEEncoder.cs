using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using Weight of Evidence (WOE).
/// </summary>
/// <remarks>
/// <para>
/// Weight of Evidence is commonly used in credit scoring and binary classification.
/// It measures the strength of the relationship between a category and the binary target.
/// WOE = ln(Distribution of Events / Distribution of Non-Events)
/// </para>
/// <para>
/// Higher WOE values indicate categories more associated with the positive class,
/// while lower (negative) values indicate association with the negative class.
/// </para>
/// <para><b>For Beginners:</b> WOE tells you how "good" or "bad" a category is for prediction:
/// - WOE &gt; 0: Category is more likely to have positive outcomes
/// - WOE &lt; 0: Category is more likely to have negative outcomes
/// - WOE ≈ 0: Category has no predictive power
///
/// Example in loan default prediction:
/// - "Employed" might have WOE = -0.5 (less likely to default)
/// - "Unemployed" might have WOE = +0.8 (more likely to default)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class WOEEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _regularization;
    private readonly WOEHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, double>>? _woeValues;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the regularization parameter to prevent infinite WOE values.
    /// </summary>
    public double Regularization => _regularization;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public WOEHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the WOE values for each category.
    /// </summary>
    public Dictionary<int, Dictionary<double, double>>? WOEValues => _woeValues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="WOEEncoder{T}"/>.
    /// </summary>
    /// <param name="regularization">Regularization to add to counts to prevent division by zero. Defaults to 0.5.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseZero.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public WOEEncoder(
        double regularization = 0.5,
        WOEHandleUnknown handleUnknown = WOEHandleUnknown.UseZero,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (regularization < 0)
        {
            throw new ArgumentException("Regularization must be non-negative.", nameof(regularization));
        }

        _regularization = regularization;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires binary target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WOEEncoder requires binary target values (0/1) for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing WOE values for each category.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The binary target values (0 or 1).</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Count total events (1s) and non-events (0s)
        int totalEvents = 0;
        int totalNonEvents = 0;

        for (int i = 0; i < target.Length; i++)
        {
            double t = NumOps.ToDouble(target[i]);
            if (Math.Abs(t - 1) < 1e-10)
            {
                totalEvents++;
            }
            else if (Math.Abs(t) < 1e-10)
            {
                totalNonEvents++;
            }
            else
            {
                throw new ArgumentException($"Target must be binary (0 or 1). Found value: {t}");
            }
        }

        if (totalEvents == 0 || totalNonEvents == 0)
        {
            throw new ArgumentException("Target must contain both classes (0 and 1).");
        }

        _woeValues = new Dictionary<int, Dictionary<double, double>>();

        foreach (int col in columnsToProcess)
        {
            // Count events and non-events per category
            var categoryStats = new Dictionary<double, (int Events, int NonEvents)>();

            for (int i = 0; i < data.Rows; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!categoryStats.TryGetValue(categoryValue, out var stats))
                {
                    stats = (0, 0);
                }

                if (Math.Abs(targetValue - 1) < 1e-10)
                {
                    categoryStats[categoryValue] = (stats.Events + 1, stats.NonEvents);
                }
                else
                {
                    categoryStats[categoryValue] = (stats.Events, stats.NonEvents + 1);
                }
            }

            // Calculate WOE for each category
            var woeMap = new Dictionary<double, double>();

            foreach (var kvp in categoryStats)
            {
                double categoryValue = kvp.Key;
                int events = kvp.Value.Events;
                int nonEvents = kvp.Value.NonEvents;

                // Add regularization to prevent log(0)
                double distEvents = (events + _regularization) / (totalEvents + _regularization * categoryStats.Count);
                double distNonEvents = (nonEvents + _regularization) / (totalNonEvents + _regularization * categoryStats.Count);

                double woe = Math.Log(distEvents / distNonEvents);

                // Clamp to prevent extreme values
                woe = Math.Max(-5, Math.Min(5, woe));

                woeMap[categoryValue] = woe;
            }

            _woeValues[col] = woeMap;
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by replacing categories with WOE values.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_woeValues is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (!processSet.Contains(j))
                {
                    result[i, j] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);

                if (_woeValues.TryGetValue(j, out var woeMap) &&
                    woeMap.TryGetValue(categoryValue, out double woe))
                {
                    result[i, j] = NumOps.FromDouble(woe);
                }
                else
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case WOEHandleUnknown.UseZero:
                            result[i, j] = NumOps.Zero;
                            break;
                        case WOEHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            result[i, j] = NumOps.Zero;
                            break;
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Calculates Information Value (IV) for each feature.
    /// </summary>
    /// <remarks>
    /// IV measures the overall predictive power of a feature.
    /// IV &lt; 0.02: Not useful for prediction
    /// 0.02 &lt; IV &lt; 0.1: Weak predictor
    /// 0.1 &lt; IV &lt; 0.3: Medium predictor
    /// 0.3 &lt; IV &lt; 0.5: Strong predictor
    /// IV &gt; 0.5: Suspicious (possible overfitting)
    /// </remarks>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The binary target.</param>
    /// <returns>Dictionary mapping column index to IV value.</returns>
    public Dictionary<int, double> CalculateInformationValue(Matrix<T> data, Vector<T> target)
    {
        if (_woeValues is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        // Count total events and non-events
        int totalEvents = 0;
        int totalNonEvents = 0;

        for (int i = 0; i < target.Length; i++)
        {
            double t = NumOps.ToDouble(target[i]);
            if (Math.Abs(t - 1) < 1e-10)
            {
                totalEvents++;
            }
            else
            {
                totalNonEvents++;
            }
        }

        var ivValues = new Dictionary<int, double>();

        foreach (var colKvp in _woeValues)
        {
            int col = colKvp.Key;
            var woeMap = colKvp.Value;

            // Count events/non-events per category
            var categoryStats = new Dictionary<double, (int Events, int NonEvents)>();

            for (int i = 0; i < data.Rows; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!categoryStats.TryGetValue(categoryValue, out var stats))
                {
                    stats = (0, 0);
                }

                if (Math.Abs(targetValue - 1) < 1e-10)
                {
                    categoryStats[categoryValue] = (stats.Events + 1, stats.NonEvents);
                }
                else
                {
                    categoryStats[categoryValue] = (stats.Events, stats.NonEvents + 1);
                }
            }

            // Calculate IV
            double iv = 0;
            foreach (var catKvp in categoryStats)
            {
                double categoryValue = catKvp.Key;
                int events = catKvp.Value.Events;
                int nonEvents = catKvp.Value.NonEvents;

                double distEvents = (events + _regularization) / (totalEvents + _regularization * categoryStats.Count);
                double distNonEvents = (nonEvents + _regularization) / (totalNonEvents + _regularization * categoryStats.Count);

                double woe = woeMap.GetValueOrDefault(categoryValue, 0);
                iv += (distEvents - distNonEvents) * woe;
            }

            ivValues[col] = iv;
        }

        return ivValues;
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WOEEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum WOEHandleUnknown
{
    /// <summary>
    /// Use WOE = 0 for unknown categories (neutral evidence).
    /// </summary>
    UseZero,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
