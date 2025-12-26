using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.PowerTransforms;

/// <summary>
/// Specifies the target output distribution for quantile transformation.
/// </summary>
public enum OutputDistributionType
{
    /// <summary>
    /// Map to a uniform distribution in [0, 1].
    /// </summary>
    Uniform,

    /// <summary>
    /// Map to a standard normal distribution (Gaussian).
    /// </summary>
    Normal
}

/// <summary>
/// Transforms features to follow a uniform or normal distribution using quantile information.
/// </summary>
/// <remarks>
/// <para>
/// QuantileTransformer applies a non-linear transformation that maps the input distribution to
/// either a uniform or normal (Gaussian) distribution. This is effective at reducing the impact
/// of outliers and normalizing non-Gaussian distributions.
/// </para>
/// <para><b>For Beginners:</b> This transformer redistributes your data to match a desired pattern:
/// - Uniform: Spreads values evenly across [0, 1]
/// - Normal: Creates a bell curve distribution
///
/// Example with uniform output:
/// [1, 1, 2, 3, 5, 8, 13, 100, 1000] → [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
/// Notice how extreme values (100, 1000) no longer dominate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class QuantileTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly OutputDistributionType _outputDistribution;
    private readonly int _nQuantiles;

    // Fitted parameters: quantiles for each column
    private List<T[]>? _quantiles;

    /// <summary>
    /// Gets the target output distribution.
    /// </summary>
    public OutputDistributionType OutputDistribution => _outputDistribution;

    /// <summary>
    /// Gets the number of quantiles used.
    /// </summary>
    public int NQuantiles => _nQuantiles;

    /// <summary>
    /// Gets the quantiles for each feature computed during fitting.
    /// </summary>
    public List<T[]>? Quantiles => _quantiles;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="QuantileTransformer{T}"/>.
    /// </summary>
    /// <param name="outputDistribution">The target distribution. Defaults to Uniform.</param>
    /// <param name="nQuantiles">The number of quantiles to compute. Defaults to 1000.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public QuantileTransformer(
        OutputDistributionType outputDistribution = OutputDistributionType.Uniform,
        int nQuantiles = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nQuantiles < 10)
        {
            throw new ArgumentException("nQuantiles must be at least 10.", nameof(nQuantiles));
        }

        _outputDistribution = outputDistribution;
        _nQuantiles = nQuantiles;
    }

    /// <summary>
    /// Computes the quantiles for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        _quantiles = new List<T[]>();

        for (int col = 0; col < numColumns; col++)
        {
            if (!columnsToProcess.Contains(col))
            {
                _quantiles.Add(Array.Empty<T>());
                continue;
            }

            var column = data.GetColumn(col);
            var sortedData = column.ToArray();
            Array.Sort(sortedData, (a, b) => NumOps.Compare(a, b));

            var quantiles = new T[_nQuantiles];
            for (int i = 0; i < _nQuantiles; i++)
            {
                double quantilePosition = (double)i / (_nQuantiles - 1);
                int index = (int)(quantilePosition * (sortedData.Length - 1));
                quantiles[i] = sortedData[index];
            }

            _quantiles.Add(quantiles);
        }
    }

    /// <summary>
    /// Transforms the data by mapping to the target distribution.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data following the target distribution.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_quantiles is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
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
                T value = data[i, j];

                if (processSet.Contains(j) && _quantiles[j].Length > 0)
                {
                    value = TransformValue(value, _quantiles[j]);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    private T TransformValue(T value, T[] quantiles)
    {
        // Handle degenerate distribution (constant feature)
        if (NumOps.Compare(quantiles[0], quantiles[quantiles.Length - 1]) == 0)
        {
            T midpoint = NumOps.FromDouble(0.5);
            return _outputDistribution == OutputDistributionType.Uniform
                ? midpoint
                : InverseNormalCDF(midpoint);
        }

        // Handle values outside the range
        if (NumOps.Compare(value, quantiles[0]) <= 0)
        {
            return _outputDistribution == OutputDistributionType.Uniform
                ? NumOps.Zero
                : NumOps.FromDouble(-8.0);
        }
        if (NumOps.Compare(value, quantiles[quantiles.Length - 1]) >= 0)
        {
            return _outputDistribution == OutputDistributionType.Uniform
                ? NumOps.One
                : NumOps.FromDouble(8.0);
        }

        // Binary search for position
        int left = 0;
        int right = quantiles.Length - 2;
        int lowerIndex = 0;
        int upperIndex = quantiles.Length - 1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (NumOps.Compare(value, quantiles[mid + 1]) <= 0)
            {
                if (NumOps.Compare(value, quantiles[mid]) > 0)
                {
                    lowerIndex = mid;
                    upperIndex = mid + 1;
                    break;
                }
                else
                {
                    right = mid - 1;
                }
            }
            else
            {
                left = mid + 1;
            }
        }

        // Linear interpolation
        T lowerValue = quantiles[lowerIndex];
        T upperValue = quantiles[upperIndex];
        T lowerPercentile = NumOps.Divide(NumOps.FromDouble(lowerIndex), NumOps.FromDouble(quantiles.Length - 1));
        T upperPercentile = NumOps.Divide(NumOps.FromDouble(upperIndex), NumOps.FromDouble(quantiles.Length - 1));

        T percentile;
        if (NumOps.Compare(upperValue, lowerValue) == 0)
        {
            percentile = NumOps.Divide(NumOps.Add(lowerPercentile, upperPercentile), NumOps.FromDouble(2.0));
        }
        else
        {
            T fraction = NumOps.Divide(
                NumOps.Subtract(value, lowerValue),
                NumOps.Subtract(upperValue, lowerValue));
            percentile = NumOps.Add(lowerPercentile, NumOps.Multiply(fraction, NumOps.Subtract(upperPercentile, lowerPercentile)));
        }

        return _outputDistribution == OutputDistributionType.Uniform
            ? percentile
            : InverseNormalCDF(percentile);
    }

    /// <summary>
    /// Reverses the quantile transformation.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_quantiles is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
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
                T value = data[i, j];

                if (processSet.Contains(j) && _quantiles[j].Length > 0)
                {
                    value = InverseTransformValue(value, _quantiles[j]);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    private T InverseTransformValue(T transformedValue, T[] quantiles)
    {
        T percentile = _outputDistribution == OutputDistributionType.Uniform
            ? transformedValue
            : NormalCDF(transformedValue);

        // Clamp percentile to [0, 1]
        if (NumOps.Compare(percentile, NumOps.Zero) < 0)
            percentile = NumOps.Zero;
        if (NumOps.Compare(percentile, NumOps.One) > 0)
            percentile = NumOps.One;

        // Map percentile back to original value
        double percentileDouble = NumOps.ToDouble(percentile);
        double position = percentileDouble * (quantiles.Length - 1);
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = Math.Min(lowerIndex + 1, quantiles.Length - 1);

        if (lowerIndex == upperIndex)
        {
            return quantiles[lowerIndex];
        }

        T fraction = NumOps.FromDouble(position - lowerIndex);
        return NumOps.Add(
            NumOps.Multiply(quantiles[lowerIndex], NumOps.Subtract(NumOps.One, fraction)),
            NumOps.Multiply(quantiles[upperIndex], fraction));
    }

    private T InverseNormalCDF(T p)
    {
        double pDouble = NumOps.ToDouble(p);

        if (pDouble <= 0.0) return NumOps.FromDouble(-8.0);
        if (pDouble >= 1.0) return NumOps.FromDouble(8.0);

        // Beasley-Springer-Moro approximation
        double q = pDouble - 0.5;

        if (Math.Abs(q) <= 0.425)
        {
            double r = 0.180625 - q * q;
            double num = ((((((2.5090809287301226727e3 * r +
                              3.3430575583588128105e4) * r +
                             6.7265770927008700853e4) * r +
                            4.5921953931549871457e4) * r +
                           1.3731693765509461125e4) * r +
                          1.9715909503065514427e3) * r +
                         1.3314166789178437745e2) * r +
                        3.3871328727963666080e0;

            double den = (((((((5.2264952788528545610e3 * r +
                               2.8729085735721942674e4) * r +
                              3.9307895800092710610e4) * r +
                             2.1213794301586595867e4) * r +
                            5.3941960214247511077e3) * r +
                           6.8718700749205790830e2) * r +
                          4.2313330701600911252e1) * r +
                         1.0);

            return NumOps.FromDouble(q * num / den);
        }
        else
        {
            double r = q < 0 ? pDouble : 1.0 - pDouble;
            r = Math.Sqrt(-Math.Log(r));

            double num, den;
            if (r <= 5.0)
            {
                r = r - 1.6;
                num = ((((((7.74545014278341407640e-4 * r +
                           2.27238449892691845833e-2) * r +
                          2.41780725177450611770e-1) * r +
                         1.27045825245236838258e0) * r +
                        3.64784832476320460504e0) * r +
                       5.76949722146069140550e0) * r +
                      4.63033784615654529590e0) * r +
                     1.42343711074968357734e0;

                den = ((((((1.05075007164441684324e-9 * r +
                           5.47593808499534494600e-4) * r +
                          1.51986665636164571966e-2) * r +
                         1.48103976427480074590e-1) * r +
                        6.89767334985100004550e-1) * r +
                       1.67638483018380384940e0) * r +
                      2.05319162663775882187e0) * r +
                     1.0;
            }
            else
            {
                r = r - 5.0;
                num = ((((((2.01033439929228813265e-7 * r +
                           2.71155556874348757815e-5) * r +
                          1.24266094738807843860e-3) * r +
                         2.65321895265761230930e-2) * r +
                        2.96560571828504891230e-1) * r +
                       1.78482653991729133580e0) * r +
                      5.46378491116411436990e0) * r +
                     6.65790464350110377720e0;

                den = ((((((2.04426310338993978564e-15 * r +
                           1.42151175831644588870e-7) * r +
                          1.84631831751005468180e-5) * r +
                         7.86869131145613259100e-4) * r +
                        1.48753612908506148525e-2) * r +
                       1.36929880922735805310e-1) * r +
                      5.99832206555887937690e-1) * r +
                     1.0;
            }

            double result = num / den;
            if (q < 0) result = -result;
            return NumOps.FromDouble(result);
        }
    }

    private T NormalCDF(T x)
    {
        double xDouble = NumOps.ToDouble(x);
        double result = 0.5 * (1.0 + Erf(xDouble / Math.Sqrt(2.0)));
        return NumOps.FromDouble(result);
    }

    private double Erf(double x)
    {
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (QuantileTransformer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
