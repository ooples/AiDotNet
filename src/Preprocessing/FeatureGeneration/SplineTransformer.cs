using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureGeneration;

/// <summary>
/// Generates B-spline basis functions for features.
/// </summary>
/// <remarks>
/// <para>
/// SplineTransformer creates B-spline basis functions from input features. B-splines
/// are piecewise polynomials that provide a flexible way to model non-linear relationships
/// while maintaining smoothness.
/// </para>
/// <para>
/// The knots can be placed uniformly across the feature range or at quantile positions
/// to ensure roughly equal numbers of samples between knots.
/// </para>
/// <para><b>For Beginners:</b> B-splines let your model capture curved (non-linear) patterns:
/// - Linear models only find straight-line relationships
/// - Splines create multiple smooth curves that join at "knots"
/// - Each input feature becomes multiple features representing different curve segments
///
/// Example: Age effect on income might be curved (rises until 50, then plateaus).
/// Splines capture this without needing polynomial features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SplineTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nKnots;
    private readonly int _degree;
    private readonly SplineKnotStrategy _knotStrategy;
    private readonly bool _includeIntercept;
    private readonly SplineExtrapolation _extrapolation;

    // Fitted parameters
    private double[][]? _knots; // Knots for each feature
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private int _nSplineFeatures; // Features per input column

    /// <summary>
    /// Gets the number of internal knots.
    /// </summary>
    public int NKnots => _nKnots;

    /// <summary>
    /// Gets the degree of the spline.
    /// </summary>
    public int Degree => _degree;

    /// <summary>
    /// Gets the knot placement strategy.
    /// </summary>
    public SplineKnotStrategy KnotStrategy => _knotStrategy;

    /// <summary>
    /// Gets whether the intercept term is included.
    /// </summary>
    public bool IncludeIntercept => _includeIntercept;

    /// <summary>
    /// Gets the extrapolation strategy.
    /// </summary>
    public SplineExtrapolation Extrapolation => _extrapolation;

    /// <summary>
    /// Gets the fitted knots for each feature.
    /// </summary>
    public double[][]? Knots => _knots;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SplineTransformer{T}"/>.
    /// </summary>
    /// <param name="nKnots">Number of internal knots. Defaults to 5.</param>
    /// <param name="degree">Degree of the spline (1=linear, 2=quadratic, 3=cubic). Defaults to 3.</param>
    /// <param name="knotStrategy">Strategy for placing knots. Defaults to Uniform.</param>
    /// <param name="includeIntercept">Whether to include the intercept (first basis function). Defaults to true.</param>
    /// <param name="extrapolation">How to handle values outside the knot range. Defaults to Constant.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public SplineTransformer(
        int nKnots = 5,
        int degree = 3,
        SplineKnotStrategy knotStrategy = SplineKnotStrategy.Uniform,
        bool includeIntercept = true,
        SplineExtrapolation extrapolation = SplineExtrapolation.Constant,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nKnots < 2)
        {
            throw new ArgumentException("Number of knots must be at least 2.", nameof(nKnots));
        }

        if (degree < 0 || degree > 5)
        {
            throw new ArgumentException("Degree must be between 0 and 5.", nameof(degree));
        }

        _nKnots = nKnots;
        _degree = degree;
        _knotStrategy = knotStrategy;
        _includeIntercept = includeIntercept;
        _extrapolation = extrapolation;
    }

    /// <summary>
    /// Computes the knot positions for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _knots = new double[_nInputFeatures][];

        // Number of spline basis functions per feature
        // For B-splines: nKnots + degree - 1 (if including intercept)
        // Without intercept: nKnots + degree - 2
        _nSplineFeatures = _nKnots + _degree - 1;
        if (!_includeIntercept)
        {
            _nSplineFeatures -= 1;
        }

        int numOutputCols = 0;

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (!processSet.Contains(col))
            {
                // Pass-through columns
                _knots[col] = Array.Empty<double>();
                numOutputCols += 1;
                continue;
            }

            // Collect all values for this column
            var values = new List<double>();
            for (int i = 0; i < data.Rows; i++)
            {
                values.Add(NumOps.ToDouble(data[i, col]));
            }

            values.Sort();

            // Create knots based on strategy
            _knots[col] = CreateKnots(values);
            numOutputCols += _nSplineFeatures;
        }

        _nOutputFeatures = numOutputCols;
    }

    private double[] CreateKnots(List<double> sortedValues)
    {
        double minVal = sortedValues[0];
        double maxVal = sortedValues[sortedValues.Count - 1];

        // Create internal knots
        var internalKnots = new double[_nKnots];

        if (_knotStrategy == SplineKnotStrategy.Uniform)
        {
            // Evenly spaced knots
            for (int i = 0; i < _nKnots; i++)
            {
                internalKnots[i] = minVal + (i + 1) * (maxVal - minVal) / (_nKnots + 1);
            }
        }
        else // Quantile
        {
            // Knots at quantile positions
            for (int i = 0; i < _nKnots; i++)
            {
                double percentile = (i + 1) * 100.0 / (_nKnots + 1);
                internalKnots[i] = CalculatePercentile(sortedValues, percentile);
            }
        }

        // Create full knot vector with boundary knots
        // For B-splines, we need (degree + 1) repeated knots at boundaries
        var fullKnots = new double[_nKnots + 2 * (_degree + 1)];

        // Lower boundary knots (repeated)
        for (int i = 0; i <= _degree; i++)
        {
            fullKnots[i] = minVal;
        }

        // Internal knots
        for (int i = 0; i < _nKnots; i++)
        {
            fullKnots[_degree + 1 + i] = internalKnots[i];
        }

        // Upper boundary knots (repeated)
        for (int i = 0; i <= _degree; i++)
        {
            fullKnots[_nKnots + _degree + 1 + i] = maxVal;
        }

        return fullKnots;
    }

    private static double CalculatePercentile(List<double> sortedValues, double percentile)
    {
        if (sortedValues.Count == 0) return 0;
        if (sortedValues.Count == 1) return sortedValues[0];

        double index = (percentile / 100.0) * (sortedValues.Count - 1);
        int lowerIndex = (int)Math.Floor(index);
        int upperIndex = (int)Math.Ceiling(index);

        if (lowerIndex == upperIndex) return sortedValues[lowerIndex];

        double fraction = index - lowerIndex;
        return sortedValues[lowerIndex] + fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
    }

    /// <summary>
    /// Transforms the data by computing B-spline basis values.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The B-spline basis features.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_knots is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        if (data.Columns != _nInputFeatures)
        {
            throw new ArgumentException(
                $"Input has {data.Columns} features, but SplineTransformer was fitted with {_nInputFeatures} features.",
                nameof(data));
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];
        var columnsToProcess = GetColumnsToProcess(data.Columns);
        var processSet = new HashSet<int>(columnsToProcess);

        int outputCol = 0;

        for (int col = 0; col < data.Columns; col++)
        {
            if (!processSet.Contains(col) || _knots[col].Length == 0)
            {
                // Pass-through column
                for (int i = 0; i < numRows; i++)
                {
                    result[i, outputCol] = data[i, col];
                }
                outputCol++;
                continue;
            }

            // Compute B-spline basis for this column
            for (int i = 0; i < numRows; i++)
            {
                double x = NumOps.ToDouble(data[i, col]);
                double[] basisValues = ComputeBSplineBasis(x, _knots[col]);

                int startIdx = _includeIntercept ? 0 : 1;
                for (int j = startIdx; j < basisValues.Length; j++)
                {
                    result[i, outputCol + (j - startIdx)] = NumOps.FromDouble(basisValues[j]);
                }
            }

            outputCol += _nSplineFeatures;
        }

        return new Matrix<T>(result);
    }

    private double[] ComputeBSplineBasis(double x, double[] knots)
    {
        int n = knots.Length - _degree - 1; // Number of basis functions
        var basis = new double[n];

        // Handle extrapolation
        double minKnot = knots[_degree];
        double maxKnot = knots[knots.Length - _degree - 1];

        if (x < minKnot)
        {
            if (_extrapolation == SplineExtrapolation.Constant)
            {
                x = minKnot;
            }
            // For Linear extrapolation, x stays as is (basis will extrapolate)
        }
        else if (x > maxKnot)
        {
            if (_extrapolation == SplineExtrapolation.Constant)
            {
                x = maxKnot;
            }
        }

        // De Boor's algorithm for B-spline basis
        // Start with degree 0 (step functions)
        var N = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (i + 1 < knots.Length)
            {
                N[i] = (x >= knots[i] && x < knots[i + 1]) ? 1.0 : 0.0;
            }
        }

        // Handle the last interval (include right endpoint)
        if (x == knots[knots.Length - 1])
        {
            N[n - 1] = 1.0;
        }

        // Recursively build higher degree basis functions
        for (int d = 1; d <= _degree; d++)
        {
            var newN = new double[n];

            for (int i = 0; i < n; i++)
            {
                if (i + d < knots.Length)
                {
                    double denom1 = knots[i + d] - knots[i];
                    double denom2 = knots[i + d + 1] - knots[i + 1];

                    double term1 = 0;
                    if (denom1 > 1e-10)
                    {
                        term1 = (x - knots[i]) / denom1 * N[i];
                    }

                    double term2 = 0;
                    if (denom2 > 1e-10 && i + 1 < n)
                    {
                        term2 = (knots[i + d + 1] - x) / denom2 * N[i + 1];
                    }

                    newN[i] = term1 + term2;
                }
            }

            N = newN;
        }

        return N;
    }

    /// <summary>
    /// Inverse transformation is not supported for spline basis functions.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SplineTransformer does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_knots is null)
        {
            return Array.Empty<string>();
        }

        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        var names = new List<string>();

        for (int col = 0; col < _nInputFeatures; col++)
        {
            string baseName = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";

            if (!processSet.Contains(col) || _knots[col].Length == 0)
            {
                names.Add(baseName);
                continue;
            }

            int startIdx = _includeIntercept ? 0 : 1;
            for (int i = startIdx; i < _nSplineFeatures + startIdx; i++)
            {
                names.Add($"{baseName}_spline{i}");
            }
        }

        return names.ToArray();
    }
}

/// <summary>
/// Strategy for placing spline knots.
/// </summary>
public enum SplineKnotStrategy
{
    /// <summary>
    /// Place knots uniformly across the feature range.
    /// </summary>
    Uniform,

    /// <summary>
    /// Place knots at quantile positions (equal number of samples between knots).
    /// </summary>
    Quantile
}

/// <summary>
/// How to handle values outside the knot range.
/// </summary>
public enum SplineExtrapolation
{
    /// <summary>
    /// Use the boundary value (clip to knot range).
    /// </summary>
    Constant,

    /// <summary>
    /// Linearly extrapolate beyond the knots.
    /// </summary>
    Linear
}
