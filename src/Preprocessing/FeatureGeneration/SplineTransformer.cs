using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureGenerator)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SplineTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nKnots;
    private readonly int _degree;
    private readonly SplineKnotStrategy _knotStrategy;
    private readonly bool _includeIntercept;
    private readonly SplineExtrapolation _extrapolation;

    // Fitted parameters
    private Tensor<T>[]? _knots; // Knots for each feature
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
    public Vector<T>[]? Knots => _knots?.Select(k => k.ToVector()).ToArray();

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
        if (data.Rows == 0)
        {
            throw new ArgumentException(
                "Input data must have at least one sample to fit spline knots.",
                nameof(data));
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        _knots = new Tensor<T>[_nInputFeatures];

        // Number of spline basis functions per feature
        // For B-splines with m knots and degree d: number of basis = m - d - 1
        // Our knot vector has: (degree+1) lower + nKnots internal + (degree+1) upper = nKnots + 2*(degree+1) knots
        // So: nBasis = nKnots + 2*(degree+1) - degree - 1 = nKnots + degree + 1
        // Without intercept (drop first basis): nKnots + degree
        _nSplineFeatures = _nKnots + _degree + 1;
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
                _knots[col] = new Tensor<T>(new[] { 0 });
                numOutputCols += 1;
                continue;
            }

            // Sorted values of this column
            var values = Engine.TensorSort(PreprocessingTensorOps.Column(data, col)).Item1;

            // Create knots based on strategy
            _knots[col] = CreateKnots(values);
            numOutputCols += _nSplineFeatures;
        }

        _nOutputFeatures = numOutputCols;
    }

    /// <summary>
    /// Builds the full knot vector of a column from its ascending values: (degree + 1) repeated boundary
    /// knots at each end around the uniform or quantile internal knots.
    /// </summary>
    private Tensor<T> CreateKnots(Tensor<T> sortedValues)
    {
        var engine = Engine;
        int count = sortedValues.Length;
        T minVal = sortedValues.GetFlat(0);
        T maxVal = sortedValues.GetFlat(count - 1);
        var positions = engine.TensorAddScalar(PreprocessingTensorOps.Range<T>(_nKnots), NumOps.One); // i + 1

        // Create internal knots
        Tensor<T> internalKnots;
        if (_knotStrategy == SplineKnotStrategy.Uniform)
        {
            // Evenly spaced knots: min + (i + 1) * (max - min) / (nKnots + 1)
            internalKnots = engine.TensorAddScalar(
                engine.TensorDivideScalar(
                    engine.TensorMultiplyScalar(positions, NumOps.Subtract(maxVal, minVal)),
                    NumOps.FromDouble(_nKnots + 1)),
                minVal);
        }
        else // Quantile
        {
            // Knots at quantile positions, linearly interpolated between the neighbouring order statistics
            var percentile = engine.TensorDivideScalar(
                engine.TensorMultiplyScalar(positions, NumOps.FromDouble(100.0)), NumOps.FromDouble(_nKnots + 1));
            var index = engine.TensorMultiplyScalar(
                engine.TensorDivideScalar(percentile, NumOps.FromDouble(100.0)), NumOps.FromDouble(count - 1));
            var lower = engine.TensorFloor(index);
            var fraction = engine.TensorSubtract(index, lower);
            var lowerValues = engine.TensorIndexSelect(sortedValues, lower.Cast<int>(), 0);
            var upperValues = engine.TensorIndexSelect(sortedValues, engine.TensorCeiling(index).Cast<int>(), 0);
            internalKnots = engine.TensorAdd(lowerValues,
                engine.TensorMultiply(fraction, engine.TensorSubtract(upperValues, lowerValues)));
        }

        // Create full knot vector with boundary knots
        // For B-splines, we need (degree + 1) repeated knots at boundaries
        return engine.TensorConcatenate(new[]
        {
            PreprocessingTensorOps.Filled(new[] { _degree + 1 }, minVal),
            internalKnots,
            PreprocessingTensorOps.Filled(new[] { _degree + 1 }, maxVal),
        }, 0);
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

        var engine = Engine;
        int numRows = data.Rows;
        var processSet = new HashSet<int>(GetColumnsToProcess(data.Columns));
        var blocks = new List<Tensor<T>>(data.Columns);
        int startIdx = _includeIntercept ? 0 : 1;

        for (int col = 0; col < data.Columns; col++)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            if (!processSet.Contains(col) || _knots[col].Length == 0)
            {
                // Pass-through column
                blocks.Add(engine.Reshape(column, new[] { numRows, 1 }));
                continue;
            }

            // Compute B-spline basis for this column
            var basis = ComputeBSplineBasis(column, _knots[col]);
            int nBasis = basis.Shape[1];
            blocks.Add(engine.TensorSlice(basis, new[] { 0, startIdx }, new[] { numRows, nBasis - startIdx }));
        }

        if (blocks.Count == 0)
            return new Matrix<T>(numRows, 0);
        return engine.TensorConcatenate(blocks.ToArray(), 1).ToMatrix();
    }

    /// <summary>
    /// The <c>[rows, basis]</c> B-spline basis values of every entry of <paramref name="x"/> (Cox-de Boor recursion).
    /// </summary>
    private Tensor<T> ComputeBSplineBasis(Tensor<T> x, Tensor<T> knots)
    {
        var engine = Engine;
        int rows = x.Length;
        int nKnotValues = knots.Length;
        int n = nKnotValues - _degree - 1; // Number of basis functions
        var shape = new[] { rows, n };

        // Handle extrapolation: Constant clamps to the boundary knots, Linear leaves x as is
        if (_extrapolation == SplineExtrapolation.Constant)
            x = engine.TensorClamp(x, knots.GetFlat(_degree), knots.GetFlat(nKnotValues - _degree - 1));

        var xs = engine.TensorBroadcastTo(engine.Reshape(x, new[] { rows, 1 }), shape);
        Tensor<T> KnotRow(int offset) =>
            engine.TensorBroadcastTo(engine.Reshape(engine.TensorSlice(knots, new[] { offset }, new[] { n }), new[] { 1, n }), shape);

        // Start with degree 0 (step functions): knots[i] <= x < knots[i + 1]
        var basis = engine.TensorMultiply(
            PreprocessingTensorOps.Not(engine.TensorLessThan(xs, KnotRow(0))),
            engine.TensorLessThan(xs, KnotRow(1)));

        // Handle the last interval (include right endpoint)
        var atEnd = engine.TensorEquals(x, knots.GetFlat(nKnotValues - 1));
        var lastColumn = new Tensor<T>(new[] { 1, n });
        lastColumn.AsSpan()[n - 1] = NumOps.One;
        var setLast = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.Reshape(atEnd, new[] { rows, 1 }), shape),
            engine.TensorBroadcastTo(lastColumn, shape));
        basis = engine.TensorAdd(basis, engine.TensorMultiply(setLast, PreprocessingTensorOps.OneMinus(basis)));

        // Recursively build higher degree basis functions
        T epsilon = NumOps.FromDouble(1e-10);
        T one = NumOps.One;
        var hasNext = engine.TensorBroadcastTo(engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { 1, n }, one), lastColumn), shape);
        for (int d = 1; d <= _degree; d++)
        {
            var left = KnotRow(0);
            var leftNext = KnotRow(1);
            var right = KnotRow(d);
            var rightNext = KnotRow(d + 1);
            var denom1 = engine.TensorSubtract(right, left);
            var denom2 = engine.TensorSubtract(rightNext, leftNext);
            var valid1 = engine.TensorGreaterThan(denom1, epsilon);
            var valid2 = engine.TensorMultiply(engine.TensorGreaterThan(denom2, epsilon), hasNext);

            // term1 = (x - knots[i]) / denom1 * N[i]; term2 = (knots[i + d + 1] - x) / denom2 * N[i + 1]
            var safe1 = engine.TensorWhere(valid1, denom1, PreprocessingTensorOps.Filled(shape, one));
            var safe2 = engine.TensorWhere(valid2, denom2, PreprocessingTensorOps.Filled(shape, one));
            var nextBasis = n > 1
                ? engine.TensorConcatenate(new[]
                {
                    engine.TensorSlice(basis, new[] { 0, 1 }, new[] { rows, n - 1 }),
                    new Tensor<T>(new[] { rows, 1 }),
                }, 1)
                : new Tensor<T>(shape);
            var term1 = engine.TensorMultiply(
                engine.TensorMultiply(engine.TensorDivide(engine.TensorSubtract(xs, left), safe1), basis), valid1);
            var term2 = engine.TensorMultiply(
                engine.TensorMultiply(engine.TensorDivide(engine.TensorSubtract(rightNext, xs), safe2), nextBasis), valid2);
            basis = engine.TensorAdd(term1, term2);
        }

        return basis;
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
