using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Imputes missing values using K-Nearest Neighbors.
/// </summary>
/// <remarks>
/// <para>
/// KNNImputer replaces missing values with the mean (or weighted mean) of the K nearest
/// neighbors found in the training set. Each sample's missing values are imputed using
/// the values from the K most similar samples that have non-missing values for that feature.
/// </para>
/// <para><b>For Beginners:</b> This imputer fills in missing values by looking at similar data points:
/// - Finds the K most similar rows that have the value you need
/// - Uses their average to fill in the missing value
/// - "Similar" is measured using Euclidean distance on non-missing features
///
/// Example: If you're missing someone's income, KNN finds K similar people
/// (same age, education, etc.) and uses their average income.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KNNImputer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nNeighbors;
    private readonly KNNWeights _weights;
    private readonly double _missingValue;

    // Fitted parameters
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _fitData;

    /// <summary>
    /// Gets the number of neighbors to use for imputation.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the weighting scheme used for neighbors.
    /// </summary>
    public KNNWeights Weights => _weights;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="KNNImputer{T}"/>.
    /// </summary>
    /// <param name="nNeighbors">Number of neighbors to use. Defaults to 5.</param>
    /// <param name="weights">Weight function used in prediction. Defaults to Uniform.</param>
    /// <param name="missingValue">The value to treat as missing. Defaults to NaN.</param>
    /// <param name="columnIndices">The column indices to impute, or null for all columns.</param>
    public KNNImputer(
        int nNeighbors = 5,
        KNNWeights weights = KNNWeights.Uniform,
        double missingValue = double.NaN,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nNeighbors = nNeighbors;
        _weights = weights;
        _missingValue = missingValue;
    }

    /// <summary>
    /// Stores the training data for neighbor lookup.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        // Store a copy of the training data for neighbor lookup
        _fitData = data.Clone();
    }

    /// <summary>
    /// Imputes missing values using K-nearest neighbors.
    /// </summary>
    /// <param name="data">The data to impute.</param>
    /// <returns>The data with missing values imputed.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_fitData is null)
        {
            throw new InvalidOperationException("Imputer has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        ScalerTensorOps.EnsureColumnCount(data, _fitData.Columns);
        var x = Tensor<T>.FromMatrix(data);
        var fit = Tensor<T>.FromMatrix(_fitData);
        var missing = MissingMask(x);
        var fitMissing = MissingMask(fit);

        // scikit-learn KNNImputer: nan_euclidean distances between the query rows and the training rows;
        // for each missing entry the donors are the n_neighbors nearest training rows that have that
        // column, averaged uniformly or by inverse distance.
        var distances = _fitData.Rows == 0 ? null : NanEuclideanDistances(x, missing, fit, fitMissing);
        var columns = new List<Tensor<T>>(numColumns);
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        for (int col = 0; col < numColumns; col++)
        {
            var column = PreprocessingTensorOps.ColumnVector(x, col);
            var receivers = processSet.Contains(col)
                ? PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.ColumnVector(missing, col))
                : Array.Empty<int>();
            if (receivers.Length == 0)
            {
                columns.Add(column);
                continue;
            }

            var imputed = ImputeColumn(col, receivers, distances, fit, fitMissing);
            var receiverIndex = new Tensor<int>(receivers, new[] { receivers.Length });
            columns.Add(Engine.TensorScatter(column.Clone(), receiverIndex, imputed, 0));
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// The imputed values of column <paramref name="col"/> for the <paramref name="receivers"/> rows.
    /// </summary>
    private Tensor<T> ImputeColumn(int col, int[] receivers, Tensor<T>? distances, Tensor<T> fit, Tensor<T> fitMissing)
    {
        var fitColumn = PreprocessingTensorOps.ColumnVector(fit, col);
        var donors = PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.Not(PreprocessingTensorOps.ColumnVector(fitMissing, col)));
        var columnMean = CalculateColumnMean(fitColumn, donors);
        if (distances is null || donors.Length == 0)
        {
            return PreprocessingTensorOps.Filled(new[] { receivers.Length }, columnMean);
        }

        var donorIndex = new Tensor<int>(donors, new[] { donors.Length });
        var donorValues = Engine.TensorIndexSelect(fitColumn, donorIndex, 0);
        var donorDistances = Engine.TensorIndexSelect(
            Engine.TensorIndexSelect(distances, new Tensor<int>(receivers, new[] { receivers.Length }), 0), donorIndex, 1);

        var values = new Tensor<T>[receivers.Length];
        for (int r = 0; r < receivers.Length; r++)
        {
            values[r] = ImputeValue(PreprocessingTensorOps.RowVector(donorDistances, r), donorValues, columnMean);
        }

        return Engine.TensorConcatenate(values, 0);
    }

    /// <summary>
    /// 1 where an entry is missing: NaN when the missing value is NaN, otherwise an entry exactly equal to it
    /// (scikit-learn compares with <c>X == missing_values</c>, no tolerance).
    /// </summary>
    private Tensor<T> MissingMask(Tensor<T> x)
    {
        if (double.IsNaN(_missingValue))
        {
            return ScalerTensorOps.NanMask(x);
        }

        return Engine.TensorMax(ScalerTensorOps.NanMask(x), Engine.TensorEquals(x, NumOps.FromDouble(_missingValue)));
    }

    /// <summary>
    /// scikit-learn's <c>nan_euclidean_distances</c> between every query row and every training row:
    /// <c>sqrt(p / present * sum over coordinates present in both of (a - b)^2)</c>, NaN when the rows share no
    /// present coordinate. Returns [queries, training rows].
    /// </summary>
    private Tensor<T> NanEuclideanDistances(Tensor<T> x, Tensor<T> missing, Tensor<T> fit, Tensor<T> fitMissing)
    {
        T zero = NumOps.Zero;
        var present = PreprocessingTensorOps.Not(missing);
        var fitPresent = PreprocessingTensorOps.Not(fitMissing);
        var x0 = Engine.TensorWhere(missing, new Tensor<T>(x.Shape.ToArray()), x);
        var f0 = Engine.TensorWhere(fitMissing, new Tensor<T>(fit.Shape.ToArray()), fit);

        // sum present(a)present(b)(a - b)^2 = (a0^2) . present(b) + present(a) . (b0^2) - 2 a0 . b0
        var sumSq = Engine.TensorAdd(
            Engine.TensorMatMul(Engine.TensorSquare(x0), Engine.TensorTranspose(fitPresent)),
            Engine.TensorMatMul(present, Engine.TensorTranspose(Engine.TensorSquare(f0))));
        sumSq = Engine.TensorSubtract(
            sumSq, Engine.TensorMultiplyScalar(Engine.TensorMatMul(x0, Engine.TensorTranspose(f0)), NumOps.FromDouble(2.0)));
        sumSq = Engine.TensorMax(sumSq, zero);

        var count = Engine.TensorMatMul(present, Engine.TensorTranspose(fitPresent));
        var noOverlap = Engine.TensorEquals(count, zero);
        var safeCount = Engine.TensorWhere(noOverlap, PreprocessingTensorOps.Filled(count.Shape.ToArray(), NumOps.One), count);
        var scaled = Engine.TensorDivide(Engine.TensorMultiplyScalar(sumSq, NumOps.FromDouble(x.Shape[1])), safeCount);
        return Engine.TensorWhere(
            noOverlap, PreprocessingTensorOps.Filled(count.Shape.ToArray(), NumOps.FromDouble(double.NaN)), Engine.TensorSqrt(scaled));
    }

    /// <summary>
    /// One imputed value (as a [1] tensor) from the distances to the donors and their values: the
    /// n_neighbors nearest donors (ties to the lower row), excluding donors at NaN distance; uniform mean, or
    /// inverse-distance weights where donors at distance 0 get weight 1 and the others 0 (scikit-learn's
    /// <c>_get_weights</c>). With no usable donor, the column mean.
    /// </summary>
    private Tensor<T> ImputeValue(Tensor<T> donorDistances, Tensor<T> donorValues, T columnMean)
    {
        var usable = PreprocessingTensorOps.Not(ScalerTensorOps.NanMask(donorDistances));
        if (PreprocessingTensorOps.CountNonZero(usable) == 0)
        {
            return PreprocessingTensorOps.Filled(new[] { 1 }, columnMean);
        }

        int k = Math.Min(_nNeighbors, donorDistances.Length);
        var order = PreprocessingTensorOps.StableOrder(ScalerTensorOps.FillNan(donorDistances, NumOps.MaxValue), descending: false);
        var nearest = new Tensor<int>(order.AsSpan().Slice(0, k).ToArray(), new[] { k });
        var distances = Engine.TensorIndexSelect(donorDistances, nearest, 0);
        var values = Engine.TensorIndexSelect(donorValues, nearest, 0);
        var weights = Engine.TensorIndexSelect(usable, nearest, 0);

        if (_weights == KNNWeights.Distance)
        {
            var atZero = Engine.TensorMultiply(weights, Engine.TensorEquals(distances, NumOps.Zero));
            if (PreprocessingTensorOps.CountNonZero(atZero) > 0)
            {
                weights = atZero;
            }
            else
            {
                var safe = Engine.TensorWhere(weights, distances, PreprocessingTensorOps.Filled(new[] { k }, NumOps.One));
                weights = Engine.TensorMultiply(weights, Engine.TensorReciprocal(safe));
            }
        }

        var numerator = Engine.ReduceSum(Engine.TensorMultiply(weights, values), new[] { 0 }, true);
        var denominator = Engine.ReduceSum(weights, new[] { 0 }, true);
        return Engine.TensorDivide(numerator, denominator);
    }

    /// <summary>Mean of the training column over its non-missing rows (0 when there are none).</summary>
    private T CalculateColumnMean(Tensor<T> fitColumn, int[] donors)
    {
        if (donors.Length == 0)
        {
            return NumOps.Zero;
        }

        return Engine.TensorMean(Engine.TensorIndexSelect(fitColumn, new Tensor<int>(donors, new[] { donors.Length }), 0));
    }

    /// <summary>
    /// Inverse transformation is not supported for KNN imputation.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KNNImputer does not support inverse transformation.");
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
/// Weight function used in KNN prediction.
/// </summary>
public enum KNNWeights
{
    /// <summary>
    /// All neighbors are weighted equally.
    /// </summary>
    Uniform,

    /// <summary>
    /// Neighbors are weighted by the inverse of their distance.
    /// </summary>
    Distance
}
