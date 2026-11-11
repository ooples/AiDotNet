using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using James-Stein shrinkage estimation.
/// </summary>
/// <remarks>
/// <para>
/// JamesSteinEncoder uses Bayesian shrinkage to blend category-specific target means
/// with the global mean. Categories with more samples get weights closer to their
/// own mean, while rare categories shrink toward the global mean.
/// </para>
/// <para>
/// The shrinkage formula: encoded = (1 - B) * category_mean + B * global_mean
/// where B is the shrinkage factor based on sample size and variance.
/// </para>
/// <para><b>For Beginners:</b> This encoder balances between:
/// - Trusting category-specific averages (when we have lots of data)
/// - Falling back to the overall average (when category data is sparse)
/// - The balance is determined automatically using statistical theory
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class JamesSteinEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly JamesSteinHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, CategoryValueMap<T>>? _encodingMap;
    private T _globalMean = MathHelper.GetNumericOperations<T>().Zero;
    private int _nInputFeatures;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public JamesSteinHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public T GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="JamesSteinEncoder{T}"/>.
    /// </summary>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public JamesSteinEncoder(
        JamesSteinHandleUnknown handleUnknown = JamesSteinHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "JamesSteinEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing James-Stein shrinkage estimates.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var engine = Engine;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Compute global mean and variance
        int n = target.Length;
        var y = Tensor<T>.FromVector(target);
        _globalMean = engine.TensorMean(y);
        T globalMean = _globalMean;
        T globalVariance = NumOps.Divide(
            engine.TensorSumOfSquares(engine.TensorSubtractScalar(y, globalMean)), NumOps.FromDouble(n - 1));
        bool degenerateVariance = NumOps.LessThan(globalVariance, NumOps.FromDouble(1e-10));
        T eps = NumOps.FromDouble(1e-10);
        T two = NumOps.FromDouble(2);

        _encodingMap = new Dictionary<int, CategoryValueMap<T>>();
        foreach (int col in columnsToProcess)
        {
            _encodingMap[col] = CategoryValueMap<T>.FromTargetStatistics(PreprocessingTensorOps.Column(data, col), y, (counts, sums) =>
            {
                var categoryMean = engine.TensorDivide(sums, counts);
                var full = PreprocessingTensorOps.Filled(new[] { counts.Length }, NumOps.One);

                // James-Stein shrinkage B = clamp(1 - (count - 2) * sigma^2 / (count * (mean - grand_mean)^2), 0, 1);
                // full shrinkage to the global mean for count <= 2, a degenerate variance or a vanishing denominator
                Tensor<T> shrinkage = full;
                if (!degenerateVariance)
                {
                    var denominator = engine.TensorMultiply(counts, engine.TensorSquare(engine.TensorSubtractScalar(categoryMean, globalMean)));
                    var partial = engine.TensorClamp(
                        PreprocessingTensorOps.OneMinus(engine.TensorDivide(
                            engine.TensorMultiplyScalar(engine.TensorSubtractScalar(counts, two), globalVariance), denominator)),
                        NumOps.Zero, NumOps.One);
                    var shrinkable = engine.TensorMultiply(
                        engine.TensorGreaterThan(counts, two), PreprocessingTensorOps.AtLeast(denominator, eps));
                    shrinkage = engine.TensorWhere(shrinkable, partial, full);
                }

                // (1 - B) * category_mean + B * global_mean
                return engine.TensorAdd(
                    engine.TensorMultiply(PreprocessingTensorOps.OneMinus(shrinkage), categoryMean),
                    engine.TensorMultiplyScalar(shrinkage, globalMean));
            });
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
    /// Transforms the data using fitted encodings.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_encodingMap is null)
        {
            throw new InvalidOperationException("JamesSteinEncoder has not been fitted.");
        }

        var processSet = new HashSet<int>(GetColumnsToProcess(data.Columns));
        return CategoryValueMap<T>.EncodeColumns(data, processSet, _encodingMap, _ => _globalMean,
            _handleUnknown == JamesSteinHandleUnknown.Error);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("JamesSteinEncoder does not support inverse transformation.");
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
/// Specifies how to handle unknown categories in JamesSteinEncoder.
/// </summary>
public enum JamesSteinHandleUnknown
{
    /// <summary>
    /// Use the global target mean for unknown categories.
    /// </summary>
    UseGlobalMean,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
