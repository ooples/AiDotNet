using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using M-estimate regularization.
/// </summary>
/// <remarks>
/// <para>
/// MEstimateEncoder applies M-estimate smoothing to target encoding, which adds
/// a regularization parameter 'm' that controls shrinkage toward the global mean.
/// </para>
/// <para>
/// The formula: encoded = (n * category_mean + m * global_mean) / (n + m)
/// where n is the count of samples in the category and m is the smoothing parameter.
/// </para>
/// <para><b>For Beginners:</b> M-estimate is like adding 'm' fake samples:
/// - Each fake sample has the global mean as its target
/// - Categories with few samples get pulled toward the global mean
/// - Categories with many samples stay close to their actual mean
/// - Higher m = more smoothing toward global mean
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MEstimateEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _m;
    private readonly MEstimateHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, CategoryValueMap<T>>? _encodingMap;
    private T _globalMean = MathHelper.GetNumericOperations<T>().Zero;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the smoothing parameter m.
    /// </summary>
    public double M => _m;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public MEstimateHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public T GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="MEstimateEncoder{T}"/>.
    /// </summary>
    /// <param name="m">Smoothing parameter (number of virtual samples). Defaults to 1.0.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public MEstimateEncoder(
        double m = 1.0,
        MEstimateHandleUnknown handleUnknown = MEstimateHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (m < 0)
        {
            throw new ArgumentException("Smoothing parameter m must be non-negative.", nameof(m));
        }

        _m = m;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MEstimateEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing M-estimate encodings.
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

        // Compute global mean
        var y = Tensor<T>.FromVector(target);
        _globalMean = engine.TensorMean(y);
        T m = NumOps.FromDouble(_m);
        T prior = NumOps.Multiply(m, _globalMean);

        // M-estimate per category: (sum + m * prior) / (count + m)
        _encodingMap = new Dictionary<int, CategoryValueMap<T>>();
        foreach (int col in columnsToProcess)
        {
            _encodingMap[col] = CategoryValueMap<T>.FromTargetStatistics(PreprocessingTensorOps.Column(data, col), y,
                (counts, sums) => engine.TensorDivide(engine.TensorAddScalar(sums, prior), engine.TensorAddScalar(counts, m)));
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
            throw new InvalidOperationException("MEstimateEncoder has not been fitted.");
        }

        var processSet = new HashSet<int>(GetColumnsToProcess(data.Columns));
        return CategoryValueMap<T>.EncodeColumns(data, processSet, _encodingMap, _ => _globalMean,
            _handleUnknown == MEstimateHandleUnknown.Error);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MEstimateEncoder does not support inverse transformation.");
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
/// Specifies how to handle unknown categories in MEstimateEncoder.
/// </summary>
public enum MEstimateHandleUnknown
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
