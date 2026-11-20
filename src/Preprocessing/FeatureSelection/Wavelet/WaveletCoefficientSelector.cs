using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wavelet;

/// <summary>
/// Wavelet Coefficient based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the magnitude and correlation of their wavelet
/// coefficients with the target variable across multiple decomposition levels.
/// </para>
/// <para><b>For Beginners:</b> This decomposes features into different frequency
/// bands and checks which features have strong relationships with the target
/// at various scales, helping identify features with multi-scale predictive power.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WaveletCoefficientSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _decompositionLevel;

    private Vector<T>? _waveletScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int DecompositionLevel => _decompositionLevel;
    public Vector<T>? WaveletScores => _waveletScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WaveletCoefficientSelector(
        int nFeaturesToSelect = 10,
        int decompositionLevel = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decompositionLevel = decompositionLevel;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WaveletCoefficientSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Decompose every feature and the target (last column) together with the Haar DWT; an odd-length signal is
        // extended by repeating its last sample (pywt's default "symmetric" mode), and the final approximation
        // counts as one more level. The score is the mean |Pearson r| between feature and target coefficients.
        var current = Engine.TensorConcatenate(new[]
        {
            Tensor<T>.FromMatrix(data), Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })
        }, 1);
        T sqrt2 = NumOps.FromDouble(Math.Sqrt(2));
        var totalScore = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        int levels = 0;

        for (int level = 0; level < _decompositionLevel && current.Shape[0] >= 2; level++)
        {
            int len = current.Shape[0];
            if (len % 2 == 1)
            {
                current = Engine.TensorConcatenate(new[]
                {
                    current, Engine.TensorSlice(current, new[] { len - 1, 0 }, new[] { 1, p + 1 })
                }, 0);
                len++;
            }

            var pairs = Engine.Reshape(current, new[] { len / 2, 2, p + 1 });
            var first = Engine.TensorSliceAxis(pairs, 1, 0);
            var second = Engine.TensorSliceAxis(pairs, 1, 1);
            var detail = Engine.TensorDivideScalar(Engine.TensorSubtract(first, second), sqrt2);
            totalScore = Engine.TensorAdd(totalScore, AbsTargetCorrelations(detail, p));
            levels++;
            current = Engine.TensorDivideScalar(Engine.TensorAdd(first, second), sqrt2);
        }

        totalScore = Engine.TensorAdd(totalScore, AbsTargetCorrelations(current, p));
        levels++;

        var scores = Engine.TensorDivideScalar(totalScore, NumOps.FromDouble(levels));
        _waveletScores = scores.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// |Pearson r| of the first <paramref name="p"/> columns of <paramref name="coefficients"/> ([m, p + 1]) with
    /// its last column, 0 when a column is constant or there are fewer than two coefficients, as [p].
    /// </summary>
    private Tensor<T> AbsTargetCorrelations(Tensor<T> coefficients, int p)
    {
        int m = coefficients.Shape[0];
        if (m < 2)
            return PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        var features = Engine.TensorSlice(coefficients, new[] { 0, 0 }, new[] { m, p });
        var target = Engine.Reshape(Engine.TensorSlice(coefficients, new[] { 0, p }, new[] { m, 1 }), new[] { m });
        return Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(features, target));
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletCoefficientSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WaveletCoefficientSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletCoefficientSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
