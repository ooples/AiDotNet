using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Distance;

/// <summary>
/// Chebyshev Distance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize class separation based on Chebyshev (L-infinity)
/// distance between class centroids.
/// </para>
/// <para><b>For Beginners:</b> Chebyshev distance is the maximum difference in any
/// single dimension. It's useful when you care about the worst-case difference
/// between classes. Features with large maximum separations are preferred.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ChebyshevDistanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _distanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DistanceScores => _distanceScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ChebyshevDistanceSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChebyshevDistanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

    }
@@NEW
        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Compute min and max of every feature for each class
        var classMin = new Tensor<T>[k];
        var classMax = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
        {
            var rows = FeatureSelectionTensorOps.Rows(X, FeatureSelectionTensorOps.GroupMembers(classIndex, c));
            classMin[c] = PreprocessingTensorOps.ColumnMin(rows);
            classMax[c] = PreprocessingTensorOps.ColumnMax(rows);
        }

        // Maximum separation between class ranges
        T zero = NumOps.Zero;
        var maxSeparation = new Tensor<T>(new[] { p });
        for (int ci = 0; ci < k; ci++)
        {
            for (int cj = ci + 1; cj < k; cj++)
            {
                // Gap between ranges
                var gap = engine.TensorMax(engine.TensorMax(
                    engine.TensorSubtract(classMin[ci], classMax[cj]),
                    engine.TensorSubtract(classMin[cj], classMax[ci])), zero);

                // Or use maximum absolute difference
                var maxDiff = engine.TensorMax(
                    engine.TensorAbs(engine.TensorSubtract(classMax[ci], classMin[cj])),
                    engine.TensorAbs(engine.TensorSubtract(classMax[cj], classMin[ci])));

                maxSeparation = engine.TensorMax(maxSeparation,
                    engine.TensorWhere(engine.TensorGreaterThan(gap, zero), gap, maxDiff));
            }
        }

        _distanceScores = maxSeparation.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(maxSeparation, numToSelect);

        IsFitted = true;
    }
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
            throw new InvalidOperationException("ChebyshevDistanceSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ChebyshevDistanceSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChebyshevDistanceSelector has not been fitted.");

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
