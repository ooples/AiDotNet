using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Kernel Density Estimation (KDE) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses kernel density estimation to select features that maximize the
/// separation of class-conditional density estimates.
/// </para>
/// <para><b>For Beginners:</b> KDE estimates the probability distribution of
/// your data smoothly. This selector finds features where different classes
/// have clearly different distributions, making it easier to tell classes apart.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KDEBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _bandwidth;

    private Vector<T>? _kdeSeparationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Bandwidth => _bandwidth;
    public Vector<T>? KDESeparationScores => _kdeSeparationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KDEBasedSelector(
        int nFeaturesToSelect = 10,
        double bandwidth = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _bandwidth = bandwidth;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KDEBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Feature values per class
        var classValues = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
            classValues[c] = FeatureSelectionTensorOps.Rows(X, FeatureSelectionTensorOps.GroupMembers(classIndex, c));

        // For each feature, compute the Hellinger distance between class densities, averaged over class pairs
        var totalSeparation = new Tensor<T>(new[] { p });
        int pairCount = 0;
        for (int c1 = 0; c1 < k; c1++)
        {
            for (int c2 = c1 + 1; c2 < k; c2++)
            {
                totalSeparation = engine.TensorAdd(totalSeparation,
                    ComputeHellingerDistance(classValues[c1], classValues[c2], p));
                pairCount++;
            }
        }

        var scores = pairCount > 0
            ? engine.TensorDivideScalar(totalSeparation, NumOps.FromDouble(pairCount))
            : totalSeparation;
        _kdeSeparationScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeHellingerDistance(Tensor<T> values1, Tensor<T> values2, int p)
    {
        // Determine evaluation points per feature: 100 points from min - 3h to max + 3h over both classes
        var engine = Engine;
        T h = NumOps.FromDouble(_bandwidth);
        T margin = NumOps.FromDouble(3 * _bandwidth);
        var minVal = engine.TensorSubtractScalar(engine.TensorMin(
            PreprocessingTensorOps.ColumnMin(values1), PreprocessingTensorOps.ColumnMin(values2)), margin);
        var maxVal = engine.TensorAddScalar(engine.TensorMax(
            PreprocessingTensorOps.ColumnMax(values1), PreprocessingTensorOps.ColumnMax(values2)), margin);

        int nPoints = 100;
        var span = engine.TensorSubtract(maxVal, minVal);
        var step = engine.TensorDivideScalar(span, NumOps.FromDouble(nPoints - 1));
        var grid = FeatureSelectionTensorOps.EvenGrid(minVal, span, nPoints);

        var density1 = ComputeKDE(grid, values1, h);
        var density2 = ComputeKDE(grid, values2, h);
        var gap = engine.TensorSubtract(engine.TensorSqrt(density1), engine.TensorSqrt(density2));
        var hellingerSum = engine.ReduceSum(engine.TensorSquare(gap), new[] { 0 }, false);

        return engine.TensorSqrt(engine.TensorDivideScalar(engine.TensorMultiply(hellingerSum, step), NumOps.FromDouble(2)));
    }

    private Tensor<T> ComputeKDE(Tensor<T> grid, Tensor<T> values, T h)
    {
        var engine = Engine;
        var sums = engine.TensorDivideScalar(
            FeatureSelectionTensorOps.GaussianKernelSums(grid, values, h), NumOps.FromDouble(Math.Sqrt(2 * Math.PI)));
        T denominator = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(values.Shape[0]), h), PreprocessingTensorOps.Tolerance<T>(1e-10));
        return engine.TensorDivideScalar(sums, denominator);
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
            throw new InvalidOperationException("KDEBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KDEBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KDEBasedSelector has not been fitted.");

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
