using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Kernel Fisher Discriminant-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Applies the Fisher criterion in kernel space to find features that maximize
/// class separation using non-linear transformations.
/// </para>
/// <para><b>For Beginners:</b> Fisher's criterion tries to maximize the distance
/// between class means while minimizing the spread within each class. Doing this
/// in kernel space allows us to find non-linear boundaries between classes,
/// selecting features that help separate classes in complex ways.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KernelFisherSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;

    private Vector<T>? _fisherScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public Vector<T>? FisherScores => _fisherScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KernelFisherSelector(
        int nFeaturesToSelect = 10,
        double gamma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gamma = gamma;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KernelFisherSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var x = Tensor<T>.FromMatrix(data);

        // Class labels (rounded to the nearest integer)
        var labels = engine.TensorRound(Tensor<T>.FromVector(target)).Cast<int>().AsSpan().ToArray();
        var classes = labels.Distinct().OrderBy(c => c).ToArray();
        var classIndex = new int[n];
        for (int i = 0; i < n; i++)
            classIndex[i] = Array.BinarySearch(classes, labels[i]);
        var membership = engine.TensorOneHot<T>(new Tensor<int>(classIndex, new[] { n }), classes.Length); // [n, c]
        var classCounts = engine.ReduceSum(membership, new[] { 0 }, false); // [c]

        // Each sample's kernel row sum sum_j k(x_i, x_j) under the single-feature RBF kernel
        T negativeGamma = NumOps.FromDouble(-_gamma);
        var rowSums = FeatureSelectionTensorOps.KernelRowSums(
            x, column => FeatureSelectionTensorOps.RbfColumnKernel(column, negativeGamma)); // [n, p]

        // Fisher score of the row sums: between-class over within-class scatter
        var overallMean = engine.ReduceMean(rowSums, new[] { 0 }, true); // [1, p]
        var classMeans = engine.TensorDivide(
            engine.TensorMatMul(engine.TensorTranspose(membership), rowSums),
            engine.Reshape(classCounts, new[] { classes.Length, 1 })); // [c, p]
        var between = engine.Reshape(engine.TensorMatMul(
            engine.Reshape(classCounts, new[] { 1, classes.Length }),
            engine.TensorSquare(engine.TensorSubtract(classMeans, overallMean))), new[] { p });
        var within = engine.ReduceSum(
            engine.TensorSquare(engine.TensorSubtract(rowSums, engine.TensorMatMul(membership, classMeans))), new[] { 0 }, false);
        var defined = engine.TensorGreaterThan(within, NumOps.FromDouble(1e-10));
        var safeWithin = engine.TensorWhere(defined, within, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var scores = engine.TensorWhere(defined, engine.TensorDivide(between, safeWithin), new Tensor<T>(new[] { p }));
        _fisherScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("KernelFisherSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KernelFisherSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelFisherSelector has not been fitted.");

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
