using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Maximum Mean Discrepancy (MMD) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses MMD to measure the distribution difference between classes in kernel space,
/// selecting features that maximize this discrepancy.
/// </para>
/// <para><b>For Beginners:</b> MMD measures how different two probability distributions
/// are. By computing MMD between different classes, we can find features that make
/// the classes most distinguishable. Features with high MMD are good at separating
/// classes.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MMDBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;

    private Vector<T>? _mmdScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public Vector<T>? MMDScores => _mmdScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MMDBasedSelector(
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
            "MMDBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Class labels (rounded to the nearest integer) and the class-mean weights 1{y_i = c} / n_c
        var labels = engine.TensorRound(Tensor<T>.FromVector(target)).Cast<int>().AsSpan().ToArray();
        var classes = labels.Distinct().OrderBy(c => c).ToArray();
        var classIndex = new int[n];
        for (int i = 0; i < n; i++)
            classIndex[i] = Array.BinarySearch(classes, labels[i]);

        var scores = new Tensor<T>(new[] { p });
        if (classes.Length >= 2)
        {
            var membership = engine.TensorOneHot<T>(new Tensor<int>(classIndex, new[] { n }), classes.Length); // [n, c]
            var classMeans = engine.TensorDivide(membership, engine.ReduceSum(membership, new[] { 0 }, true));

            // MMD^2(a, b) = E[k(A, A')] + E[k(B, B')] - 2 E[k(A, B)] = sum K .* (d d') with d = w_a - w_b
            var pairWeights = new List<Tensor<T>>();
            for (int a = 0; a < classes.Length; a++)
            {
                for (int b = a + 1; b < classes.Length; b++)
                {
                    var difference = engine.TensorSubtract(
                        engine.TensorSlice(classMeans, new[] { 0, a }, new[] { n, 1 }),
                        engine.TensorSlice(classMeans, new[] { 0, b }, new[] { n, 1 }));
                    pairWeights.Add(engine.TensorMatMul(difference, engine.TensorTranspose(difference)));
                }
            }

            T negativeGamma = NumOps.FromDouble(-_gamma);
            var mmd = FeatureSelectionTensorOps.KernelWeightedSums(
                x,
                column => FeatureSelectionTensorOps.RbfColumnKernel(column, negativeGamma),
                pairWeights.ToArray(),
                Array.Empty<Tensor<T>>());

            // Average of the (non-negative) MMD^2 over the class pairs
            scores = engine.ReduceMean(engine.TensorMax(mmd, NumOps.Zero), new[] { 1 }, false);
        }

        _mmdScores = scores.ToVector();

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
            throw new InvalidOperationException("MMDBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MMDBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMDBasedSelector has not been fitted.");

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
