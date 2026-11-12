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
/// Selects features based on how well their density distributions separate
/// between different classes.
/// </para>
/// <para><b>For Beginners:</b> KDE estimates how data points are spread out.
/// Features where different classes have clearly separate density patterns
/// are better for classification.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KDESelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _bandwidth;

    private Vector<T>? _separationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Bandwidth => _bandwidth;
    public Vector<T>? SeparationScores => _separationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KDESelector(
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
            "KDESelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize features (population std; a feature with std not above 1e-10 is left as is)
        var mean = engine.ReduceMean(X, new[] { 0 }, true);
        var centered = engine.TensorSubtract(X, engine.TensorBroadcastTo(mean, new[] { n, p }));
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, true));
        var standardized = engine.TensorWhere(
            engine.TensorBroadcastTo(engine.TensorGreaterThan(std, NumOps.FromDouble(1e-10)), new[] { n, p }),
            engine.TensorDivide(centered, engine.TensorBroadcastTo(std, new[] { n, p })), X);

        // Compute class-specific densities at evaluation points (50 points from each feature's min to max)
        int nEvalPoints = 50;
        var minVal = PreprocessingTensorOps.ColumnMin(standardized);
        var evalPoints = FeatureSelectionTensorOps.EvenGrid(
            minVal, engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(standardized), minVal), nEvalPoints);

        // Compute KDE for each class
        var classDensities = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
            classDensities[c] = ComputeKDE(
                FeatureSelectionTensorOps.Rows(standardized, FeatureSelectionTensorOps.GroupMembers(classIndex, c)), evalPoints);

        // Compute separation score: average symmetric KL divergence between class densities
        var totalSeparation = new Tensor<T>(new[] { p });
        int pairs = 0;
        for (int ci = 0; ci < k; ci++)
        {
            for (int cj = ci + 1; cj < k; cj++)
            {
                totalSeparation = engine.TensorAdd(totalSeparation, ComputeSymmetricKL(classDensities[ci], classDensities[cj]));
                pairs++;
            }
        }

        var scores = pairs > 0 ? engine.TensorDivideScalar(totalSeparation, NumOps.FromDouble(pairs)) : totalSeparation;
        _separationScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }
    }

    private Tensor<T> ComputeKDE(Tensor<T> data, Tensor<T> evalPoints)
    {
        var engine = Engine;
        T normFactor = NumOps.Divide(NumOps.One,
            NumOps.FromDouble(data.Shape[0] * _bandwidth * Math.Sqrt(2 * Math.PI)));
        var sums = FeatureSelectionTensorOps.GaussianKernelSums(evalPoints, data, NumOps.FromDouble(_bandwidth));

        // Add small constant to avoid log(0)
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(sums, normFactor), NumOps.FromDouble(1e-10));
    }

    private Tensor<T> ComputeSymmetricKL(Tensor<T> p, Tensor<T> q)
    {
        // Sum over the evaluation points where both densities are above 1e-10
        var engine = Engine;
        T floor = NumOps.FromDouble(1e-10);
        var both = engine.TensorMultiply(engine.TensorGreaterThan(p, floor), engine.TensorGreaterThan(q, floor));
        var klPQ = engine.TensorMultiply(p, engine.TensorLog(engine.TensorDivide(p, q)));
        var klQP = engine.TensorMultiply(q, engine.TensorLog(engine.TensorDivide(q, p)));
        var terms = engine.TensorWhere(both, engine.TensorAdd(klPQ, klQP), new Tensor<T>(p.Shape.ToArray()));
        return engine.TensorDivideScalar(engine.ReduceSum(terms, new[] { 0 }, false), NumOps.FromDouble(2));
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
            throw new InvalidOperationException("KDESelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KDESelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KDESelector has not been fitted.");

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
