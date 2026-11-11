using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// MMPC (Max-Min Parents and Children) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using the MMPC algorithm, which identifies the Markov
/// blanket of the target - the minimal set of features needed for prediction.
/// </para>
/// <para><b>For Beginners:</b> MMPC finds features that are direct causes or
/// effects of the target (parents/children in a causal graph). It uses conditional
/// independence tests to identify these relationships. The result is a minimal
/// set of truly relevant features, removing spurious correlations.
/// </para>
/// <para>
/// <b>Note:</b> The alpha parameter in this implementation is used as a minimum association
/// threshold (correlation magnitude), not as a p-value significance level as in traditional MMPC.
/// Features with association below this threshold are considered conditionally independent.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MMPCSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minAssociationThreshold;

    private Vector<T>? _associationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MinAssociationThreshold => _minAssociationThreshold;
    public Vector<T>? AssociationScores => _associationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MMPCSelector(
        int nFeaturesToSelect = 10,
        double minAssociationThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minAssociationThreshold <= 0 || minAssociationThreshold >= 1)
            throw new ArgumentException("Minimum association threshold must be between 0 and 1.", nameof(minAssociationThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minAssociationThreshold = minAssociationThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MMPCSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var xc = engine.TensorSubtract(X, engine.ReduceMean(X, new[] { 0 }, true));
        var yc = engine.TensorSubtractScalar(y, engine.TensorMean(y));
        T threshold = NumOps.FromDouble(_minAssociationThreshold);

        // Marginal association with the target
        var scores = engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(X, y));
        _associationScores = scores.ToVector();

        // Phase 1: Forward - greedy min-max selection. minAssoc[j] is the smallest association of j with
        // the target over the empty set and every single current candidate (partial association).
        var candidates = new List<int>();
        var isCandidate = new bool[p];
        var partialColumns = new Dictionary<int, Tensor<T>>();
        var excluded = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(-1.0));
        var minAssoc = scores;

        while (candidates.Count < Math.Min(_nFeaturesToSelect, p))
        {
            var open = engine.TensorWhere(isCandidate, excluded, minAssoc);
            int bestFeature = PreprocessingTensorOps.FirstArgMax(open);
            if (PreprocessingTensorOps.IsBelow(engine.TensorMaxValue(open), threshold))
                break;

            candidates.Add(bestFeature);
            isCandidate[bestFeature] = true;
            var partial = PartialAssociations(xc, yc, bestFeature);
            partialColumns[bestFeature] = partial;
            minAssoc = engine.TensorMin(minAssoc, partial);
        }

        // Phase 2: Backward - remove candidates whose association falls below the threshold given
        // every other candidate.
        if (candidates.Count > 1)
        {
            int m = candidates.Count;
            var candidateIndex = new Tensor<int>(candidates.ToArray(), new[] { m });
            var columns = new Tensor<T>[m];
            for (int b = 0; b < m; b++)
                columns[b] = engine.TensorIndexSelect(partialColumns[candidates[b]], candidateIndex, 0);
            var pairwise = engine.TensorStack(columns, 1); // [a, b] = partial association of a given b
            var offDiagonal = PreprocessingTensorOps.Not(engine.TensorEye<T>(m));
            var dependent = engine.TensorMultiply(PreprocessingTensorOps.AtLeast(pairwise, threshold), offDiagonal);
            var anyDependent = engine.ReduceMax(dependent, new[] { 1 }, false);
            var independent = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(anyDependent));
            var remove = new HashSet<int>(independent.Select(a => candidates[a]));
            candidates.RemoveAll(remove.Contains);
            foreach (int j in remove)
                isCandidate[j] = false;
        }

        // If we have fewer than requested, add more by score
        if (candidates.Count < _nFeaturesToSelect)
        {
            int extra = Math.Min(_nFeaturesToSelect - candidates.Count, p - candidates.Count);
            var open = engine.TensorWhere(isCandidate, excluded, scores);
            candidates.AddRange(PreprocessingTensorOps.TopKIndicesSorted(open, extra));
        }

        _selectedIndices = candidates.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Tensor<T> PartialAssociations(Tensor<T> xc, Tensor<T> yc, int conditioning)
    {
        // |corr(resid(x_j | z), resid(y | z))| for every column j, with z the conditioning column
        var engine = Engine;
        int n = xc.Shape[0];
        var z = PreprocessingTensorOps.ColumnVector(xc, conditioning);
        T varZ = engine.TensorSumOfSquares(z);
        var residX = xc;
        var residY = yc;
        if (NumOps.GreaterThan(varZ, NumOps.FromDouble(1e-10)))
        {
            var zColumn = engine.Reshape(z, new[] { n, 1 });
            var slopeX = engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(zColumn), xc), varZ);
            residX = engine.TensorSubtract(xc, engine.TensorMatMul(zColumn, slopeX));
            T slopeY = NumOps.Divide(engine.TensorSum(engine.TensorMultiply(yc, z)), varZ);
            residY = engine.TensorSubtract(yc, engine.TensorMultiplyScalar(z, slopeY));
        }

        return engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(residX, residY));
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMPCSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MMPCSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMPCSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMPCSelector has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
