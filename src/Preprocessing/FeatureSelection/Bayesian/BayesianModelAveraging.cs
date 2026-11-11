using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Bayesian Model Averaging Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their posterior inclusion probabilities across
/// all possible models, weighted by model posterior probabilities.
/// </para>
/// <para><b>For Beginners:</b> Instead of picking one best model, BMA considers
/// all possible models and weights them by how likely each is. Features that
/// appear in many high-probability models are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BayesianModelAveraging<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nModels;
    private readonly double _priorInclusion;

    private Vector<T>? _inclusionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? InclusionProbabilities => _inclusionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BayesianModelAveraging(
        int nFeaturesToSelect = 10,
        int nModels = 1000,
        double priorInclusion = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nModels = nModels;
        _priorInclusion = priorInclusion;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BayesianModelAveraging requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Sample every model (subset of features) up front; the draw order matches one model at a time.
        var included = new bool[_nModels * p];
        for (int m = 0; m < _nModels; m++)
        {
            bool any = false;
            for (int j = 0; j < p; j++)
            {
                if (rand.NextDouble() < _priorInclusion)
                {
                    included[m * p + j] = true;
                    any = true;
                }
            }

            if (!any)
                included[m * p + rand.Next(p)] = true;
        }

        var ones = PreprocessingTensorOps.Filled(new[] { _nModels * p }, NumOps.One);
        var inclusion = engine.Reshape(
            engine.TensorWhere(included, ones, new Tensor<T>(new[] { _nModels * p })), new[] { _nModels, p });

        // Model likelihood (R-squared proxy from per-feature squared correlations) with a BIC-like penalty.
        T eps = NumOps.FromDouble(1e-10);
        Tensor<T> modelWeights;
        T ssTot = engine.TensorSumOfSquares(engine.TensorSubtractScalar(y, engine.TensorMean(y)));
        if (PreprocessingTensorOps.IsBelow(ssTot, eps))
        {
            modelWeights = PreprocessingTensorOps.Filled(new[] { _nModels, 1 }, eps);
        }
        else
        {
            var rSquaredPerFeature = engine.Reshape(
                engine.TensorSquare(PreprocessingTensorOps.TargetCorrelations(X, y)), new[] { p, 1 });
            var modelSizes = engine.TensorMatMul(inclusion, PreprocessingTensorOps.Filled(new[] { p, 1 }, NumOps.One));
            var rSquared = engine.TensorMin(
                engine.TensorDivide(engine.TensorMatMul(inclusion, rSquaredPerFeature), modelSizes), NumOps.FromDouble(0.99));
            var bic = engine.TensorAdd(
                engine.TensorMultiplyScalar(
                    engine.TensorLog(engine.TensorAddScalar(engine.TensorNegate(rSquared), NumOps.FromDouble(1 + 1e-10))),
                    NumOps.FromDouble(n)),
                engine.TensorMultiplyScalar(modelSizes, NumOps.FromDouble(Math.Log(n))));
            modelWeights = engine.TensorExp(engine.TensorMultiplyScalar(bic, NumOps.FromDouble(-0.5)));
        }

        // Posterior inclusion probabilities: weighted inclusion over the total model weight
        T totalWeight = engine.TensorSum(modelWeights);
        var inclusionCounts = engine.Reshape(
            engine.TensorMatMul(engine.TensorTranspose(inclusion), modelWeights), new[] { p });
        var probabilities = engine.TensorDivideScalar(inclusionCounts, NumOps.Add(totalWeight, eps));
        _inclusionProbabilities = probabilities.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(probabilities, numToSelect);

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
            throw new InvalidOperationException("BayesianModelAveraging has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BayesianModelAveraging does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianModelAveraging has not been fitted.");

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
