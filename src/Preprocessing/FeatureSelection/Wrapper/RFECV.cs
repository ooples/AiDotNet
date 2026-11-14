using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Recursive Feature Elimination with Cross-Validation (RFECV).
/// </summary>
/// <remarks>
/// <para>
/// RFECV uses cross-validation to find the optimal number of features.
/// It performs RFE for different feature counts and selects the count
/// that maximizes cross-validated performance.
/// </para>
/// <para><b>For Beginners:</b> Regular RFE requires you to specify how many
/// features to keep. RFECV figures out the best number automatically by
/// testing different counts and seeing which gives the best validation score.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RFECV<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _minFeatures;
    private readonly int _step;
    private readonly int _nFolds;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, T>? _scoringFunc;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _importanceFunc;

    private Vector<T>? _cvScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private int _optimalFeatureCount;

    public int MinFeatures => _minFeatures;
    public Vector<T>? CVScores => _cvScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    public int OptimalFeatureCount => _optimalFeatureCount;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RFECV(
        Func<Matrix<T>, Vector<T>, T>? scoringFunc = null,
        Func<Matrix<T>, Vector<T>, Vector<T>>? importanceFunc = null,
        int minFeatures = 1,
        int step = 1,
        int nFolds = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (minFeatures < 1)
            throw new ArgumentException("Minimum features must be at least 1.", nameof(minFeatures));
        if (step < 1)
            throw new ArgumentException("Step must be at least 1.", nameof(step));
        if (nFolds < 2)
            throw new ArgumentException("Number of folds must be at least 2.", nameof(nFolds));

        _scoringFunc = scoringFunc;
        _importanceFunc = importanceFunc;
        _minFeatures = minFeatures;
        _step = step;
        _nFolds = nFolds;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RFECV requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Generate fold indices
        var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
        var foldIndices = new int[n];
        for (int i = 0; i < n; i++)
            foldIndices[indices[i]] = i % _nFolds;

        // Track which features are still active
        var activeFeatures = Enumerable.Range(0, p).ToList();
        var featureRankings = new int[p];
        var cvScoresList = new List<T>();
        T foldCount = NumOps.FromDouble(_nFolds);
        int currentRank = p;

        while (activeFeatures.Count >= _minFeatures)
        {
            // Cross-validation for current feature set
            T totalScore = NumOps.Zero;

            for (int fold = 0; fold < _nFolds; fold++)
            {
                var trainIdx = new List<int>();
                var valIdx = new List<int>();

                for (int i = 0; i < n; i++)
                {
                    if (foldIndices[i] == fold)
                        valIdx.Add(i);
                    else
                        trainIdx.Add(i);
                }

                var trainData = ExtractSubset(data, trainIdx, activeFeatures);
                var trainTarget = ExtractVector(target, trainIdx);
                var valData = ExtractSubset(data, valIdx, activeFeatures);
                var valTarget = ExtractVector(target, valIdx);

                T score = EvaluateScore(trainData, trainTarget, valData, valTarget);
                totalScore = WrapperScoringOps.Add(totalScore, score);
            }

            cvScoresList.Add(WrapperScoringOps.Divide(totalScore, foldCount));

            if (activeFeatures.Count <= _minFeatures)
                break;

            // Get feature importances and eliminate least important
            var importances = GetImportances(data, target, activeFeatures, n);
            int nToRemove = Math.Min(_step, activeFeatures.Count - _minFeatures);

            var toRemove = PreprocessingTensorOps.StableOrder(Tensor<T>.FromVector(importances), descending: false)
                .Take(nToRemove)
                .Select(position => activeFeatures[position])
                .ToList();

            foreach (int f in toRemove)
            {
                featureRankings[f] = currentRank--;
                activeFeatures.Remove(f);
            }
        }

        // Assign remaining features rank 1
        foreach (int f in activeFeatures)
            featureRankings[f] = 1;

        var cvScores = WrapperScoringOps.FromList(cvScoresList);
        _cvScores = cvScores.ToVector();

        // Find optimal feature count (the first best score)
        int bestIdx = PreprocessingTensorOps.FirstArgMax(cvScores);

        _optimalFeatureCount = p - bestIdx * _step;
        if (_optimalFeatureCount < _minFeatures)
            _optimalFeatureCount = _minFeatures;

        // Select features with rank <= optimal count
        _selectedIndices = featureRankings
            .Select((rank, idx) => (Rank: rank, Index: idx))
            .Where(x => x.Rank <= _optimalFeatureCount)
            .OrderBy(x => x.Index)
            .Select(x => x.Index)
            .ToArray();

        IsFitted = true;
    }

    private Matrix<T> ExtractSubset(Matrix<T> data, List<int> rows, List<int> cols)
    {
        if (rows.Count == 0 || cols.Count == 0)
            return new Matrix<T>(rows.Count, cols.Count);

        var selectedRows = Engine.TensorIndexSelect(Tensor<T>.FromMatrix(data),
            new Tensor<int>(rows.ToArray(), new[] { rows.Count }), 0);
        return Engine.TensorIndexSelect(selectedRows, new Tensor<int>(cols.ToArray(), new[] { cols.Count }), 1).ToMatrix();
    }

    private Vector<T> ExtractVector(Vector<T> vec, List<int> indices)
    {
        if (indices.Count == 0)
            return new Vector<T>(0);

        return Engine.TensorIndexSelect(Tensor<T>.FromVector(vec),
            new Tensor<int>(indices.ToArray(), new[] { indices.Count }), 0).ToVector();
    }

    private T EvaluateScore(Matrix<T> trainData, Vector<T> trainTarget,
        Matrix<T> valData, Vector<T> valTarget)
    {
        if (_scoringFunc is not null)
            return _scoringFunc(valData, valTarget);

        // Default: validation R^2 of a least-squares fit on the training folds
        return ComputeValidationRSquared(trainData, trainTarget, valData, valTarget);
    }

    /// <summary>
    /// R^2 on the validation fold of the least-squares linear regression (with intercept) fitted on the training
    /// folds, as scikit-learn's RFECV scores a LinearRegression; 0 when either part has no rows. (The score used to
    /// be the mean |Pearson r| on the training folds, which ignored the validation fold.)
    /// </summary>
    private T ComputeValidationRSquared(Matrix<T> trainData, Vector<T> trainTarget,
        Matrix<T> valData, Vector<T> valTarget)
    {
        if (trainData.Rows == 0 || valData.Rows == 0)
            return NumOps.Zero;

        var slopes = WrapperScoringOps.LeastSquaresSlopes(
            Tensor<T>.FromMatrix(trainData), Tensor<T>.FromVector(trainTarget), out var xMean, out T yMean);
        var predictions = WrapperScoringOps.Predict(Tensor<T>.FromMatrix(valData), slopes, xMean, yMean);
        return WrapperScoringOps.RSquared(Tensor<T>.FromVector(valTarget), predictions);
    }

    private Vector<T> GetImportances(Matrix<T> data, Vector<T> target, List<int> activeFeatures, int n)
    {
        if (_importanceFunc is not null)
        {
            var subsetData = ExtractSubset(data, Enumerable.Range(0, n).ToList(), activeFeatures);
            return _importanceFunc(subsetData, target);
        }

        // Default: absolute correlation
        return WrapperScoringOps.AbsTargetCorrelations(data, target, activeFeatures.ToArray()).ToVector();
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
            throw new InvalidOperationException("RFECV has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RFECV does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RFECV has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
