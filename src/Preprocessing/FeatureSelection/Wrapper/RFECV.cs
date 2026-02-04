using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class RFECV<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _minFeatures;
    private readonly int _step;
    private readonly int _nFolds;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, double>? _scoringFunc;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _importanceFunc;

    private double[]? _cvScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private int _optimalFeatureCount;

    public int MinFeatures => _minFeatures;
    public double[]? CVScores => _cvScores;
    public int[]? SelectedIndices => _selectedIndices;
    public int OptimalFeatureCount => _optimalFeatureCount;
    public override bool SupportsInverseTransform => false;

    public RFECV(
        Func<Matrix<T>, Vector<T>, double>? scoringFunc = null,
        Func<Matrix<T>, Vector<T>, double[]>? importanceFunc = null,
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RFECV requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

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
        var cvScoresList = new List<double>();
        int currentRank = p;

        while (activeFeatures.Count >= _minFeatures)
        {
            // Cross-validation for current feature set
            double totalScore = 0;

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

                double score = EvaluateScore(trainData, trainTarget, valData, valTarget);
                totalScore += score;
            }

            cvScoresList.Add(totalScore / _nFolds);

            if (activeFeatures.Count <= _minFeatures)
                break;

            // Get feature importances and eliminate least important
            var importances = GetImportances(data, target, activeFeatures, n);
            int nToRemove = Math.Min(_step, activeFeatures.Count - _minFeatures);

            var toRemove = importances
                .Select((imp, idx) => (Importance: imp, FeatureIdx: activeFeatures[idx]))
                .OrderBy(x => x.Importance)
                .Take(nToRemove)
                .Select(x => x.FeatureIdx)
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

        _cvScores = cvScoresList.ToArray();

        // Find optimal feature count
        int bestIdx = 0;
        double bestScore = _cvScores[0];
        for (int i = 1; i < _cvScores.Length; i++)
        {
            if (_cvScores[i] > bestScore)
            {
                bestScore = _cvScores[i];
                bestIdx = i;
            }
        }

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
        var result = new T[rows.Count, cols.Count];
        for (int i = 0; i < rows.Count; i++)
            for (int j = 0; j < cols.Count; j++)
                result[i, j] = data[rows[i], cols[j]];
        return new Matrix<T>(result);
    }

    private Vector<T> ExtractVector(Vector<T> vec, List<int> indices)
    {
        var result = new T[indices.Count];
        for (int i = 0; i < indices.Count; i++)
            result[i] = vec[indices[i]];
        return new Vector<T>(result);
    }

    private double EvaluateScore(Matrix<T> trainData, Vector<T> trainTarget,
        Matrix<T> valData, Vector<T> valTarget)
    {
        if (_scoringFunc is not null)
            return _scoringFunc(valData, valTarget);

        // Default: correlation-based score
        return ComputeCorrelationScore(trainData, trainTarget, valData, valTarget);
    }

    private double ComputeCorrelationScore(Matrix<T> trainData, Vector<T> trainTarget,
        Matrix<T> valData, Vector<T> valTarget)
    {
        int p = trainData.Columns;
        int nTrain = trainData.Rows;

        double totalCorr = 0;
        for (int j = 0; j < p; j++)
        {
            double xMean = 0, yMean = 0;
            for (int i = 0; i < nTrain; i++)
            {
                xMean += NumOps.ToDouble(trainData[i, j]);
                yMean += NumOps.ToDouble(trainTarget[i]);
            }
            xMean /= nTrain;
            yMean /= nTrain;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < nTrain; i++)
            {
                double dx = NumOps.ToDouble(trainData[i, j]) - xMean;
                double dy = NumOps.ToDouble(trainTarget[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                totalCorr += Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        return totalCorr / p;
    }

    private double[] GetImportances(Matrix<T> data, Vector<T> target, List<int> activeFeatures, int n)
    {
        if (_importanceFunc is not null)
        {
            var subsetData = ExtractSubset(data, Enumerable.Range(0, n).ToList(), activeFeatures);
            return _importanceFunc(subsetData, target);
        }

        // Default: absolute correlation
        var importances = new double[activeFeatures.Count];
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < activeFeatures.Count; j++)
        {
            int f = activeFeatures[j];
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, f]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, f]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                importances[j] = Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        return importances;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RFECV has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RFECV does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RFECV has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
