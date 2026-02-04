using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Bidirectional Feature Search for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Bidirectional search combines forward selection and backward elimination in an
/// interleaved manner. It can add or remove features at each step, potentially
/// finding better solutions than either method alone.
/// </para>
/// <para><b>For Beginners:</b> Instead of just adding features (forward) or just
/// removing them (backward), this method does both alternately. This can escape
/// local optima and find feature combinations that neither approach would find alone.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BidirectionalSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxIterations;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxIterations => _maxIterations;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BidirectionalSearch(
        int nFeaturesToSelect = 10,
        int maxIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxIterations = maxIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BidirectionalSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _featureScores = new double[p];
        var selected = new HashSet<int>();
        var notSelected = new HashSet<int>(Enumerable.Range(0, p));

        // Start with half features randomly selected
        int initialCount = Math.Min(_nFeaturesToSelect, p / 2 + 1);
        var initialFeatures = Enumerable.Range(0, p).OrderBy(_ => random.Next()).Take(initialCount);
        foreach (int f in initialFeatures)
        {
            selected.Add(f);
            notSelected.Remove(f);
        }

        double currentScore = EvaluateSubset(data, target, selected.ToArray(), n, p);
        bool improved = true;

        for (int iter = 0; iter < _maxIterations && improved; iter++)
        {
            improved = false;

            // Forward step: try adding features
            if (selected.Count < _nFeaturesToSelect)
            {
                int bestAdd = -1;
                double bestAddScore = currentScore;

                foreach (int candidate in notSelected)
                {
                    selected.Add(candidate);
                    double score = EvaluateSubset(data, target, selected.ToArray(), n, p);
                    selected.Remove(candidate);

                    if (score > bestAddScore)
                    {
                        bestAddScore = score;
                        bestAdd = candidate;
                    }
                }

                if (bestAdd >= 0)
                {
                    selected.Add(bestAdd);
                    notSelected.Remove(bestAdd);
                    currentScore = bestAddScore;
                    improved = true;
                }
            }

            // Backward step: try removing features
            if (selected.Count > 1)
            {
                int bestRemove = -1;
                double bestRemoveScore = currentScore;

                foreach (int candidate in selected.ToList())
                {
                    selected.Remove(candidate);
                    double score = EvaluateSubset(data, target, selected.ToArray(), n, p);
                    selected.Add(candidate);

                    if (score > bestRemoveScore)
                    {
                        bestRemoveScore = score;
                        bestRemove = candidate;
                    }
                }

                if (bestRemove >= 0)
                {
                    selected.Remove(bestRemove);
                    notSelected.Add(bestRemove);
                    currentScore = bestRemoveScore;
                    improved = true;
                }
            }
        }

        // Adjust to exact number of features requested
        while (selected.Count > _nFeaturesToSelect)
        {
            int weakest = selected.OrderBy(f => _featureScores[f]).First();
            selected.Remove(weakest);
            notSelected.Add(weakest);
        }

        while (selected.Count < _nFeaturesToSelect && notSelected.Any())
        {
            var scoresForNotSelected = notSelected
                .Select(f => (f, EvaluateFeature(data, target, f, n)))
                .OrderByDescending(x => x.Item2);
            var best = scoresForNotSelected.First().f;
            selected.Add(best);
            notSelected.Remove(best);
        }

        foreach (int s in selected)
            _featureScores[s] = EvaluateFeature(data, target, s, n);

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] features, int n, int p)
    {
        if (features.Length == 0) return 0;

        double score = 0;
        foreach (int f in features)
            score += EvaluateFeature(data, target, f, n);

        return score / features.Length;
    }

    private double EvaluateFeature(Matrix<T> data, Vector<T> target, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, j]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BidirectionalSearch has not been fitted.");

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
        throw new NotSupportedException("BidirectionalSearch does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BidirectionalSearch has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
