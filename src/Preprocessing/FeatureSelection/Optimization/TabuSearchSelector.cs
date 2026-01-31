using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Optimization;

/// <summary>
/// Tabu Search based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses tabu search optimization with a memory structure that prevents revisiting
/// recently explored solutions, helping escape local optima.
/// </para>
/// <para><b>For Beginners:</b> Tabu search keeps a "forbidden list" of recent moves.
/// This forces the algorithm to explore new areas instead of cycling back to old
/// solutions, helping find better feature combinations.
/// </para>
/// </remarks>
public class TabuSearchSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _tabuTenure;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int TabuTenure => _tabuTenure;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TabuSearchSelector(
        int nFeaturesToSelect = 10,
        int tabuTenure = 10,
        int nIterations = 500,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _tabuTenure = tabuTenure;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TabuSearchSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize current solution
        var current = new HashSet<int>();
        var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect).ToList();
        foreach (int idx in indices)
            current.Add(idx);

        double currentScore = EvaluateFitness(X, y, current, n, p);
        var best = new HashSet<int>(current);
        double bestScore = currentScore;

        // Tabu list: feature -> iteration when it becomes non-tabu
        var tabuList = new Dictionary<int, int>();

        _featureImportances = new double[p];

        for (int iter = 0; iter < _nIterations; iter++)
        {
            int bestMoveFeature = -1;
            double bestMoveScore = double.MinValue;

            // Try all possible moves
            // 1. Remove a feature and add another
            foreach (int toRemove in current)
            {
                if (tabuList.TryGetValue(toRemove, out int tabuUntil) && iter < tabuUntil)
                    continue; // Tabu

                foreach (int toAdd in Enumerable.Range(0, p).Except(current))
                {
                    if (tabuList.TryGetValue(toAdd, out int addTabuUntil) && iter < addTabuUntil)
                        continue; // Tabu

                    var neighbor = new HashSet<int>(current);
                    neighbor.Remove(toRemove);
                    neighbor.Add(toAdd);

                    double score = EvaluateFitness(X, y, neighbor, n, p);

                    // Aspiration criterion: accept if better than best
                    if (score > bestMoveScore || score > bestScore)
                    {
                        bestMoveScore = score;
                        bestMoveFeature = toAdd;
                    }
                }
            }

            // Apply best move
            if (bestMoveFeature >= 0)
            {
                // Find which feature to swap out
                int toRemove = -1;
                double lowestContrib = double.MaxValue;
                foreach (int j in current)
                {
                    var test = new HashSet<int>(current);
                    test.Remove(j);
                    double contrib = currentScore - EvaluateFitness(X, y, test, n, p);
                    if (contrib < lowestContrib)
                    {
                        lowestContrib = contrib;
                        toRemove = j;
                    }
                }

                if (toRemove >= 0)
                {
                    current.Remove(toRemove);
                    current.Add(bestMoveFeature);
                    tabuList[toRemove] = iter + _tabuTenure;
                    tabuList[bestMoveFeature] = iter + _tabuTenure;

                    currentScore = bestMoveScore;

                    if (currentScore > bestScore)
                    {
                        best = new HashSet<int>(current);
                        bestScore = currentScore;
                    }
                }
            }

            // Track feature importance
            foreach (int j in current)
                _featureImportances[j] += currentScore;
        }

        _selectedIndices = best.OrderBy(x => x).ToArray();

        // Normalize feature importances
        double maxImportance = _featureImportances.Max();
        if (maxImportance > 0)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= maxImportance;
        }

        IsFitted = true;
    }

    private double EvaluateFitness(double[,] X, double[] y, HashSet<int> selected, int n, int p)
    {
        if (selected.Count == 0) return 0;

        double totalCorr = 0;
        foreach (int j in selected)
        {
            double corr = ComputeCorrelation(X, y, j, n);
            totalCorr += Math.Abs(corr);
        }

        return totalCorr / selected.Count;
    }

    private double ComputeCorrelation(double[,] X, double[] y, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += X[i, j];
            yMean += y[i];
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = X[i, j] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TabuSearchSelector has not been fitted.");

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
        throw new NotSupportedException("TabuSearchSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TabuSearchSelector has not been fitted.");

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
