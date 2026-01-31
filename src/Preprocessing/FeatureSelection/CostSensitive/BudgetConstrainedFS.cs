using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.CostSensitive;

/// <summary>
/// Budget-Constrained Feature Selection with a maximum cost budget.
/// </summary>
/// <remarks>
/// <para>
/// Budget-Constrained Feature Selection selects the most informative features
/// while staying within a specified cost budget. It uses a greedy approach
/// to maximize value per unit cost.
/// </para>
/// <para><b>For Beginners:</b> Imagine you have $100 to spend on data collection.
/// This selector picks the most valuable features you can afford within your budget.
/// It's like a knapsack problem - get the most value without exceeding your limit.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BudgetConstrainedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _budget;
    private readonly double[]? _featureCosts;

    private double[]? _relevanceScores;
    private double[]? _valuePerCost;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _usedBudget;

    public double Budget => _budget;
    public double[]? FeatureCosts => _featureCosts;
    public double[]? RelevanceScores => _relevanceScores;
    public double[]? ValuePerCost => _valuePerCost;
    public int[]? SelectedIndices => _selectedIndices;
    public double UsedBudget => _usedBudget;
    public override bool SupportsInverseTransform => false;

    public BudgetConstrainedFS(
        double budget = 100.0,
        double[]? featureCosts = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (budget <= 0)
            throw new ArgumentException("Budget must be positive.", nameof(budget));

        _budget = budget;
        _featureCosts = featureCosts;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BudgetConstrainedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute relevance scores
        _relevanceScores = ComputeRelevanceScores(data, target, n, p);

        // Get or create feature costs
        var costs = _featureCosts ?? CreateUnitCosts(p);
        if (costs.Length != p)
            throw new ArgumentException("Feature costs length must match number of features.");

        // Compute value per cost
        _valuePerCost = new double[p];
        for (int j = 0; j < p; j++)
        {
            _valuePerCost[j] = costs[j] > 1e-10 ? _relevanceScores[j] / costs[j] : _relevanceScores[j] * 1000;
        }

        // Greedy selection within budget
        var selected = new List<int>();
        _usedBudget = 0;

        // Sort by value per cost
        var sortedFeatures = Enumerable.Range(0, p)
            .OrderByDescending(j => _valuePerCost[j])
            .ToList();

        foreach (int j in sortedFeatures)
        {
            if (_usedBudget + costs[j] <= _budget)
            {
                selected.Add(j);
                _usedBudget += costs[j];
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    public void Fit(Matrix<T> data, Vector<T> target, double[] featureCosts)
    {
        if (featureCosts.Length != data.Columns)
            throw new ArgumentException("Feature costs must match number of columns.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute relevance scores
        _relevanceScores = ComputeRelevanceScores(data, target, n, p);

        // Compute value per cost
        _valuePerCost = new double[p];
        for (int j = 0; j < p; j++)
        {
            _valuePerCost[j] = featureCosts[j] > 1e-10
                ? _relevanceScores[j] / featureCosts[j]
                : _relevanceScores[j] * 1000;
        }

        // Greedy selection within budget
        var selected = new List<int>();
        _usedBudget = 0;

        var sortedFeatures = Enumerable.Range(0, p)
            .OrderByDescending(j => _valuePerCost[j])
            .ToList();

        foreach (int j in sortedFeatures)
        {
            if (_usedBudget + featureCosts[j] <= _budget)
            {
                selected.Add(j);
                _usedBudget += featureCosts[j];
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private double[] ComputeRelevanceScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    private double[] CreateUnitCosts(int p)
    {
        var costs = new double[p];
        for (int j = 0; j < p; j++)
            costs[j] = 1.0;
        return costs;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BudgetConstrainedFS has not been fitted.");

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
        throw new NotSupportedException("BudgetConstrainedFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BudgetConstrainedFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public double GetRemainingBudget()
    {
        return _budget - _usedBudget;
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
