using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.CostSensitive;

/// <summary>
/// Cost-Sensitive Feature Selection considering feature acquisition costs.
/// </summary>
/// <remarks>
/// <para>
/// Cost-Sensitive Feature Selection balances the predictive value of features against
/// their acquisition costs. This is important in scenarios where collecting certain
/// features is expensive, time-consuming, or invasive.
/// </para>
/// <para><b>For Beginners:</b> Sometimes getting data for a feature costs money or time.
/// This selector picks features that give you the most predictive power for the least cost.
/// It's like shopping for ingredients - you want quality ingredients that don't break
/// your budget.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CostSensitiveFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _costWeight;
    private readonly double[]? _featureCosts;

    private double[]? _relevanceScores;
    private double[]? _costBenefitScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CostWeight => _costWeight;
    public double[]? FeatureCosts => _featureCosts;
    public double[]? RelevanceScores => _relevanceScores;
    public double[]? CostBenefitScores => _costBenefitScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CostSensitiveFS(
        int nFeaturesToSelect = 10,
        double costWeight = 0.5,
        double[]? featureCosts = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (costWeight < 0 || costWeight > 1)
            throw new ArgumentException("Cost weight must be between 0 and 1.", nameof(costWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _costWeight = costWeight;
        _featureCosts = featureCosts;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CostSensitiveFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute relevance scores (correlation with target)
        _relevanceScores = ComputeRelevanceScores(data, target, n, p);

        // Get or create feature costs
        var costs = _featureCosts ?? CreateUniformCosts(p);
        if (costs.Length != p)
            throw new ArgumentException("Feature costs length must match number of features.");

        // Normalize costs to [0, 1]
        var normalizedCosts = NormalizeCosts(costs, p);

        // Compute cost-benefit scores
        _costBenefitScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            // Higher relevance and lower cost = better score
            double relevance = _relevanceScores[j];
            double cost = normalizedCosts[j];

            // Weighted combination: (1-w) * relevance - w * cost
            _costBenefitScores[j] = (1 - _costWeight) * relevance - _costWeight * cost;
        }

        // Select features with best cost-benefit scores
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _costBenefitScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public void Fit(Matrix<T> data, Vector<T> target, double[] featureCosts)
    {
        if (featureCosts.Length != data.Columns)
            throw new ArgumentException("Feature costs must match number of columns.");

        // Create a new instance with the provided costs and fit
        var temp = new CostSensitiveFS<T>(_nFeaturesToSelect, _costWeight, featureCosts);
        temp.Fit(data, target);

        // Copy results
        _relevanceScores = temp._relevanceScores;
        _costBenefitScores = temp._costBenefitScores;
        _selectedIndices = temp._selectedIndices;

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

    private double[] CreateUniformCosts(int p)
    {
        var costs = new double[p];
        for (int j = 0; j < p; j++)
            costs[j] = 1.0;
        return costs;
    }

    private double[] NormalizeCosts(double[] costs, int p)
    {
        double min = double.MaxValue, max = double.MinValue;
        for (int j = 0; j < p; j++)
        {
            if (costs[j] < min) min = costs[j];
            if (costs[j] > max) max = costs[j];
        }

        var normalized = new double[p];
        double range = max - min;

        for (int j = 0; j < p; j++)
        {
            normalized[j] = range > 1e-10 ? (costs[j] - min) / range : 0.5;
        }

        return normalized;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CostSensitiveFS has not been fitted.");

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
        throw new NotSupportedException("CostSensitiveFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CostSensitiveFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public double GetTotalCost()
    {
        if (_selectedIndices is null || _featureCosts is null)
            return 0;

        return _selectedIndices.Sum(j => _featureCosts[j]);
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
