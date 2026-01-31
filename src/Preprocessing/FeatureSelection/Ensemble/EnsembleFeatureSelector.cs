using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Ensemble Feature Selection combining multiple selection methods.
/// </summary>
/// <remarks>
/// <para>
/// Ensemble Feature Selection aggregates results from multiple feature selection
/// methods. Features consistently selected across methods are more likely to be
/// truly important.
/// </para>
/// <para>
/// Aggregation methods:
/// - Voting: Select features chosen by majority of methods
/// - Ranking: Aggregate rankings using Borda count or similar
/// - Weighted: Weight methods by their reliability
/// </para>
/// <para><b>For Beginners:</b> Different selection methods have different strengths.
/// By combining them, we get more robust and reliable feature selection. Features
/// that multiple methods agree on are more trustworthy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EnsembleFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _votingThreshold;
    private readonly AggregationMethod _method;
    private readonly List<Func<Matrix<T>, Vector<T>, int[]>> _selectors;

    private double[]? _aggregateScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum AggregationMethod { Voting, BordaCount, WeightedVoting }

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? AggregateScores => _aggregateScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public EnsembleFeatureSelector(
        int nFeaturesToSelect = 10,
        double votingThreshold = 0.5,
        AggregationMethod method = AggregationMethod.BordaCount,
        List<Func<Matrix<T>, Vector<T>, int[]>>? selectors = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (votingThreshold <= 0 || votingThreshold > 1)
            throw new ArgumentException("Voting threshold must be between 0 and 1.", nameof(votingThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _votingThreshold = votingThreshold;
        _method = method;
        _selectors = selectors ?? new List<Func<Matrix<T>, Vector<T>, int[]>>();
    }

    /// <summary>
    /// Adds a selector function to the ensemble.
    /// </summary>
    public void AddSelector(Func<Matrix<T>, Vector<T>, int[]> selector)
    {
        _selectors.Add(selector);
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EnsembleFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // If no selectors, use default ones
        if (_selectors.Count == 0)
            AddDefaultSelectors();

        // Run all selectors
        var allSelections = new List<int[]>();
        foreach (var selector in _selectors)
        {
            try
            {
                var selected = selector(data, target);
                if (selected.Length > 0)
                    allSelections.Add(selected);
            }
            catch
            {
                // Skip failed selectors
            }
        }

        if (allSelections.Count == 0)
        {
            // Fallback: select first n features
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            _aggregateScores = new double[p];
            IsFitted = true;
            return;
        }

        // Aggregate based on method
        switch (_method)
        {
            case AggregationMethod.Voting:
                AggregateByVoting(allSelections, p);
                break;
            case AggregationMethod.BordaCount:
                AggregateByBordaCount(allSelections, p);
                break;
            case AggregationMethod.WeightedVoting:
                AggregateByWeightedVoting(allSelections, p);
                break;
        }

        IsFitted = true;
    }

    private void AddDefaultSelectors()
    {
        int k = _nFeaturesToSelect;

        // Correlation-based selector
        _selectors.Add((data, target) =>
        {
            int n = data.Rows;
            int p = data.Columns;
            var correlations = new double[p];

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

                double ssXY = 0, ssXX = 0, ssYY = 0;
                for (int i = 0; i < n; i++)
                {
                    double dx = NumOps.ToDouble(data[i, j]) - xMean;
                    double dy = NumOps.ToDouble(target[i]) - yMean;
                    ssXY += dx * dy;
                    ssXX += dx * dx;
                    ssYY += dy * dy;
                }

                if (ssXX > 1e-10 && ssYY > 1e-10)
                    correlations[j] = Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
            }

            return correlations
                .Select((c, idx) => (c, idx))
                .OrderByDescending(x => x.c)
                .Take(k)
                .Select(x => x.idx)
                .ToArray();
        });

        // Variance-based selector
        _selectors.Add((data, target) =>
        {
            int n = data.Rows;
            int p = data.Columns;
            var variances = new double[p];

            for (int j = 0; j < p; j++)
            {
                double mean = 0;
                for (int i = 0; i < n; i++)
                    mean += NumOps.ToDouble(data[i, j]);
                mean /= n;

                double variance = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - mean;
                    variance += diff * diff;
                }
                variances[j] = variance / n;
            }

            return variances
                .Select((v, idx) => (v, idx))
                .OrderByDescending(x => x.v)
                .Take(k)
                .Select(x => x.idx)
                .ToArray();
        });

        // Mutual information approximation
        _selectors.Add((data, target) =>
        {
            int n = data.Rows;
            int p = data.Columns;
            var miScores = new double[p];

            for (int j = 0; j < p; j++)
            {
                // Simplified MI using correlation squared as proxy
                double xMean = 0, yMean = 0;
                for (int i = 0; i < n; i++)
                {
                    xMean += NumOps.ToDouble(data[i, j]);
                    yMean += NumOps.ToDouble(target[i]);
                }
                xMean /= n;
                yMean /= n;

                double ssXY = 0, ssXX = 0, ssYY = 0;
                for (int i = 0; i < n; i++)
                {
                    double dx = NumOps.ToDouble(data[i, j]) - xMean;
                    double dy = NumOps.ToDouble(target[i]) - yMean;
                    ssXY += dx * dy;
                    ssXX += dx * dx;
                    ssYY += dy * dy;
                }

                if (ssXX > 1e-10 && ssYY > 1e-10)
                {
                    double r = ssXY / Math.Sqrt(ssXX * ssYY);
                    miScores[j] = -0.5 * Math.Log(1 - r * r + 1e-10);
                }
            }

            return miScores
                .Select((mi, idx) => (mi, idx))
                .OrderByDescending(x => x.mi)
                .Take(k)
                .Select(x => x.idx)
                .ToArray();
        });
    }

    private void AggregateByVoting(List<int[]> allSelections, int p)
    {
        var votes = new int[p];
        foreach (var selection in allSelections)
            foreach (int idx in selection)
                if (idx >= 0 && idx < p)
                    votes[idx]++;

        _aggregateScores = votes.Select(v => (double)v / allSelections.Count).ToArray();

        int threshold = (int)Math.Ceiling(_votingThreshold * allSelections.Count);
        var selected = votes
            .Select((v, idx) => (v, idx))
            .Where(x => x.v >= threshold)
            .OrderByDescending(x => x.v)
            .Take(_nFeaturesToSelect)
            .Select(x => x.idx)
            .ToList();

        if (selected.Count == 0)
        {
            selected = votes
                .Select((v, idx) => (v, idx))
                .OrderByDescending(x => x.v)
                .Take(_nFeaturesToSelect)
                .Select(x => x.idx)
                .ToList();
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
    }

    private void AggregateByBordaCount(List<int[]> allSelections, int p)
    {
        var bordaScores = new double[p];

        foreach (var selection in allSelections)
        {
            for (int rank = 0; rank < selection.Length; rank++)
            {
                int idx = selection[rank];
                if (idx >= 0 && idx < p)
                    bordaScores[idx] += selection.Length - rank;
            }
        }

        _aggregateScores = bordaScores;

        _selectedIndices = bordaScores
            .Select((s, idx) => (s, idx))
            .OrderByDescending(x => x.s)
            .Take(_nFeaturesToSelect)
            .Select(x => x.idx)
            .OrderBy(x => x)
            .ToArray();
    }

    private void AggregateByWeightedVoting(List<int[]> allSelections, int p)
    {
        // Equal weights for now (can be extended with method weights)
        var weights = Enumerable.Repeat(1.0 / allSelections.Count, allSelections.Count).ToArray();

        var scores = new double[p];
        for (int m = 0; m < allSelections.Count; m++)
        {
            foreach (int idx in allSelections[m])
                if (idx >= 0 && idx < p)
                    scores[idx] += weights[m];
        }

        _aggregateScores = scores;

        _selectedIndices = scores
            .Select((s, idx) => (s, idx))
            .OrderByDescending(x => x.s)
            .Take(_nFeaturesToSelect)
            .Select(x => x.idx)
            .OrderBy(x => x)
            .ToArray();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EnsembleFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("EnsembleFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EnsembleFeatureSelector has not been fitted.");

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
