using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Ensemble Feature Selection combining multiple methods.
/// </summary>
/// <remarks>
/// <para>
/// Ensemble Feature Selection combines the results of multiple feature selection
/// methods using voting or rank aggregation. Features consistently selected across
/// multiple methods are more likely to be truly important.
/// </para>
/// <para><b>For Beginners:</b> Different feature selection methods have different
/// biases and catch different patterns. By combining multiple methods (like asking
/// multiple experts), we get a more reliable answer. Features that everyone agrees
/// are important are more trustworthy than those only one method likes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EnsembleFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _threshold;
    private readonly AggregationMethod _aggregation;

    private double[]? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum AggregationMethod
    {
        MajorityVoting,
        WeightedVoting,
        RankAggregation,
        BordaCount
    }

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public AggregationMethod Aggregation => _aggregation;
    public double[]? AggregatedScores => _aggregatedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public EnsembleFS(
        int nFeaturesToSelect = 10,
        double threshold = 0.5,
        AggregationMethod aggregation = AggregationMethod.MajorityVoting,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (threshold <= 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _threshold = threshold;
        _aggregation = aggregation;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EnsembleFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Run multiple base methods and collect rankings
        var methodRankings = new List<int[]>();
        var methodScores = new List<double[]>();

        // Method 1: Correlation-based
        var corrScores = ComputeCorrelationScores(data, target);
        methodRankings.Add(GetRanking(corrScores));
        methodScores.Add(corrScores);

        // Method 2: Variance-based (combined with correlation)
        var varianceScores = ComputeVarianceScores(data);
        var combinedVar = corrScores.Zip(varianceScores, (c, v) => c * v).ToArray();
        methodRankings.Add(GetRanking(combinedVar));
        methodScores.Add(combinedVar);

        // Method 3: Fisher score approximation
        var fisherScores = ComputeFisherScoreApprox(data, target);
        methodRankings.Add(GetRanking(fisherScores));
        methodScores.Add(fisherScores);

        // Method 4: Information gain approximation
        var igScores = ComputeInformationGainApprox(data, target);
        methodRankings.Add(GetRanking(igScores));
        methodScores.Add(igScores);

        // Aggregate results
        _aggregatedScores = _aggregation switch
        {
            AggregationMethod.MajorityVoting => AggregateMajorityVoting(methodRankings, p),
            AggregationMethod.WeightedVoting => AggregateWeightedVoting(methodScores, p),
            AggregationMethod.RankAggregation => AggregateRanks(methodRankings, p),
            AggregationMethod.BordaCount => AggregateBordaCount(methodRankings, p),
            _ => AggregateMajorityVoting(methodRankings, p)
        };

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _aggregatedScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeCorrelationScores(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
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

    private double[] ComputeVarianceScores(Matrix<T> data)
    {
        int n = data.Rows;
        int p = data.Columns;
        var scores = new double[p];

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
            scores[j] = variance / n;
        }

        // Normalize
        double maxVar = scores.Max();
        if (maxVar > 0)
            for (int j = 0; j < p; j++)
                scores[j] /= maxVar;

        return scores;
    }

    private double[] ComputeFisherScoreApprox(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        var scores = new double[p];

        // Binary split based on target median
        double yMedian = 0;
        for (int i = 0; i < n; i++)
            yMedian += NumOps.ToDouble(target[i]);
        yMedian /= n;

        for (int j = 0; j < p; j++)
        {
            double mean0 = 0, mean1 = 0;
            int count0 = 0, count1 = 0;

            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(target[i]) <= yMedian)
                {
                    mean0 += NumOps.ToDouble(data[i, j]);
                    count0++;
                }
                else
                {
                    mean1 += NumOps.ToDouble(data[i, j]);
                    count1++;
                }
            }

            if (count0 > 0) mean0 /= count0;
            if (count1 > 0) mean1 /= count1;

            double var0 = 0, var1 = 0;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (NumOps.ToDouble(target[i]) <= yMedian)
                {
                    var0 += (val - mean0) * (val - mean0);
                }
                else
                {
                    var1 += (val - mean1) * (val - mean1);
                }
            }

            double between = (mean0 - mean1) * (mean0 - mean1);
            double within = var0 / Math.Max(1, count0) + var1 / Math.Max(1, count1);
            scores[j] = within > 1e-10 ? between / within : 0;
        }

        return scores;
    }

    private double[] ComputeInformationGainApprox(Matrix<T> data, Vector<T> target)
    {
        // Use correlation as IG approximation (they're related for continuous data)
        return ComputeCorrelationScores(data, target);
    }

    private int[] GetRanking(double[] scores)
    {
        return scores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Select((x, rank) => (x.Index, Rank: rank))
            .OrderBy(x => x.Index)
            .Select(x => x.Rank)
            .ToArray();
    }

    private double[] AggregateMajorityVoting(List<int[]> rankings, int p)
    {
        var votes = new double[p];
        int topK = _nFeaturesToSelect;

        foreach (var ranking in rankings)
        {
            var topFeatures = ranking
                .Select((r, idx) => (Rank: r, Index: idx))
                .OrderBy(x => x.Rank)
                .Take(topK)
                .Select(x => x.Index);

            foreach (int idx in topFeatures)
                votes[idx]++;
        }

        return votes;
    }

    private double[] AggregateWeightedVoting(List<double[]> allScores, int p)
    {
        var aggregated = new double[p];

        foreach (var scores in allScores)
        {
            double max = scores.Max();
            if (max > 0)
            {
                for (int j = 0; j < p; j++)
                    aggregated[j] += scores[j] / max;
            }
        }

        return aggregated;
    }

    private double[] AggregateRanks(List<int[]> rankings, int p)
    {
        var avgRanks = new double[p];

        foreach (var ranking in rankings)
        {
            for (int j = 0; j < p; j++)
                avgRanks[j] += ranking[j];
        }

        // Convert to scores (lower rank = higher score)
        double maxRank = avgRanks.Max();
        for (int j = 0; j < p; j++)
            avgRanks[j] = maxRank - avgRanks[j];

        return avgRanks;
    }

    private double[] AggregateBordaCount(List<int[]> rankings, int p)
    {
        var bordaScores = new double[p];

        foreach (var ranking in rankings)
        {
            for (int j = 0; j < p; j++)
                bordaScores[j] += p - ranking[j];
        }

        return bordaScores;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EnsembleFS has not been fitted.");

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
        throw new NotSupportedException("EnsembleFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EnsembleFS has not been fitted.");

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
