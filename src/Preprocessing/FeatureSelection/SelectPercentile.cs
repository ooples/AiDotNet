using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Select features based on percentile of the highest scores.
/// </summary>
/// <remarks>
/// <para>
/// SelectPercentile is similar to SelectKBest but instead of selecting a fixed number
/// of features, it selects features that score in the top percentile (e.g., top 10%).
/// </para>
/// <para><b>For Beginners:</b> Instead of saying "give me the best 5 features", you say
/// "give me the best 10% of features". This is useful when you don't know the total
/// number of features ahead of time or want a proportional selection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SelectPercentile<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _percentile;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _scoreFunc;
    private readonly string _defaultScoreFunc;

    private double[]? _scores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Percentile => _percentile;
    public double[]? Scores => _scores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SelectPercentile(
        double percentile = 10.0,
        Func<Matrix<T>, Vector<T>, double[]>? scoreFunc = null,
        string defaultScoreFunc = "f_classif",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be in (0, 100].", nameof(percentile));

        _percentile = percentile;
        _scoreFunc = scoreFunc;
        _defaultScoreFunc = defaultScoreFunc;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectPercentile requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute scores
        if (_scoreFunc != null)
        {
            _scores = _scoreFunc(data, target);
        }
        else
        {
            _scores = _defaultScoreFunc switch
            {
                "f_classif" => ComputeFClassif(data, target, n, p),
                "f_regression" => ComputeFRegression(data, target, n, p),
                "chi2" => ComputeChi2(data, target, n, p),
                "mutual_info_classif" => ComputeMutualInfo(data, target, n, p),
                _ => ComputeFClassif(data, target, n, p)
            };
        }

        // Determine threshold for percentile
        var sortedScores = _scores.OrderByDescending(s => s).ToArray();
        int numToSelect = Math.Max(1, (int)Math.Ceiling(p * _percentile / 100.0));
        double threshold = sortedScores[Math.Min(numToSelect - 1, sortedScores.Length - 1)];

        // Select features at or above threshold
        _selectedIndices = _scores
            .Select((s, idx) => (Score: s, Index: idx))
            .Where(x => x.Score >= threshold)
            .OrderBy(x => x.Index)
            .Select(x => x.Index)
            .ToArray();

        // If we got more than expected due to ties, trim to exact percentile count
        if (_selectedIndices.Length > numToSelect)
        {
            _selectedIndices = _scores
                .Select((s, idx) => (Score: s, Index: idx))
                .OrderByDescending(x => x.Score)
                .Take(numToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double[] ComputeFClassif(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        _pValues = new double[p];

        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(label))
                classGroups[label] = [];
            classGroups[label].Add(i);
        }

        int k = classGroups.Count;

        for (int j = 0; j < p; j++)
        {
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            double ssb = 0, ssw = 0;

            foreach (var kvp in classGroups)
            {
                double classMean = 0;
                foreach (int i in kvp.Value)
                    classMean += NumOps.ToDouble(data[i, j]);
                classMean /= kvp.Value.Count;

                ssb += kvp.Value.Count * (classMean - overallMean) * (classMean - overallMean);

                foreach (int i in kvp.Value)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - classMean;
                    ssw += diff * diff;
                }
            }

            double msb = ssb / (k - 1);
            double msw = ssw / (n - k);
            scores[j] = msw > 1e-10 ? msb / msw : 0;
            _pValues[j] = 0.05;
        }

        return scores;
    }

    private double[] ComputeFRegression(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        _pValues = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double ssTotal = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(target[i]) - yMean;
            ssTotal += diff * diff;
        }

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
            }

            double beta = sxx > 1e-10 ? sxy / sxx : 0;
            double ssResidual = 0;

            for (int i = 0; i < n; i++)
            {
                double pred = yMean + beta * (NumOps.ToDouble(data[i, j]) - xMean);
                double residual = NumOps.ToDouble(target[i]) - pred;
                ssResidual += residual * residual;
            }

            double ssRegression = ssTotal - ssResidual;
            double msRegression = ssRegression;
            double msResidual = ssResidual / Math.Max(1, n - 2);

            scores[j] = msResidual > 1e-10 ? msRegression / msResidual : 0;
            _pValues[j] = 0.05;
        }

        return scores;
    }

    private double[] ComputeChi2(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var classSums = new Dictionary<int, double>();
            var classCounts = new Dictionary<int, int>();
            double totalSum = 0;

            for (int i = 0; i < n; i++)
            {
                int label = (int)Math.Round(NumOps.ToDouble(target[i]));
                double val = Math.Max(0, NumOps.ToDouble(data[i, j]));

                if (!classSums.ContainsKey(label))
                {
                    classSums[label] = 0;
                    classCounts[label] = 0;
                }
                classSums[label] += val;
                classCounts[label]++;
                totalSum += val;
            }

            double chi2 = 0;
            foreach (var kvp in classSums)
            {
                double expected = totalSum * classCounts[kvp.Key] / n;
                if (expected > 1e-10)
                    chi2 += (kvp.Value - expected) * (kvp.Value - expected) / expected;
            }

            scores[j] = chi2;
            _pValues[j] = 0.05;
        }

        return scores;
    }

    private double[] ComputeMutualInfo(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        int nBins = 10;

        for (int j = 0; j < p; j++)
        {
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            var jointCounts = new Dictionary<(int, int), int>();
            var featureCounts = new int[nBins];
            var targetCounts = new Dictionary<int, int>();

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int bin = Math.Min((int)(((val - minVal) / range) * nBins), nBins - 1);
                int label = (int)Math.Round(NumOps.ToDouble(target[i]));

                featureCounts[bin]++;
                if (!targetCounts.ContainsKey(label))
                    targetCounts[label] = 0;
                targetCounts[label]++;

                var key = (bin, label);
                if (!jointCounts.ContainsKey(key))
                    jointCounts[key] = 0;
                jointCounts[key]++;
            }

            double mi = 0;
            foreach (var kvp in jointCounts)
            {
                int bin = kvp.Key.Item1;
                int label = kvp.Key.Item2;
                int joint = kvp.Value;

                double pJoint = (double)joint / n;
                double pF = (double)featureCounts[bin] / n;
                double pT = (double)targetCounts[label] / n;

                if (pJoint > 0 && pF > 0 && pT > 0)
                    mi += pJoint * Math.Log(pJoint / (pF * pT));
            }

            scores[j] = mi;
        }

        return scores;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectPercentile has not been fitted.");

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
        throw new NotSupportedException("SelectPercentile does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectPercentile has not been fitted.");

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
