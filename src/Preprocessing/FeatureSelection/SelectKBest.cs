using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Select K Best features using a pluggable scoring function.
/// </summary>
/// <remarks>
/// <para>
/// SelectKBest is a simple feature selector that computes a score for each feature
/// using a provided scoring function and selects the k features with highest scores.
/// </para>
/// <para><b>For Beginners:</b> This is a versatile "pick the best k features" tool.
/// You provide a scoring function (like correlation, F-statistic, or mutual information),
/// and it simply picks the top k scoring features. It's scikit-learn compatible in concept.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SelectKBest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _k;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _scoreFunc;
    private readonly string _defaultScoreFunc;

    private double[]? _scores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int K => _k;
    public double[]? Scores => _scores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SelectKBest(
        int k = 10,
        Func<Matrix<T>, Vector<T>, double[]>? scoreFunc = null,
        string defaultScoreFunc = "f_classif",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (k < 1)
            throw new ArgumentException("k must be at least 1.", nameof(k));

        _k = k;
        _scoreFunc = scoreFunc;
        _defaultScoreFunc = defaultScoreFunc;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectKBest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
            // Use default scoring function
            _scores = _defaultScoreFunc switch
            {
                "f_classif" => ComputeFClassif(data, target, n, p),
                "f_regression" => ComputeFRegression(data, target, n, p),
                "chi2" => ComputeChi2(data, target, n, p),
                "mutual_info_classif" => ComputeMutualInfo(data, target, n, p),
                _ => ComputeFClassif(data, target, n, p)
            };
        }

        // Select top k features
        int kActual = Math.Min(_k, p);
        _selectedIndices = _scores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(kActual)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeFClassif(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        _pValues = new double[p];

        // Group by class
        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(label))
                classGroups[label] = new List<int>();
            classGroups[label].Add(i);
        }

        int k = classGroups.Count;

        for (int j = 0; j < p; j++)
        {
            // Overall mean
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            // Between-group and within-group sum of squares
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

            // F-statistic
            double msb = ssb / (k - 1);
            double msw = ssw / (n - k);
            scores[j] = msw > 1e-10 ? msb / msw : 0;
            _pValues[j] = 0.05; // Placeholder
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
            // Simple chi-square: sum of (observed - expected)^2 / expected
            var classSums = new Dictionary<int, double>();
            var classCounts = new Dictionary<int, int>();
            double totalSum = 0;

            for (int i = 0; i < n; i++)
            {
                int label = (int)Math.Round(NumOps.ToDouble(target[i]));
                double val = Math.Max(0, NumOps.ToDouble(data[i, j])); // Chi2 needs non-negative

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
            // Discretize feature
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
            throw new InvalidOperationException("SelectKBest has not been fitted.");

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
        throw new NotSupportedException("SelectKBest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectKBest has not been fitted.");

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
