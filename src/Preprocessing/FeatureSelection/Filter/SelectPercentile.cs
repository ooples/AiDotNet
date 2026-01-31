using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Selects features based on a percentile of the highest scores.
/// </summary>
/// <remarks>
/// <para>
/// Similar to SelectKBest but selects a percentage of features rather than a fixed
/// number. Useful when you want a consistent selection ratio across datasets of
/// different sizes.
/// </para>
/// <para><b>For Beginners:</b> Instead of saying "keep the top 10 features," you say
/// "keep the top 25% of features." This is useful when you want a consistent reduction
/// ratio that scales with the number of features in your dataset.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SelectPercentile<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _percentile;
    private readonly Func<Matrix<T>, Vector<T>, int, double>? _scoringFunction;

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
        double percentile = 25,
        Func<Matrix<T>, Vector<T>, int, double>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be between 0 and 100.", nameof(percentile));

        _percentile = percentile;
        _scoringFunction = scoringFunction;
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

        _scores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            if (_scoringFunction is not null)
            {
                _scores[j] = _scoringFunction(data, target, j);
                _pValues[j] = 1.0 / (1.0 + _scores[j]);
            }
            else
            {
                var (fStat, pVal) = ComputeFStatistic(data, target, j, n);
                _scores[j] = fStat;
                _pValues[j] = pVal;
            }
        }

        // Calculate threshold based on percentile
        int nToSelect = Math.Max(1, (int)Math.Ceiling(p * _percentile / 100.0));
        nToSelect = Math.Min(nToSelect, p);

        _selectedIndices = _scores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private (double FStat, double PValue) ComputeFStatistic(Matrix<T> data, Vector<T> target, int featureIdx, int n)
    {
        var classes = new Dictionary<double, List<double>>();

        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            double x = NumOps.ToDouble(data[i, featureIdx]);

            if (!classes.ContainsKey(y))
                classes[y] = new List<double>();
            classes[y].Add(x);
        }

        if (classes.Count <= 1)
        {
            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += NumOps.ToDouble(data[i, featureIdx]);
                yMean += NumOps.ToDouble(target[i]);
            }
            xMean /= n;
            yMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, featureIdx]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX < 1e-10 || ssYY < 1e-10)
                return (0, 1);

            double r2 = (ssXY * ssXY) / (ssXX * ssYY);
            double fStat = r2 * (n - 2) / (1 - r2 + 1e-10);
            return (fStat, 1.0 / (1 + fStat));
        }

        double grandMean = 0;
        foreach (var kv in classes)
            grandMean += kv.Value.Sum();
        grandMean /= n;

        double ssBetween = 0;
        double ssWithin = 0;

        foreach (var kv in classes)
        {
            double classMean = kv.Value.Average();
            int nk = kv.Value.Count;
            ssBetween += nk * Math.Pow(classMean - grandMean, 2);

            foreach (double x in kv.Value)
                ssWithin += Math.Pow(x - classMean, 2);
        }

        int k = classes.Count;
        double dfBetween = k - 1;
        double dfWithin = n - k;

        if (dfBetween <= 0 || dfWithin <= 0 || ssWithin < 1e-10)
            return (0, 1);

        double msBetween = ssBetween / dfBetween;
        double msWithin = ssWithin / dfWithin;
        double fStat2 = msBetween / msWithin;

        return (fStat2, 1.0 / (1 + fStat2));
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
