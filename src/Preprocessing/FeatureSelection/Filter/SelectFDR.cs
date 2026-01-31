using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Select features using Benjamini-Hochberg False Discovery Rate control.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Benjamini-Hochberg procedure to control the expected proportion of
/// false discoveries among selected features. This provides more power than
/// FWER control while still limiting false positives.
/// </para>
/// <para><b>For Beginners:</b> When testing many features, some will look significant
/// by chance. FDR control says "among all the features we select, we want at most
/// 5% (or your alpha) to be false positives." This is less strict than FWER but
/// more practical for high-dimensional data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SelectFDR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _alpha;
    private readonly Func<Matrix<T>, Vector<T>, int, double>? _scoringFunction;

    private double[]? _scores;
    private double[]? _pValues;
    private double[]? _adjustedPValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Alpha => _alpha;
    public double[]? Scores => _scores;
    public double[]? PValues => _pValues;
    public double[]? AdjustedPValues => _adjustedPValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SelectFDR(
        double alpha = 0.05,
        Func<Matrix<T>, Vector<T>, int, double>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _alpha = alpha;
        _scoringFunction = scoringFunction;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectFDR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Benjamini-Hochberg procedure
        _adjustedPValues = BenjaminiHochberg(_pValues);

        // Select features with adjusted p-value < alpha
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_adjustedPValues[j] < _alpha)
                selected.Add(j);
        }

        // If none selected, select the best one
        if (selected.Count == 0)
        {
            int best = 0;
            for (int j = 1; j < p; j++)
                if (_adjustedPValues[j] < _adjustedPValues[best])
                    best = j;
            selected.Add(best);
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[] BenjaminiHochberg(double[] pValues)
    {
        int n = pValues.Length;
        var adjustedP = new double[n];

        // Sort p-values with indices
        var sorted = pValues
            .Select((p, idx) => (PValue: p, Index: idx))
            .OrderBy(x => x.PValue)
            .ToList();

        // Adjust p-values
        adjustedP[sorted[n - 1].Index] = sorted[n - 1].PValue;

        for (int i = n - 2; i >= 0; i--)
        {
            double rank = i + 1;
            double adjusted = sorted[i].PValue * n / rank;
            adjusted = Math.Min(adjusted, adjustedP[sorted[i + 1].Index]);
            adjustedP[sorted[i].Index] = Math.Min(1.0, adjusted);
        }

        return adjustedP;
    }

    private (double FStat, double PValue) ComputeFStatistic(Matrix<T> data, Vector<T> target, int featureIdx, int n)
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
        double pValue = 1.0 / (1 + fStat);  // Approximate

        return (fStat, pValue);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectFDR has not been fitted.");

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
        throw new NotSupportedException("SelectFDR does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectFDR has not been fitted.");

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
