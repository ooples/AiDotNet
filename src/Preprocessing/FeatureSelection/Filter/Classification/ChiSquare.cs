using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Classification;

/// <summary>
/// Chi-Square (χ²) test for feature selection in classification.
/// </summary>
/// <remarks>
/// <para>
/// Chi-Square feature selection measures the dependence between features and
/// class labels using contingency tables. Features with high chi-square scores
/// have a significant statistical relationship with the target.
/// </para>
/// <para><b>For Beginners:</b> The Chi-Square test checks if the distribution of
/// a feature is different for different classes. If the values of a feature look
/// very different between classes, it gets a high score because that difference
/// can help predict class membership.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChiSquare<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _chiSquareScores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? ChiSquareScores => _chiSquareScores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ChiSquare(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChiSquare requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique class labels
        var classLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
            classLabels.Add((int)Math.Round(NumOps.ToDouble(target[i])));

        int nClasses = classLabels.Count;
        var labelList = classLabels.OrderBy(x => x).ToList();

        _chiSquareScores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize continuous feature
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }

            double range = maxVal - minVal;

            // Build contingency table
            var contingency = new int[_nBins, nClasses];
            var featureCounts = new int[_nBins];
            var classCounts = new int[nClasses];

            for (int i = 0; i < n; i++)
            {
                int fBin = range > 1e-10
                    ? Math.Min(_nBins - 1, (int)((NumOps.ToDouble(data[i, j]) - minVal) / range * (_nBins - 1)))
                    : 0;
                int cIdx = labelList.IndexOf((int)Math.Round(NumOps.ToDouble(target[i])));

                contingency[fBin, cIdx]++;
                featureCounts[fBin]++;
                classCounts[cIdx]++;
            }

            // Compute chi-square statistic
            double chiSq = 0;
            int df = 0;

            for (int f = 0; f < _nBins; f++)
            {
                for (int c = 0; c < nClasses; c++)
                {
                    double expected = (double)featureCounts[f] * classCounts[c] / n;
                    if (expected > 0)
                    {
                        double observed = contingency[f, c];
                        chiSq += Math.Pow(observed - expected, 2) / expected;
                        if (featureCounts[f] > 0 && classCounts[c] > 0)
                            df++;
                    }
                }
            }

            df = (_nBins - 1) * (nClasses - 1);
            _chiSquareScores[j] = chiSq;
            _pValues[j] = ComputePValue(chiSq, df);
        }

        // Select top features by chi-square score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _chiSquareScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputePValue(double chiSquare, int df)
    {
        if (chiSquare <= 0 || df <= 0) return 1.0;

        // Approximation using Wilson-Hilferty transformation
        double k = df;
        double x = chiSquare;

        if (k > 100)
        {
            double z = Math.Pow(x / k, 1.0 / 3.0) - (1 - 2.0 / (9 * k));
            double se = Math.Sqrt(2.0 / (9 * k));
            double standardZ = z / se;

            return 0.5 * (1 - Erf(standardZ / Math.Sqrt(2)));
        }

        // Simple approximation for smaller df
        return Math.Exp(-0.5 * chiSquare / Math.Max(1, df));
    }

    private double Erf(double x)
    {
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquare has not been fitted.");

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
        throw new NotSupportedException("ChiSquare does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquare has not been fitted.");

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
