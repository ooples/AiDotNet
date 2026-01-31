using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Chi-Square test for feature selection in classification problems.
/// </summary>
/// <remarks>
/// <para>
/// The Chi-Square test measures the independence between each feature and the target
/// class. Features that are strongly associated with the target (high chi-square statistic)
/// are considered more relevant for classification.
/// </para>
/// <para><b>For Beginners:</b> Chi-Square asks: "Is there a relationship between this
/// feature and the target class, or could the observed pattern be due to chance?"
/// High chi-square values indicate a strong association. It works best with categorical
/// or discretized features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChiSquare<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _chiSquareScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ChiSquareScores => _chiSquareScores;
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

        // Get unique classes
        var classSet = new HashSet<int>();
        for (int i = 0; i < n; i++)
            classSet.Add((int)Math.Round(NumOps.ToDouble(target[i])));
        var classes = classSet.OrderBy(x => x).ToArray();
        int nClasses = classes.Length;

        _chiSquareScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize feature into bins
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            var bins = new int[n];
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                bins[i] = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
            }

            // Build contingency table
            var observed = new int[_nBins, nClasses];
            var rowTotals = new int[_nBins];
            var colTotals = new int[nClasses];

            for (int i = 0; i < n; i++)
            {
                int bin = bins[i];
                int classIdx = Array.IndexOf(classes, (int)Math.Round(NumOps.ToDouble(target[i])));
                observed[bin, classIdx]++;
                rowTotals[bin]++;
                colTotals[classIdx]++;
            }

            // Compute chi-square statistic
            double chiSq = 0;
            for (int b = 0; b < _nBins; b++)
            {
                for (int c = 0; c < nClasses; c++)
                {
                    double expected = (double)rowTotals[b] * colTotals[c] / n;
                    if (expected > 0)
                    {
                        double diff = observed[b, c] - expected;
                        chiSq += (diff * diff) / expected;
                    }
                }
            }

            _chiSquareScores[j] = chiSq;
        }

        // Select top features by chi-square score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _chiSquareScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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
