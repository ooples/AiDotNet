using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// ANOVA-based feature selection for multi-class problems.
/// </summary>
/// <remarks>
/// <para>
/// Analysis of Variance (ANOVA) extends the t-test to multiple classes. It measures
/// how much the mean of each feature varies across classes compared to within-class
/// variation.
/// </para>
/// <para><b>For Beginners:</b> ANOVA checks if the average value of a feature is
/// significantly different across multiple groups. If a feature has very different
/// averages for each class, it's good at distinguishing between them. Unlike the
/// t-test, ANOVA works with any number of classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ANOVASelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ANOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classSamples.ContainsKey(label))
                classSamples[label] = new List<int>();
            classSamples[label].Add(i);
        }

        int nClasses = classSamples.Count;
        _fStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Overall mean
            double grandMean = 0;
            for (int i = 0; i < n; i++)
                grandMean += NumOps.ToDouble(data[i, j]);
            grandMean /= n;

            // Between-group sum of squares (SSB)
            double ssb = 0;
            foreach (var kvp in classSamples)
            {
                double classMean = kvp.Value.Sum(i => NumOps.ToDouble(data[i, j])) / kvp.Value.Count;
                ssb += kvp.Value.Count * Math.Pow(classMean - grandMean, 2);
            }

            // Within-group sum of squares (SSW)
            double ssw = 0;
            foreach (var kvp in classSamples)
            {
                double classMean = kvp.Value.Sum(i => NumOps.ToDouble(data[i, j])) / kvp.Value.Count;
                foreach (int i in kvp.Value)
                    ssw += Math.Pow(NumOps.ToDouble(data[i, j]) - classMean, 2);
            }

            // F-statistic
            int dfb = nClasses - 1;
            int dfw = n - nClasses;

            if (dfw > 0 && ssw > 1e-10)
                _fStatistics[j] = (ssb / dfb) / (ssw / dfw);
            else
                _fStatistics[j] = 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fStatistics[j])
            .Take(numToSelect)
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
            throw new InvalidOperationException("ANOVASelector has not been fitted.");

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
        throw new NotSupportedException("ANOVASelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVASelector has not been fitted.");

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
