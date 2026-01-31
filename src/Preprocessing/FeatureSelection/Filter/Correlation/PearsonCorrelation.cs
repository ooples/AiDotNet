using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Pearson Correlation-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Pearson Correlation measures the linear relationship between each feature
/// and the target variable. Features with higher absolute correlation are
/// considered more predictive.
/// </para>
/// <para><b>For Beginners:</b> Pearson correlation measures how well a straight line
/// fits the relationship between a feature and the target. A value of 1 means perfect
/// positive linear relationship, -1 means perfect negative, and 0 means no linear
/// relationship. Good for finding features with linear dependencies.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PearsonCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _useAbsoluteValue;

    private double[]? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool UseAbsoluteValue => _useAbsoluteValue;
    public double[]? CorrelationScores => _correlationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PearsonCorrelation(
        int nFeaturesToSelect = 10,
        bool useAbsoluteValue = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _useAbsoluteValue = useAbsoluteValue;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PearsonCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _correlationScores = new double[p];

        // Compute target statistics
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double syy = 0;
        for (int i = 0; i < n; i++)
        {
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            syy += yDiff * yDiff;
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

            double correlation = (sxx > 1e-10 && syy > 1e-10)
                ? sxy / Math.Sqrt(sxx * syy)
                : 0;

            _correlationScores[j] = _useAbsoluteValue ? Math.Abs(correlation) : correlation;
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _useAbsoluteValue ? _correlationScores[j] : Math.Abs(_correlationScores[j]))
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
            throw new InvalidOperationException("PearsonCorrelation has not been fitted.");

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
        throw new NotSupportedException("PearsonCorrelation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PearsonCorrelation has not been fitted.");

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
