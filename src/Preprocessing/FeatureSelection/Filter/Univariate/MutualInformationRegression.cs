using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Mutual Information for regression feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Measures the mutual information between each feature and a continuous target.
/// Uses binning to estimate the probability distributions for computing MI.
/// </para>
/// <para><b>For Beginners:</b> For regression problems where the target is a number
/// (not a category), mutual information still works by grouping both the feature
/// and target into bins. High MI means knowing the feature helps predict the target
/// value, capturing both linear and nonlinear relationships.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MutualInformationRegression<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _mutualInfoScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MutualInfoScores => _mutualInfoScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MutualInformationRegression(
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
            "MutualInformationRegression requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Bin the target variable
        double yMin = double.MaxValue, yMax = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(target[i]);
            yMin = Math.Min(yMin, val);
            yMax = Math.Max(yMax, val);
        }

        double yRange = yMax - yMin;
        if (yRange < 1e-10) yRange = 1;

        var yBins = new int[n];
        var yProb = new double[_nBins];
        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(target[i]);
            yBins[i] = Math.Min((int)(((val - yMin) / yRange) * _nBins), _nBins - 1);
            yProb[yBins[i]] += 1.0 / n;
        }

        _mutualInfoScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Bin feature
            double xMin = double.MaxValue, xMax = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                xMin = Math.Min(xMin, val);
                xMax = Math.Max(xMax, val);
            }

            double xRange = xMax - xMin;
            if (xRange < 1e-10) xRange = 1;

            var xProb = new double[_nBins];
            var jointProb = new double[_nBins, _nBins];

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int xBin = Math.Min((int)(((val - xMin) / xRange) * _nBins), _nBins - 1);
                xProb[xBin] += 1.0 / n;
                jointProb[xBin, yBins[i]] += 1.0 / n;
            }

            // Compute mutual information
            double mi = 0;
            for (int xb = 0; xb < _nBins; xb++)
            {
                for (int yb = 0; yb < _nBins; yb++)
                {
                    if (jointProb[xb, yb] > 1e-10 && xProb[xb] > 1e-10 && yProb[yb] > 1e-10)
                    {
                        mi += jointProb[xb, yb] * Math.Log(jointProb[xb, yb] / (xProb[xb] * yProb[yb]));
                    }
                }
            }

            _mutualInfoScores[j] = Math.Max(0, mi);
        }

        // Select top features by MI
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _mutualInfoScores
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
            .Take(nToSelect)
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
            throw new InvalidOperationException("MutualInformationRegression has not been fitted.");

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
        throw new NotSupportedException("MutualInformationRegression does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInformationRegression has not been fitted.");

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
