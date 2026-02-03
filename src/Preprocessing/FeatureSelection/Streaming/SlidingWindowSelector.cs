using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Sliding Window Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Maintains a sliding window of recent samples and performs feature selection
/// based on the most recent data, adapting to concept drift.
/// </para>
/// <para><b>For Beginners:</b> Data patterns can change over time (concept drift).
/// This method only looks at the most recent samples in a sliding window, so it
/// adapts to changes. Features that were important last month might not be
/// important now, and this method handles that.
/// </para>
/// </remarks>
public class SlidingWindowSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _windowSize;

    private readonly Queue<double[]> _windowX;
    private readonly Queue<double> _windowY;
    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int WindowSize => _windowSize;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SlidingWindowSelector(
        int nFeaturesToSelect = 10,
        int windowSize = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (windowSize < 10)
            throw new ArgumentException("Window size must be at least 10.", nameof(windowSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _windowSize = windowSize;
        _windowX = new Queue<double[]>();
        _windowY = new Queue<double>();
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SlidingWindowSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        _nInputFeatures = p;

        // Add new samples to window
        for (int i = 0; i < n; i++)
        {
            var row = new double[p];
            for (int j = 0; j < p; j++)
                row[j] = NumOps.ToDouble(data[i, j]);

            _windowX.Enqueue(row);
            _windowY.Enqueue(NumOps.ToDouble(target[i]));

            // Remove old samples if window is full
            while (_windowX.Count > _windowSize)
            {
                _windowX.Dequeue();
                _windowY.Dequeue();
            }
        }

        // Compute feature scores from window
        ComputeScoresFromWindow(p);

        IsFitted = true;
    }

    private void ComputeScoresFromWindow(int p)
    {
        int windowN = _windowX.Count;
        if (windowN < 2) return;

        var X = _windowX.ToArray();
        var y = _windowY.ToArray();

        _featureScores = new double[p];

        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < windowN; i++)
                xMean += X[i][j];
            xMean /= windowN;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < windowN; i++)
            {
                double xd = X[i][j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }

            _featureScores[j] = (sxx > 1e-10 && syy > 1e-10)
                ? Math.Abs(sxy / Math.Sqrt(sxx * syy))
                : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();
    }

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SlidingWindowSelector has not been fitted.");

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
        throw new NotSupportedException("SlidingWindowSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SlidingWindowSelector has not been fitted.");

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
