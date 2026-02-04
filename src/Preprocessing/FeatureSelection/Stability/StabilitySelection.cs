using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Stability Selection for robust feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Stability selection runs feature selection on multiple random subsets of the
/// data and keeps features that are consistently selected across runs. This
/// reduces the variance of feature selection and produces more reliable results.
/// </para>
/// <para><b>For Beginners:</b> Different samples of your data might give different
/// feature rankings. Stability selection checks which features are consistently
/// chosen across many random samples. Features that are always selected are
/// likely to be truly important, not just lucky in one particular sample.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StabilitySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _subsampleRatio;
    private readonly double _selectionThreshold;
    private readonly int? _randomState;

    private double[]? _selectionFrequencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double SubsampleRatio => _subsampleRatio;
    public double SelectionThreshold => _selectionThreshold;
    public double[]? SelectionFrequencies => _selectionFrequencies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StabilitySelection(
        int nFeaturesToSelect = 10,
        int nIterations = 50,
        double subsampleRatio = 0.5,
        double selectionThreshold = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));
        if (subsampleRatio <= 0 || subsampleRatio > 1)
            throw new ArgumentException("Subsample ratio must be between 0 and 1.", nameof(subsampleRatio));
        if (selectionThreshold < 0 || selectionThreshold > 1)
            throw new ArgumentException("Selection threshold must be between 0 and 1.", nameof(selectionThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _subsampleRatio = subsampleRatio;
        _selectionThreshold = selectionThreshold;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StabilitySelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int subsampleSize = (int)(n * _subsampleRatio);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var selectionCounts = new int[p];

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Create subsample
            var subsampleIndices = Enumerable.Range(0, n)
                .OrderBy(_ => random.Next())
                .Take(subsampleSize)
                .ToList();

            // Compute correlation scores on subsample
            var scores = new double[p];
            double yMean = subsampleIndices.Average(i => NumOps.ToDouble(target[i]));

            for (int j = 0; j < p; j++)
            {
                double xMean = subsampleIndices.Average(i => NumOps.ToDouble(data[i, j]));

                double sxy = 0, sxx = 0, syy = 0;
                foreach (int i in subsampleIndices)
                {
                    double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                    double yDiff = NumOps.ToDouble(target[i]) - yMean;
                    sxy += xDiff * yDiff;
                    sxx += xDiff * xDiff;
                    syy += yDiff * yDiff;
                }

                scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            }

            // Select top features in this iteration
            var topFeatures = Enumerable.Range(0, p)
                .OrderByDescending(j => scores[j])
                .Take(_nFeaturesToSelect)
                .ToList();

            foreach (int f in topFeatures)
                selectionCounts[f]++;
        }

        // Compute selection frequencies
        _selectionFrequencies = new double[p];
        for (int j = 0; j < p; j++)
            _selectionFrequencies[j] = (double)selectionCounts[j] / _nIterations;

        // Select features above threshold
        var stableFeatures = Enumerable.Range(0, p)
            .Where(j => _selectionFrequencies[j] >= _selectionThreshold)
            .ToList();

        // If not enough features meet threshold, take top by frequency
        if (stableFeatures.Count < _nFeaturesToSelect)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _selectionFrequencies[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = stableFeatures
                .OrderByDescending(j => _selectionFrequencies[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

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
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

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
        throw new NotSupportedException("StabilitySelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

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
