using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Stability Selection for robust feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// Runs a base feature selector on many bootstrap samples and selects
/// features that are consistently chosen across subsamples. Provides
/// error rate control.
/// </para>
/// <para><b>For Beginners:</b> By running selection on many random subsets
/// of data, we find features that are "stably" selected regardless of
/// which samples we use. These are more likely to be truly important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StabilitySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _sampleFraction;
    private readonly double _threshold;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, int[]>? _baseSelector;

    private double[]? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SelectionProbabilities => _selectionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StabilitySelection(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double sampleFraction = 0.5,
        double threshold = 0.6,
        Func<Matrix<T>, Vector<T>, int[]>? baseSelector = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBootstraps < 10)
            throw new ArgumentException("Number of bootstraps must be at least 10.", nameof(nBootstraps));
        if (sampleFraction <= 0 || sampleFraction > 1)
            throw new ArgumentException("Sample fraction must be between 0 and 1.", nameof(sampleFraction));
        if (threshold <= 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _sampleFraction = sampleFraction;
        _threshold = threshold;
        _baseSelector = baseSelector;
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
        int sampleSize = (int)(n * _sampleFraction);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var selectionCounts = new int[p];

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Bootstrap sample
            var sampleIndices = Enumerable.Range(0, n)
                .OrderBy(_ => random.Next())
                .Take(sampleSize)
                .ToArray();

            // Create subsample
            var subData = new T[sampleSize, p];
            var subTarget = new T[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                for (int j = 0; j < p; j++)
                    subData[i, j] = data[sampleIndices[i], j];
                subTarget[i] = target[sampleIndices[i]];
            }

            var subMatrix = new Matrix<T>(subData);
            var subVector = new Vector<T>(subTarget);

            // Run base selector
            int[] selected;
            if (_baseSelector is not null)
            {
                selected = _baseSelector(subMatrix, subVector);
            }
            else
            {
                // Default: correlation-based selection
                selected = DefaultSelector(subMatrix, subVector, Math.Min(_nFeaturesToSelect, p));
            }

            // Count selections
            foreach (int idx in selected)
                if (idx >= 0 && idx < p)
                    selectionCounts[idx]++;
        }

        // Compute selection probabilities
        _selectionProbabilities = new double[p];
        for (int j = 0; j < p; j++)
            _selectionProbabilities[j] = (double)selectionCounts[j] / _nBootstraps;

        // Select features above threshold
        var stableFeatures = _selectionProbabilities
            .Select((prob, idx) => (Prob: prob, Index: idx))
            .Where(x => x.Prob >= _threshold)
            .OrderByDescending(x => x.Prob)
            .Take(_nFeaturesToSelect)
            .ToList();

        if (stableFeatures.Count == 0)
        {
            // Fall back to top by probability
            stableFeatures = _selectionProbabilities
                .Select((prob, idx) => (Prob: prob, Index: idx))
                .OrderByDescending(x => x.Prob)
                .Take(_nFeaturesToSelect)
                .ToList();
        }

        _selectedIndices = stableFeatures.Select(x => x.Index).OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private int[] DefaultSelector(Matrix<T> data, Vector<T> target, int k)
    {
        int n = data.Rows;
        int p = data.Columns;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var correlations = new double[p];
        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                correlations[j] = Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        return correlations
            .Select((c, idx) => (c, idx))
            .OrderByDescending(x => x.c)
            .Take(k)
            .Select(x => x.idx)
            .ToArray();
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
