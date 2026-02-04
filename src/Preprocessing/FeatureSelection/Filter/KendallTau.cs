using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Kendall Tau correlation for feature selection based on concordant pairs.
/// </summary>
/// <remarks>
/// <para>
/// Kendall's Tau measures the ordinal association between features and target by
/// comparing pairs of observations. It counts concordant pairs (both values increase
/// or both decrease) versus discordant pairs (one increases while the other decreases).
/// </para>
/// <para><b>For Beginners:</b> Kendall Tau looks at every pair of data points and asks:
/// when the feature goes up, does the target also go up? If most pairs agree (concordant),
/// the correlation is high. Unlike Pearson, it doesn't assume linear relationships and
/// is very robust to outliers. It's especially good for ordinal or ranked data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KendallTau<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private double[]? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Correlations => _correlations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KendallTau(
        int nFeaturesToSelect = 10,
        double minCorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCorrelation = minCorrelation;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KendallTau requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Extract target values
        var targetValues = new double[n];
        for (int i = 0; i < n; i++)
            targetValues[i] = NumOps.ToDouble(target[i]);

        _correlations = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract feature values
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
                featureValues[i] = NumOps.ToDouble(data[i, j]);

            // Count concordant and discordant pairs
            long concordant = 0;
            long discordant = 0;
            long tiesX = 0;
            long tiesY = 0;

            for (int i = 0; i < n - 1; i++)
            {
                for (int k = i + 1; k < n; k++)
                {
                    double xDiff = featureValues[k] - featureValues[i];
                    double yDiff = targetValues[k] - targetValues[i];

                    if (Math.Abs(xDiff) < 1e-10 && Math.Abs(yDiff) < 1e-10)
                    {
                        // Tie in both
                        continue;
                    }
                    else if (Math.Abs(xDiff) < 1e-10)
                    {
                        // Tie in X only
                        tiesX++;
                    }
                    else if (Math.Abs(yDiff) < 1e-10)
                    {
                        // Tie in Y only
                        tiesY++;
                    }
                    else if ((xDiff > 0 && yDiff > 0) || (xDiff < 0 && yDiff < 0))
                    {
                        concordant++;
                    }
                    else
                    {
                        discordant++;
                    }
                }
            }

            // Kendall Tau-b (adjusted for ties)
            long nPairs = (long)n * (n - 1) / 2;
            double numerator = concordant - discordant;
            double denominator = Math.Sqrt((nPairs - tiesX) * (double)(nPairs - tiesY));

            _correlations[j] = denominator > 1e-10 ? numerator / denominator : 0;
        }

        // Select features above threshold or top by absolute correlation
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (Math.Abs(_correlations[j]) >= _minCorrelation)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => Math.Abs(_correlations[j]))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _correlations
                .Select((c, idx) => (Corr: Math.Abs(c), Index: idx))
                .OrderByDescending(x => x.Corr)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
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
            throw new InvalidOperationException("KendallTau has not been fitted.");

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
        throw new NotSupportedException("KendallTau does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTau has not been fitted.");

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
