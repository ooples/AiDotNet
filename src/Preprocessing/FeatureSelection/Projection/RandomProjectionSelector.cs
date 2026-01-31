using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Random Projection-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses random projections to identify features that contribute most to
/// preserving distances in the projected space (Johnson-Lindenstrauss lemma).
/// </para>
/// <para><b>For Beginners:</b> Random projections preserve important structure
/// in data when projecting to lower dimensions. By looking at which original
/// features contribute most to these projections, we can identify the most
/// important features without expensive computations.
/// </para>
/// </remarks>
public class RandomProjectionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nProjections;
    private readonly int _projectionDim;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RandomProjectionSelector(
        int nFeaturesToSelect = 10,
        int nProjections = 50,
        int projectionDim = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nProjections = nProjections;
        _projectionDim = projectionDim;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        int k = Math.Min(_projectionDim, p);
        _featureScores = new double[p];

        for (int proj = 0; proj < _nProjections; proj++)
        {
            // Generate sparse random projection matrix
            var R = new double[p, k];
            double sqrtP = Math.Sqrt(p);
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double u = rand.NextDouble();
                    if (u < 1.0 / 6)
                        R[i, j] = sqrtP;
                    else if (u < 2.0 / 6)
                        R[i, j] = -sqrtP;
                    // else 0 (4/6 probability)
                }
            }

            // Project data
            var projected = new double[n, k];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    for (int l = 0; l < p; l++)
                        projected[i, j] += X[i, l] * R[l, j];

            // Compute variance in each projected dimension
            for (int j = 0; j < k; j++)
            {
                double mean = 0;
                for (int i = 0; i < n; i++) mean += projected[i, j];
                mean /= n;

                double variance = 0;
                for (int i = 0; i < n; i++)
                    variance += (projected[i, j] - mean) * (projected[i, j] - mean);
                variance /= (n - 1);

                // Weight features by their contribution to high-variance projections
                for (int l = 0; l < p; l++)
                    _featureScores[l] += Math.Abs(R[l, j]) * variance;
            }
        }

        // Normalize
        double maxScore = _featureScores.Max();
        if (maxScore > 0)
            for (int j = 0; j < p; j++)
                _featureScores[j] /= maxScore;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomProjectionSelector has not been fitted.");

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
        throw new NotSupportedException("RandomProjectionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomProjectionSelector has not been fitted.");

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
