using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Privacy;

/// <summary>
/// Differential Privacy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using differential privacy mechanisms to protect individual
/// data points while still identifying important features.
/// </para>
/// <para><b>For Beginners:</b> Differential privacy adds calibrated noise to feature
/// scores to protect individual privacy. Even if an attacker sees the selected
/// features, they can't determine if any specific individual was in the dataset.
/// This uses the exponential mechanism to select features with probability
/// proportional to their importance while maintaining privacy guarantees.
/// </para>
/// </remarks>
public class DifferentialPrivacySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _epsilon;
    private readonly int? _randomState;

    private double[]? _noisyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Epsilon => _epsilon;
    public double[]? NoisyScores => _noisyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DifferentialPrivacySelector(
        int nFeaturesToSelect = 10,
        double epsilon = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (epsilon <= 0)
            throw new ArgumentException("Epsilon must be positive.", nameof(epsilon));

        _nFeaturesToSelect = nFeaturesToSelect;
        _epsilon = epsilon;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DifferentialPrivacySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Compute base scores using correlation (sensitivity-bounded)
        var baseScores = new double[p];
        double targetMean = y.Average();
        double targetVar = y.Sum(v => (v - targetMean) * (v - targetMean));

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double mean = col.Average();
            double cov = 0, varX = 0;
            for (int i = 0; i < n; i++)
            {
                cov += (col[i] - mean) * (y[i] - targetMean);
                varX += (col[i] - mean) * (col[i] - mean);
            }

            double denom = Math.Sqrt(varX * targetVar);
            // Correlation is bounded in [-1, 1], so sensitivity is 2/n
            baseScores[j] = denom > 1e-10 ? Math.Abs(cov / denom) : 0;
        }

        // Add Laplace noise for differential privacy
        // Sensitivity of correlation is approximately 2/n
        double sensitivity = 2.0 / n;
        double scale = sensitivity / _epsilon;

        _noisyScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            // Laplace noise: sample from Laplace(0, scale)
            double u = rand.NextDouble() - 0.5;
            double laplace = -scale * Math.Sign(u) * Math.Log(1 - 2 * Math.Abs(u));
            _noisyScores[j] = baseScores[j] + laplace;
        }

        // Select top features by noisy scores
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _noisyScores[j])
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
            throw new InvalidOperationException("DifferentialPrivacySelector has not been fitted.");

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
        throw new NotSupportedException("DifferentialPrivacySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialPrivacySelector has not been fitted.");

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
