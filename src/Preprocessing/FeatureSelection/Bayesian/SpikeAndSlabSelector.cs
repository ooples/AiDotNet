using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Spike-and-Slab Feature Selection using Bayesian variable selection.
/// </summary>
/// <remarks>
/// <para>
/// Implements the spike-and-slab prior model where each feature coefficient
/// has a mixture prior: a "spike" (concentrated at zero) and a "slab" (diffuse).
/// Features with high probability of being in the slab are selected.
/// </para>
/// <para><b>For Beginners:</b> Imagine a spike-and-slab as two possibilities for
/// each feature: either it has no effect (spike at zero) or it has some effect
/// (spread out slab). The method estimates which possibility is more likely for
/// each feature. If the slab is more likely, the feature is important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpikeAndSlabSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _slabVariance;
    private readonly double _spikeVariance;
    private readonly double _priorInclusionProbability;
    private readonly int _nIterations;

    private double[]? _inclusionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? InclusionProbabilities => _inclusionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SpikeAndSlabSelector(
        int nFeaturesToSelect = 10,
        double slabVariance = 1.0,
        double spikeVariance = 0.001,
        double priorInclusionProbability = 0.5,
        int nIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (slabVariance <= 0)
            throw new ArgumentException("Slab variance must be positive.", nameof(slabVariance));
        if (spikeVariance <= 0)
            throw new ArgumentException("Spike variance must be positive.", nameof(spikeVariance));

        _nFeaturesToSelect = nFeaturesToSelect;
        _slabVariance = slabVariance;
        _spikeVariance = spikeVariance;
        _priorInclusionProbability = priorInclusionProbability;
        _nIterations = nIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SpikeAndSlabSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _inclusionProbabilities = new double[p];
        var gamma = new double[p];

        // Initialize inclusion indicators
        for (int j = 0; j < p; j++)
            gamma[j] = _priorInclusionProbability;

        // Convert data to arrays for faster access
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Estimate residual variance
        double yMean = y.Average();
        double residualVariance = y.Sum(yi => (yi - yMean) * (yi - yMean)) / (n - 1) + 1e-10;

        // Gibbs sampling iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            for (int j = 0; j < p; j++)
            {
                // Compute marginal likelihood ratio
                double xjNorm = 0;
                double xjy = 0;
                for (int i = 0; i < n; i++)
                {
                    xjNorm += X[i, j] * X[i, j];
                    xjy += X[i, j] * y[i];
                }

                // Bayes factor approximation
                double betaHat = xjNorm > 1e-10 ? xjy / xjNorm : 0;
                double logBFSlab = -0.5 * Math.Log(_slabVariance + residualVariance / xjNorm)
                    + 0.5 * betaHat * betaHat * xjNorm / (residualVariance + 1e-10);
                double logBFSpike = -0.5 * Math.Log(_spikeVariance + residualVariance / xjNorm)
                    + 0.5 * betaHat * betaHat * xjNorm / (residualVariance * _spikeVariance / _slabVariance + 1e-10);

                double logBF = logBFSlab - logBFSpike;
                double priorOdds = _priorInclusionProbability / (1 - _priorInclusionProbability);
                double posteriorOdds = priorOdds * Math.Exp(Math.Min(logBF, 50));

                gamma[j] = posteriorOdds / (1 + posteriorOdds);
            }

            // Accumulate for posterior mean
            for (int j = 0; j < p; j++)
                _inclusionProbabilities[j] += gamma[j];
        }

        // Average over iterations
        for (int j = 0; j < p; j++)
            _inclusionProbabilities[j] /= _nIterations;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _inclusionProbabilities[j])
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
            throw new InvalidOperationException("SpikeAndSlabSelector has not been fitted.");

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
        throw new NotSupportedException("SpikeAndSlabSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpikeAndSlabSelector has not been fitted.");

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
