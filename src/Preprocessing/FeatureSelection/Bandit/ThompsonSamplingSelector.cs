using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bandit;

/// <summary>
/// Thompson Sampling based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using Thompson Sampling, a Bayesian approach to the
/// multi-armed bandit problem that maintains probability distributions over
/// feature quality.
/// </para>
/// <para><b>For Beginners:</b> Thompson Sampling treats each feature as a "slot machine"
/// with unknown payout. Instead of tracking a single estimate, we maintain a probability
/// distribution (Beta distribution) representing our belief about each feature's quality.
/// We sample from these distributions and pick the feature with the highest sample.
/// This naturally balances exploration (trying uncertain features) and exploitation
/// (using features we know are good).
/// </para>
/// </remarks>
public class ThompsonSamplingSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _alphas;
    private double[]? _betas;
    private double[]? _posteriorMeans;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double[]? Alphas => _alphas;
    public double[]? Betas => _betas;
    public double[]? PosteriorMeans => _posteriorMeans;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ThompsonSamplingSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ThompsonSamplingSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Precompute feature-target correlations as true reward probabilities
        double targetMean = y.Average();
        double targetVar = y.Sum(v => (v - targetMean) * (v - targetMean));
        var baseCorrelations = new double[p];

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
            baseCorrelations[j] = denom > 1e-10 ? Math.Abs(cov / denom) : 0;
        }

        // Initialize Beta distribution parameters (uniform prior: alpha=1, beta=1)
        _alphas = new double[p];
        _betas = new double[p];
        for (int j = 0; j < p; j++)
        {
            _alphas[j] = 1.0;
            _betas[j] = 1.0;
        }

        // Thompson Sampling iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Sample from each arm's Beta distribution
            var samples = new double[p];
            int bestArm = 0;
            double bestSample = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                samples[j] = SampleBeta(_alphas[j], _betas[j], rand);
                if (samples[j] > bestSample)
                {
                    bestSample = samples[j];
                    bestArm = j;
                }
            }

            // Pull the selected arm and observe reward
            double noise = (rand.NextDouble() - 0.5) * 0.2;
            double rewardProb = Math.Max(0, Math.Min(1, baseCorrelations[bestArm] + noise));
            bool success = rand.NextDouble() < rewardProb;

            // Update Beta distribution parameters
            if (success)
                _alphas[bestArm] += 1;
            else
                _betas[bestArm] += 1;
        }

        // Compute posterior means: alpha / (alpha + beta)
        _posteriorMeans = new double[p];
        for (int j = 0; j < p; j++)
            _posteriorMeans[j] = _alphas[j] / (_alphas[j] + _betas[j]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _posteriorMeans[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    /// <summary>
    /// Sample from a Beta(alpha, beta) distribution using the ratio of Gamma samples.
    /// </summary>
    private double SampleBeta(double alpha, double beta, Random rand)
    {
        double x = SampleGamma(alpha, rand);
        double gammaY = SampleGamma(beta, rand);
        return x / (x + gammaY);
    }

    /// <summary>
    /// Sample from a Gamma(shape, 1) distribution using Marsaglia-Tsang method.
    /// </summary>
    private double SampleGamma(double shape, Random rand)
    {
        if (shape < 1)
        {
            // Use Gamma(shape+1) / U^(1/shape)
            double u = rand.NextDouble();
            return SampleGamma(shape + 1, rand) * Math.Pow(u, 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            double x, v;
            do
            {
                x = SampleStandardNormal(rand);
                v = 1 + c * x;
            } while (v <= 0);

            v = v * v * v;
            double u = rand.NextDouble();

            if (u < 1 - 0.0331 * (x * x) * (x * x))
                return d * v;

            if (Math.Log(u) < 0.5 * x * x + d * (1 - v + Math.Log(v)))
                return d * v;
        }
    }

    /// <summary>
    /// Sample from standard normal distribution using Box-Muller transform.
    /// </summary>
    private double SampleStandardNormal(Random rand)
    {
        double u1, u2;
        do
        {
            u1 = rand.NextDouble();
            u2 = rand.NextDouble();
        } while (u1 <= double.Epsilon);

        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ThompsonSamplingSelector has not been fitted.");

        if (data.Columns < _nInputFeatures)
            throw new ArgumentException(
                $"Input data has {data.Columns} columns but selector was fitted on {_nInputFeatures} columns.",
                nameof(data));

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
        throw new NotSupportedException("ThompsonSamplingSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ThompsonSamplingSelector has not been fitted.");

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
