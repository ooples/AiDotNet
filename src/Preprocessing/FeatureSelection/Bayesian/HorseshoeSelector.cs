using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Horseshoe Prior Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the horseshoe prior which has heavy tails (allowing large coefficients
/// for important features) and infinite mass at zero (shrinking irrelevant
/// features to exactly zero).
/// </para>
/// <para><b>For Beginners:</b> The horseshoe prior is shaped like a horseshoe -
/// most feature weights get pushed to zero (irrelevant), but truly important
/// features are allowed to have large weights. It's very good at finding
/// sparse solutions where only a few features matter.
/// </para>
/// </remarks>
public class HorseshoeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _tau;

    private double[]? _posteriorMeans;
    private double[]? _shrinkageFactors;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? PosteriorMeans => _posteriorMeans;
    public double[]? ShrinkageFactors => _shrinkageFactors;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HorseshoeSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        double tau = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _tau = tau;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HorseshoeSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Standardize
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        // Initialize
        _posteriorMeans = new double[p];
        _shrinkageFactors = new double[p];
        var lambda = new double[p];
        for (int j = 0; j < p; j++) lambda[j] = 1.0;

        double sigma2 = 1.0;
        var rand = RandomHelper.CreateSecureRandom();

        // Gibbs sampling
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Update beta
            for (int j = 0; j < p; j++)
            {
                double xjy = 0, xjxj = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = y[i];
                    for (int k = 0; k < p; k++)
                        if (k != j) residual -= X[i, k] * _posteriorMeans[k];
                    xjy += X[i, j] * residual;
                    xjxj += X[i, j] * X[i, j];
                }

                double priorVar = lambda[j] * lambda[j] * _tau * _tau;
                double postVar = 1.0 / (xjxj / sigma2 + 1.0 / priorVar);
                double postMean = postVar * xjy / sigma2;
                _posteriorMeans[j] = postMean;
            }

            // Update lambda (local shrinkage)
            for (int j = 0; j < p; j++)
            {
                double beta2 = _posteriorMeans[j] * _posteriorMeans[j];
                double rate = 1.0 + beta2 / (2 * _tau * _tau * sigma2);
                lambda[j] = Math.Sqrt(1.0 / SampleInverseGamma(0.5, rate, rand));
            }

            // Update sigma2
            double sse = 0;
            for (int i = 0; i < n; i++)
            {
                double pred = 0;
                for (int j = 0; j < p; j++) pred += X[i, j] * _posteriorMeans[j];
                sse += (y[i] - pred) * (y[i] - pred);
            }
            sigma2 = SampleInverseGamma((n + p) / 2.0, sse / 2.0, rand);
        }

        // Compute shrinkage factors
        for (int j = 0; j < p; j++)
        {
            double kappa = 1.0 / (1.0 + lambda[j] * lambda[j] * _tau * _tau);
            _shrinkageFactors[j] = 1.0 - kappa;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_posteriorMeans[j]) * _shrinkageFactors[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double SampleInverseGamma(double shape, double rate, Random rand)
    {
        double x = SampleGamma(shape, 1.0 / rate, rand);
        return 1.0 / (x + 1e-10);
    }

    private double SampleGamma(double shape, double scale, Random rand)
    {
        if (shape < 1)
        {
            double u = rand.NextDouble();
            return SampleGamma(shape + 1, scale, rand) * Math.Pow(u, 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = SampleNormal(rand);
                v = 1.0 + c * x;
            } while (v <= 0);

            v = v * v * v;
            double u = rand.NextDouble();
            if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v * scale;
            if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v))) return d * v * scale;
        }
    }

    private double SampleNormal(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HorseshoeSelector has not been fitted.");

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
        throw new NotSupportedException("HorseshoeSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HorseshoeSelector has not been fitted.");

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
