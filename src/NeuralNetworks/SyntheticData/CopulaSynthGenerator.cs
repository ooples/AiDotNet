using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Copula-Based Synthesis generator that models marginal distributions independently
/// and couples them with a Gaussian copula to capture inter-feature dependencies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The generator operates in three phases:
/// 1. Fit marginal distributions for each feature using empirical CDF / kernel density estimation
/// 2. Transform data to uniform [0,1] via marginal CDFs, then to standard normal via inverse CDF
/// 3. Fit a Gaussian copula (correlation matrix) on the normal-transformed data
/// </para>
/// <para>
/// To generate:
/// 1. Sample from multivariate normal using the learned correlation matrix
/// 2. Transform back to uniform via standard normal CDF
/// 3. Transform to original scale via inverse marginal CDFs (quantile functions)
/// </para>
/// <para>
/// <b>For Beginners:</b> This method works like building synthetic data from two ingredients:
///
/// Ingredient 1 — Marginal shapes:
///   Each feature's histogram is learned independently (e.g., "Age is bell-shaped around 40").
///
/// Ingredient 2 — Correlation structure:
///   How features move together (e.g., "Age and Income go up together").
///
/// To generate: sample correlated random values, then map each one to its feature's histogram.
/// The result preserves both individual distributions and pairwise correlations.
/// </para>
/// </remarks>
public class CopulaSynthGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private readonly CopulaSynthOptions<T> _options;

    // Marginal parameters per feature
    private double[][] _sortedValues = Array.Empty<double[]>();

    // Gaussian copula: correlation matrix (Cholesky-decomposed for sampling)
    private double[,]? _choleskyCorr;

    private int _numFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="CopulaSynthGenerator{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the Copula Synthesis model.</param>
    public CopulaSynthGenerator(CopulaSynthOptions<T> options) : base(options.Seed)
    {
        _options = options;
    }

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numFeatures = data.Columns;
        int n = data.Rows;

        // Step 1: Store sorted values per feature for empirical CDF/quantile
        _sortedValues = new double[_numFeatures][];
        for (int j = 0; j < _numFeatures; j++)
        {
            var vals = new double[n];
            for (int i = 0; i < n; i++)
                vals[i] = NumOps.ToDouble(data[i, j]);
            Array.Sort(vals);
            _sortedValues[j] = vals;
        }

        // Step 2: Transform to standard normal via empirical CDF → inverse normal CDF
        var normalData = new double[n, _numFeatures];
        for (int j = 0; j < _numFeatures; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                double u = EmpiricalCDF(j, val);
                // Clamp to avoid infinities at boundaries
                u = Math.Min(Math.Max(u, 1e-6), 1.0 - 1e-6);
                normalData[i, j] = InverseNormalCDF(u);
            }
        }

        // Step 3: Compute correlation matrix on normal-transformed data
        var corrMatrix = ComputeCorrelation(normalData, n, _numFeatures);

        // Step 4: Cholesky decomposition for sampling
        _choleskyCorr = CholeskyDecompose(corrMatrix, _numFeatures);
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        if (_choleskyCorr is null)
            throw new InvalidOperationException("Generator is not fitted.");

        var result = new Matrix<T>(numSamples, _numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            // Sample standard normal vector
            var z = new double[_numFeatures];
            for (int j = 0; j < _numFeatures; j++)
                z[j] = NumOps.ToDouble(SampleStandardNormal());

            // Multiply by Cholesky factor to get correlated normals
            var correlated = new double[_numFeatures];
            for (int j = 0; j < _numFeatures; j++)
            {
                double sum = 0;
                for (int k = 0; k <= j; k++)
                    sum += _choleskyCorr[j, k] * z[k];
                correlated[j] = sum;
            }

            // Transform: standard normal → uniform → original scale via quantile function
            for (int j = 0; j < _numFeatures; j++)
            {
                double u = NormalCDF(correlated[j]);
                u = Math.Min(Math.Max(u, 1e-6), 1.0 - 1e-6);
                double val = InverseEmpiricalCDF(j, u);
                result[i, j] = NumOps.FromDouble(val);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the empirical CDF value for a given feature and value.
    /// P(X &lt;= x) approximated by the proportion of sorted values &lt;= x.
    /// </summary>
    private double EmpiricalCDF(int featureIndex, double value)
    {
        var sorted = _sortedValues[featureIndex];
        int n = sorted.Length;
        if (n == 0) return 0.5;

        // Binary search for the position
        int idx = Array.BinarySearch(sorted, value);
        if (idx < 0) idx = ~idx;
        else idx++; // Include the found element

        return (double)idx / n;
    }

    /// <summary>
    /// Inverse empirical CDF (quantile function): given probability u, returns value.
    /// Uses linear interpolation between sorted values.
    /// </summary>
    private double InverseEmpiricalCDF(int featureIndex, double u)
    {
        var sorted = _sortedValues[featureIndex];
        int n = sorted.Length;
        if (n == 0) return 0;

        double idx = u * (n - 1);
        int lo = (int)Math.Floor(idx);
        int hi = Math.Min(lo + 1, n - 1);
        lo = Math.Max(lo, 0);

        double frac = idx - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
    }

    /// <summary>
    /// Computes the correlation matrix from normal-transformed data.
    /// </summary>
    private static double[,] ComputeCorrelation(double[,] data, int n, int d)
    {
        var means = new double[d];
        var stds = new double[d];
        var corr = new double[d, d];

        // Compute means
        for (int j = 0; j < d; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += data[i, j];
            means[j] = sum / n;
        }

        // Compute standard deviations
        for (int j = 0; j < d; j++)
        {
            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = data[i, j] - means[j];
                sumSq += diff * diff;
            }
            stds[j] = n > 1 ? Math.Sqrt(sumSq / (n - 1)) : 1.0;
            if (stds[j] < 1e-10) stds[j] = 1e-10;
        }

        // Compute correlation
        for (int j1 = 0; j1 < d; j1++)
        {
            corr[j1, j1] = 1.0;
            for (int j2 = j1 + 1; j2 < d; j2++)
            {
                double cov = 0;
                for (int i = 0; i < n; i++)
                    cov += (data[i, j1] - means[j1]) * (data[i, j2] - means[j2]);
                cov /= (n - 1);
                double r = cov / (stds[j1] * stds[j2]);
                r = Math.Min(Math.Max(r, -0.999), 0.999);
                corr[j1, j2] = r;
                corr[j2, j1] = r;
            }
        }

        return corr;
    }

    /// <summary>
    /// Performs Cholesky decomposition: A = L * L^T, returns L.
    /// Falls back to identity if decomposition fails (non-positive-definite).
    /// </summary>
    private static double[,] CholeskyDecompose(double[,] matrix, int d)
    {
        var L = new double[d, d];

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                    sum += L[i, k] * L[j, k];

                if (i == j)
                {
                    double val = matrix[i, i] - sum;
                    if (val <= 0) val = 1e-6; // Ensure positive definiteness
                    L[i, j] = Math.Sqrt(val);
                }
                else
                {
                    double denom = L[j, j];
                    if (Math.Abs(denom) < 1e-10) denom = 1e-10;
                    L[i, j] = (matrix[i, j] - sum) / denom;
                }
            }
        }

        return L;
    }

    /// <summary>
    /// Standard normal CDF using the error function approximation.
    /// </summary>
    private static double NormalCDF(double x)
    {
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    /// <summary>
    /// Inverse standard normal CDF using rational approximation (Beasley-Springer-Moro).
    /// </summary>
    private static double InverseNormalCDF(double p)
    {
        if (p <= 0) return -10.0;
        if (p >= 1) return 10.0;

        double q = p - 0.5;
        double r;
        if (Math.Abs(q) <= 0.425)
        {
            r = 0.180625 - q * q;
            return q * (((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r +
                6.7265770927008700853e4) * r + 4.5921953931549871457e4) * r +
                1.3731693765509461125e4) * r + 1.9715909503065514427e3) * r +
                1.3314166764078226174e2) * r + 3.3871328727963666080e0) /
                (((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r +
                3.9307895800092710610e4) * r + 2.1213794301586595867e4) * r +
                5.3941960214247511077e3) * r + 6.8718700749205790830e2) * r +
                4.2313330701600911252e1) * r + 1.0);
        }
        else
        {
            r = q < 0 ? p : 1 - p;
            r = Math.Sqrt(-Math.Log(r));

            double result;
            if (r <= 5.0)
            {
                r -= 1.6;
                result = (((((((7.7454501427834140764e-4 * r + 2.2723844989269184187e-2) * r +
                    7.2235882094086552433e-1) * r + 1.3045513272014416523e1) * r +
                    6.7726035681048076932e1) * r + 1.0831971006903470000e2) * r +
                    1.1725700325035845468e2) * r + 1.1986131097775042662e2) /
                    (((((((1.0507500716444169339e-4 * r + 1.0532057034436329024e-2) * r +
                    1.6882755560235047313e-1) * r + 7.1753246312516124959e-1) * r +
                    1.0563096406028367585e1) * r + 4.0984797509990525498e1) * r +
                    6.0191721792796699071e1) * r + 4.8180017248005942159e1);
            }
            else
            {
                r -= 5.0;
                result = (((((((2.0103102997076720227e-7 * r + 2.7623188653471891095e-5) * r +
                    1.2425092002673407267e-3) * r + 2.6520252785698296534e-2) * r +
                    2.9694239383093398677e-1) * r + 1.7832908668613600000e0) * r +
                    5.1908257258867402690e0) * r + 6.0511655467684700000e0) /
                    (((((((2.0442819666417094592e-7 * r + 1.4215117665672720000e-5) * r +
                    1.8463183175054972690e-4) * r + 7.8686960064460000000e-3) * r +
                    1.4862913816091900000e-1) * r + 6.8085697098073400000e-1) * r +
                    1.6780028166927800000e0) * r + 1.0);
            }

            if (q < 0) result = -result;
            return result;
        }
    }

    /// <summary>
    /// Error function approximation using Horner form.
    /// </summary>
    private static double Erf(double x)
    {
        double sign = x >= 0 ? 1.0 : -1.0;
        x = Math.Abs(x);
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }
}
