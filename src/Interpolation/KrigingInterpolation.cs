namespace AiDotNet.Interpolation;

public class KrigingInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _z;
    private Vector<T> _weights;
    private readonly INumericOperations<T> _numOps;
    private readonly IMatrixDecomposition<T>? _decomposition;
    private readonly IKernelFunction<T> _kernel;
    private T _nugget;
    private T _sill;
    private T _range;

    public KrigingInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, 
        IKernelFunction<T>? kernel = null, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 3)
            throw new ArgumentException("At least 3 points are required for Kriging interpolation.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = Vector<T>.Empty();
        _decomposition = decomposition;
        _kernel = kernel ?? new GaussianKernel<T>();
        _nugget = _numOps.Zero;
        _sill = _numOps.Zero;
        _range = _numOps.Zero;

        EstimateVariogramParameters();
        CalculateWeights();
    }

    public T Interpolate(T x, T y)
    {
        Vector<T> point = new Vector<T>(new[] { x, y });
        Vector<T> k = new Vector<T>(_x.Length);
        for (int i = 0; i < _x.Length; i++)
        {
            Vector<T> dataPoint = new Vector<T>(new[] { _x[i], _y[i] });
            k[i] = _kernel.Calculate(point, dataPoint);
        }

        T result = _numOps.Zero;
        for (int i = 0; i < _weights.Length; i++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_weights[i], k[i]));
        }

        return result;
    }

    private void CalculateWeights()
    {
        int n = _x.Length;
        Matrix<T> K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> point1 = new Vector<T>(new[] { _x[i], _y[i] });
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    K[i, j] = _numOps.Add(_sill, _nugget);
                }
                else
                {
                    Vector<T> point2 = new Vector<T>([_x[j], _y[j]]);
                    K[i, j] = _kernel.Calculate(point1, point2);
                }
            }
        }

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(K);
        _weights = MatrixSolutionHelper.SolveLinearSystem(_z, decomposition);
    }

    private void EstimateVariogramParameters()
    {
        const int numBins = 10;
        var distances = new List<T>();
        var gammas = new List<T>();

        // Calculate distances and semivariances
        for (int i = 0; i < _x.Length; i++)
        {
            for (int j = i + 1; j < _x.Length; j++)
            {
                T distance = CalculateDistance(_x[i], _y[i], _x[j], _y[j]);
                T diff = _numOps.Subtract(_z[i], _z[j]);
                T squaredDiff = _numOps.Multiply(diff, diff);
                T gamma = _numOps.Divide(squaredDiff, _numOps.FromDouble(2));

                distances.Add(distance);
                gammas.Add(gamma);
            }
        }

        // Sort distances and gammas
        var sortedPairs = distances.Zip(gammas, (d, g) => new { Distance = d, Gamma = g })
                                   .OrderBy(pair => Convert.ToDouble(pair.Distance))
                                   .ToList();

        // Bin the data
        var binnedDistances = new List<T>();
        var binnedGammas = new List<T>();
        int pairsPerBin = sortedPairs.Count / numBins;

        for (int i = 0; i < numBins; i++)
        {
            var bin = sortedPairs.Skip(i * pairsPerBin).Take(pairsPerBin);
            binnedDistances.Add(_numOps.FromDouble(bin.Average(p => Convert.ToDouble(p.Distance))));
            binnedGammas.Add(_numOps.FromDouble(bin.Average(p => Convert.ToDouble(p.Gamma))));
        }

        // Fit exponential variogram model using least squares
        FitExponentialVariogram(binnedDistances, binnedGammas);
    }

    private void FitExponentialVariogram(List<T> distances, List<T> gammas)
    {
        int n = distances.Count;
        T sumX = _numOps.Zero, sumY = _numOps.Zero, sumXY = _numOps.Zero, sumX2 = _numOps.Zero;
        T maxGamma = gammas.Max() ?? _numOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T x = distances[i];
            T y = _numOps.Log(_numOps.Subtract(maxGamma, gammas[i]));

            sumX = _numOps.Add(sumX, x);
            sumY = _numOps.Add(sumY, y);
            sumXY = _numOps.Add(sumXY, _numOps.Multiply(x, y));
            sumX2 = _numOps.Add(sumX2, _numOps.Multiply(x, x));
        }

        T slope = _numOps.Divide(
            _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(n), sumXY), _numOps.Multiply(sumX, sumY)),
            _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(n), sumX2), _numOps.Multiply(sumX, sumX))
        );

        T intercept = _numOps.Divide(
            _numOps.Subtract(sumY, _numOps.Multiply(slope, sumX)),
            _numOps.FromDouble(n)
        );

        _range = _numOps.Divide(_numOps.FromDouble(-1), slope);
        _sill = maxGamma;
        _nugget = _numOps.Subtract(_sill, _numOps.Exp(intercept));

        // Ensure nugget is non-negative
        _nugget = MathHelper.Max(_nugget, _numOps.Zero);
    }

    private T CalculateDistance(T x1, T y1, T x2, T y2)
    {
        T dx = _numOps.Subtract(x1, x2);
        T dy = _numOps.Subtract(y1, y2);

        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
    }
}