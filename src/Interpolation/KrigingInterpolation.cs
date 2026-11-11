namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Kriging interpolation for two-dimensional data points.
/// </summary>
/// <remarks>
/// Kriging is a geostatistical interpolation technique that predicts unknown values
/// based on the spatial correlation between known data points. It's particularly useful
/// for creating smooth surfaces from scattered data points.
/// 
/// <b>For Beginners:</b> Kriging is like predicting the height of a landscape at any point
/// when you only know the heights at certain locations. It works by assuming that points
/// closer together are more likely to have similar values than points far apart.
/// This method is widely used in geography, mining, and environmental science.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public partial class KrigingInterpolation<T> : I2DInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The z-values (heights) of the known data points.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _z;

    /// <summary>
    /// The calculated weights used in the Kriging interpolation.
    /// </summary>
    private Vector<T> _weights;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The matrix decomposition method used for solving linear systems.
    /// </summary>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The kernel function that determines how the influence of points decreases with distance.
    /// </summary>
    private readonly IKernelFunction<T> _kernel;

    /// <summary>
    /// The nugget parameter of the variogram model, representing measurement error or micro-scale variation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The nugget represents the "noise" in your data - small variations that occur
    /// even when measuring the same location multiple times.
    /// </remarks>
    private T _nugget;

    /// <summary>
    /// The sill parameter of the variogram model, representing the maximum variance between points.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The sill is like the "ceiling" of difference between any two points in your data,
    /// no matter how far apart they are.
    /// </remarks>
    private T _sill;

    /// <summary>
    /// The range parameter of the variogram model, representing the distance at which points become uncorrelated.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The range tells you how far apart two points can be before they stop influencing each other.
    /// Beyond this distance, knowing the value at one point doesn't help predict the value at the other.
    /// </remarks>
    private T _range;

    /// <summary>
    /// Creates a new instance of the Kriging interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the Kriging interpolator with your data points and
    /// automatically calculates the necessary parameters for interpolation.
    /// 
    /// <b>For Beginners:</b> When you create a Kriging interpolator, you provide the locations (x,y)
    /// and values (z) of your known data points. The constructor then analyzes the data to
    /// understand the spatial patterns and prepares the model for making predictions at new locations.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates of the known data points.</param>
    /// <param name="z">The z-values (heights) of the known data points.</param>
    /// <param name="kernel">Optional kernel function that determines how influence decreases with distance.</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving the linear system.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths or fewer than 3 points are provided.</exception>
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

    /// <summary>
    /// Interpolates the z-value at a given (x,y) coordinate.
    /// </summary>
    /// <remarks>
    /// This method predicts the value at any point using the Kriging algorithm.
    /// 
    /// <b>For Beginners:</b> Once you've set up the Kriging model with your known data points,
    /// this method lets you ask "What would the value be at this specific location?"
    /// It uses the spatial patterns detected in your data to make the best possible guess.
    /// </remarks>
    /// <param name="x">The x-coordinate of the point to interpolate.</param>
    /// <param name="y">The y-coordinate of the point to interpolate.</param>
    /// <returns>The interpolated z-value at the specified point.</returns>
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

    /// <summary>
    /// Calculates the weights used in the Kriging interpolation.
    /// </summary>
    /// <remarks>
    /// This method builds and solves the Kriging system of equations to determine
    /// the optimal weights for each known data point.
    /// 
    /// <b>For Beginners:</b> Kriging needs to figure out how much influence each known point
    /// should have when predicting values at new locations. This method calculates those
    /// influence factors (weights) by solving a system of equations that takes into account
    /// the spatial relationships between all points.
    /// </remarks>
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

    /// <summary>
    /// Estimates the parameters of the variogram model from the data.
    /// </summary>
    /// <remarks>
    /// This method analyzes the spatial relationships in the data to determine
    /// the nugget, sill, and range parameters of the variogram model.
    /// 
    /// <b>For Beginners:</b> To make accurate predictions, Kriging needs to understand
    /// how values change with distance in your specific dataset. This method analyzes
    /// your data to figure out these patterns. It groups pairs of points by distance,
    /// calculates how different their values tend to be, and fits a mathematical model
    /// to this relationship.
    /// </remarks>
    private void EstimateVariogramParameters()
    {
        const int numBins = 10;
        var engine = AiDotNetEngine.Current;
        int n = _x.Length;
        Tensor<T> xs = _x.AsTensor();
        Tensor<T> ys = _y.AsTensor();
        Tensor<T> zs = _z.AsTensor();

        // Calculate distances and semivariances for every pair i < j, kept in row-major (i, then j) order
        Tensor<T> dx = engine.TensorSubtract(xs.Reshape(n, 1), xs.Reshape(1, n));
        Tensor<T> dy = engine.TensorSubtract(ys.Reshape(n, 1), ys.Reshape(1, n));
        Tensor<T> dz = engine.TensorSubtract(zs.Reshape(n, 1), zs.Reshape(1, n));
        Tensor<T> pairDistances = engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy)));
        Tensor<T> pairGammas = engine.TensorDivideScalar(engine.TensorMultiply(dz, dz), _numOps.FromDouble(2));
        var aboveDiagonal = engine.TensorEqScalar(engine.TensorTriangularMask<T>(n, upper: true, diagonal: 1), _numOps.One);

        var distances = new List<T>(engine.TensorMaskedSelect(pairDistances, aboveDiagonal).ToVector());
        var gammas = new List<T>(engine.TensorMaskedSelect(pairGammas, aboveDiagonal).ToVector());

        // Sort distances and gammas
        var sortedPairs = distances.Zip(gammas, (d, g) => new { Distance = d, Gamma = g })
                                   .OrderBy(pair => Convert.ToDouble(pair.Distance))
                                   .ToList();

        // Bin the data: numBins consecutive bins of pairsPerBin sorted pairs each, averaged per bin
        int pairsPerBin = sortedPairs.Count / numBins;
        if (pairsPerBin == 0)
        {
            // An empty bin has no average (Enumerable.Average threw this for it before the bins were vectorized)
            throw new InvalidOperationException("Sequence contains no elements");
        }

        var binnedDistances = new List<T>(BinMeans(new Vector<T>(sortedPairs.Select(pair => pair.Distance)), numBins, pairsPerBin));
        var binnedGammas = new List<T>(BinMeans(new Vector<T>(sortedPairs.Select(pair => pair.Gamma)), numBins, pairsPerBin));

        // Fit exponential variogram model using least squares
        FitExponentialVariogram(binnedDistances, binnedGammas);
    }

    /// <summary>
    /// Averages the first <paramref name="binCount"/> * <paramref name="binSize"/> values in consecutive bins.
    /// </summary>
    /// <param name="values">The values, in bin order.</param>
    /// <param name="binCount">The number of bins.</param>
    /// <param name="binSize">The number of values in each bin; values past the last full bin are ignored.</param>
    /// <returns>The mean of each bin.</returns>
    private static Vector<T> BinMeans(Vector<T> values, int binCount, int binSize)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> binned = engine.TensorSlice(values.AsTensor(), new[] { 0 }, new[] { binCount * binSize })
            .Reshape(binCount, binSize);
        return engine.ReduceMean(binned, new[] { 1 }, keepDims: false).ToVector();
    }

    /// <summary>
    /// Fits an exponential variogram model to the provided distance and gamma value pairs.
    /// </summary>
    /// <remarks>
    /// This method uses linear regression to fit an exponential model to the experimental variogram data,
    /// which helps determine how spatial correlation changes with distance between points.
    /// 
    /// <b>For Beginners:</b> This method is like finding the best mathematical formula that describes
    /// how values become more different as points get farther apart. It's similar to drawing
    /// the best-fitting line through scattered points on a graph, but using a special curve
    /// shape (exponential) that works well for spatial data. The results help the algorithm
    /// understand the patterns in your data.
    /// </remarks>
    /// <param name="distances">List of binned distances between point pairs.</param>
    /// <param name="gammas">List of binned semi-variance values corresponding to the distances.</param>
    private void FitExponentialVariogram(List<T> distances, List<T> gammas)
    {
        var engine = AiDotNetEngine.Current;
        int n = distances.Count;
        T maxGamma = gammas.Max() ?? _numOps.Zero;

        Tensor<T> x = new Vector<T>(distances).AsTensor();
        Tensor<T> y = engine.TensorLog(engine.ScalarMinusTensor(maxGamma, new Vector<T>(gammas).AsTensor()));

        T sumX = engine.TensorSum(x);
        T sumY = engine.TensorSum(y);
        T sumXY = engine.TensorSum(engine.TensorMultiply(x, y));
        T sumX2 = engine.TensorSum(engine.TensorMultiply(x, x));

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
}
