namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Base class for time series decomposition algorithms that break down time series data into component parts.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series decomposition is like taking apart a complex toy to see its individual pieces.
/// It breaks down a sequence of data points (like daily sales, monthly temperatures, etc.) into simpler
/// components that are easier to understand. Common components include:
/// - Trend: The long-term direction (going up, down, or staying flat)
/// - Seasonal: Regular patterns that repeat (like higher sales during holidays)
/// - Residual: The leftover "noise" after removing trend and seasonal patterns
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public abstract class TimeSeriesDecompositionBase<T> : ITimeSeriesDecomposition<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The original time series data to be decomposed.
    /// </summary>
    public Vector<T> TimeSeries { get; }

    /// <summary>
    /// Dictionary storing the different components extracted from the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Think of this as a container that holds all the separate pieces
    /// after we've taken apart our time series. Each piece (trend, seasonal pattern, etc.)
    /// is stored with a label so we can easily retrieve it later.
    /// </para>
    /// </remarks>
    protected Dictionary<DecompositionComponentType, object> Components { get; }

    /// <summary>
    /// Initializes a new instance of the time series decomposition class.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    protected TimeSeriesDecompositionBase(Vector<T> timeSeries)
    {
        TimeSeries = timeSeries;
        NumOps = MathHelper.GetNumericOperations<T>();
        Components = new Dictionary<DecompositionComponentType, object>();
    }

    /// <summary>
    /// Performs the decomposition of the time series into its components.
    /// This method must be implemented by derived classes.
    /// </summary>
    protected abstract void Decompose();

    /// <summary>
    /// Returns all components extracted from the time series.
    /// </summary>
    /// <returns>A dictionary containing all decomposed components.</returns>
    public Dictionary<DecompositionComponentType, object> GetComponents() => Components;

    /// <summary>
    /// Adds a component to the decomposition results.
    /// </summary>
    /// <param name="componentType">The type of component (e.g., Trend, Seasonal).</param>
    /// <param name="component">The component data as either a vector or matrix.</param>
    /// <exception cref="ArgumentException">Thrown when the component is not a Vector or Matrix.</exception>
    protected void AddComponent(DecompositionComponentType componentType, object component)
    {
        if (component is Vector<T> vector)
        {
            Components[componentType] = vector;
        }
        else if (component is Matrix<T> matrix)
        {
            Components[componentType] = matrix;
        }
        else
        {
            throw new ArgumentException("Component must be either Vector<T> or Matrix<T>", nameof(component));
        }
    }

    /// <summary>
    /// Retrieves a specific component from the decomposition results.
    /// </summary>
    /// <param name="componentType">The type of component to retrieve.</param>
    /// <returns>The requested component, or null if it doesn't exist.</returns>
    public object? GetComponent(DecompositionComponentType componentType)
    {
        if (Components.TryGetValue(componentType, out var component))
        {
            return component;
        }
        else
        {
            return null;
        }
    }

    /// <summary>
    /// Retrieves a specific component as a vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A vector is simply a list of numbers in a specific order.
    /// For time series data, each number typically represents a value at a specific point in time.
    /// </para>
    /// </remarks>
    /// <param name="componentType">The type of component to retrieve.</param>
    /// <returns>The component as a vector, or an empty vector if the component doesn't exist or isn't a vector.</returns>
    public Vector<T> GetComponentAsVector(DecompositionComponentType componentType)
    {
        if (Components.TryGetValue(componentType, out var component))
        {
            return component as Vector<T> ?? Vector<T>.Empty();
        }
        else
        {
            return Vector<T>.Empty();
        }
    }

    /// <summary>
    /// Retrieves a specific component as a matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A matrix is like a table or grid of numbers with rows and columns.
    /// In time series analysis, a matrix might store multiple related series or different 
    /// versions of the same series (e.g., seasonal patterns for different years).
    /// </para>
    /// </remarks>
    /// <param name="componentType">The type of component to retrieve.</param>
    /// <returns>The component as a matrix, or an empty matrix if the component doesn't exist or isn't a matrix.</returns>
    public Matrix<T> GetComponentAsMatrix(DecompositionComponentType componentType)
    {
        if (Components.TryGetValue(componentType, out var component))
        {
            return component as Matrix<T> ?? Matrix<T>.Empty();
        }
        else
        {
            return Matrix<T>.Empty();
        }
    }

    /// <summary>
    /// Checks if a specific component exists in the decomposition results.
    /// </summary>
    /// <param name="componentType">The type of component to check for.</param>
    /// <returns>True if the component exists; otherwise, false.</returns>
    public bool HasComponent(DecompositionComponentType componentType)
    {
        return Components.ContainsKey(componentType);
    }

    /// <summary>
    /// Calculates the residual component by subtracting trend and seasonal components from the original time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The residual is what's left over after we remove the main patterns from our data.
    /// Think of it like this:
    /// Original data = Trend + Seasonal pattern + Residual
    /// 
    /// So: Residual = Original data - (Trend + Seasonal pattern)
    /// 
    /// Ideally, the residual should look like random noise with no obvious patterns.
    /// If patterns remain in the residual, it suggests our decomposition missed something important.
    /// </para>
    /// </remarks>
    /// <param name="trend">The trend component.</param>
    /// <param name="seasonal">The seasonal component.</param>
    /// <returns>The residual component as a vector.</returns>
    protected Vector<T> CalculateResidual(Vector<T> trend, Vector<T> seasonal)
    {
        return Engine.Subtract(TimeSeries, Engine.Add(trend, seasonal));
    }

    /// <summary>
    /// Weighted average of <paramref name="data"/> over the centred window of the odd-length,
    /// symmetric <paramref name="kernel"/>: positions outside the series are dropped and the result is
    /// renormalized by the kernel weight that falls inside the series.
    /// </summary>
    protected Vector<T> CenteredWindowAverage(Vector<T> data, Tensor<T> kernel)
    {
        int n = data.Length;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        int width = kernel.Length;
        int half = width / 2;
        var k = kernel.Reshape(new[] { 1, 1, width });
        var x = Tensor<T>.FromVector(data, new[] { 1, 1, n });
        var sums = Engine.Conv1D(x, k, 1, half);
        var weights = Engine.Conv1D(Engine.TensorOnesLike(x), k, 1, half);
        return Engine.TensorMultiply(sums, Engine.TensorReciprocal(weights)).Reshape(new[] { n }).ToVector();
    }

    /// <summary>Unit weights over the window [-halfWidth, halfWidth] (a plain moving average).</summary>
    protected Tensor<T> BoxKernel(int halfWidth)
    {
        return Engine.TensorOnesLike(new Tensor<T>(new[] { 2 * halfWidth + 1 }));
    }

    /// <summary>
    /// Tri-cube weights (1 - |d| / halfWidth)^3 for the offsets d in [-halfWidth, halfWidth].
    /// </summary>
    protected Tensor<T> TriCubeKernel(int halfWidth)
    {
        var offsets = Engine.TensorArange<T>(-halfWidth, halfWidth + 1, 1);
        var scaled = Engine.TensorMultiplyScalar(Engine.TensorAbs(offsets), NumOps.FromDouble(1.0 / halfWidth));
        return TriCube(scaled);
    }

    /// <summary>
    /// Tri-cube weight function, element-wise: 0 where x &gt; 1, otherwise (1 - x)^3.
    /// </summary>
    protected Tensor<T> TriCube(Tensor<T> x)
    {
        var oneMinusX = Engine.ScalarMinusTensor(NumOps.One, x);
        var cube = Engine.TensorMultiply(Engine.TensorMultiply(oneMinusX, oneMinusX), oneMinusX);
        return Engine.TensorWhere(Engine.TensorGreaterThan(x, NumOps.One), Engine.TensorZerosLike(x), cube);
    }

    /// <summary>
    /// Lays the series out row-major in a [ceil(n / period), period] grid (zero padded), so that
    /// column p holds data[p], data[p + period], ...
    /// </summary>
    protected Tensor<T> ToPhaseGrid(Vector<T> data, int period)
    {
        int n = data.Length;
        int rows = (n + period - 1) / period;
        var padded = Engine.TensorConstantPad(Tensor<T>.FromVector(data), new[] { 0, rows * period - n }, NumOps.Zero);
        return padded.Reshape(new[] { rows, period });
    }

    /// <summary>Mean of data[p], data[p + period], ... for every phase p in [0, period).</summary>
    protected Vector<T> PhaseMeans(Vector<T> data, int period)
    {
        int n = data.Length;
        if (n == 0)
        {
            return new Vector<T>(period);
        }

        var grid = ToPhaseGrid(data, period);
        var counts = ToPhaseGrid(Engine.TensorOnesLike(Tensor<T>.FromVector(data)).ToVector(), period);
        var sums = Engine.ReduceSum(grid, new[] { 0 }, false);
        var totals = Engine.ReduceSum(counts, new[] { 0 }, false);
        return Engine.TensorMultiply(sums, Engine.TensorReciprocal(totals)).ToVector();
    }

    /// <summary>Repeats <paramref name="pattern"/> cyclically to length <paramref name="length"/>.</summary>
    protected Vector<T> RepeatPeriodic(Vector<T> pattern, int length)
    {
        int period = pattern.Length;
        if (length == 0 || period == 0)
        {
            return new Vector<T>(length);
        }

        int repeats = (length + period - 1) / period;
        var tiled = Engine.TensorTile(Tensor<T>.FromVector(pattern), new[] { repeats });
        return Engine.TensorSlice(tiled, new[] { 0 }, new[] { length }).ToVector();
    }

    /// <summary>True when any element of <paramref name="value"/> is below <paramref name="threshold"/>.</summary>
    protected bool AnyBelow(Tensor<T> value, T threshold)
    {
        return Engine.TensorCountNonzero(Engine.TensorLessThan(value, threshold)) > 0;
    }

    /// <summary>True when any element of <paramref name="value"/> is above <paramref name="threshold"/>.</summary>
    protected bool AnyAbove(Tensor<T> value, T threshold)
    {
        return Engine.TensorCountNonzero(Engine.TensorGreaterThan(value, threshold)) > 0;
    }
}
