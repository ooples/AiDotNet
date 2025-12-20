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
        Vector<T> residual = new Vector<T>(TimeSeries.Length);

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            residual[i] = NumOps.Subtract(
                TimeSeries[i],
                NumOps.Add(trend[i], seasonal[i])
            );
        }

        return residual;
    }
}
