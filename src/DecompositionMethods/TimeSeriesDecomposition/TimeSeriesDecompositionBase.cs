namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public abstract class TimeSeriesDecompositionBase<T> : ITimeSeriesDecomposition<T>
{
    protected readonly INumericOperations<T> NumOps;

    public Vector<T> TimeSeries { get; }
    protected Dictionary<DecompositionComponentType, object> Components { get; }

    protected TimeSeriesDecompositionBase(Vector<T> timeSeries)
    {
        TimeSeries = timeSeries;
        NumOps = MathHelper.GetNumericOperations<T>();
        Components = new Dictionary<DecompositionComponentType, object>();
    }

    protected abstract void Decompose();

    public Dictionary<DecompositionComponentType, object> GetComponents() => Components;

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

    public bool HasComponent(DecompositionComponentType componentType)
    {
        return Components.ContainsKey(componentType);
    }

    protected Vector<T> CalculateResidual(Vector<T> trend, Vector<T> seasonal)
    {
        Vector<T> residual = new Vector<T>(TimeSeries.Length, NumOps);

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