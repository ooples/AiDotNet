namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public abstract class TimeSeriesDecompositionBase<T> : ITimeSeriesDecomposition<T>
{
    protected readonly INumericOperations<T> NumOps;

    public Vector<T> TimeSeries { get; }
    protected Dictionary<DecompositionComponentType, Vector<T>> Components { get; }

    protected TimeSeriesDecompositionBase(Vector<T> timeSeries)
    {
        TimeSeries = timeSeries;
        NumOps = MathHelper.GetNumericOperations<T>();
        Components = new Dictionary<DecompositionComponentType, Vector<T>>();
    }

    public Dictionary<DecompositionComponentType, Vector<T>> GetComponents() => Components;

    protected void AddComponent(DecompositionComponentType componentType, Vector<T> component)
    {
        Components[componentType] = component;
    }

    public Vector<T> GetComponent(DecompositionComponentType componentType)
    {
        return Components.TryGetValue(componentType, out var component) ? component : Vector<T>.Empty();
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