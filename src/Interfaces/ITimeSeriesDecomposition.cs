namespace AiDotNet.Interfaces;

public interface ITimeSeriesDecomposition<T>
{
    Vector<T> TimeSeries { get; }
    Dictionary<DecompositionComponentType, object> GetComponents();
    object? GetComponent(DecompositionComponentType componentType);
    bool HasComponent(DecompositionComponentType componentType);
}