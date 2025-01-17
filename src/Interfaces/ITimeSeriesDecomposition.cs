namespace AiDotNet.Interfaces;

public interface ITimeSeriesDecomposition<T>
{
    Vector<T> TimeSeries { get; }
    Dictionary<DecompositionComponentType, Vector<T>> GetComponents();
    Vector<T> GetComponent(DecompositionComponentType componentType);
    bool HasComponent(DecompositionComponentType componentType);
}