namespace AiDotNet.LinearAlgebra;

public class Sample<T>
{
    public Vector<T> Features { get; set; }
    public T Target { get; set; }

    public Sample(Vector<T> features, T target)
    {
        Features = features;
        Target = target;
    }
}