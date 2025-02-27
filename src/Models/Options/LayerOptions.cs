namespace AiDotNet.Models.Options;

public class LayerOptions<T>
{
    public int PoolSize { get; set; }
    public int Stride { get; set; }
    public PoolingType PoolingType { get; set; }
    public int InputChannels { get; set; }
    public int OutputChannels { get; set; }
    public int KernelSize { get; set; }
    public int Padding { get; set; }
    public int InputSize { get; set; }
    public int OutputSize { get; set; }
    public IActivationFunction<T>? ScalarActivationFunction { get; set; }
    public IVectorActivationFunction<T>? VectorActivationFunction { get; set; }
}