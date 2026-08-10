using AiDotNet.ActivationFunctions;

namespace AiDotNet.NeuralNetworks.Layers;

public enum GLUGateType { Sigmoid, Swish, GELU, ReLU, Bilinear }

public partial class SwiGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public SwiGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new SwishActivation<T>())
    {
        _outputSize = outputSize;
    }
}

public partial class GeGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public GeGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new GELUActivation<T>())
    {
        _outputSize = outputSize;
    }
}

public partial class ReGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public ReGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new ReLUActivation<T>())
    {
        _outputSize = outputSize;
    }
}

public partial class BilinearGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public BilinearGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outputSize = outputSize;
    }
}
