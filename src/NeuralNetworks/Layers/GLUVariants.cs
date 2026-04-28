using AiDotNet.ActivationFunctions;

namespace AiDotNet.NeuralNetworks.Layers;

public enum GLUGateType { Sigmoid, Swish, GELU, ReLU, Bilinear }

public class SwiGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    public SwiGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new SwishActivation<T>()) { }
}

public class GeGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    public GeGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new GELUActivation<T>()) { }
}

public class ReGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    public ReGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new ReLUActivation<T>()) { }
}

public class BilinearGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>
{
    public BilinearGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()) { }
}
