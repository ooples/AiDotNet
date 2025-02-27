namespace AiDotNet.Factories;

public static class ActivationFunctionFactory<T>
{
    public static IActivationFunction<T> CreateActivationFunction(ActivationFunction activationFunction)
    {
        return activationFunction switch
        {
            ActivationFunction.ReLU => new ReLUActivation<T>(),
            //ActivationFunction.Sigmoid => new SigmoidActivation<T>(),
            //ActivationFunction.Tanh => new TanhActivation<T>(),
            //ActivationFunction.Linear => new LinearActivation<T>(),
            //ActivationFunction.LeakyReLU => new LeakyReLUActivation<T>(),
            //ActivationFunction.ELU => new ELUActivation<T>(),
            //ActivationFunction.SELU => new SELUActivation<T>(),
            //ActivationFunction.Softplus => new SoftplusActivation<T>(),
            //ActivationFunction.SoftSign => new SoftSignActivation<T>(),
            //ActivationFunction.Swish => new SwishActivation<T>(),
            //ActivationFunction.GELU => new GELUActivation<T>(),
            ActivationFunction.Softmax => throw new NotSupportedException("Softmax is not applicable to single values. Use CreateVectorActivationFunction for Softmax."),
            _ => throw new NotImplementedException($"Activation function {activationFunction} not implemented.")
        };
    }

    public static IVectorActivationFunction<T> CreateVectorActivationFunction(ActivationFunction activationFunction)
    {
        return activationFunction switch
        {
            ActivationFunction.Softmax => new SoftmaxActivation<T>(),
            _ => throw new NotImplementedException($"Vector activation function {activationFunction} not implemented.")
        };
    }
}