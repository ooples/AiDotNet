namespace AiDotNet.ActivationFunctions;

public class SiLUActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // SiLU: x * sigmoid(x)
        T sigmoid = MathHelper.Sigmoid(input);
        return NumOps.Multiply(input, sigmoid);
    }

    public override T Derivative(T input)
    {
        // Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        T sigmoid = MathHelper.Sigmoid(input);
        T sigmoidDerivative = NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
        T xSigmoidDerivative = NumOps.Multiply(input, sigmoidDerivative);

        return NumOps.Add(sigmoid, xSigmoidDerivative);
    }
}