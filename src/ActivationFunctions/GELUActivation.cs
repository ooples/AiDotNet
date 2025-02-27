namespace AiDotNet.ActivationFunctions;

public class GELUActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        T sqrt2OverPi = NumOps.Sqrt(NumOps.FromDouble(2.0 / Math.PI));
        T x3 = NumOps.Multiply(NumOps.Multiply(input, input), input);
        T inner = NumOps.Add(input, NumOps.Multiply(NumOps.FromDouble(0.044715), x3));
        T tanhTerm = NumOps.Add(NumOps.One, MathHelper.Tanh(NumOps.Multiply(sqrt2OverPi, inner)));

        return NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(input, tanhTerm));
    }

    public override T Derivative(T input)
    {
        // d/dx GELU(x) = 0.5 * tanh(0.0356774 * x^3 + 0.797885 * x) + 
        //                (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.0356774 * x^3 + 0.797885 * x) + 0.5
        T x2 = NumOps.Multiply(input, input);
        T x3 = NumOps.Multiply(x2, input);
        
        T term1 = NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(0.0356774), x3),
            NumOps.Multiply(NumOps.FromDouble(0.797885), input)
        );
        
        T term2 = NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(0.0535161), x3),
            NumOps.Multiply(NumOps.FromDouble(0.398942), input)
        );

        T tanhTerm = MathHelper.Tanh(term1);
        T sech2Term = NumOps.Subtract(NumOps.One, NumOps.Multiply(tanhTerm, tanhTerm));

        return NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(0.5), tanhTerm),
                NumOps.Multiply(term2, sech2Term)
            ),
            NumOps.FromDouble(0.5)
        );
    }
}