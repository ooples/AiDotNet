namespace AiDotNet.ActivationFunctions;

public class MishActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        T softplus = NumOps.Log(NumOps.Add(NumOps.One, NumOps.Exp(input)));
        T tanh = MathHelper.Tanh(softplus);

        return NumOps.Multiply(input, tanh);
    }

    public override T Derivative(T input)
    {
        T exp_x = NumOps.Exp(input);
        T exp_2x = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(2), input));
        T exp_3x = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(3), input));
        
        T omega = NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(4), NumOps.Add(input, NumOps.One)),
                NumOps.Multiply(NumOps.FromDouble(4), exp_2x)
            ),
            NumOps.Add(
                exp_3x,
                NumOps.Multiply(exp_x, NumOps.Add(NumOps.Multiply(NumOps.FromDouble(4), input), NumOps.FromDouble(6)))
            )
        );
        
        T delta = NumOps.Add(
            NumOps.Add(NumOps.Multiply(NumOps.FromDouble(2), exp_2x), NumOps.FromDouble(2)),
            NumOps.Multiply(exp_2x, NumOps.Square(NumOps.Add(input, NumOps.FromDouble(2))))
        );
        
        return NumOps.Divide(NumOps.Multiply(exp_x, omega), NumOps.Square(delta));
    }
}