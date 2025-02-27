namespace AiDotNet.ActivationFunctions;

public class GumbelSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _temperature;
    private readonly Random _random;

    public GumbelSoftmaxActivation(double temperature = 1.0, int? seed = null)
    {
        _temperature = NumOps.FromDouble(temperature);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> gumbel = SampleGumbel(input.Length);
        Vector<T> logits = input.Add(gumbel);

        return Softmax(logits);
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> output = Activate(input);
        int d = input.Length;
        Matrix<T> jacobian = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(output[i], NumOps.Subtract(NumOps.One, output[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Multiply(NumOps.Negate(output[i]), output[j]);
                }
            }
        }

        // Scale the Jacobian by the inverse temperature
        T invTemp = NumOps.Divide(NumOps.One, _temperature);
        return jacobian.Transform((x, row, col) => NumOps.Multiply(x, invTemp));
    }

    private Vector<T> SampleGumbel(int size)
    {
        Vector<T> uniform = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            uniform[i] = NumOps.FromDouble(_random.NextDouble());
        }

        return uniform.Transform(u => 
            NumOps.Multiply(
                NumOps.Negate(
                    NumOps.Log(
                        NumOps.Negate(
                            NumOps.Log(u)
                        )
                    )
                ),
                _temperature
            )
        );
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        Vector<T> expValues = logits.Transform(x => NumOps.Exp(NumOps.Divide(x, _temperature)));
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }
}