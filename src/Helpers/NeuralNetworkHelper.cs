namespace AiDotNet.Helpers;

public static class NeuralNetworkHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Activation functions
    public static T ReLU(T x) => MathHelper.Max(x, NumOps.Zero);
    public static T ReLUDerivative(T x) => NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Zero;

    public static T Sigmoid(T x) => NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(x))));
    public static T SigmoidDerivative(T x)
    {
        T sigmoid = Sigmoid(x);
        return NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
    }

    public static T TanH(T x)
    {
        T exp2x = NumOps.Exp(NumOps.Multiply(x, NumOps.FromDouble(2)));
        return NumOps.Divide(NumOps.Subtract(exp2x, NumOps.One), NumOps.Add(exp2x, NumOps.One));
    }
    public static T TanHDerivative(T x)
    {
        T tanh = TanH(x);
        return NumOps.Subtract(NumOps.One, NumOps.Multiply(tanh, tanh));
    }

    public static T Linear(T x) => x;
    public static T LinearDerivative(T x) => NumOps.One;

    // Loss functions
    public static T MeanSquaredError(Vector<T> predicted, Vector<T> actual)
    {
        return StatisticsHelper<T>.CalculateMeanSquaredError(predicted, actual);
    }
    public static Vector<T> MeanSquaredErrorDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Transform(x => NumOps.Multiply(NumOps.FromDouble(2), x)).Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T MeanAbsoluteError(Vector<T> predicted, Vector<T> actual)
    {
        return StatisticsHelper<T>.CalculateMeanAbsoluteError(predicted, actual);
    }
    public static Vector<T> MeanAbsoluteErrorDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Transform(x => NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One)).Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T BinaryCrossEntropy(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], NumOps.FromDouble(1e-15), NumOps.FromDouble(1 - 1e-15));
            sum = NumOps.Add(sum, NumOps.Add(
                NumOps.Multiply(actual[i], NumOps.Log(p)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, actual[i]), NumOps.Log(NumOps.Subtract(NumOps.One, p)))
            ));
        }

        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Length)));
    }
    public static Vector<T> BinaryCrossEntropyDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        Vector<T> derivative = new Vector<T>(predicted.Length, NumOps);
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], NumOps.FromDouble(1e-15), NumOps.FromDouble(1 - 1e-15));
            derivative[i] = NumOps.Divide(
                NumOps.Subtract(p, actual[i]),
                NumOps.Multiply(p, NumOps.Subtract(NumOps.One, p))
            );
        }
        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T CategoricalCrossEntropy(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], NumOps.FromDouble(1e-15), NumOps.FromDouble(1 - 1e-15));
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumOps.Log(p)));
        }
        return NumOps.Negate(sum);
    }

    public static Vector<T> CategoricalCrossEntropyDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T HuberLoss(Vector<T> predicted, Vector<T> actual, T? delta = default)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        delta = delta ?? NumOps.FromDouble(1.0);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Abs(NumOps.Subtract(predicted[i], actual[i]));
            if (NumOps.LessThanOrEquals(diff, delta))
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(diff, diff)));
            }
            else
            {
                sum = NumOps.Add(sum, NumOps.Subtract(
                    NumOps.Multiply(delta, diff),
                    NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(delta, delta))
                ));
            }
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> HuberLossDerivative(Vector<T> predicted, Vector<T> actual, T? delta = default)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        delta = delta ?? NumOps.FromDouble(1.0);

        Vector<T> derivative = new(predicted.Length, NumOps);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            if (NumOps.LessThanOrEquals(NumOps.Abs(diff), delta))
            {
                derivative[i] = diff;
            }
            else
            {
                derivative[i] = NumOps.Multiply(delta, NumOps.GreaterThan(diff, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One));
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }
}