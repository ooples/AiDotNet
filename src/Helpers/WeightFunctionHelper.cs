namespace AiDotNet.Helpers;

public static class WeightFunctionHelper
{
    public static Vector<T> CalculateWeights<T>(Vector<T> residuals, WeightFunction weightFunction, double tuningConstant, INumericOperations<T> numOps)
    {
        return weightFunction switch
        {
            WeightFunction.Huber => CalculateHuberWeights(residuals, tuningConstant, numOps),
            WeightFunction.Bisquare => CalculateBisquareWeights(residuals, tuningConstant, numOps),
            WeightFunction.Andrews => CalculateAndrewsWeights(residuals, tuningConstant, numOps),
            _ => throw new ArgumentException("Unsupported weight function"),
        };
    }

    private static Vector<T> CalculateHuberWeights<T>(Vector<T> residuals, double tuningConstant, INumericOperations<T> numOps)
    {
        T k = numOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length, numOps);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = numOps.Abs(residuals[i]);
            weights[i] = numOps.LessThanOrEquals(absRes, k) ? numOps.One : numOps.Divide(k, absRes);
        }

        return weights;
    }

    private static Vector<T> CalculateBisquareWeights<T>(Vector<T> residuals, double tuningConstant, INumericOperations<T> numOps)
    {
        T k = numOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length, numOps);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = numOps.Abs(residuals[i]);
            if (numOps.LessThanOrEquals(absRes, k))
            {
                T u = numOps.Divide(residuals[i], k);
                T w = numOps.Subtract(numOps.One, numOps.Multiply(u, u));
                weights[i] = numOps.Multiply(w, w);
            }
            else
            {
                weights[i] = numOps.Zero;
            }
        }

        return weights;
    }

    private static Vector<T> CalculateAndrewsWeights<T>(Vector<T> residuals, double tuningConstant, INumericOperations<T> numOps)
    {
        T k = numOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length, numOps);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = numOps.Abs(residuals[i]);
            if (numOps.LessThanOrEquals(absRes, numOps.Multiply(k, MathHelper.Pi<T>())))
            {
                T u = numOps.Divide(residuals[i], k);
                weights[i] = numOps.Divide(MathHelper.Sin(u), u);
            }
            else
            {
                weights[i] = numOps.Zero;
            }
        }

        return weights;
    }
}