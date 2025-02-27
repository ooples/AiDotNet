namespace AiDotNet.Helpers;

public static class WeightFunctionHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static Vector<T> CalculateWeights(Vector<T> residuals, WeightFunction weightFunction, double tuningConstant)
    {
        return weightFunction switch
        {
            WeightFunction.Huber => CalculateHuberWeights(residuals, tuningConstant),
            WeightFunction.Bisquare => CalculateBisquareWeights(residuals, tuningConstant),
            WeightFunction.Andrews => CalculateAndrewsWeights(residuals, tuningConstant),
            _ => throw new ArgumentException("Unsupported weight function"),
        };
    }

    private static Vector<T> CalculateHuberWeights(Vector<T> residuals, double tuningConstant)
    {
        T k = NumOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = NumOps.Abs(residuals[i]);
            weights[i] = NumOps.LessThanOrEquals(absRes, k) ? NumOps.One : NumOps.Divide(k, absRes);
        }

        return weights;
    }

    private static Vector<T> CalculateBisquareWeights(Vector<T> residuals, double tuningConstant)
    {
        T k = NumOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = NumOps.Abs(residuals[i]);
            if (NumOps.LessThanOrEquals(absRes, k))
            {
                T u = NumOps.Divide(residuals[i], k);
                T w = NumOps.Subtract(NumOps.One, NumOps.Multiply(u, u));
                weights[i] = NumOps.Multiply(w, w);
            }
            else
            {
                weights[i] = NumOps.Zero;
            }
        }

        return weights;
    }

    private static Vector<T> CalculateAndrewsWeights(Vector<T> residuals, double tuningConstant)
    {
        T k = NumOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = NumOps.Abs(residuals[i]);
            if (NumOps.LessThanOrEquals(absRes, NumOps.Multiply(k, MathHelper.Pi<T>())))
            {
                T u = NumOps.Divide(residuals[i], k);
                weights[i] = NumOps.Divide(MathHelper.Sin(u), u);
            }
            else
            {
                weights[i] = NumOps.Zero;
            }
        }

        return weights;
    }
}