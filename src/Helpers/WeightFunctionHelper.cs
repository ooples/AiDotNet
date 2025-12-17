namespace AiDotNet.Helpers;

/// <summary>
/// Provides methods for calculating weights used in robust regression techniques.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> In standard regression, all data points are treated equally. However, in real-world data,
/// some points may be outliers (unusual values that don't follow the general pattern). Robust regression
/// techniques handle these outliers by assigning different "weights" to different data points.
/// 
/// Think of weights like importance scores:
/// - Normal data points get high weights (close to 1), meaning they have full influence on the model
/// - Outliers get low weights (close to 0), reducing their influence on the model
/// 
/// This helper class calculates these weights using different mathematical formulas (Huber, Bisquare, Andrews)
/// that determine how aggressively to downweight outliers.
/// </remarks>
public static class WeightFunctionHelper<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Calculates weights for data points based on their residuals using the specified weight function.
    /// </summary>
    /// <param name="residuals">The vector of residuals (differences between predicted and actual values).</param>
    /// <param name="weightFunction">The weight function to use (Huber, Bisquare, or Andrews).</param>
    /// <param name="tuningConstant">A parameter that controls how aggressively to downweight outliers.</param>
    /// <returns>A vector of weights for each data point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes the errors in your model's predictions (residuals) and calculates
    /// how much each data point should influence your model.
    /// 
    /// The residual for a data point is the difference between what your model predicted and the actual value.
    /// Large residuals often indicate outliers.
    /// 
    /// The weight function determines the mathematical formula used to convert residuals to weights:
    /// - Huber: Moderately reduces the influence of outliers
    /// - Bisquare: More aggressively reduces the influence of outliers
    /// - Andrews: Similar to Bisquare but uses a different mathematical approach
    /// 
    /// The tuning constant controls the threshold for what's considered an outlier - smaller values
    /// will treat more points as outliers.
    /// </remarks>
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

    /// <summary>
    /// Calculates weights using Huber's method.
    /// </summary>
    /// <param name="residuals">The vector of residuals.</param>
    /// <param name="tuningConstant">The tuning constant that controls the threshold for outliers.</param>
    /// <returns>A vector of weights for each data point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Huber weights work like this:
    /// - If a residual is small (less than the tuning constant), the data point gets a weight of 1 (full influence)
    /// - If a residual is large, the weight decreases proportionally to 1/|residual|
    /// 
    /// This is a moderate approach that reduces the influence of outliers without removing them completely.
    /// </remarks>
    private static Vector<T> CalculateHuberWeights(Vector<T> residuals, double tuningConstant)
    {
        T k = _numOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = _numOps.Abs(residuals[i]);
            weights[i] = _numOps.LessThanOrEquals(absRes, k) ? _numOps.One : _numOps.Divide(k, absRes);
        }

        return weights;
    }

    /// <summary>
    /// Calculates weights using Tukey's bisquare (biweight) method.
    /// </summary>
    /// <param name="residuals">The vector of residuals.</param>
    /// <param name="tuningConstant">The tuning constant that controls the threshold for outliers.</param>
    /// <returns>A vector of weights for each data point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Bisquare weights are more aggressive than Huber weights:
    /// - If a residual is small (less than the tuning constant), the weight is calculated using a formula
    ///   that gradually decreases as the residual gets larger
    /// - If a residual is large (greater than the tuning constant), the data point gets a weight of 0
    ///   (no influence at all)
    /// 
    /// This approach completely removes the influence of extreme outliers, making it more robust but
    /// potentially discarding useful information if the tuning constant is too small.
    /// </remarks>
    private static Vector<T> CalculateBisquareWeights(Vector<T> residuals, double tuningConstant)
    {
        T k = _numOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = _numOps.Abs(residuals[i]);
            if (_numOps.LessThanOrEquals(absRes, k))
            {
                T u = _numOps.Divide(residuals[i], k);
                T w = _numOps.Subtract(_numOps.One, _numOps.Multiply(u, u));
                weights[i] = _numOps.Multiply(w, w);
            }
            else
            {
                weights[i] = _numOps.Zero;
            }
        }

        return weights;
    }

    /// <summary>
    /// Calculates weights using Andrews' sine wave method.
    /// </summary>
    /// <param name="residuals">The vector of residuals.</param>
    /// <param name="tuningConstant">The tuning constant that controls the threshold for outliers.</param>
    /// <returns>A vector of weights for each data point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Andrews weights use a sine function to determine weights:
    /// - If a residual is small (less than p times the tuning constant), the weight is calculated using
    ///   a sine function that gradually decreases as the residual gets larger
    /// - If a residual is large, the data point gets a weight of 0 (no influence at all)
    /// 
    /// This approach is similar to bisquare but uses a different mathematical function (sine wave)
    /// to determine the transition from full influence to no influence.
    /// </remarks>
    private static Vector<T> CalculateAndrewsWeights(Vector<T> residuals, double tuningConstant)
    {
        T k = _numOps.FromDouble(tuningConstant);
        Vector<T> weights = new(residuals.Length);

        for (int i = 0; i < residuals.Length; i++)
        {
            T absRes = _numOps.Abs(residuals[i]);
            if (_numOps.LessThanOrEquals(absRes, _numOps.Multiply(k, MathHelper.Pi<T>())))
            {
                T u = _numOps.Divide(residuals[i], k);
                weights[i] = _numOps.Divide(MathHelper.Sin(u), u);
            }
            else
            {
                weights[i] = _numOps.Zero;
            }
        }

        return weights;
    }
}
