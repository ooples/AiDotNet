
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Shared helper methods for distillation strategies.
/// </summary>
internal static class DistillationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Applies temperature-scaled softmax to logits.
    /// </summary>
    public static Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaledLogits = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(logits[i]) / temperature;
            scaledLogits[i] = NumOps.FromDouble(val);
        }

        T maxLogit = scaledLogits[0];
        for (int i = 1; i < n; i++)
        {
            if (NumOps.GreaterThan(scaledLogits[i], maxLogit))
                maxLogit = scaledLogits[i];
        }

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(NumOps.Subtract(scaledLogits[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.Divide(expValues[i], sum);
        }

        return result;
    }

    /// <summary>
    /// Computes KL divergence: KL(p || q) = sum(p * log(p / q)).
    /// </summary>
    public static T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = NumOps.Zero;
        const double epsilon = 1e-10;

        for (int i = 0; i < p.Length; i++)
        {
            double pVal = Math.Max(Convert.ToDouble(p[i]), epsilon);
            double qVal = Math.Max(Convert.ToDouble(q[i]), epsilon);
            double term = pVal * Math.Log(pVal / qVal);
            divergence = NumOps.Add(divergence, NumOps.FromDouble(term));
        }

        return divergence;
    }

    /// <summary>
    /// Computes cross-entropy: H(p, q) = -sum(p * log(q)).
    /// </summary>
    public static T CrossEntropy(Vector<T> p, Vector<T> q)
    {
        T loss = NumOps.Zero;
        const double epsilon = 1e-10;

        for (int i = 0; i < p.Length; i++)
        {
            double pVal = Convert.ToDouble(p[i]);
            double qVal = Math.Max(Convert.ToDouble(q[i]), epsilon);
            double term = -pVal * Math.Log(qVal);
            loss = NumOps.Add(loss, NumOps.FromDouble(term));
        }

        return loss;
    }
}
