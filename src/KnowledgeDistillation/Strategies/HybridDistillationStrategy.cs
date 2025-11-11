using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Hybrid distillation strategy.
/// </summary>
public class HybridDistillationStrategy<T> : DistillationStrategyBase<Vector<T>, T>
{
    public HybridDistillationStrategy(double temperature = 3.0, double alpha = 0.3)
        : base(temperature, alpha) { }

    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(studentSoft, teacherSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);
            return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
        }
        return softLoss;
    }

    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);
        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);

        for (int i = 0; i < n; i++)
            gradient[i] = NumOps.Multiply(NumOps.Subtract(studentSoft[i], teacherSoft[i]), NumOps.FromDouble(Temperature * Temperature));

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            for (int i = 0; i < n; i++)
            {
                var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);
                gradient[i] = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[i]));
            }
        }
        return gradient;
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];
        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(NumOps.ToDouble(logits[i]) / temperature);
        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];
        T sum = NumOps.Zero;
        var expValues = new T[n];
        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }
        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);
        return result;
    }

    private T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = NumOps.Zero;
        for (int i = 0; i < p.Length; i++)
        {
            double pVal = NumOps.ToDouble(p[i]);
            double qVal = NumOps.ToDouble(q[i]);
            if (pVal > 1e-10)
                divergence = NumOps.Add(divergence, NumOps.FromDouble(pVal * Math.Log(pVal / (qVal + 1e-10))));
        }
        return divergence;
    }

    private T CrossEntropy(Vector<T> predictions, Vector<T> trueLabels)
    {
        T entropy = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = NumOps.ToDouble(predictions[i]);
            double label = NumOps.ToDouble(trueLabels[i]);
            if (label > 1e-10)
                entropy = NumOps.Add(entropy, NumOps.FromDouble(-label * Math.Log(pred + 1e-10)));
        }
        return entropy;
    }
}
