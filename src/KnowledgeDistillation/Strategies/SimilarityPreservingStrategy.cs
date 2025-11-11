using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Similarity-preserving distillation that preserves pairwise similarity structure.
/// </summary>
public class SimilarityPreservingStrategy<T> : DistillationStrategyBase<Vector<T>, T>
{
    private readonly double _similarityWeight;

    public SimilarityPreservingStrategy(
        double similarityWeight = 0.5,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        _similarityWeight = similarityWeight;
    }

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
            return NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
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
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
        }

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);

            for (int i = 0; i < n; i++)
            {
                var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);
                gradient[i] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[i]));
            }
        }

        return gradient;
    }

    public T ComputeSimilarityLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        int n = studentEmbeddings.Length;
        T totalLoss = NumOps.Zero;
        int pairCount = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double teacherSim = CosineSimilarity(teacherEmbeddings[i], teacherEmbeddings[j]);
                double studentSim = CosineSimilarity(studentEmbeddings[i], studentEmbeddings[j]);
                double diff = teacherSim - studentSim;
                totalLoss = NumOps.Add(totalLoss, NumOps.FromDouble(diff * diff));
                pairCount++;
            }
        }

        var loss = pairCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(pairCount)) : NumOps.Zero;
        return NumOps.Multiply(loss, NumOps.FromDouble(_similarityWeight));
    }

    private double CosineSimilarity(Vector<T> v1, Vector<T> v2)
    {
        T dot = NumOps.Zero, norm1 = NumOps.Zero, norm2 = NumOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(v1[i], v2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(v1[i], v1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(v2[i], v2[i]));
        }

        return NumOps.ToDouble(dot) / (Math.Sqrt(NumOps.ToDouble(norm1)) * Math.Sqrt(NumOps.ToDouble(norm2)) + Epsilon);
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

            if (pVal > Epsilon)
            {
                double contrib = pVal * Math.Log(pVal / (qVal + Epsilon));
                divergence = NumOps.Add(divergence, NumOps.FromDouble(contrib));
            }
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

            if (label > Epsilon)
            {
                double contrib = -label * Math.Log(pred + Epsilon);
                entropy = NumOps.Add(entropy, NumOps.FromDouble(contrib));
            }
        }

        return entropy;
    }
}
