using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Implements Contrastive Representation Distillation (CRD) which transfers knowledge through
/// contrastive learning of sample relationships rather than just matching outputs.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Contrastive distillation teaches the student to understand which
/// samples are similar and which are different, not just to copy the teacher's predictions.
/// It's like learning to group things by their similarities rather than just memorizing labels.</para>
///
/// <para><b>Real-world Analogy:</b>
/// Instead of just teaching a student "This is a dog," you teach them "Dogs are more similar
/// to wolves than to cats" and "Retrievers are more similar to Labs than to Chihuahuas."
/// This relational understanding helps the student generalize better to new examples.</para>
///
/// <para><b>How CRD Works:</b>
/// 1. Extract embeddings/features from teacher and student
/// 2. For each sample (anchor), identify:
///    - Positive samples: Same class or similar features
///    - Negative samples: Different class or dissimilar features
/// 3. Pull student embeddings of anchor and positives together
/// 4. Push student embeddings of anchor and negatives apart
/// 5. Ensure student's embedding space has same structure as teacher's</para>
///
/// <para><b>Key Differences from Standard Distillation:</b>
/// - **Standard**: Match output probabilities [0.1, 0.7, 0.2]
/// - **CRD**: Match embedding similarities and distances
/// - **Benefit**: Better generalization, especially for few-shot learning</para>
///
/// <para><b>Mathematical Foundation:</b>
/// CRD uses InfoNCE loss (Noise Contrastive Estimation):
/// L = -log(exp(sim(t_i, s_i)/τ) / Σ_j exp(sim(t_i, s_j)/τ))
/// where:
/// - t_i, s_i are teacher/student embeddings of sample i
/// - τ is temperature
/// - j ranges over all samples in batch</para>
///
/// <para><b>Benefits:</b>
/// - **Better Features**: Student learns richer representations
/// - **Few-Shot Learning**: Transfers better to new classes
/// - **Robustness**: Less sensitive to label noise
/// - **Interpretability**: Embedding space is more structured
/// - **Complementary**: Can combine with standard distillation</para>
///
/// <para><b>Use Cases:</b>
/// - Few-shot/zero-shot learning
/// - Transfer learning across domains
/// - Learning with noisy labels
/// - Metric learning tasks (face recognition, image retrieval)
/// - Self-supervised pre-training</para>
///
/// <para><b>Performance Improvements:</b>
/// - CRD often gives 2-4% better accuracy than standard distillation
/// - Particularly strong for small student models
/// - Excellent for tasks requiring good embeddings</para>
///
/// <para><b>References:</b>
/// - Tian et al. (2020). Contrastive Representation Distillation. ICLR.
/// - Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.</para>
/// </remarks>
public class ContrastiveDistillationStrategy<T> : DistillationStrategyBase<Vector<T>, T>
{
    private readonly double _contrastiveWeight;
    private readonly int _negativesSampleSize;
    private readonly ContrastiveMode _mode;

    /// <summary>
    /// Initializes a new instance of the ContrastiveDistillationStrategy class.
    /// </summary>
    /// <param name="contrastiveWeight">Weight for contrastive loss vs standard output loss (default: 0.8).</param>
    /// <param name="temperature">Temperature for contrastive softmax (default: 0.07).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.2).</param>
    /// <param name="negativesSampleSize">Number of negative samples to use (default: 1024).</param>
    /// <param name="mode">Contrastive mode (default: InfoNCE).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configure how much to weight contrastive learning:</para>
    /// <para>- contrastiveWeight 0.6-0.9: More focus on learning representations
    /// - temperature 0.05-0.1: Lower = sharper distinctions between similar/dissimilar
    /// - negativesSampleSize 512-2048: More negatives = better discrimination</para>
    ///
    /// <para>Example:
    /// <code>
    /// var strategy = new ContrastiveDistillationStrategy&lt;double&gt;(
    ///     contrastiveWeight: 0.8,  // 80% contrastive, 20% standard
    ///     temperature: 0.07,        // Standard for contrastive learning
    ///     alpha: 0.2,              // Mostly teacher knowledge
    ///     negativesSampleSize: 1024 // Large negative set for better discrimination
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public ContrastiveDistillationStrategy(
        double contrastiveWeight = 0.8,
        double temperature = 0.07,
        double alpha = 0.2,
        int negativesSampleSize = 1024,
        ContrastiveMode mode = ContrastiveMode.InfoNCE)
        : base(temperature, alpha)
    {
        if (contrastiveWeight < 0 || contrastiveWeight > 1)
            throw new ArgumentException("Contrastive weight must be between 0 and 1", nameof(contrastiveWeight));
        if (negativesSampleSize < 1)
            throw new ArgumentException("Negatives sample size must be at least 1", nameof(negativesSampleSize));

        _contrastiveWeight = contrastiveWeight;
        _negativesSampleSize = negativesSampleSize;
        _mode = mode;
    }

    /// <summary>
    /// Computes standard output loss (contrastive loss computed separately on embeddings).
    /// </summary>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // Standard distillation loss (scaled by 1 - contrastiveWeight)
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(studentSoft, teacherSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);

            var alphaT = NumOps.FromDouble(Alpha);
            var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

            var combinedLoss = NumOps.Add(
                NumOps.Multiply(alphaT, hardLoss),
                NumOps.Multiply(oneMinusAlpha, softLoss));

            // Scale by (1 - contrastiveWeight) to make room for contrastive loss
            return NumOps.Multiply(combinedLoss, NumOps.FromDouble(1.0 - _contrastiveWeight));
        }

        return NumOps.Multiply(softLoss, NumOps.FromDouble(1.0 - _contrastiveWeight));
    }

    /// <summary>
    /// Computes gradient of standard loss.
    /// </summary>
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
            var hardGradient = new Vector<T>(n);

            for (int i = 0; i < n; i++)
            {
                hardGradient[i] = NumOps.Subtract(studentProbs[i], trueLabels[i]);
            }

            var alphaT = NumOps.FromDouble(Alpha);
            var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

            for (int i = 0; i < n; i++)
            {
                gradient[i] = NumOps.Add(
                    NumOps.Multiply(alphaT, hardGradient[i]),
                    NumOps.Multiply(oneMinusAlpha, gradient[i]));
            }
        }

        // Scale by (1 - contrastiveWeight)
        var scale = NumOps.FromDouble(1.0 - _contrastiveWeight);
        for (int i = 0; i < n; i++)
        {
            gradient[i] = NumOps.Multiply(gradient[i], scale);
        }

        return gradient;
    }

    /// <summary>
    /// Computes contrastive loss on embeddings/features.
    /// </summary>
    /// <param name="studentEmbeddings">Student embeddings for batch.</param>
    /// <param name="teacherEmbeddings">Teacher embeddings for batch.</param>
    /// <param name="labels">Sample labels for determining positives/negatives.</param>
    /// <returns>Contrastive loss value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well the student's embedding space
    /// matches the teacher's structural relationships. Lower loss means better match.</para>
    ///
    /// <para>The embeddings should be from intermediate layers (not final outputs),
    /// as those contain richer representation information.</para>
    /// </remarks>
    public T ComputeContrastiveLoss(
        Vector<T>[] studentEmbeddings,
        Vector<T>[] teacherEmbeddings,
        int[] labels)
    {
        if (studentEmbeddings.Length != teacherEmbeddings.Length || studentEmbeddings.Length != labels.Length)
            throw new ArgumentException("All arrays must have same length");

        T totalLoss = NumOps.Zero;
        int batchSize = studentEmbeddings.Length;

        switch (_mode)
        {
            case ContrastiveMode.InfoNCE:
                totalLoss = ComputeInfoNCELoss(studentEmbeddings, teacherEmbeddings, labels);
                break;

            case ContrastiveMode.TripletLoss:
                totalLoss = ComputeTripletLoss(studentEmbeddings, teacherEmbeddings, labels);
                break;

            case ContrastiveMode.NTXent:
                totalLoss = ComputeNTXentLoss(studentEmbeddings, teacherEmbeddings);
                break;
        }

        // Apply contrastive weight
        return NumOps.Multiply(totalLoss, NumOps.FromDouble(_contrastiveWeight));
    }

    /// <summary>
    /// Computes InfoNCE (Noise Contrastive Estimation) loss.
    /// </summary>
    private T ComputeInfoNCELoss(Vector<T>[] studentEmbs, Vector<T>[] teacherEmbs, int[] labels)
    {
        int batchSize = studentEmbs.Length;
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Positive: teacher-student pair for same sample
            double positiveSim = CosineSimilarity(teacherEmbs[i], studentEmbs[i]);
            double positiveScore = Math.Exp(positiveSim / Temperature);

            // Negatives: all other samples
            double negativeSum = 0;
            int negativeCount = 0;

            for (int j = 0; j < Math.Min(batchSize, _negativesSampleSize); j++)
            {
                if (i != j)
                {
                    double negativeSim = CosineSimilarity(teacherEmbs[i], studentEmbs[j]);
                    negativeSum += Math.Exp(negativeSim / Temperature);
                    negativeCount++;
                }
            }

            // InfoNCE loss
            double loss = -Math.Log(positiveScore / (positiveScore + negativeSum + Epsilon));
            totalLoss = NumOps.Add(totalLoss, NumOps.FromDouble(loss));
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes triplet loss.
    /// </summary>
    private T ComputeTripletLoss(Vector<T>[] studentEmbs, Vector<T>[] teacherEmbs, int[] labels)
    {
        int batchSize = studentEmbs.Length;
        T totalLoss = NumOps.Zero;
        int tripletCount = 0;

        double margin = 0.1;

        for (int i = 0; i < batchSize; i++)
        {
            // Find positive: same class
            int positiveIdx = -1;
            for (int j = 0; j < batchSize; j++)
            {
                if (j != i && labels[j] == labels[i])
                {
                    positiveIdx = j;
                    break;
                }
            }

            if (positiveIdx == -1) continue;

            // Find negative: different class
            int negativeIdx = -1;
            for (int j = 0; j < batchSize; j++)
            {
                if (labels[j] != labels[i])
                {
                    negativeIdx = j;
                    break;
                }
            }

            if (negativeIdx == -1) continue;

            // Triplet loss: max(0, d(anchor,positive) - d(anchor,negative) + margin)
            double posDist = EuclideanDistance(studentEmbs[i], studentEmbs[positiveIdx]);
            double negDist = EuclideanDistance(studentEmbs[i], studentEmbs[negativeIdx]);
            double loss = Math.Max(0, posDist - negDist + margin);

            totalLoss = NumOps.Add(totalLoss, NumOps.FromDouble(loss));
            tripletCount++;
        }

        return tripletCount > 0 ?
            NumOps.Divide(totalLoss, NumOps.FromDouble(tripletCount)) :
            NumOps.Zero;
    }

    /// <summary>
    /// Computes NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    /// </summary>
    private T ComputeNTXentLoss(Vector<T>[] studentEmbs, Vector<T>[] teacherEmbs)
    {
        int batchSize = studentEmbs.Length;
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Positive: corresponding teacher embedding
            double positiveSim = CosineSimilarity(studentEmbs[i], teacherEmbs[i]) / Temperature;

            // Denominator: all pairs
            double denominator = 0;
            for (int j = 0; j < batchSize; j++)
            {
                if (i != j)
                {
                    double sim = CosineSimilarity(studentEmbs[i], teacherEmbs[j]) / Temperature;
                    denominator += Math.Exp(sim);
                }
            }

            double loss = -positiveSim + Math.Log(denominator + Epsilon);
            totalLoss = NumOps.Add(totalLoss, NumOps.FromDouble(loss));
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    private double CosineSimilarity(Vector<T> v1, Vector<T> v2)
    {
        T dot = NumOps.Zero;
        T norm1 = NumOps.Zero;
        T norm2 = NumOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(v1[i], v2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(v1[i], v1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(v2[i], v2[i]));
        }

        double dotVal = Convert.ToDouble(dot);
        double norm1Val = Math.Sqrt(Convert.ToDouble(norm1));
        double norm2Val = Math.Sqrt(Convert.ToDouble(norm2));

        return dotVal / (norm1Val * norm2Val + Epsilon);
    }

    private double EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        T sumSq = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = NumOps.Subtract(v1[i], v2[i]);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
        }
        return Math.Sqrt(Convert.ToDouble(sumSq));
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);

        var scaledLogits = new T[n];
        for (int i = 0; i < n; i++)
        {
            scaledLogits[i] = NumOps.FromDouble(Convert.ToDouble(logits[i]) / temperature);
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

    private T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = NumOps.Zero;

        for (int i = 0; i < p.Length; i++)
        {
            double pVal = Convert.ToDouble(p[i]);
            double qVal = Convert.ToDouble(q[i]);

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
            double pred = Convert.ToDouble(predictions[i]);
            double label = Convert.ToDouble(trueLabels[i]);

            if (label > Epsilon)
            {
                double contrib = -label * Math.Log(pred + Epsilon);
                entropy = NumOps.Add(entropy, NumOps.FromDouble(contrib));
            }
        }

        return entropy;
    }
}

/// <summary>
/// Defines the contrastive learning mode.
/// </summary>
public enum ContrastiveMode
{
    /// <summary>
    /// InfoNCE (Noise Contrastive Estimation) - most common, used in SimCLR, MoCo.
    /// </summary>
    InfoNCE,

    /// <summary>
    /// Triplet loss - anchor-positive-negative triplets.
    /// </summary>
    TripletLoss,

    /// <summary>
    /// NT-Xent (Normalized Temperature-scaled Cross Entropy) - symmetric version.
    /// </summary>
    NTXent
}
