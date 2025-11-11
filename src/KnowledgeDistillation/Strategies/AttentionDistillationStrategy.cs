using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Implements attention-based knowledge distillation for transformer models.
/// Transfers knowledge through attention patterns rather than just final outputs.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Attention mechanisms in transformers tell us "what the model is focusing on."
/// Instead of just copying the teacher's final answers, attention distillation teaches the student to
/// focus on the same things the teacher focuses on.</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine learning to play chess from a grandmaster. Instead of just copying their moves (outputs),
/// you also learn where they look on the board and what pieces they pay attention to. This deeper
/// understanding helps you think like the master, not just mimic their moves.</para>
///
/// <para><b>Why Attention Distillation?</b>
/// - **Richer Knowledge**: Attention patterns reveal reasoning process
/// - **Better for Transformers**: Transformers rely heavily on attention
/// - **Interpretability**: Can see what student learned to focus on
/// - **Complementary**: Works with response-based distillation</para>
///
/// <para><b>How It Works:</b>
/// 1. Extract attention weights from teacher layers
/// 2. Extract attention weights from student layers
/// 3. Minimize MSE between attention distributions
/// 4. Combine with standard output distillation loss</para>
///
/// <para><b>Attention Matching Strategies:</b>
/// - **Layer-wise**: Match corresponding layers (layer 6→layer 3)
/// - **Head-wise**: Match individual attention heads
/// - **Global**: Match averaged attention across all heads
/// - **Selective**: Match only the most important heads</para>
///
/// <para><b>Common Applications:</b>
/// - **DistilBERT**: Used attention distillation to compress BERT
/// - **TinyBERT**: Attention transfer + representation transfer
/// - **MobileBERT**: Layer-wise attention matching
/// - **Vision Transformers**: Attention distillation for ViT compression</para>
///
/// <para><b>Benefits:</b>
/// - Preserves model's "reasoning" process
/// - Improves student's interpretability
/// - Often yields 2-5% better accuracy than output-only distillation
/// - Helps with few-shot and zero-shot transfer</para>
///
/// <para><b>References:</b>
/// - Sanh et al. (2019). DistilBERT: A Distilled Version of BERT. arXiv:1910.01108
/// - Jiao et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. EMNLP.
/// - Wang et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression.</para>
/// </remarks>
public class AttentionDistillationStrategy<T> : DistillationStrategyBase<Vector<T>, T>
{
    private readonly string[] _attentionLayers;
    private readonly double _attentionWeight;
    private readonly AttentionMatchingMode _matchingMode;

    /// <summary>
    /// Initializes a new instance of the AttentionDistillationStrategy class.
    /// </summary>
    /// <param name="attentionLayers">Names of attention layers to match (e.g., ["layer.0.attention", "layer.1.attention"]).</param>
    /// <param name="attentionWeight">Weight for attention loss vs. output loss (default: 0.3).</param>
    /// <param name="temperature">Temperature for softmax scaling (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="matchingMode">How to match attention patterns (default: MSE).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Specify which attention layers to match and how much weight
    /// to give to attention matching vs. output matching.</para>
    ///
    /// <para>Example for BERT-like model:
    /// <code>
    /// var strategy = new AttentionDistillationStrategy&lt;double&gt;(
    ///     attentionLayers: new[] {
    ///         "encoder.layer.0.attention",
    ///         "encoder.layer.3.attention",
    ///         "encoder.layer.6.attention"
    ///     },
    ///     attentionWeight: 0.3,  // 30% attention, 70% output
    ///     temperature: 2.0,
    ///     alpha: 0.5
    /// );
    /// </code>
    /// </para>
    ///
    /// <para><b>Layer Selection Tips:</b>
    /// - **Early layers**: Low-level patterns (syntax, local features)
    /// - **Middle layers**: Mid-level concepts (phrases, object parts)
    /// - **Late layers**: High-level semantics (meaning, objects)
    /// - **All layers**: Most comprehensive but computationally expensive
    /// - **Selective**: Match 2-3 key layers for efficiency</para>
    ///
    /// <para><b>Weight Selection:</b>
    /// - 0.1-0.2: Slight attention guidance
    /// - 0.3-0.4: Balanced (recommended for most cases)
    /// - 0.5-0.7: Strong attention focus
    /// - 0.8+: Primarily attention-driven (risky)</para>
    /// </remarks>
    public AttentionDistillationStrategy(
        string[] attentionLayers,
        double attentionWeight = 0.3,
        double temperature = 3.0,
        double alpha = 0.3,
        AttentionMatchingMode matchingMode = AttentionMatchingMode.MSE)
        : base(temperature, alpha)
    {
        if (attentionWeight < 0 || attentionWeight > 1)
            throw new ArgumentException("Attention weight must be between 0 and 1", nameof(attentionWeight));

        _attentionLayers = attentionLayers ?? throw new ArgumentNullException(nameof(attentionLayers));
        _attentionWeight = attentionWeight;
        _matchingMode = matchingMode;

        if (_attentionLayers.Length == 0)
            throw new ArgumentException("At least one attention layer must be specified", nameof(attentionLayers));
    }

    /// <summary>
    /// Computes combined distillation loss (output loss + attention loss).
    /// </summary>
    /// <param name="studentOutput">Student's output logits.</param>
    /// <param name="teacherOutput">Teacher's output logits.</param>
    /// <param name="trueLabels">Optional ground truth labels.</param>
    /// <returns>Combined loss value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This combines two types of loss:
    /// 1. Standard distillation loss on final outputs
    /// 2. Attention matching loss on intermediate attention patterns</para>
    ///
    /// <para>Formula: L = (1 - w) × L_output + w × L_attention
    /// where w is attentionWeight.</para>
    /// </remarks>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // This method only computes output loss
        // Attention loss is computed separately via ComputeAttentionLoss
        // and combined externally

        // Soft loss with temperature scaling
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(studentSoft, teacherSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        // Hard loss if labels provided
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

            // Scale by (1 - attentionWeight) to leave room for attention loss
            combinedLoss = NumOps.Multiply(combinedLoss, NumOps.FromDouble(1.0 - _attentionWeight));
            return combinedLoss;
        }

        // Scale soft loss by (1 - attentionWeight)
        return NumOps.Multiply(softLoss, NumOps.FromDouble(1.0 - _attentionWeight));
    }

    /// <summary>
    /// Computes gradient of the combined loss.
    /// </summary>
    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);

        // Soft gradient
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);

        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
        }

        // Add hard gradient if labels provided
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

        // Scale by (1 - attentionWeight)
        var scale = NumOps.FromDouble(1.0 - _attentionWeight);
        for (int i = 0; i < n; i++)
        {
            gradient[i] = NumOps.Multiply(gradient[i], scale);
        }

        return gradient;
    }

    /// <summary>
    /// Computes attention matching loss between teacher and student attention patterns.
    /// </summary>
    /// <param name="teacherAttentionExtractor">Function to extract teacher attention for a layer.</param>
    /// <param name="studentAttentionExtractor">Function to extract student attention for a layer.</param>
    /// <returns>Attention matching loss.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how different the attention patterns are.
    /// Lower loss means student is focusing on the same things as the teacher.</para>
    ///
    /// <para>The extractors should return attention weights as vectors, typically
    /// flattened from [num_heads, seq_len, seq_len] matrices.</para>
    /// </remarks>
    public T ComputeAttentionLoss(
        Func<string, Vector<T>> teacherAttentionExtractor,
        Func<string, Vector<T>> studentAttentionExtractor)
    {
        if (teacherAttentionExtractor == null) throw new ArgumentNullException(nameof(teacherAttentionExtractor));
        if (studentAttentionExtractor == null) throw new ArgumentNullException(nameof(studentAttentionExtractor));

        T totalLoss = NumOps.Zero;

        foreach (var layerName in _attentionLayers)
        {
            var teacherAttention = teacherAttentionExtractor(layerName);
            var studentAttention = studentAttentionExtractor(layerName);

            if (teacherAttention.Length != studentAttention.Length)
            {
                throw new ArgumentException(
                    $"Attention dimensions must match for layer {layerName}. " +
                    $"Teacher: {teacherAttention.Length}, Student: {studentAttention.Length}");
            }

            T layerLoss = ComputeAttentionMatchingLoss(studentAttention, teacherAttention);
            totalLoss = NumOps.Add(totalLoss, layerLoss);
        }

        // Average across layers
        totalLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_attentionLayers.Length));

        // Apply attention weight
        totalLoss = NumOps.Multiply(totalLoss, NumOps.FromDouble(_attentionWeight));

        return totalLoss;
    }

    /// <summary>
    /// Computes loss for matching a single attention layer.
    /// </summary>
    private T ComputeAttentionMatchingLoss(Vector<T> studentAttention, Vector<T> teacherAttention)
    {
        switch (_matchingMode)
        {
            case AttentionMatchingMode.MSE:
                return ComputeMSE(studentAttention, teacherAttention);

            case AttentionMatchingMode.KL:
                return KLDivergence(teacherAttention, studentAttention);

            case AttentionMatchingMode.Cosine:
                return ComputeCosineLoss(studentAttention, teacherAttention);

            default:
                throw new NotImplementedException($"Matching mode {_matchingMode} not implemented");
        }
    }

    private T ComputeMSE(Vector<T> student, Vector<T> teacher)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < student.Length; i++)
        {
            var diff = NumOps.Subtract(student[i], teacher[i]);
            var squared = NumOps.Multiply(diff, diff);
            sum = NumOps.Add(sum, squared);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(student.Length));
    }

    private T ComputeCosineLoss(Vector<T> student, Vector<T> teacher)
    {
        // Cosine similarity: dot(a,b) / (||a|| * ||b||)
        // Loss: 1 - similarity

        T dot = NumOps.Zero;
        T normStudent = NumOps.Zero;
        T normTeacher = NumOps.Zero;

        for (int i = 0; i < student.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(student[i], teacher[i]));
            normStudent = NumOps.Add(normStudent, NumOps.Multiply(student[i], student[i]));
            normTeacher = NumOps.Add(normTeacher, NumOps.Multiply(teacher[i], teacher[i]));
        }

        double dotVal = Convert.ToDouble(dot);
        double normStudentVal = Math.Sqrt(Convert.ToDouble(normStudent));
        double normTeacherVal = Math.Sqrt(Convert.ToDouble(normTeacher));

        double similarity = dotVal / (normStudentVal * normTeacherVal + Epsilon);
        double loss = 1.0 - similarity;

        return NumOps.FromDouble(loss);
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
/// Defines how to match attention patterns between teacher and student.
/// </summary>
public enum AttentionMatchingMode
{
    /// <summary>
    /// Mean Squared Error - simple, fast, treats all attention weights equally.
    /// </summary>
    MSE,

    /// <summary>
    /// KL Divergence - treats attention as probability distribution, preserves structure.
    /// </summary>
    KL,

    /// <summary>
    /// Cosine similarity - focuses on direction/pattern rather than magnitude.
    /// </summary>
    Cosine
}
