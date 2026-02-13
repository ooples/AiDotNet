
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

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
public class AttentionDistillationStrategy<T> : DistillationStrategyBase<T>, IIntermediateActivationStrategy<T>
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

        Guard.NotNull(attentionLayers);
        _attentionLayers = attentionLayers;
        _attentionWeight = attentionWeight;
        _matchingMode = matchingMode;

        if (_attentionLayers.Length == 0)
            throw new ArgumentException("At least one attention layer must be specified", nameof(attentionLayers));
    }

    /// <summary>
    /// Computes combined distillation loss (output loss + attention loss).
    /// </summary>
    /// <param name="studentBatchOutput">Student batch output [batchSize x outputDim].</param>
    /// <param name="teacherBatchOutput">Teacher batch output [batchSize x outputDim].</param>
    /// <param name="trueLabelsBatch">Optional batch labels [batchSize x outputDim].</param>
    /// <returns>Average loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This combines two types of loss:
    /// 1. Standard distillation loss on final outputs
    /// 2. Attention matching loss on intermediate attention patterns</para>
    ///
    /// <para>Formula: L = (1 - w) × L_output + w × L_attention
    /// where w is attentionWeight.</para>
    /// </remarks>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);

        if (studentBatchOutput == null) throw new ArgumentNullException(nameof(studentBatchOutput));
        if (teacherBatchOutput == null) throw new ArgumentNullException(nameof(teacherBatchOutput));

        if (studentBatchOutput.Rows != teacherBatchOutput.Rows || studentBatchOutput.Columns != teacherBatchOutput.Columns)
            throw new ArgumentException("Student and teacher batch outputs must have matching dimensions");

        if (trueLabelsBatch != null && (trueLabelsBatch.Rows != studentBatchOutput.Rows || trueLabelsBatch.Columns != studentBatchOutput.Columns))
            throw new ArgumentException("Batch labels must match student output dimensions");

        int batchSize = studentBatchOutput.Rows;
        T totalLoss = NumOps.Zero;

        // This method only computes output loss
        // Attention loss must be computed separately via ComputeAttentionLoss
        // and manually combined by the user

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            // Soft loss with temperature scaling
            var studentSoft = DistillationHelper<T>.Softmax(studentRow, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherRow, Temperature);
            var softLoss = DistillationHelper<T>.KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            T sampleLoss = softLoss;

            // Hard loss if labels provided
            if (labelRow != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentRow, 1.0);
                var hardLoss = DistillationHelper<T>.CrossEntropy(studentProbs, labelRow);

                var alphaT = NumOps.FromDouble(Alpha);
                var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

                sampleLoss = NumOps.Add(
                    NumOps.Multiply(alphaT, hardLoss),
                    NumOps.Multiply(oneMinusAlpha, softLoss));
            }

            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient of the combined loss.
    /// </summary>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);

        if (studentBatchOutput == null) throw new ArgumentNullException(nameof(studentBatchOutput));
        if (teacherBatchOutput == null) throw new ArgumentNullException(nameof(teacherBatchOutput));

        if (studentBatchOutput.Rows != teacherBatchOutput.Rows || studentBatchOutput.Columns != teacherBatchOutput.Columns)
            throw new ArgumentException("Student and teacher batch outputs must have matching dimensions");

        if (trueLabelsBatch != null && (trueLabelsBatch.Rows != studentBatchOutput.Rows || trueLabelsBatch.Columns != studentBatchOutput.Columns))
            throw new ArgumentException("Batch labels must match student output dimensions");

        int batchSize = studentBatchOutput.Rows;
        int outputDim = studentBatchOutput.Columns;
        var batchGradient = new Matrix<T>(batchSize, outputDim);

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            int n = studentRow.Length;
            var gradient = new Vector<T>(n);

            // Soft gradient
            var studentSoft = DistillationHelper<T>.Softmax(studentRow, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherRow, Temperature);

            for (int i = 0; i < n; i++)
            {
                var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
            }

            // Add hard gradient if labels provided
            if (labelRow != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentRow, 1.0);
                var hardGradient = new Vector<T>(n);

                for (int i = 0; i < n; i++)
                {
                    hardGradient[i] = NumOps.Subtract(studentProbs[i], labelRow[i]);
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

            // Store in batch gradient matrix
            for (int c = 0; c < outputDim; c++)
            {
                batchGradient[r, c] = gradient[c];
            }
        }

        return batchGradient;
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
    /// Computes intermediate activation loss by matching attention patterns between teacher and student.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate layer activations (must include attention layers).</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate layer activations (must include attention layers).</param>
    /// <returns>The attention matching loss (already weighted by attentionWeight).</returns>
    /// <remarks>
    /// <para>This implements the IIntermediateActivationStrategy interface to properly integrate
    /// attention matching into the training loop. The loss is computed from attention patterns stored
    /// in the intermediate activations for layers specified in the constructor.</para>
    ///
    /// <para>If any target layer is not found, it is skipped. Returns zero if no layers are found.</para>
    /// </remarks>
    public T ComputeIntermediateLoss(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations)
    {
        if (studentIntermediateActivations == null)
            throw new ArgumentNullException(nameof(studentIntermediateActivations));
        if (teacherIntermediateActivations == null)
            throw new ArgumentNullException(nameof(teacherIntermediateActivations));

        T totalLoss = NumOps.Zero;
        int layersFound = 0;

        foreach (var layerName in _attentionLayers)
        {
            var studentMatrix = studentIntermediateActivations.Get(layerName);
            var teacherMatrix = teacherIntermediateActivations.Get(layerName);

            // Skip if layer not found in either model
            if (studentMatrix == null || teacherMatrix == null)
                continue;

            // Validate dimensions match
            if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
            {
                throw new ArgumentException(
                    $"Student and teacher attention dimensions must match for layer '{layerName}'. " +
                    $"Student: [{studentMatrix.Rows} x {studentMatrix.Columns}], " +
                    $"Teacher: [{teacherMatrix.Rows} x {teacherMatrix.Columns}]");
            }

            int batchSize = studentMatrix.Rows;
            if (batchSize == 0)
                continue;

            // Average attention matching loss across all samples in batch
            T layerBatchLoss = NumOps.Zero;
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                var studentAttention = studentMatrix.GetRow(sampleIdx);
                var teacherAttention = teacherMatrix.GetRow(sampleIdx);

                T sampleLoss = ComputeAttentionMatchingLoss(studentAttention, teacherAttention);
                layerBatchLoss = NumOps.Add(layerBatchLoss, sampleLoss);
            }

            // Average over batch
            layerBatchLoss = NumOps.Divide(layerBatchLoss, NumOps.FromDouble(batchSize));
            totalLoss = NumOps.Add(totalLoss, layerBatchLoss);
            layersFound++;
        }

        // Return zero if no layers found
        if (layersFound == 0)
            return NumOps.Zero;

        // Average across layers
        totalLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(layersFound));

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
                return DistillationHelper<T>.KLDivergence(teacherAttention, studentAttention);

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

    /// <summary>
    /// Computes gradients of intermediate activation loss with respect to student activations.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate layer activations.</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate layer activations.</param>
    /// <returns>Gradients for each attention layer (already weighted by attentionWeight).</returns>
    public IntermediateActivations<T> ComputeIntermediateGradient(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations)
    {
        if (studentIntermediateActivations == null)
            throw new ArgumentNullException(nameof(studentIntermediateActivations));
        if (teacherIntermediateActivations == null)
            throw new ArgumentNullException(nameof(teacherIntermediateActivations));

        var gradients = new IntermediateActivations<T>();
        int layersFound = 0;

        foreach (var layerName in _attentionLayers)
        {
            var studentMatrix = studentIntermediateActivations.Get(layerName);
            var teacherMatrix = teacherIntermediateActivations.Get(layerName);

            // Skip if layer not found in either model
            if (studentMatrix == null || teacherMatrix == null)
                continue;

            // Validate dimensions match
            if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
            {
                throw new ArgumentException(
                    $"Student and teacher activation dimensions must match for layer '{layerName}'. " +
                    $"Student: [{studentMatrix.Rows} x {studentMatrix.Columns}], " +
                    $"Teacher: [{teacherMatrix.Rows} x {teacherMatrix.Columns}]");
            }

            int batchSize = studentMatrix.Rows;
            if (batchSize == 0)
                continue;

            int attentionDim = studentMatrix.Columns;
            var layerGradient = new Matrix<T>(batchSize, attentionDim);

            // Compute gradient for each sample in batch
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                var studentAttention = studentMatrix.GetRow(sampleIdx);
                var teacherAttention = teacherMatrix.GetRow(sampleIdx);

                var sampleGradient = ComputeAttentionMatchingGradient(studentAttention, teacherAttention);
                layerGradient.SetRow(sampleIdx, sampleGradient);
            }

            // Average over batch and layers (will divide by layersFound at the end)
            for (int r = 0; r < batchSize; r++)
            {
                for (int c = 0; c < attentionDim; c++)
                {
                    layerGradient[r, c] = NumOps.Divide(layerGradient[r, c], NumOps.FromDouble(batchSize));
                }
            }

            gradients.Add(layerName, layerGradient);
            layersFound++;
        }

        // If no layers found, return empty gradients
        if (layersFound == 0)
            return gradients;

        // Scale by 1/layersFound to average across layers, and by attentionWeight
        var totalScale = _attentionWeight / layersFound;
        foreach (var layerName in _attentionLayers)
        {
            var layerGrad = gradients.Get(layerName);
            if (layerGrad != null)
            {
                for (int r = 0; r < layerGrad.Rows; r++)
                {
                    for (int c = 0; c < layerGrad.Columns; c++)
                    {
                        layerGrad[r, c] = NumOps.Multiply(layerGrad[r, c], NumOps.FromDouble(totalScale));
                    }
                }
            }
        }

        return gradients;
    }

    /// <summary>
    /// Computes gradient of attention matching loss for a single sample.
    /// </summary>
    private Vector<T> ComputeAttentionMatchingGradient(Vector<T> studentAttention, Vector<T> teacherAttention)
    {
        var gradient = new Vector<T>(studentAttention.Length);

        switch (_matchingMode)
        {
            case AttentionMatchingMode.MSE:
                // MSE gradient: ∂L/∂student = 2 * (student - teacher) / dim
                for (int i = 0; i < studentAttention.Length; i++)
                {
                    var diff = NumOps.Subtract(studentAttention[i], teacherAttention[i]);
                    gradient[i] = NumOps.Multiply(NumOps.FromDouble(2.0 / studentAttention.Length), diff);
                }
                break;

            case AttentionMatchingMode.KL:
                // KL divergence gradient: ∂L/∂student = log(student/teacher) + 1
                // This assumes student and teacher are probability distributions
                for (int i = 0; i < studentAttention.Length; i++)
                {
                    double s = Convert.ToDouble(studentAttention[i]) + Epsilon;
                    double t = Convert.ToDouble(teacherAttention[i]) + Epsilon;
                    double grad = Math.Log(s / t) + 1.0;
                    gradient[i] = NumOps.FromDouble(grad);
                }
                break;

            case AttentionMatchingMode.Cosine:
                // Cosine loss gradient: ∂(1 - cos)/∂student
                // cos(s,t) = dot(s,t) / (||s|| * ||t||)
                // ∂cos/∂s_i = (t_i / (||s|| * ||t||)) - (dot(s,t) * s_i) / (||s||^3 * ||t||)

                T dot = NumOps.Zero;
                T normStudent = NumOps.Zero;
                T normTeacher = NumOps.Zero;

                for (int i = 0; i < studentAttention.Length; i++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(studentAttention[i], teacherAttention[i]));
                    normStudent = NumOps.Add(normStudent, NumOps.Multiply(studentAttention[i], studentAttention[i]));
                    normTeacher = NumOps.Add(normTeacher, NumOps.Multiply(teacherAttention[i], teacherAttention[i]));
                }

                double dotVal = Convert.ToDouble(dot);
                double normS = Math.Sqrt(Convert.ToDouble(normStudent)) + Epsilon;
                double normT = Math.Sqrt(Convert.ToDouble(normTeacher)) + Epsilon;

                for (int i = 0; i < studentAttention.Length; i++)
                {
                    double s_i = Convert.ToDouble(studentAttention[i]);
                    double t_i = Convert.ToDouble(teacherAttention[i]);

                    // ∂cos/∂s_i
                    double dcos_dsi = (t_i / (normS * normT)) - (dotVal * s_i) / (normS * normS * normS * normT);

                    // Loss is (1 - cos), so ∂L/∂s_i = -∂cos/∂s_i
                    gradient[i] = NumOps.FromDouble(-dcos_dsi);
                }
                break;

            default:
                throw new NotImplementedException($"Gradient for matching mode {_matchingMode} not implemented");
        }

        return gradient;
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


