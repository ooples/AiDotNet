
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter",
    "https://arxiv.org/abs/1910.01108",
    Year = 2019,
    Authors = "Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
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

        // This method only computes output loss
        // Attention loss must be computed separately via ComputeAttentionLoss
        // and manually combined by the user

        // Soft loss with temperature scaling, blended with the hard loss if labels provided
        T totalLoss = Engine.Sum(DistillationHelper<T>.StandardSampleLosses(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha));

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

        // Soft gradient, blended with the hard gradient if labels provided
        var batchGradient = DistillationHelper<T>.StandardSampleGradients(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha);

        // Scale by (1 - attentionWeight)
        var scale = NumOps.FromDouble(1.0 - _attentionWeight);
        batchGradient = Engine.MatrixMultiplyScalar(batchGradient, scale);

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

        var layerLosses = new List<T>(_attentionLayers.Length);

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

            layerLosses.Add(ComputeAttentionMatchingLoss(studentAttention, teacherAttention));
        }

        T totalLoss = Engine.Sum(new Vector<T>(layerLosses));

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

        var layerLosses = new List<T>(_attentionLayers.Length);

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
            layerLosses.Add(ComputeLayerAttentionLoss(studentMatrix, teacherMatrix));
        }

        int layersFound = layerLosses.Count;
        T totalLoss = Engine.Sum(new Vector<T>(layerLosses));

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
    /// Mean over the batch rows of the per-sample attention matching loss for one layer.
    /// </summary>
    private T ComputeLayerAttentionLoss(Matrix<T> studentMatrix, Matrix<T> teacherMatrix)
    {
        T layerBatchLoss = Engine.Sum(ComputeAttentionMatchingLossRows(studentMatrix, teacherMatrix));

        // Average over batch
        return NumOps.Divide(layerBatchLoss, NumOps.FromDouble(studentMatrix.Rows));
    }

    /// <summary>
    /// <see cref="ComputeAttentionMatchingLoss"/> for every row (sample) of the two attention matrices.
    /// </summary>
    private Vector<T> ComputeAttentionMatchingLossRows(Matrix<T> studentMatrix, Matrix<T> teacherMatrix)
    {
        switch (_matchingMode)
        {
            case AttentionMatchingMode.MSE:
            {
                var diff = Engine.TensorSubtract(Tensor<T>.FromMatrix(studentMatrix), Tensor<T>.FromMatrix(teacherMatrix));
                return Engine.Divide(
                    DistillationHelper<T>.RowSums(Engine.TensorMultiply(diff, diff)), NumOps.FromDouble(studentMatrix.Columns));
            }

            case AttentionMatchingMode.KL:
                return DistillationHelper<T>.KLDivergenceRows(teacherMatrix, studentMatrix);

            case AttentionMatchingMode.Cosine:
            {
                var student = Tensor<T>.FromMatrix(studentMatrix);
                var teacher = Tensor<T>.FromMatrix(teacherMatrix);
                var dot = DistillationHelper<T>.RowSums(Engine.TensorMultiply(student, teacher));
                var normStudent = Engine.Sqrt(DistillationHelper<T>.RowSums(Engine.TensorMultiply(student, student)));
                var normTeacher = Engine.Sqrt(DistillationHelper<T>.RowSums(Engine.TensorMultiply(teacher, teacher)));
                var denominator = Engine.Add(
                    Engine.Multiply(normStudent, normTeacher), Vector<T>.CreateDefault(dot.Length, NumOps.FromDouble(Epsilon)));
                // Loss: 1 - similarity
                return Engine.Subtract(Vector<T>.CreateDefault(dot.Length, NumOps.One), Engine.Divide(dot, denominator));
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(_matchingMode), _matchingMode, $"Unrecognised AttentionMatchingMode '{_matchingMode}'. Valid modes: MSE, KL, Cosine.");
        }
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
                throw new ArgumentOutOfRangeException(nameof(_matchingMode), _matchingMode, $"Unrecognised AttentionMatchingMode '{_matchingMode}'. Valid modes: MSE, KL, Cosine.");
        }
    }

    private T ComputeMSE(Vector<T> student, Vector<T> teacher)
    {
        var diff = Engine.Subtract(student, teacher);
        T sum = Engine.DotProduct(diff, diff);
        return NumOps.Divide(sum, NumOps.FromDouble(student.Length));
    }

    private T ComputeCosineLoss(Vector<T> student, Vector<T> teacher)
    {
        // Cosine similarity: dot(a,b) / (||a|| * ||b||)
        // Loss: 1 - similarity

        T dot = Engine.DotProduct(student, teacher);
        T normStudent = Engine.DotProduct(student, student);
        T normTeacher = Engine.DotProduct(teacher, teacher);

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

            // Gradient for each sample in batch, averaged over batch (layers are averaged below)
            gradients.Add(layerName, ComputeLayerAttentionGradient(studentMatrix, teacherMatrix));
            layersFound++;
        }

        // If no layers found, return empty gradients
        if (layersFound == 0)
            return gradients;

        // Scale by 1/layersFound to average across layers, and by attentionWeight. Get returns a copy, so
        // the scaled matrix is stored back (each distinct layer is scaled once).
        var totalScale = NumOps.FromDouble(_attentionWeight / layersFound);
        foreach (var layerName in _attentionLayers.Distinct())
        {
            var layerGrad = gradients.Get(layerName);
            if (layerGrad != null)
            {
                gradients.Add(layerName, Engine.MatrixMultiplyScalar(layerGrad, totalScale));
            }
        }

        return gradients;
    }

    /// <summary>
    /// Gradient of the attention matching loss for one layer, averaged over the batch rows.
    /// </summary>
    private Matrix<T> ComputeLayerAttentionGradient(Matrix<T> studentMatrix, Matrix<T> teacherMatrix)
    {
        var layerGradient = ComputeAttentionMatchingGradientRows(studentMatrix, teacherMatrix);

        // Average over batch
        return Engine.TensorDivideScalar(layerGradient, NumOps.FromDouble(studentMatrix.Rows)).ToMatrix();
    }

    /// <summary>
    /// Per-sample gradient of the attention matching loss for every row of the two attention matrices.
    /// </summary>
    private Tensor<T> ComputeAttentionMatchingGradientRows(Matrix<T> studentMatrix, Matrix<T> teacherMatrix)
    {
        var student = Tensor<T>.FromMatrix(studentMatrix);
        var teacher = Tensor<T>.FromMatrix(teacherMatrix);
        int attentionDim = studentMatrix.Columns;

        switch (_matchingMode)
        {
            case AttentionMatchingMode.MSE:
                // MSE gradient: ∂L/∂student = 2 * (student - teacher) / dim
                return Engine.TensorMultiplyScalar(
                    Engine.TensorSubtract(student, teacher), NumOps.FromDouble(2.0 / attentionDim));

            case AttentionMatchingMode.KL:
            {
                // KL divergence gradient: ∂L/∂student = log(student/teacher) + 1
                // This assumes student and teacher are probability distributions
                T epsilon = NumOps.FromDouble(Epsilon);
                var ratio = Engine.TensorDivide(
                    Engine.TensorAddScalar(student, epsilon), Engine.TensorAddScalar(teacher, epsilon));
                return Engine.TensorAddScalar(Engine.TensorLog(ratio), NumOps.One);
            }

            case AttentionMatchingMode.Cosine:
            {
                // Cosine loss gradient: ∂(1 - cos)/∂student
                // cos(s,t) = dot(s,t) / (||s|| * ||t||)
                // ∂cos/∂s_i = (t_i / (||s|| * ||t||)) - (dot(s,t) * s_i) / (||s||^3 * ||t||)
                T epsilon = NumOps.FromDouble(Epsilon);
                int[] shape = student.Shape.ToArray();
                var dot = Engine.ReduceSum(Engine.TensorMultiply(student, teacher), [1], keepDims: true);
                var normS = Engine.TensorAddScalar(
                    Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(student, student), [1], keepDims: true)), epsilon);
                var normT = Engine.TensorAddScalar(
                    Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(teacher, teacher), [1], keepDims: true)), epsilon);
                var normProduct = Engine.TensorMultiply(normS, normT);
                var cubedProduct = Engine.TensorMultiply(Engine.TensorMultiply(Engine.TensorMultiply(normS, normS), normS), normT);

                var dcosDs = Engine.TensorSubtract(
                    Engine.TensorDivide(teacher, Engine.TensorBroadcastTo(normProduct, shape)),
                    Engine.TensorMultiply(student, Engine.TensorBroadcastTo(Engine.TensorDivide(dot, cubedProduct), shape)));

                // Loss is (1 - cos), so ∂L/∂s_i = -∂cos/∂s_i
                return Engine.TensorNegate(dcosDs);
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(_matchingMode), _matchingMode, $"Gradient is not defined for AttentionMatchingMode '{_matchingMode}'. Valid modes: MSE, KL, Cosine.");
        }
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


