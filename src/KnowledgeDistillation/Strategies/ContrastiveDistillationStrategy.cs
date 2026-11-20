
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Contrastive Representation Distillation",
    "https://arxiv.org/abs/1910.10699",
    Year = 2020,
    Authors = "Yonglong Tian, Dilip Krishnan, Phillip Isola")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class ContrastiveDistillationStrategy<T> : DistillationStrategyBase<T>, IIntermediateActivationStrategy<T>
{
    private readonly double _contrastiveWeight;
    private readonly int _negativesSampleSize;
    private readonly ContrastiveMode _mode;
    private readonly string _embeddingLayerName;

    /// <summary>
    /// Initializes a new instance of the ContrastiveDistillationStrategy class.
    /// </summary>
    /// <param name="embeddingLayerName">Name of the layer to extract embeddings from (e.g., "layer_before_classifier").</param>
    /// <param name="contrastiveWeight">Weight for contrastive loss vs standard output loss (default: 0.8).</param>
    /// <param name="temperature">Temperature for contrastive softmax (default: 0.07).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.2).</param>
    /// <param name="negativesSampleSize">Number of negative samples to use (default: 1024).</param>
    /// <param name="mode">Contrastive mode (default: NTXent for label-free operation).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configure how much to weight contrastive learning:</para>
    /// <para>- contrastiveWeight 0.6-0.9: More focus on learning representations
    /// - temperature 0.05-0.1: Lower = sharper distinctions between similar/dissimilar
    /// - negativesSampleSize 512-2048: More negatives = better discrimination</para>
    ///
    /// <para>Example:
    /// <code>
    /// var strategy = new ContrastiveDistillationStrategy&lt;double&gt;(
    ///     embeddingLayerName: "embedding_layer",
    ///     contrastiveWeight: 0.8,  // 80% contrastive, 20% standard
    ///     temperature: 0.07,        // Standard for contrastive learning
    ///     alpha: 0.2,              // Mostly teacher knowledge
    ///     negativesSampleSize: 1024 // Large negative set for better discrimination
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public ContrastiveDistillationStrategy(
        string embeddingLayerName = "embeddings",
        double contrastiveWeight = 0.8,
        double temperature = 0.07,
        double alpha = 0.2,
        int negativesSampleSize = 1024,
        ContrastiveMode mode = ContrastiveMode.NTXent)
        : base(temperature, alpha)
    {
        if (string.IsNullOrWhiteSpace(embeddingLayerName))
            throw new ArgumentException("Embedding layer name cannot be null or whitespace", nameof(embeddingLayerName));
        if (contrastiveWeight < 0 || contrastiveWeight > 1)
            throw new ArgumentException("Contrastive weight must be between 0 and 1", nameof(contrastiveWeight));
        if (negativesSampleSize < 1)
            throw new ArgumentException("Negatives sample size must be at least 1", nameof(negativesSampleSize));

        _embeddingLayerName = embeddingLayerName;
        _contrastiveWeight = contrastiveWeight;
        _negativesSampleSize = negativesSampleSize;
        _mode = mode;
    }

    /// <summary>
    /// Computes standard output loss (contrastive loss computed separately on embeddings).
    /// </summary>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Standard distillation loss (scaled by 1 - contrastiveWeight to make room for contrastive loss)
        var sampleLosses = Engine.Multiply(
            DistillationHelper<T>.StandardSampleLosses(studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha),
            NumOps.FromDouble(1.0 - _contrastiveWeight));
        T totalLoss = Engine.Sum(sampleLosses);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient of standard loss.
    /// </summary>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        var gradientBatch = DistillationHelper<T>.StandardSampleGradients(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha);

        // Scale by (1 - contrastiveWeight)
        var scale = NumOps.FromDouble(1.0 - _contrastiveWeight);
        gradientBatch = Engine.MatrixMultiplyScalar(gradientBatch, scale);

        // Average gradients over the batch
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        gradientBatch = Engine.MatrixMultiplyScalar(gradientBatch, oneOverBatch);

        return gradientBatch;
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
    /// Computes intermediate activation loss by matching embedding space structure between teacher and student.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate layer activations (must include embedding layer).</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate layer activations (must include embedding layer).</param>
    /// <returns>The contrastive loss (already weighted by contrastiveWeight).</returns>
    /// <remarks>
    /// <para>This implements the IIntermediateActivationStrategy interface to properly integrate
    /// contrastive learning into the training loop. The loss is computed from embeddings stored
    /// in the intermediate activations for the layer specified in the constructor.</para>
    ///
    /// <para>Uses NTXent mode by default as it doesn't require labels. Each teacher-student pair
    /// for the same sample is treated as a positive pair, while other samples are negatives.</para>
    ///
    /// <para>If the embedding layer is not found, returns zero loss.</para>
    /// </remarks>
    public T ComputeIntermediateLoss(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations)
    {
        if (studentIntermediateActivations == null)
            throw new ArgumentNullException(nameof(studentIntermediateActivations));
        if (teacherIntermediateActivations == null)
            throw new ArgumentNullException(nameof(teacherIntermediateActivations));

        var studentMatrix = studentIntermediateActivations.Get(_embeddingLayerName);
        var teacherMatrix = teacherIntermediateActivations.Get(_embeddingLayerName);

        // If layer not found, return zero loss
        if (studentMatrix == null || teacherMatrix == null)
            return NumOps.Zero;

        // Validate dimensions match
        if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
        {
            throw new ArgumentException(
                $"Student and teacher embedding dimensions must match for layer '{_embeddingLayerName}'. " +
                $"Student: [{studentMatrix.Rows} x {studentMatrix.Columns}], " +
                $"Teacher: [{teacherMatrix.Rows} x {teacherMatrix.Columns}]");
        }

        int batchSize = studentMatrix.Rows;
        if (batchSize == 0)
            return NumOps.Zero;

        // Convert matrices to vector arrays for contrastive loss computation
        var studentEmbeddings = new Vector<T>[batchSize];
        var teacherEmbeddings = new Vector<T>[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            studentEmbeddings[i] = studentMatrix.GetRow(i);
            teacherEmbeddings[i] = teacherMatrix.GetRow(i);
        }

        // Use NTXent mode since we don't have labels in IntermediateActivations
        T contrastiveLoss = ComputeNTXentLoss(studentEmbeddings, teacherEmbeddings);

        // Apply contrastive weight
        return NumOps.Multiply(contrastiveLoss, NumOps.FromDouble(_contrastiveWeight));
    }

    /// <summary>
    /// Computes InfoNCE (Noise Contrastive Estimation) loss.
    /// </summary>
    private T ComputeInfoNCELoss(Vector<T>[] studentEmbs, Vector<T>[] teacherEmbs, int[] labels)
    {
        int batchSize = studentEmbs.Length;
        if (batchSize == 0)
            return NumOps.Zero;

        var teachers = DistillationHelper<T>.StackRows(teacherEmbs);
        var students = DistillationHelper<T>.StackRows(studentEmbs);
        double inverseTemperature = 1.0 / Temperature;

        // Positive logit per anchor, row-wise: cos(teacher_i, student_i) / τ -> [B, 1]
        var positive = Engine.Reshape(Engine.TensorMultiplyScalar(
            DistillationHelper<T>.RowwiseCosineSimilarityWide(teachers, students), inverseTemperature), [batchSize, 1]);

        // Negative logits only against the first min(batchSize, negativesSampleSize) students, so the
        // working set is [B, negativeLimit] rather than [B, B]; each anchor's own student is excluded.
        int negativeLimit = Math.Min(batchSize, _negativesSampleSize);
        var negatives = Engine.TensorMultiplyScalar(
            DistillationHelper<T>.CosineSimilarityMatrixWide(teachers, Engine.TensorNarrow(students, 0, 0, negativeLimit)),
            inverseTemperature);
        var rowIndex = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<double>(0, batchSize, 1), [batchSize, 1]), [batchSize, negativeLimit]);
        var columnIndex = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<double>(0, negativeLimit, 1), [1, negativeLimit]), [batchSize, negativeLimit]);
        negatives = Engine.TensorWhere(
            Engine.TensorEquals(rowIndex, columnIndex),
            Tensor<double>.CreateDefault([batchSize, negativeLimit], double.NegativeInfinity),
            negatives);

        // -log(e^pos / (e^pos + Σ e^neg + ε)) = log(e^pos + Σ e^neg + ε) - pos, as a shifted log-sum-exp
        var logits = Engine.TensorConcatenate([positive, negatives], 1);
        var losses = Engine.TensorSubtract(
            DistillationHelper<T>.RowLogSumExp(logits, Epsilon), Engine.Reshape(positive, [batchSize]));

        return NumOps.FromDouble(Engine.TensorSum(losses) / batchSize);
    }

    /// <summary>
    /// Computes triplet loss.
    /// </summary>
    private T ComputeTripletLoss(Vector<T>[] studentEmbs, Vector<T>[] teacherEmbs, int[] labels)
    {
        int batchSize = studentEmbs.Length;
        var anchors = new List<int>();
        var negatives = new List<int>();

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

            anchors.Add(i);
            negatives.Add(negativeIdx);
        }

        int tripletCount = anchors.Count;
        if (tripletCount == 0)
            return NumOps.Zero;

        // Teacher-guided triplet loss:
        // Positive distance: student to its corresponding teacher
        // Negative distance: student to teacher of different class
        // This ensures student embeddings align with teacher's embedding space
        var anchorIndices = new Tensor<int>(anchors.ToArray(), [tripletCount]);
        var negativeIndices = new Tensor<int>(negatives.ToArray(), [tripletCount]);
        var students = Engine.TensorIndexSelect(DistillationHelper<T>.StackRows(studentEmbs), anchorIndices, 0);
        var teachers = DistillationHelper<T>.StackRows(teacherEmbs);
        var posDist = RowDistances(students, Engine.TensorIndexSelect(teachers, anchorIndices, 0));
        var negDist = RowDistances(students, Engine.TensorIndexSelect(teachers, negativeIndices, 0));
        var losses = Engine.TensorMax(
            new Tensor<T>([tripletCount]),
            Engine.TensorAddScalar(Engine.TensorSubtract(posDist, negDist), NumOps.FromDouble(margin)));

        return NumOps.Divide(Engine.TensorSum(losses), NumOps.FromDouble(tripletCount));
    }

    /// <summary>
    /// Euclidean distance between corresponding rows of two [k, d] tensors.
    /// </summary>
    private Tensor<T> RowDistances(Tensor<T> a, Tensor<T> b)
    {
        var diff = Engine.TensorSubtract(a, b);
        return Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(diff, diff), [1], keepDims: false));
    }

    /// <summary>
    /// Computes NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    /// </summary>
    private T ComputeNTXentLoss(Vector<T>[] studentEmbs, Vector<T>[] teacherEmbs)
    {
        int batchSize = studentEmbs.Length;
        if (batchSize == 0)
            return NumOps.Zero;

        var students = DistillationHelper<T>.StackRows(studentEmbs);
        var teachers = DistillationHelper<T>.StackRows(teacherEmbs);
        double inverseTemperature = 1.0 / Temperature;

        // sims[i, j] = cos(student_i, teacher_j) / τ in double; positive pairs on the diagonal
        var sims = Engine.TensorMultiplyScalar(
            DistillationHelper<T>.CosineSimilarityMatrixWide(students, teachers), inverseTemperature);
        var positiveSim = Engine.TensorMultiplyScalar(
            DistillationHelper<T>.RowwiseCosineSimilarityWide(students, teachers), inverseTemperature);

        // NT-Xent: -log(exp(pos) / (Σ_all exp + ε)) = log(Σ_all exp + ε) - pos, as a shifted log-sum-exp
        var losses = Engine.TensorSubtract(DistillationHelper<T>.RowLogSumExp(sims, Epsilon), positiveSim);

        return NumOps.FromDouble(Engine.TensorSum(losses) / batchSize);
    }



    /// <summary>
    /// Computes gradients of intermediate activation loss with respect to student embeddings.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate layer activations (must include embedding layer).</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate layer activations (must include embedding layer).</param>
    /// <returns>Gradients for the embedding layer (already weighted by contrastiveWeight).</returns>
    /// <remarks>
    /// <para>Computes analytical gradients for NT-Xent loss with respect to student embeddings.
    /// Uses cosine similarity gradients derived from the quotient rule.</para>
    ///
    /// <para>For NT-Xent loss L_i = -sim(s_i, t_i)/τ + log(Σ_j exp(sim(s_i, t_j)/τ)),
    /// the gradient involves both positive and negative pair contributions.</para>
    /// </remarks>
    public IntermediateActivations<T> ComputeIntermediateGradient(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations)
    {
        if (studentIntermediateActivations == null)
            throw new ArgumentNullException(nameof(studentIntermediateActivations));
        if (teacherIntermediateActivations == null)
            throw new ArgumentNullException(nameof(teacherIntermediateActivations));

        var gradients = new IntermediateActivations<T>();

        var studentMatrix = studentIntermediateActivations.Get(_embeddingLayerName);
        var teacherMatrix = teacherIntermediateActivations.Get(_embeddingLayerName);

        // If layer not found, return empty gradients
        if (studentMatrix == null || teacherMatrix == null)
            return gradients;

        // Validate dimensions match
        if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
        {
            throw new ArgumentException(
                $"Student and teacher embedding dimensions must match for layer '{_embeddingLayerName}'. " +
                $"Student: [{studentMatrix.Rows} x {studentMatrix.Columns}], " +
                $"Teacher: [{teacherMatrix.Rows} x {teacherMatrix.Columns}]");
        }

        int batchSize = studentMatrix.Rows;
        if (batchSize == 0)
            return gradients;

        // Convert matrices to vector arrays for gradient computation
        var studentEmbeddings = new Vector<T>[batchSize];
        var teacherEmbeddings = new Vector<T>[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            studentEmbeddings[i] = studentMatrix.GetRow(i);
            teacherEmbeddings[i] = teacherMatrix.GetRow(i);
        }

        // Compute NT-Xent gradients for each sample
        var gradientMatrix = ComputeNTXentGradients(studentEmbeddings, teacherEmbeddings);

        // Average over batch
        gradientMatrix = Engine.TensorDivideScalar(gradientMatrix, NumOps.FromDouble(batchSize));

        // Scale by contrastive weight
        gradientMatrix = Engine.TensorMultiplyScalar(gradientMatrix, NumOps.FromDouble(_contrastiveWeight));

        gradients.Add(_embeddingLayerName, gradientMatrix.ToMatrix());
        return gradients;
    }

    /// <summary>
    /// Computes the NT-Xent gradient for every anchor sample.
    /// </summary>
    /// <param name="studentEmbeddings">All student embeddings in the batch.</param>
    /// <param name="teacherEmbeddings">All teacher embeddings in the batch.</param>
    /// <returns>[batch, dim] gradient; row i is the gradient with respect to student embedding i.</returns>
    /// <remarks>
    /// <para>NT-Xent loss for sample i: L_i = -sim(s_i, t_i)/τ + log(Σ_j exp(sim(s_i, t_j)/τ))</para>
    ///
    /// <para>Gradient: ∂L_i/∂s_i = (-∂sim(s_i, t_i)/∂s_i + Σ_j softmax_j * ∂sim(s_i, t_j)/∂s_i) / τ</para>
    ///
    /// <para>Where softmax_j = exp(sim(s_i, t_j)/τ) / Σ_k exp(sim(s_i, t_k)/τ), and
    /// ∂sim(s, t)/∂s = t / (||s|| ||t||) - dot(s, t) s / (||s||³ ||t||) with ε-padded norms.</para>
    /// </remarks>
    private Tensor<T> ComputeNTXentGradients(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        int batchSize = studentEmbeddings.Length;
        var studentRows = DistillationHelper<T>.StackRows(studentEmbeddings);
        var teacherRows = DistillationHelper<T>.StackRows(teacherEmbeddings);
        var students = studentRows.Cast<double>();
        var teachers = teacherRows.Cast<double>();
        double epsilon = Epsilon;

        // Softmax weights exp(sim/τ) / (Σ exp(sim/τ) + ε), evaluated as exp(sim/τ - logSumExp) so
        // small temperatures cannot overflow the exponentials.
        var sims = Engine.TensorMultiplyScalar(
            DistillationHelper<T>.CosineSimilarityMatrixWide(studentRows, teacherRows), 1.0 / Temperature);
        var logNormalizer = Engine.Reshape(DistillationHelper<T>.RowLogSumExp(sims, Epsilon), [batchSize, 1]);
        var softmaxWeights = Engine.TensorExp(
            Engine.TensorSubtract(sims, Engine.TensorBroadcastTo(logNormalizer, [batchSize, batchSize])));

        // -∂sim(s_i, t_i)/∂s_i + Σ_j softmax_j ∂sim(s_i, t_j)/∂s_i = Σ_j (softmax_ij - δ_ij) ∂sim(s_i, t_j)/∂s_i
        var pairWeights = Engine.TensorSubtract(softmaxWeights, Tensor<double>.FromMatrix(Matrix<double>.CreateIdentity(batchSize)));

        var dots = Engine.TensorMatMul(students, Engine.TensorTranspose(teachers));                       // [B, B]
        var normS = Engine.TensorAddScalar(
            Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(students, students), [1], keepDims: true)), epsilon); // [B, 1]
        var normT = Engine.TensorAddScalar(
            Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(teachers, teachers), [1], keepDims: true)), epsilon); // [B, 1]
        var normTRow = Engine.TensorTranspose(normT);                                                       // [1, B]
        var normProduct = Engine.TensorMatMul(normS, normTRow);                                             // ||s_i|| ||t_j||
        var cubedProduct = Engine.TensorMatMul(Engine.TensorMultiply(Engine.TensorMultiply(normS, normS), normS), normTRow);

        // Σ_j w_ij t_j / (||s_i|| ||t_j||)  -  s_i Σ_j w_ij dot_ij / (||s_i||³ ||t_j||)
        var teacherTerm = Engine.TensorMatMul(Engine.TensorDivide(pairWeights, normProduct), teachers);
        var selfCoefficient = Engine.ReduceSum(
            Engine.TensorDivide(Engine.TensorMultiply(pairWeights, dots), cubedProduct), [1], keepDims: true);
        var gradient = Engine.TensorSubtract(
            teacherTerm, Engine.TensorMultiply(students, Engine.TensorBroadcastTo(selfCoefficient, students.Shape.ToArray())));

        // Scale by 1/temperature
        return Engine.TensorMultiplyScalar(gradient, 1.0 / Temperature).Cast<T>();
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


