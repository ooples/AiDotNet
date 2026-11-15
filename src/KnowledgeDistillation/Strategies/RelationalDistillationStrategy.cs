using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Implements Relational Knowledge Distillation (RKD) which transfers knowledge about
/// relationships between samples rather than individual sample predictions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Unlike standard distillation which teaches the student to match
/// the teacher's predictions for each sample, RKD teaches the student to match the teacher's
/// understanding of how samples relate to each other.</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine learning about animals from a zoologist. Standard distillation would teach you
/// to identify each animal individually. Relational distillation teaches you the relationships:
/// "dogs are more similar to wolves than to cats," "sparrows are closer to robins than to eagles."
/// This relational knowledge helps you understand the taxonomy and make better predictions
/// about animals you've never seen.</para>
///
/// <para><b>Key Insight:</b>
/// The relationships between data points (similarity, ordering, grouping) often contain more
/// information than individual predictions. By preserving these relationships, the student
/// learns a more robust and generalizable representation.</para>
///
/// <para><b>Types of Relations RKD Preserves:</b>
/// - **Distance-wise**: Relative distances between samples
/// - **Angle-wise**: Angular relationships (geometric structure)
/// - **Ranking**: Relative ordering of samples
/// - **Pairwise Similarity**: How similar each pair of samples is</para>
///
/// <para><b>Mathematical Foundation:</b>
/// Given samples (x1, x2, x3), standard distillation matches:
/// - f_student(x1) ≈ f_teacher(x1)
/// - f_student(x2) ≈ f_teacher(x2)
///
/// RKD additionally matches:
/// - distance(f_student(x1), f_student(x2)) ≈ distance(f_teacher(x1), f_teacher(x2))
/// - angle(f_student(x1), f_student(x2), f_student(x3)) ≈ angle(f_teacher(x1), f_teacher(x2), f_teacher(x3))</para>
///
/// <para><b>Benefits:</b>
/// - **Better Generalization**: Student learns structural knowledge
/// - **Robust to Domain Shift**: Relationships more stable than absolute values
/// - **Few-shot Learning**: Helps with small training sets
/// - **Metric Learning**: Improves learned embeddings
/// - **Complementary**: Can combine with standard distillation</para>
///
/// <para><b>When to Use:</b>
/// - Metric learning tasks (face recognition, image retrieval)
/// - Few-shot and zero-shot learning
/// - Domain adaptation (relationships transfer better)
/// - Learning embeddings/representations
/// - Classification with fine-grained distinctions</para>
///
/// <para><b>Practical Example:</b>
/// For a batch of 32 images:
/// - Standard distillation: 32 individual predictions to match
/// - RKD: 32×32 = 1024 pairwise relationships to match
/// This richer signal often yields 2-5% accuracy improvement.</para>
///
/// <para><b>Computational Cost:</b>
/// - Distance-wise: O(n²) for n samples in batch
/// - Angle-wise: O(n³) for triplets
/// - Use sampling for large batches (sample k pairs instead of all)</para>
///
/// <para><b>References:</b>
/// - Park et al. (2019). Relational Knowledge Distillation. CVPR.
/// - Tung &amp; Mori (2019). Similarity-Preserving Knowledge Distillation. ICCV.
/// - Peng et al. (2019). Correlation Congruence for Knowledge Distillation. ICCV.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Relational Knowledge Distillation",
    "https://arxiv.org/abs/1904.05068",
    Year = 2019,
    Authors = "Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class RelationalDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _distanceWeight;
    private readonly double _angleWeight;
    private readonly int _maxSamplesPerBatch;
    private readonly RelationalDistanceMetric _distanceMetric;

    /// <summary>
    /// Initializes a new instance of the RelationalDistillationStrategy class.
    /// </summary>
    /// <param name="distanceWeight">Weight for distance-wise relation loss (default: 1.0).</param>
    /// <param name="angleWeight">Weight for angle-wise relation loss (default: 2.0).</param>
    /// <param name="temperature">Temperature for softmax scaling (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="maxSamplesPerBatch">Max samples to consider for relations (default: 32).</param>
    /// <param name="distanceMetric">Distance metric to use (default: Euclidean).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configure how much to weight different types of relations:
    /// - distanceWeight: How much to preserve pairwise distances
    /// - angleWeight: How much to preserve triplet angles
    /// - Higher weights = stronger enforcement of that relation type</para>
    ///
    /// <para>Example:
    /// <code>
    /// var strategy = new RelationalDistillationStrategy&lt;double&gt;(
    ///     distanceWeight: 25.0,  // Strong distance preservation
    ///     angleWeight: 50.0,     // Very strong angle preservation (recommended higher)
    ///     temperature: 3.0,
    ///     alpha: 0.3,
    ///     maxSamplesPerBatch: 16  // Limit for efficiency
    /// );
    /// </code>
    /// </para>
    ///
    /// <para><b>Weight Selection Guidelines:</b>
    /// - **Distance-wise**: 10-50 (Park et al. used 25)
    /// - **Angle-wise**: 20-100 (Park et al. used 50, typically 2× distance)
    /// - **Max samples**: 16-32 for efficiency (full batch is O(n³) for angles)</para>
    ///
    /// <para><b>Balancing with Output Loss:</b>
    /// The final loss combines:
    /// 1. Standard output distillation (α × hard + (1-α) × soft)
    /// 2. Distance-wise relational loss × distanceWeight
    /// 3. Angle-wise relational loss × angleWeight
    /// Tune weights so each component contributes meaningfully.</para>
    /// </remarks>
    public RelationalDistillationStrategy(
        double distanceWeight = 1.0,
        double angleWeight = 2.0,
        double temperature = 3.0,
        double alpha = 0.3,
        int maxSamplesPerBatch = 32,
        RelationalDistanceMetric distanceMetric = RelationalDistanceMetric.Euclidean)
        : base(temperature, alpha)
    {
        if (distanceWeight < 0)
            throw new ArgumentException("Distance weight must be non-negative", nameof(distanceWeight));
        if (angleWeight < 0)
            throw new ArgumentException("Angle weight must be non-negative", nameof(angleWeight));
        if (maxSamplesPerBatch < 2)
            throw new ArgumentException("Max samples per batch must be at least 2", nameof(maxSamplesPerBatch));

        _distanceWeight = distanceWeight;
        _angleWeight = angleWeight;
        _maxSamplesPerBatch = maxSamplesPerBatch;
        _distanceMetric = distanceMetric;
    }

    /// <summary>
    /// Resets the strategy's internal state.
    /// </summary>
    /// <remarks>
    /// <para><b>Note:</b> With Matrix&lt;T&gt; batch processing, this strategy no longer maintains
    /// state between calls, so Reset() is a no-op. It's kept for compatibility with the trainer's
    /// OnEpochEnd() which calls Reset() on all strategies.</para>
    /// </remarks>
    public void Reset()
    {
        // No-op: Matrix<T> batch processing doesn't maintain state between batches
    }

    /// <summary>
    /// Computes combined output loss and relational loss.
    /// </summary>
    /// <remarks>
    /// <para>This method accumulates student/teacher outputs and computes relational loss
    /// when a batch is complete. The relational loss is then amortized across subsequent samples.</para>
    ///
    /// <para><b>CRITICAL:</b> Call Reset() at epoch boundaries to prevent buffer leakage.</para>
    /// </remarks>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Collect all batch outputs for relational loss computation
        var studentOutputs = DistillationHelper<T>.RowsOf(studentBatchOutput);
        var teacherOutputs = DistillationHelper<T>.RowsOf(teacherBatchOutput);

        // Compute relational loss once for entire batch
        T relationalLoss = ComputeRelationalLoss(studentOutputs, teacherOutputs);

        // Standard loss for each sample plus the amortized relational loss
        var sampleLosses = DistillationHelper<T>.StandardSampleLosses(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha);
        T relationalContribution = NumOps.Divide(relationalLoss, NumOps.FromDouble(batchSize));
        T totalLoss = NumOps.Add(
            Engine.Sum(sampleLosses), NumOps.Multiply(relationalContribution, NumOps.FromDouble(batchSize)));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient of combined output loss and relational loss.
    /// </summary>
    /// <remarks>
    /// <para>The gradient includes both the standard distillation gradient and the relational gradient
    /// computed from pairwise distances and angular relationships in the accumulated batch.</para>
    /// </remarks>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Standard soft gradient, plus the hard gradient if labels are provided
        var standardGradient = DistillationHelper<T>.StandardSampleGradients(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha);

        // Add each sample's own relational gradient (NOT averaged — each sample
        // gets its own gradient based on its specific pairwise relationships)
        var perSampleRelGradients = ComputePerSampleRelationalGradients(
            DistillationHelper<T>.RowsOf(studentBatchOutput), DistillationHelper<T>.RowsOf(teacherBatchOutput));
        var gradientBatch = Engine.MatrixAdd(standardGradient, perSampleRelGradients.ToMatrix());

        // Average gradients over the batch
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        return Engine.MatrixMultiplyScalar(gradientBatch, oneOverBatch);
    }

    /// <summary>
    /// Computes each sample's analytical gradient of <see cref="ComputeRelationalLoss"/>.
    /// </summary>
    /// <param name="studentEmbeddings">Student's output embeddings/features for batch.</param>
    /// <param name="teacherEmbeddings">Teacher's output embeddings/features for batch.</param>
    /// <returns>[batch, dim] gradient; row i is ∂L_rel/∂studentEmbeddings[i]. Rows beyond the first
    /// maxSamplesPerBatch samples are zero, because the relational loss only looks at those samples.</returns>
    private Tensor<T> ComputePerSampleRelationalGradients(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        int batchSize = studentEmbeddings.Length;
        int dim = batchSize > 0 ? studentEmbeddings[0].Length : 0;
        if (batchSize < 2 || dim == 0)
            return new Tensor<T>([batchSize, dim]); // Need at least 2 samples for relations

        int n = Math.Min(batchSize, _maxSamplesPerBatch);
        var students = DistillationHelper<T>.StackRows(studentEmbeddings.Take(n).ToArray());
        var teachers = DistillationHelper<T>.StackRows(teacherEmbeddings.Take(n).ToArray());
        var gradient = new Tensor<T>([n, dim]);

        // Distance-wise gradient
        if (_distanceWeight > 0)
        {
            gradient = Engine.TensorAdd(gradient, Engine.TensorMultiplyScalar(
                ComputeDistanceWiseGradient(students, teachers), NumOps.FromDouble(_distanceWeight)));
        }

        // Angle-wise gradient
        if (_angleWeight > 0 && n >= 3)
        {
            gradient = Engine.TensorAdd(gradient, Engine.TensorMultiplyScalar(
                ComputeAngleWiseGradient(students, teachers), NumOps.FromDouble(_angleWeight)));
        }

        if (n == batchSize)
            return gradient;

        return Engine.TensorConcatenate([gradient, new Tensor<T>([batchSize - n, dim])], 0);
    }

    /// <summary>
    /// Gradient of <see cref="ComputeDistanceWiseLoss"/> with respect to every student row.
    /// </summary>
    /// <remarks>
    /// <para>Each pair (i, j) contributes h'(d_s(i, j) - d_t(i, j)) · ∂d_s(i, j)/∂x_i to row i, where h is
    /// the Huber loss (h' = 2·diff when |diff| &lt; 1, else 2·sign(diff)), normalized by the pair count.</para>
    /// </remarks>
    private Tensor<T> ComputeDistanceWiseGradient(Tensor<T> students, Tensor<T> teachers)
    {
        int n = students.Shape[0], dimensions = students.Shape[1];
        var wideStudents = students.Cast<double>();
        var wideTeachers = teachers.Cast<double>();
        double pairCount = n * (double)(n - 1) / 2;
        var blocks = new List<Tensor<double>>();
        for (int i = 0; i < n; i += DistillationPairOps.RowBlockSize)
        {
            int count = Math.Min(DistillationPairOps.RowBlockSize, n - i);
            var a = Engine.TensorNarrow(wideStudents, 0, i, count);
            var ta = Engine.TensorNarrow(wideTeachers, 0, i, count);
            var gradient = new Tensor<double>([count, dimensions]);
            for (int j = 0; j < n; j += DistillationPairOps.RowBlockSize)
            {
                int otherCount = Math.Min(DistillationPairOps.RowBlockSize, n - j);
                var b = Engine.TensorNarrow(wideStudents, 0, j, otherCount);
                var tb = Engine.TensorNarrow(wideTeachers, 0, j, otherCount);
                var distances = MetricDistances(a, b);
                var difference = Engine.TensorSubtract(distances, MetricDistances(ta, tb));
                var weights = OffDiagonalBlock(Engine.TensorMultiplyScalar(Engine.TensorWhere(
                    Engine.TensorLessThan(Engine.TensorAbs(difference), 1.0), difference, Engine.TensorSign(difference)), 2.0 / pairCount), i, j);
                Tensor<double> contribution;
                if (_distanceMetric == RelationalDistanceMetric.Cosine)
                {
                    var dots = Engine.TensorMatMul(a, Engine.TensorTranspose(b));
                    var normA = Engine.ReduceSum(Engine.TensorMultiply(a, a), [1], keepDims: true);
                    var normB = Engine.ReduceSum(Engine.TensorMultiply(b, b), [1], keepDims: true);
                    var denominator = Engine.TensorSqrt(Engine.TensorMatMul(normA, Engine.TensorTranspose(normB)));
                    var positive = Engine.TensorGreaterThan(denominator, 1e-10);
                    var safe = Engine.TensorWhere(positive, denominator, Tensor<double>.CreateDefault([count, otherCount], 1.0));
                    var valid = Engine.TensorMultiply(positive, Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorDivide(dots, safe)), 1.0));
                    var zeros = new Tensor<double>([count, otherCount]);
                    var cross = Engine.TensorWhere(valid, Engine.TensorDivide(weights, safe), zeros);
                    var normProduct = Engine.TensorMultiply(safe, Engine.TensorBroadcastTo(normA, safe.Shape.ToArray()));
                    normProduct = Engine.TensorWhere(valid, normProduct, Tensor<double>.CreateDefault([count, otherCount], 1.0));
                    var self = Engine.TensorWhere(valid, Engine.TensorDivide(Engine.TensorMultiply(weights, dots), normProduct), zeros);
                    contribution = Engine.TensorSubtract(
                        Engine.TensorMultiply(a, Engine.TensorBroadcastTo(Engine.ReduceSum(self, [1], keepDims: true), a.Shape.ToArray())),
                        Engine.TensorMatMul(cross, b));
                }
                else
                {
                    if (_distanceMetric == RelationalDistanceMetric.Euclidean)
                    {
                        var positive = Engine.TensorGreaterThan(distances, 0.0);
                        var safe = Engine.TensorWhere(positive, distances, Tensor<double>.CreateDefault([count, otherCount], 1.0));
                        weights = Engine.TensorWhere(positive, Engine.TensorDivide(weights, safe), new Tensor<double>([count, otherCount]));
                    }
                    contribution = DistillationPairOps.WeightedDifferences(a, b, weights, _distanceMetric == RelationalDistanceMetric.Manhattan);
                }
                gradient = Engine.TensorAdd(gradient, contribution);
            }
            blocks.Add(gradient);
        }
        return Engine.TensorConcatenate(blocks.ToArray(), 0).Cast<T>();
    }

    /// <summary>
    /// Gradient of <see cref="ComputeAngleWiseLoss"/> with respect to every student row.
    /// </summary>
    /// <remarks>
    /// <para>For a triplet with sides A = x_i - x_j and B = x_k - x_j, θ = arccos(cos) with
    /// cos = (A · B) / (||A|| ||B|| + ε), so ∂θ/∂A = -1/sin(θ) · [B/D - (A · B) ||B|| A / (||A|| D²)] with
    /// D = ||A|| ||B|| + ε (and symmetrically for B). Row i receives ∂/∂A, row k receives ∂/∂B and the
    /// vertex row j receives -(∂/∂A + ∂/∂B). Clamped cosines contribute nothing.</para>
    /// </remarks>
    private Tensor<T> ComputeAngleWiseGradient(Tensor<T> students, Tensor<T> teachers)
    {
        int n = students.Shape[0];
        int dim = students.Shape[1];
        var (first, vertex, third) = SampleTriplets(n);
        int tripletCount = first.Length;
        if (tripletCount == 0)
            return new Tensor<T>([n, dim]);

        var i = new Tensor<int>(first, [tripletCount]);
        var j = new Tensor<int>(vertex, [tripletCount]);
        var k = new Tensor<int>(third, [tripletCount]);
        var ones = Tensor<T>.CreateDefault([tripletCount], NumOps.One);
        var zeros = new Tensor<T>([tripletCount]);

        var vertexRows = Engine.TensorIndexSelect(students, j, 0);
        var sideA = Engine.TensorSubtract(Engine.TensorIndexSelect(students, i, 0), vertexRows);
        var sideB = Engine.TensorSubtract(Engine.TensorIndexSelect(students, k, 0), vertexRows);
        var dot = Engine.ReduceSum(Engine.TensorMultiply(sideA, sideB), [1], keepDims: false);
        var normA = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(sideA, sideA), [1], keepDims: false));
        var normB = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(sideB, sideB), [1], keepDims: false));
        var denom = Engine.TensorAddScalar(Engine.TensorMultiply(normA, normB), NumOps.FromDouble(Epsilon));
        var rawCosine = Engine.TensorDivide(dot, denom);
        var cosine = Engine.TensorClamp(rawCosine, NumOps.FromDouble(-1.0), NumOps.One);
        var angleDiff = Engine.TensorSubtract(Engine.TensorAcos(cosine), TripletAngles(teachers, i, j, k));

        // dθ/dcos = -1/sqrt(1 - cos²) inside (-1, 1); zero where the clamp is active
        var inside = Engine.TensorLessThan(Engine.TensorAbs(rawCosine), NumOps.One);
        var angleSlope = Engine.TensorWhere(
            inside,
            Engine.TensorNegate(Engine.TensorDivide(
                ones, Engine.TensorSqrt(Engine.TensorSubtract(ones, Engine.TensorMultiply(cosine, cosine))))),
            zeros);

        // Chain rule for the mean of (θ_s - θ_t)²
        var coefficient = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(angleDiff, angleSlope), NumOps.FromDouble(2.0 / tripletCount));

        // Zero-length sides contribute no direction; guard their norms against division by zero
        var safeNormA = Engine.TensorWhere(Engine.TensorGreaterThan(normA, NumOps.Zero), normA, ones);
        var safeNormB = Engine.TensorWhere(Engine.TensorGreaterThan(normB, NumOps.Zero), normB, ones);
        var denomSquared = Engine.TensorMultiply(denom, denom);
        var crossScale = Column(Engine.TensorDivide(coefficient, denom), dim);
        var selfScaleA = Column(Engine.TensorDivide(
            Engine.TensorMultiply(Engine.TensorMultiply(coefficient, dot), normB),
            Engine.TensorMultiply(safeNormA, denomSquared)), dim);
        var selfScaleB = Column(Engine.TensorDivide(
            Engine.TensorMultiply(Engine.TensorMultiply(coefficient, dot), normA),
            Engine.TensorMultiply(safeNormB, denomSquared)), dim);
        var gradA = Engine.TensorSubtract(Engine.TensorMultiply(crossScale, sideB), Engine.TensorMultiply(selfScaleA, sideA));
        var gradB = Engine.TensorSubtract(Engine.TensorMultiply(crossScale, sideA), Engine.TensorMultiply(selfScaleB, sideB));

        // Scatter back to the rows: one-hot selectors [triplets, n] transposed onto the triplet gradients
        var identity = Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(n));
        var toI = Engine.TensorTranspose(Engine.TensorIndexSelect(identity, i, 0));
        var toJ = Engine.TensorTranspose(Engine.TensorIndexSelect(identity, j, 0));
        var toK = Engine.TensorTranspose(Engine.TensorIndexSelect(identity, k, 0));

        return Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorMatMul(toI, gradA), Engine.TensorMatMul(toK, gradB)),
            Engine.TensorMatMul(toJ, Engine.TensorAdd(gradA, gradB)));
    }
    /// <summary>
    /// Computes relational knowledge distillation loss for a batch of samples.
    /// </summary>
    /// <param name="studentEmbeddings">Student's output embeddings/features for batch.</param>
    /// <param name="teacherEmbeddings">Teacher's output embeddings/features for batch.</param>
    /// <returns>Combined relational loss (distance + angle).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes how well the student preserves the teacher's
    /// relational structure. Pass in the embeddings/features (not final classifications).</para>
    ///
    /// <para>The loss has two components:
    /// 1. **Distance-wise**: Do pairs of samples have similar distances?
    /// 2. **Angle-wise**: Do triplets of samples have similar angular relationships?</para>
    ///
    /// <para>Example usage:
    /// <code>
    /// // Get embeddings from penultimate layer
    /// var studentEmbeds = new Vector&lt;double&gt;[] { student.GetEmbedding(x1), student.GetEmbedding(x2), ... };
    /// var teacherEmbeds = new Vector&lt;double&gt;[] { teacher.GetEmbedding(x1), teacher.GetEmbedding(x2), ... };
    ///
    /// var relationalLoss = strategy.ComputeRelationalLoss(studentEmbeds, teacherEmbeds);
    /// </code>
    /// </para>
    /// </remarks>
    public T ComputeRelationalLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        if (studentEmbeddings == null || teacherEmbeddings == null)
            throw new ArgumentNullException("Embeddings cannot be null");
        if (studentEmbeddings.Length != teacherEmbeddings.Length)
            throw new ArgumentException("Student and teacher must have same number of samples");
        if (studentEmbeddings.Length < 2)
            return NumOps.Zero; // Need at least 2 samples for relations

        // Validate all vectors have same dimensions
        if (studentEmbeddings.Length > 0)
        {
            int studentDim = studentEmbeddings[0].Length;
            int teacherDim = teacherEmbeddings[0].Length;

            for (int i = 0; i < studentEmbeddings.Length; i++)
            {
                if (studentEmbeddings[i].Length != studentDim)
                    throw new ArgumentException($"All student embeddings must have same dimension. Expected {studentDim}, got {studentEmbeddings[i].Length} at index {i}");
                if (teacherEmbeddings[i].Length != teacherDim)
                    throw new ArgumentException($"All teacher embeddings must have same dimension. Expected {teacherDim}, got {teacherEmbeddings[i].Length} at index {i}");
                if (studentEmbeddings[i].Length != teacherEmbeddings[i].Length)
                    throw new ArgumentException($"Student and teacher embeddings must have matching dimensions. Got student={studentEmbeddings[i].Length}, teacher={teacherEmbeddings[i].Length} at index {i}");
                if (studentEmbeddings[i].Length == 0)
                    throw new ArgumentException($"Embedding vectors cannot be empty at index {i}");
            }
        }

        int n = Math.Min(studentEmbeddings.Length, _maxSamplesPerBatch);
        T totalLoss = NumOps.Zero;

        // Distance-wise loss
        if (_distanceWeight > 0)
        {
            T distLoss = ComputeDistanceWiseLoss(studentEmbeddings, teacherEmbeddings, n);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(distLoss, NumOps.FromDouble(_distanceWeight)));
        }

        // Angle-wise loss
        if (_angleWeight > 0 && n >= 3)
        {
            T angleLoss = ComputeAngleWiseLoss(studentEmbeddings, teacherEmbeddings, n);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(angleLoss, NumOps.FromDouble(_angleWeight)));
        }

        return totalLoss;
    }

    /// <summary>
    /// Computes distance-wise relational loss (preserves pairwise distances).
    /// </summary>
    private T ComputeDistanceWiseLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings, int n)
    {
        if (n < 2) return NumOps.Zero;
        var students = DistillationHelper<T>.StackRows(studentEmbeddings.Take(n).ToArray()).Cast<double>();
        var teachers = DistillationHelper<T>.StackRows(teacherEmbeddings.Take(n).ToArray()).Cast<double>();
        double sum = 0;
        for (int i = 0; i < n; i += DistillationPairOps.RowBlockSize)
        {
            int count = Math.Min(DistillationPairOps.RowBlockSize, n - i);
            var a = Engine.TensorNarrow(students, 0, i, count);
            var ta = Engine.TensorNarrow(teachers, 0, i, count);
            for (int j = 0; j < n; j += DistillationPairOps.RowBlockSize)
            {
                int otherCount = Math.Min(DistillationPairOps.RowBlockSize, n - j);
                var b = Engine.TensorNarrow(students, 0, j, otherCount);
                var tb = Engine.TensorNarrow(teachers, 0, j, otherCount);
                var difference = Engine.TensorSubtract(MetricDistances(a, b), MetricDistances(ta, tb));
                var absolute = Engine.TensorAbs(difference);
                var loss = Engine.TensorWhere(Engine.TensorLessThan(absolute, 1.0),
                    Engine.TensorMultiply(difference, difference),
                    Engine.TensorSubtractScalar(Engine.TensorMultiplyScalar(absolute, 2.0), 1.0));
                sum += Engine.TensorSum(OffDiagonalBlock(loss, i, j));
            }
        }
        return NumOps.FromDouble(sum / (n * (double)(n - 1)));
    }

    /// <summary>
    /// Computes angle-wise relational loss (preserves angular relationships).
    /// </summary>
    private T ComputeAngleWiseLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings, int n)
    {
        var (first, vertex, third) = SampleTriplets(n);
        int tripletCount = first.Length;
        if (tripletCount == 0)
            return NumOps.Zero;

        var i = new Tensor<int>(first, [tripletCount]);
        var j = new Tensor<int>(vertex, [tripletCount]);
        var k = new Tensor<int>(third, [tripletCount]);
        var studentAngles = TripletAngles(DistillationHelper<T>.StackRows(studentEmbeddings.Take(n).ToArray()), i, j, k);
        var teacherAngles = TripletAngles(DistillationHelper<T>.StackRows(teacherEmbeddings.Take(n).ToArray()), i, j, k);

        // Squared difference in angles
        var diff = Engine.TensorSubtract(studentAngles, teacherAngles);
        T loss = Engine.TensorSum(Engine.TensorMultiply(diff, diff));

        return NumOps.Divide(loss, NumOps.FromDouble(tripletCount));
    }

    /// <summary>
    /// Samples the triplets (i, j, k) used by the angle-wise loss with a fixed seed, so the loss and
    /// its gradient always see the same triplets.
    /// </summary>
    private static (int[] First, int[] Vertex, int[] Third) SampleTriplets(int n)
    {
        var first = new List<int>();
        var vertex = new List<int>();
        var third = new List<int>();

        // Sample triplets (all combinations would be O(n³))
        int maxTriplets = Math.Min(n * n, 1000); // Limit for efficiency
        var random = RandomHelper.CreateSeededRandom(42);

        for (int t = 0; t < maxTriplets; t++)
        {
            // Randomly sample triplet
            int i = random.Next(n);
            int j = random.Next(n);
            int k = random.Next(n);

            if (i == j || j == k || i == k)
                continue;

            first.Add(i);
            vertex.Add(j);
            third.Add(k);
        }

        return (first.ToArray(), vertex.ToArray(), third.ToArray());
    }

    /// <summary>
    /// [n, n] distances between every pair of rows, based on the selected metric.
    /// </summary>
    private Tensor<double> MetricDistances(Tensor<double> a, Tensor<double> b)
    {
        if (_distanceMetric == RelationalDistanceMetric.Cosine)
        {
            var cosine = DistillationHelper<double>.CosineSimilarityMatrix(a, b);
            return Engine.TensorSubtract(Tensor<double>.CreateDefault(cosine.Shape.ToArray(), 1.0), cosine);
        }
        if (_distanceMetric != RelationalDistanceMetric.Euclidean && _distanceMetric != RelationalDistanceMetric.Manhattan)
            throw new ArgumentOutOfRangeException(nameof(_distanceMetric));
        var distance = DistillationPairOps.DistanceBlock(a, b, _distanceMetric == RelationalDistanceMetric.Manhattan);
        return _distanceMetric == RelationalDistanceMetric.Manhattan ? distance : Engine.TensorSqrt(distance);
    }

    private Tensor<double> OffDiagonalBlock(Tensor<double> values, int rowStart, int columnStart)
    {
        var rows = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<int>(rowStart, rowStart + values.Shape[0], 1), [values.Shape[0], 1]), values.Shape.ToArray());
        var columns = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<int>(columnStart, columnStart + values.Shape[1], 1), [1, values.Shape[1]]), values.Shape.ToArray());
        return Engine.TensorWhere(Engine.TensorNotEquals(rows, columns).Cast<double>(), values, new Tensor<double>(values.Shape.ToArray()));
    }

    // Pair differences are assembled only inside bounded row/feature blocks.

    /// <summary>
    /// Broadcasts a length-m vector tensor to an [m, width] tensor whose row r repeats element r.
    /// </summary>
    private Tensor<T> Column(Tensor<T> values, int width)
    {
        return Engine.TensorBroadcastTo(Engine.Reshape(values, [values.Length, 1]), [values.Length, width]);
    }

    /// <summary>
    /// Computes the angle (in radians) at vertex j of each triplet (i, j, k) of rows.
    /// </summary>
    private Tensor<T> TripletAngles(Tensor<T> rows, Tensor<int> i, Tensor<int> j, Tensor<int> k)
    {
        // Vectors from j to i and j to k
        var vertexRows = Engine.TensorIndexSelect(rows, j, 0);
        var ji = Engine.TensorSubtract(Engine.TensorIndexSelect(rows, i, 0), vertexRows);
        var jk = Engine.TensorSubtract(Engine.TensorIndexSelect(rows, k, 0), vertexRows);

        // Dot product and norms
        var dot = Engine.ReduceSum(Engine.TensorMultiply(ji, jk), [1], keepDims: false);
        var normJi = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(ji, ji), [1], keepDims: false));
        var normJk = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(jk, jk), [1], keepDims: false));

        // cos(angle) = dot / (norm1 * norm2), clamped to [-1, 1] for numerical stability
        var cosAngle = Engine.TensorDivide(
            dot, Engine.TensorAddScalar(Engine.TensorMultiply(normJi, normJk), NumOps.FromDouble(Epsilon)));
        cosAngle = Engine.TensorClamp(cosAngle, NumOps.FromDouble(-1.0), NumOps.One);

        return Engine.TensorAcos(cosAngle); // Return angle in radians
    }






}

/// <summary>
/// Distance metrics for relational knowledge distillation.
/// </summary>
public enum RelationalDistanceMetric
{
    /// <summary>
    /// Euclidean (L2) distance - most common, geometrically intuitive.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Cosine distance (1 - cosine similarity) - good for high-dimensional embeddings.
    /// </summary>
    Cosine,

    /// <summary>
    /// Manhattan (L1) distance - more robust to outliers.
    /// </summary>
    Manhattan
}


