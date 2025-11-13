using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
/// - Tung & Mori (2019). Similarity-Preserving Knowledge Distillation. ICCV.
/// - Peng et al. (2019). Correlation Congruence for Knowledge Distillation. ICCV.</para>
/// </remarks>
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
    /// Computes combined output loss and relational loss for a batch.
    /// </summary>
    /// <remarks>
    /// <para>This method computes both standard distillation loss and relational loss
    /// directly on the batch. Relational loss is computed from the entire batch and
    /// distributed equally across all samples, fixing the previous bug where relational
    /// loss was applied to the wrong samples.</para>
    /// </remarks>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.RowCount;
        T totalLoss = NumOps.Zero;

        // Compute standard distillation loss for each sample
        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            // Standard distillation loss (soft targets with temperature scaling)
            var studentSoft = Softmax(studentRow, Temperature);
            var teacherSoft = Softmax(teacherRow, Temperature);
            var softLoss = KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            if (labelRow != null)
            {
                var studentProbs = Softmax(studentRow, 1.0);
                var hardLoss = CrossEntropy(studentProbs, labelRow);
                var sampleLoss = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
                totalLoss = NumOps.Add(totalLoss, sampleLoss);
            }
            else
            {
                totalLoss = NumOps.Add(totalLoss, softLoss);
            }
        }

        // Compute relational loss for the entire batch
        var studentEmbeddings = new Vector<T>[batchSize];
        var teacherEmbeddings = new Vector<T>[batchSize];
        for (int r = 0; r < batchSize; r++)
        {
            studentEmbeddings[r] = studentBatchOutput.GetRow(r);
            teacherEmbeddings[r] = teacherBatchOutput.GetRow(r);
        }

        T relationalLoss = ComputeRelationalLoss(studentEmbeddings, teacherEmbeddings);

        // Add relational loss to total (already accounts for batch size internally)
        totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(relationalLoss, NumOps.FromDouble(batchSize)));

        // Return average loss over batch
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient of combined output loss and relational loss for a batch.
    /// </summary>
    /// <remarks>
    /// <para>The gradient includes both the standard distillation gradient and the relational gradient
    /// computed from pairwise distances and angular relationships in the batch.</para>
    /// </remarks>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.RowCount;
        int numClasses = studentBatchOutput.ColumnCount;
        var gradient = new Matrix<T>(batchSize, numClasses);

        // Compute standard distillation gradient for each sample
        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            var studentSoft = Softmax(studentRow, Temperature);
            var teacherSoft = Softmax(teacherRow, Temperature);

            for (int c = 0; c < numClasses; c++)
            {
                var diff = NumOps.Subtract(studentSoft[c], teacherSoft[c]);
                gradient[r, c] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
            }

            if (labelRow != null)
            {
                var studentProbs = Softmax(studentRow, 1.0);

                for (int c = 0; c < numClasses; c++)
                {
                    var hardGrad = NumOps.Subtract(studentProbs[c], labelRow[c]);
                    gradient[r, c] = NumOps.Add(
                        NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                        NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[r, c]));
                }
            }
        }

        // Compute relational gradients for the entire batch
        var studentEmbeddings = new Vector<T>[batchSize];
        var teacherEmbeddings = new Vector<T>[batchSize];
        for (int r = 0; r < batchSize; r++)
        {
            studentEmbeddings[r] = studentBatchOutput.GetRow(r);
            teacherEmbeddings[r] = teacherBatchOutput.GetRow(r);
        }

        // Add relational gradient contribution to each sample
        var relationalGradients = ComputeBatchRelationalGradients(studentEmbeddings, teacherEmbeddings);
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                gradient[r, c] = NumOps.Add(gradient[r, c], relationalGradients[r][c]);
            }
        }

        // Average gradients over batch
        T oneOverBatchSize = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                gradient[r, c] = NumOps.Multiply(gradient[r, c], oneOverBatchSize);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes relational gradients for all samples in a batch.
    /// </summary>
    /// <param name="studentEmbeddings">Student's output embeddings/features for batch.</param>
    /// <param name="teacherEmbeddings">Teacher's output embeddings/features for batch.</param>
    /// <returns>Array of gradient vectors, one for each sample in the batch.</returns>
    private Vector<T>[] ComputeBatchRelationalGradients(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings)
    {
        int n = studentEmbeddings.Length;
        if (n == 0)
            return new Vector<T>[0];

        int dim = studentEmbeddings[0].Length;
        var gradients = new Vector<T>[n];

        // Initialize all gradients to zero
        for (int i = 0; i < n; i++)
        {
            gradients[i] = new Vector<T>(dim);
            for (int d = 0; d < dim; d++)
            {
                gradients[i][d] = NumOps.Zero;
            }
        }

        if (n < 2)
            return gradients;

        // Compute distance-wise gradients for all pairs
        if (_distanceWeight > 0)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;

                    var distGrad = ComputePairwiseDistanceGradient(
                        studentEmbeddings[i],
                        studentEmbeddings[j],
                        teacherEmbeddings[i],
                        teacherEmbeddings[j]);

                    for (int d = 0; d < dim; d++)
                    {
                        var weighted = NumOps.Multiply(distGrad[d], NumOps.FromDouble(_distanceWeight));
                        gradients[i][d] = NumOps.Add(gradients[i][d], weighted);
                    }
                }
            }
        }

        // Compute angle-wise gradients for triplets
        if (_angleWeight > 0 && n >= 3)
        {
            int maxTriplets = Math.Min(10, n);
            for (int i = 0; i < n; i++)
            {
                for (int t = 0; t < maxTriplets; t++)
                {
                    int j = (i + t + 1) % n;
                    int k = (i + t + 2) % n;

                    if (j == i || k == i || j == k)
                        continue;

                    var angleGrad = ComputeTripletAngleGradient(
                        studentEmbeddings[i],
                        studentEmbeddings[j],
                        studentEmbeddings[k],
                        teacherEmbeddings[i],
                        teacherEmbeddings[j],
                        teacherEmbeddings[k]);

                    for (int d = 0; d < dim; d++)
                    {
                        var weighted = NumOps.Multiply(angleGrad[d], NumOps.FromDouble(_angleWeight));
                        gradients[i][d] = NumOps.Add(gradients[i][d], weighted);
                    }
                }
            }
        }

        return gradients;
    }

    /// <summary>
    /// Computes gradient of distance-wise loss for a pair.
    /// </summary>
    private Vector<T> ComputePairwiseDistanceGradient(
        Vector<T> studentI,
        Vector<T> studentJ,
        Vector<T> teacherI,
        Vector<T> teacherJ)
    {
        int dim = studentI.Length;
        var gradient = new Vector<T>(dim);

        var studentDist = ComputeDistance(studentI, studentJ);
        var teacherDist = ComputeDistance(teacherI, teacherJ);

        var diff = NumOps.Subtract(studentDist, teacherDist);
        double diffVal = Convert.ToDouble(diff);

        // Huber loss gradient
        double gradScale;
        if (Math.Abs(diffVal) < 1.0)
        {
            gradScale = 2.0 * diffVal; // Quadratic region
        }
        else
        {
            gradScale = 2.0 * Math.Sign(diffVal); // Linear region
        }

        // Gradient of distance w.r.t. studentI
        double distVal = Convert.ToDouble(studentDist) + Epsilon;
        for (int k = 0; k < dim; k++)
        {
            double component = Convert.ToDouble(NumOps.Subtract(studentI[k], studentJ[k]));
            gradient[k] = NumOps.FromDouble(gradScale * component / distVal);
        }

        return gradient;
    }

    /// <summary>
    /// Computes gradient of angle-wise loss for a triplet (simplified approximation).
    /// </summary>
    private Vector<T> ComputeTripletAngleGradient(
        Vector<T> studentI,
        Vector<T> studentJ,
        Vector<T> studentK,
        Vector<T> teacherI,
        Vector<T> teacherJ,
        Vector<T> teacherK)
    {
        int dim = studentI.Length;
        var gradient = new Vector<T>(dim);

        var studentAngle = ComputeAngle(studentI, studentJ, studentK);
        var teacherAngle = ComputeAngle(teacherI, teacherJ, teacherK);

        var angleDiff = NumOps.Subtract(studentAngle, teacherAngle);
        double diffVal = Convert.ToDouble(angleDiff);

        // Numerical gradient approximation (for simplicity)
        double eps = 0.001;
        for (int d = 0; d < dim; d++)
        {
            var perturbed = new Vector<T>(dim);
            for (int k = 0; k < dim; k++)
            {
                perturbed[k] = k == d
                    ? NumOps.Add(studentI[k], NumOps.FromDouble(eps))
                    : studentI[k];
            }

            var perturbedAngle = ComputeAngle(perturbed, studentJ, studentK);
            var angleGrad = NumOps.Subtract(perturbedAngle, studentAngle);
            var numGrad = NumOps.Divide(angleGrad, NumOps.FromDouble(eps));

            gradient[d] = NumOps.Multiply(numGrad, NumOps.FromDouble(2.0 * diffVal));
        }

        return gradient;
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
        T loss = NumOps.Zero;
        int pairCount = 0;

        // Compute all pairwise distances for both student and teacher
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                var studentDist = ComputeDistance(studentEmbeddings[i], studentEmbeddings[j]);
                var teacherDist = ComputeDistance(teacherEmbeddings[i], teacherEmbeddings[j]);

                // Huber loss for robust distance matching
                var diff = NumOps.Subtract(studentDist, teacherDist);
                var absDiff = Math.Abs(Convert.ToDouble(diff));

                T pairLoss;
                if (absDiff < 1.0)
                {
                    // Quadratic for small differences
                    pairLoss = NumOps.Multiply(diff, diff);
                }
                else
                {
                    // Linear for large differences (more robust)
                    pairLoss = NumOps.FromDouble(2.0 * absDiff - 1.0);
                }

                loss = NumOps.Add(loss, pairLoss);
                pairCount++;
            }
        }

        // Normalize by number of pairs
        return pairCount > 0 ? NumOps.Divide(loss, NumOps.FromDouble(pairCount)) : NumOps.Zero;
    }

    /// <summary>
    /// Computes angle-wise relational loss (preserves angular relationships).
    /// </summary>
    private T ComputeAngleWiseLoss(Vector<T>[] studentEmbeddings, Vector<T>[] teacherEmbeddings, int n)
    {
        T loss = NumOps.Zero;
        int tripletCount = 0;

        // Sample triplets (all combinations would be O(n³))
        int maxTriplets = Math.Min(n * n, 1000); // Limit for efficiency
        var random = new Random(42);

        for (int t = 0; t < maxTriplets; t++)
        {
            // Randomly sample triplet
            int i = random.Next(n);
            int j = random.Next(n);
            int k = random.Next(n);

            if (i == j || j == k || i == k)
                continue;

            var studentAngle = ComputeAngle(studentEmbeddings[i], studentEmbeddings[j], studentEmbeddings[k]);
            var teacherAngle = ComputeAngle(teacherEmbeddings[i], teacherEmbeddings[j], teacherEmbeddings[k]);

            // Squared difference in angles
            var diff = NumOps.Subtract(studentAngle, teacherAngle);
            var squaredDiff = NumOps.Multiply(diff, diff);
            loss = NumOps.Add(loss, squaredDiff);
            tripletCount++;
        }

        return tripletCount > 0 ? NumOps.Divide(loss, NumOps.FromDouble(tripletCount)) : NumOps.Zero;
    }

    /// <summary>
    /// Computes distance between two vectors based on the selected metric.
    /// </summary>
    private T ComputeDistance(Vector<T> v1, Vector<T> v2)
    {
        switch (_distanceMetric)
        {
            case RelationalDistanceMetric.Euclidean:
                return ComputeEuclideanDistance(v1, v2);

            case RelationalDistanceMetric.Cosine:
                return ComputeCosineDistance(v1, v2);

            case RelationalDistanceMetric.Manhattan:
                return ComputeManhattanDistance(v1, v2);

            default:
                throw new NotImplementedException($"Distance metric {_distanceMetric} not implemented");
        }
    }

    private T ComputeEuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sum)));
    }

    private T ComputeCosineDistance(Vector<T> v1, Vector<T> v2)
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

        double similarity = Convert.ToDouble(dot) /
            (Math.Sqrt(Convert.ToDouble(norm1)) * Math.Sqrt(Convert.ToDouble(norm2)) + Epsilon);

        return NumOps.FromDouble(1.0 - similarity); // Distance = 1 - similarity
    }

    private T ComputeManhattanDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = Math.Abs(Convert.ToDouble(NumOps.Subtract(v1[i], v2[i])));
            sum = NumOps.Add(sum, NumOps.FromDouble(diff));
        }
        return sum;
    }

    /// <summary>
    /// Computes angle between three points (angle at point j).
    /// </summary>
    private T ComputeAngle(Vector<T> vi, Vector<T> vj, Vector<T> vk)
    {
        // Vectors from j to i and j to k
        var ji = Subtract(vi, vj);
        var jk = Subtract(vk, vj);

        // Dot product and norms
        T dot = DotProduct(ji, jk);
        T normJi = Norm(ji);
        T normJk = Norm(jk);

        // cos(angle) = dot / (norm1 * norm2)
        double cosAngle = Convert.ToDouble(dot) /
            (Convert.ToDouble(normJi) * Convert.ToDouble(normJk) + Epsilon);

        // Clamp to [-1, 1] for numerical stability
        cosAngle = Math.Max(-1.0, Math.Min(1.0, cosAngle));

        return NumOps.FromDouble(Math.Acos(cosAngle)); // Return angle in radians
    }

    private Vector<T> Subtract(Vector<T> v1, Vector<T> v2)
    {
        var result = new Vector<T>(v1.Length);
        for (int i = 0; i < v1.Length; i++)
        {
            result[i] = NumOps.Subtract(v1[i], v2[i]);
        }
        return result;
    }

    private T DotProduct(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(v1[i], v2[i]));
        }
        return sum;
    }

    private T Norm(Vector<T> v)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(v[i], v[i]));
        }
        return NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sum)));
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
