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
public class RelationalDistillationStrategy<T> : DistillationStrategyBase<Vector<T>, T>
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
    /// Computes standard output loss (relational loss computed separately).
    /// </summary>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // Standard distillation loss
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

            return NumOps.Add(
                NumOps.Multiply(alphaT, hardLoss),
                NumOps.Multiply(oneMinusAlpha, softLoss));
        }

        return softLoss;
    }

    /// <summary>
    /// Computes gradient of output loss.
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
                var absDiff = Math.Abs(NumOps.ToDouble(diff));

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
        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sum)));
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

        double similarity = NumOps.ToDouble(dot) /
            (Math.Sqrt(NumOps.ToDouble(norm1)) * Math.Sqrt(NumOps.ToDouble(norm2)) + Epsilon);

        return NumOps.FromDouble(1.0 - similarity); // Distance = 1 - similarity
    }

    private T ComputeManhattanDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = Math.Abs(NumOps.ToDouble(NumOps.Subtract(v1[i], v2[i])));
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
        double cosAngle = NumOps.ToDouble(dot) /
            (NumOps.ToDouble(normJi) * NumOps.ToDouble(normJk) + Epsilon);

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
        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sum)));
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);

        var scaledLogits = new T[n];
        for (int i = 0; i < n; i++)
        {
            scaledLogits[i] = NumOps.FromDouble(NumOps.ToDouble(logits[i]) / temperature);
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
            double val = NumOps.ToDouble(NumOps.Subtract(scaledLogits[i], maxLogit));
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
