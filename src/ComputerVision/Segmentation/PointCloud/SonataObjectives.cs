using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Turns the teacher's prototype scores into the soft assignment targets the student is trained to
/// predict (Sonata uses Sinkhorn-Knopp centering, <see cref="SinkhornKnoppCentering{T}"/>).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Implement this to replace how the teacher's answers are turned into targets,
/// for example with DINO-style centering and sharpening.
/// </remarks>
public interface ISonataTargetCentering<T>
{
    /// <summary>Returns <c>[matches, prototypes]</c> targets whose rows sum to one.</summary>
    /// <param name="teacherScores">Teacher prototype scores <c>[matches, prototypes]</c>.</param>
    /// <param name="temperature">The teacher temperature at this step.</param>
    Tensor<T> Center(Tensor<T> teacherScores, double temperature);
}

/// <summary>
/// The per-match distillation loss between student scores and teacher targets (Sonata uses the
/// cluster cross-entropy, <see cref="SonataClusterCrossEntropyLoss{T}"/>).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface ISonataDistillationLoss<T>
{
    /// <summary>Returns the scalar loss (shape <c>[1]</c>) over all matches.</summary>
    /// <param name="studentScores">Student prototype scores <c>[matches, prototypes]</c>.</param>
    /// <param name="targets">Teacher targets <c>[matches, prototypes]</c>, rows summing to one.</param>
    /// <param name="studentTemperature">The student softmax temperature.</param>
    /// <param name="sampleOfMatch">The scene each match belongs to, for per-scene averaging.</param>
    Tensor<T> Compute(Tensor<T> studentScores, Tensor<T> targets, double studentTemperature, int[] sampleOfMatch);
}

/// <summary>
/// An extra regulariser on the student's point embeddings, added to the distillation loss
/// (for example the KoLeo spreading term, <see cref="KoLeoRegularizer{T}"/>).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface ISonataFeatureRegularizer<T>
{
    /// <summary>Returns the scalar regulariser (shape <c>[1]</c>) for <c>[points, channels]</c> embeddings.</summary>
    /// <param name="embeddings">The student's embeddings.</param>
    Tensor<T> Compute(Tensor<T> embeddings);
}

/// <summary>
/// Sinkhorn-Knopp centering (Caron et al. 2020; Sonata <c>sinkhorn_knopp</c>): alternately normalises
/// the prototypes and the samples of <c>exp(scores / temperature)</c> so every prototype receives the
/// same total mass and every sample's row is a distribution.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Runs in single precision regardless of <typeparamref name="T"/>, as Sonata does
/// (<c>feat = feat.float()</c>): <c>exp(1 / 0.04)</c> is beyond half precision. The global maximum is
/// subtracted before the exponential, which the first normalisation cancels exactly. The result
/// carries no gradient: it is the teacher's target.
/// </para>
/// </remarks>
public sealed class SinkhornKnoppCentering<T> : ISonataTargetCentering<T>
{
    /// <summary>Creates the centering with the given number of iterations.</summary>
    /// <param name="iterations">Sinkhorn iterations (Sonata: 3).</param>
    public SinkhornKnoppCentering(int iterations = 3)
    {
        if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations), "At least one iteration is required.");
        Iterations = iterations;
    }

    /// <summary>The number of row/column normalisation rounds.</summary>
    public int Iterations { get; }

    /// <inheritdoc />
    public Tensor<T> Center(Tensor<T> teacherScores, double temperature)
    {
        if (teacherScores is null) throw new ArgumentNullException(nameof(teacherScores));
        if (teacherScores.Rank != 2) throw new ArgumentException("Teacher scores must be [matches, prototypes].", nameof(teacherScores));
        if (!(temperature > 0)) throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature must be positive.");
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = teacherScores.Shape[0], k = teacherScores.Shape[1];
        var values = new float[n * k];
        numOps.ToFloatSpan(teacherScores.ToArray(), values);
        var q = Normalize(new Tensor<float>(values, new[] { n, k }), temperature, Iterations);
        var result = new T[n * k];
        numOps.FromFloatSpan(q.ToArray(), result);
        return new Tensor<T>(result, new[] { n, k });
    }

    /// <summary>The Sinkhorn-Knopp iteration on <c>[samples, prototypes]</c> scores, in float.</summary>
    internal static Tensor<float> Normalize(Tensor<float> scores, double temperature, int iterations)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<float>();
        int n = scores.Shape[0], k = scores.Shape[1];
        var shape = new[] { n, k };
        float max = engine.TensorMaxValue(scores);
        var q = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSubtractScalar(scores, max), (float)(1.0 / temperature)));
        q = engine.TensorDivideScalar(q, engine.TensorSum(q));
        for (int i = 0; i < iterations; i++)
        {
            // Each prototype (column) to total mass 1 / k, then each sample (row) to 1 / n.
            var perPrototype = engine.TensorMultiplyScalar(engine.ReduceSum(q, new[] { 0 }, true), (float)k);
            q = engine.TensorDivide(q, engine.TensorBroadcastTo(perPrototype, shape));
            var perSample = engine.TensorMultiplyScalar(engine.ReduceSum(q, new[] { 1 }, true), (float)n);
            q = engine.TensorDivide(q, engine.TensorBroadcastTo(perSample, shape));
        }

        return engine.TensorMultiplyScalar(q, (float)n);
    }
}

/// <summary>
/// Sonata's cluster cross-entropy: <c>-sum(target * log_softmax(student / tau))</c> per match,
/// averaged within each scene and then across scenes (<c>segment_coo(..., reduce="mean").mean()</c>).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class SonataClusterCrossEntropyLoss<T> : ISonataDistillationLoss<T>
{
    /// <inheritdoc />
    public Tensor<T> Compute(Tensor<T> studentScores, Tensor<T> targets, double studentTemperature, int[] sampleOfMatch)
    {
        if (studentScores is null) throw new ArgumentNullException(nameof(studentScores));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (sampleOfMatch is null) throw new ArgumentNullException(nameof(sampleOfMatch));
        if (!(studentTemperature > 0)) throw new ArgumentOutOfRangeException(nameof(studentTemperature));
        int m = studentScores.Shape[0];
        if (sampleOfMatch.Length != m) throw new ArgumentException("One scene index per match is required.", nameof(sampleOfMatch));
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var logProbabilities = engine.TensorLogSoftmax(
            engine.TensorMultiplyScalar(studentScores, numOps.FromDouble(1.0 / studentTemperature)), -1);
        var perMatch = engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(targets, logProbabilities), new[] { 1 }, true));

        // Compact the scenes that have matches so empty scenes do not enter the mean as zeros.
        var compact = new Dictionary<int, int>();
        var segment = new int[m];
        for (int i = 0; i < m; i++)
        {
            if (!compact.TryGetValue(sampleOfMatch[i], out int id))
            {
                id = compact.Count;
                compact[sampleOfMatch[i]] = id;
            }

            segment[i] = id;
        }

        var perScene = engine.ScatterMean(perMatch, PtOps.Indices(segment), out _, 0, compact.Count);
        return engine.Reshape(engine.ReduceMean(perScene, new[] { 0, 1 }, false), new[] { 1 });
    }
}

/// <summary>
/// The Kozachenko-Leonenko entropy regulariser of DINOv2 (Sablayrolles et al. 2019):
/// <c>-mean(log(||x_i - x_nn(i)|| + eps))</c> over L2-normalised embeddings, pushing each embedding
/// away from its nearest neighbour.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// The official Sonata v1m1 configuration does not use it, so <see cref="SonataOptions.KoLeoWeight"/>
/// defaults to zero; it is provided for the DINOv2-style variant.
/// </remarks>
public sealed class KoLeoRegularizer<T> : ISonataFeatureRegularizer<T>
{
    private const double Epsilon = 1e-8;

    /// <inheritdoc />
    public Tensor<T> Compute(Tensor<T> embeddings)
    {
        if (embeddings is null) throw new ArgumentNullException(nameof(embeddings));
        if (embeddings.Rank != 2) throw new ArgumentException("Embeddings must be [points, channels].", nameof(embeddings));
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = embeddings.Shape[0], c = embeddings.Shape[1];
        if (n < 2) return new Tensor<T>(new[] { 1 });
        var shape = new[] { n, c };
        var x = engine.TensorDivide(embeddings, engine.TensorBroadcastTo(L2Norm(engine, embeddings, numOps), shape));

        Tensor<int> nearest;
        using (new NoGradScope<T>())
        {
            // dots with the diagonal pushed below -1, the smallest cosine, so a point never picks itself.
            var dots = engine.TensorMatMul(x, engine.TensorPermute(x, new[] { 1, 0 }));
            var self = engine.TensorMultiplyScalar(engine.TensorEye<T>(n), numOps.FromDouble(3.0));
            nearest = engine.TensorArgMax(engine.TensorSubtract(dots, self), 1);
        }

        var difference = engine.TensorAddScalar(
            engine.TensorSubtract(x, engine.TensorIndexSelect(x, nearest, 0)), numOps.FromDouble(Epsilon));
        var distance = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(difference), new[] { 1 }, false));
        var logs = engine.TensorLog(engine.TensorAddScalar(distance, numOps.FromDouble(Epsilon)));
        return engine.Reshape(engine.TensorNegate(engine.ReduceMean(logs, new[] { 0 }, false)), new[] { 1 });
    }

    private static Tensor<T> L2Norm(IEngine engine, Tensor<T> x, INumericOperations<T> numOps)
        => engine.TensorClampMin(engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), new[] { 1 }, true)),
            numOps.FromDouble(Epsilon));
}
