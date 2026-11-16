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
/// (<c>feat = feat.float()</c>): <c>exp(1 / 0.04)</c> is beyond half precision. The iteration runs in
/// log space: each normalisation is a log-softmax along its axis minus the log of the axis' target
/// mass, and the result is exponentiated once at the end. That is algebraically Sonata's linear
/// update, but a prototype whose every <c>exp(score / temperature)</c> underflows to zero no longer
/// divides 0 by 0 into NaN. The result carries no gradient: it is the teacher's target.
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
        float max = engine.TensorMaxValue(scores);
        // log Q = (s - max) / temperature; the global normaliser of Sonata's Q /= sum(Q) is a constant
        // shift that the first log-softmax over prototypes cancels exactly.
        var logQ = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(scores, max), (float)(1.0 / temperature));
        float logK = (float)Math.Log(k), logN = (float)Math.Log(n);
        for (int i = 0; i < iterations; i++)
        {
            // Each prototype (column) to total mass 1 / k, then each sample (row) to 1 / n.
            logQ = engine.TensorSubtractScalar(engine.TensorLogSoftmax(logQ, 0), logK);
            logQ = engine.TensorSubtractScalar(engine.TensorLogSoftmax(logQ, 1), logN);
        }

        // Q * n: every row sums to one.
        return engine.TensorExp(engine.TensorAddScalar(logQ, logN));
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
/// <para>
/// The official Sonata v1m1 configuration does not use it, so <see cref="SonataOptions.KoLeoWeight"/>
/// defaults to zero; it is provided for the DINOv2-style variant.
/// </para>
/// <para>
/// DINOv2 picks the neighbours from the full <c>[points, points]</c> cosine matrix, which for a 65,536-point
/// view is 4.3 billion entries. Here the matrix is scanned in <see cref="BlockSize"/>-square tiles that
/// keep each query's best score and global index (the first index on ties, as <c>argmax</c>), so memory
/// is bounded by one tile while the selected neighbours are the same. The selection carries no
/// gradient; the distance to the selected neighbour does.
/// </para>
/// </remarks>
public sealed class KoLeoRegularizer<T> : ISonataFeatureRegularizer<T>
{
    private const double Epsilon = 1e-8;

    /// <summary>Creates the regulariser.</summary>
    /// <param name="blockSize">Rows and columns of each nearest-neighbour tile (default 1024).</param>
    public KoLeoRegularizer(int blockSize = 1024)
    {
        if (blockSize < 1) throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "The block size must be positive.");
        BlockSize = blockSize;
    }

    /// <summary>Rows and columns of each tile of the cosine matrix scanned for nearest neighbours.</summary>
    public int BlockSize { get; }

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

        var nearest = NearestNeighbours(engine, numOps, x, n, c);
        var difference = engine.TensorAddScalar(
            engine.TensorSubtract(x, engine.TensorIndexSelect(x, PtOps.Indices(nearest), 0)), numOps.FromDouble(Epsilon));
        var distance = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(difference), new[] { 1 }, false));
        var logs = engine.TensorLog(engine.TensorAddScalar(distance, numOps.FromDouble(Epsilon)));
        return engine.Reshape(engine.TensorNegate(engine.ReduceMean(logs, new[] { 0 }, false)), new[] { 1 });
    }

    /// <summary>
    /// For every row of the normalised <paramref name="x"/>, the index of the other row with the largest
    /// cosine, scanning <see cref="BlockSize"/>-square tiles.
    /// </summary>
    internal int[] NearestNeighbours(IEngine engine, INumericOperations<T> numOps, Tensor<T> x, int n, int c)
    {
        var nearest = new int[n];
        int block = BlockSize;
        int tiles = (n + block - 1) / block;
        var selfPenalty = numOps.FromDouble(3.0);
        using (new NoGradScope<T>())
        {
            for (int queryStart = 0; queryStart < n; queryStart += block)
            {
                int queryCount = Math.Min(block, n - queryStart);
                var queries = engine.TensorSlice(x, new[] { queryStart, 0 }, new[] { queryCount, c });
                var tileMaxima = new Tensor<T>[tiles];
                var tileIndices = new int[tiles][];
                for (int t = 0; t < tiles; t++)
                {
                    int referenceStart = t * block;
                    int referenceCount = Math.Min(block, n - referenceStart);
                    var references = engine.TensorSlice(x, new[] { referenceStart, 0 }, new[] { referenceCount, c });
                    var dots = engine.TensorMatMul(queries, engine.TensorPermute(references, new[] { 1, 0 }));
                    if (referenceStart == queryStart)
                    {
                        // Tiles share one grid, so only the diagonal tile holds self pairs; push them below
                        // -1, the smallest cosine, so a point never picks itself.
                        dots = engine.TensorSubtract(dots, engine.TensorMultiplyScalar(engine.TensorEye<T>(queryCount), selfPenalty));
                    }

                    tileMaxima[t] = engine.ReduceMax(dots, new[] { 1 }, true);
                    tileIndices[t] = engine.TensorArgMax(dots, 1).ToArray();
                }

                // [queries, tiles] of per-tile maxima: argmax picks the first tile holding the row maximum,
                // and within it the first column, so ties resolve to the earliest index as one argmax would.
                var winner = engine.TensorArgMax(tiles == 1 ? tileMaxima[0] : engine.TensorConcatenate(tileMaxima, 1), 1).ToArray();
                for (int i = 0; i < queryCount; i++)
                    nearest[queryStart + i] = winner[i] * block + tileIndices[winner[i]][i];
            }
        }

        return nearest;
    }

    private static Tensor<T> L2Norm(IEngine engine, Tensor<T> x, INumericOperations<T> numOps)
        => engine.TensorClampMin(engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), new[] { 1 }, true)),
            numOps.FromDouble(Epsilon));
}
