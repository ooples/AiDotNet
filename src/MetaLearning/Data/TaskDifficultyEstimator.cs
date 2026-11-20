using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Estimates the difficulty of a meta-learning task based on geometric properties of the data:
/// inter-class separation, intra-class variance, and support/query alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This utility looks at the support and query data and estimates
/// how hard the task is. Tasks where classes overlap a lot are harder than tasks where classes
/// are clearly separated. This estimate can feed into curriculum or dynamic samplers.</para>
/// <para><b>Reference:</b> Adaptive Task Sampling for Meta-Learning (2024).</para>
/// </remarks>
public static class TaskDifficultyEstimator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Estimates difficulty of a task in [0, 1] based on support set geometry.
    /// 0 = trivially easy (well-separated classes), 1 = very hard (overlapping classes).
    /// </summary>
    /// <param name="supportX">Support set features as a flat vector (row-major, numExamples x featureDim).</param>
    /// <param name="supportY">Support set labels.</param>
    /// <param name="numWays">Number of classes.</param>
    /// <param name="numShots">Number of shots per class.</param>
    /// <returns>Difficulty score in [0, 1].</returns>
    public static T EstimateDifficulty(Vector<T> supportX, Vector<T> supportY, int numWays, int numShots)
    {
        T neutral = NumOps.FromDouble(0.5);
        if (supportX.Length == 0 || supportY.Length == 0 || numWays < 2)
            return neutral;

        int numExamples = supportY.Length;

        // Validate episode shape: numShots * numWays should match numExamples
        if (numExamples < numWays)
            return neutral; // Not enough examples to compute meaningful difficulty

        // Validate feature dimensions: supportX must be exactly divisible by numExamples
        if (supportX.Length % numExamples != 0)
            return neutral; // Feature vector not evenly divisible — can't decode reliably

        int featureDim = supportX.Length / numExamples;
        if (featureDim == 0)
            return neutral;

        var engine = AiDotNetEngine.Current;
        var features = Tensor<T>.FromVector(supportX, new[] { numExamples, featureDim });

        // 1. Compute class centroids
        // Map distinct label values to class indices instead of using modular arithmetic,
        // which can collapse distinct labels onto the same class
        var rawLabels = RoundedLabels(supportY);
        var labelToClass = new Dictionary<int, int>();
        foreach (int rawLabel in rawLabels)
        {
            if (!labelToClass.ContainsKey(rawLabel) && labelToClass.Count < numWays)
                labelToClass[rawLabel] = labelToClass.Count;
        }

        var classOf = rawLabels.Select(rawLabel =>
        {
            int classIdx = labelToClass.TryGetValue(rawLabel, out int mapped) ? mapped : rawLabel % numWays;
            if (classIdx < 0) classIdx += numWays;
            if (classIdx >= numWays) classIdx = numWays - 1;
            return classIdx;
        }).ToArray();

        var (membership, counts, centroids) = ClassCentroids(engine, features, classOf, numWays);

        // 2. Inter-class distance (average pairwise centroid distance, excluding empty classes)
        var pairDistances = engine.TensorSqrt(engine.ReduceSum(
            Squared(engine, engine.TensorSubtract(
                engine.Reshape(centroids, new[] { numWays, 1, featureDim }),
                engine.Reshape(centroids, new[] { 1, numWays, featureDim }))),
            new[] { 2 }, keepDims: false));
        var occupied = engine.TensorGreaterThan(counts, NumOps.Zero);
        var pairMask = engine.TensorTriu(
            engine.TensorMultiply(
                engine.Reshape(occupied, new[] { numWays, 1 }),
                engine.Reshape(occupied, new[] { 1, numWays })),
            1);
        T pairCount = engine.TensorSum(pairMask);
        T interClassDist = engine.TensorSum(engine.TensorMultiply(pairDistances, pairMask));
        if (NumOps.GreaterThan(pairCount, NumOps.Zero)) interClassDist = NumOps.Divide(interClassDist, pairCount);

        // 3. Intra-class variance (average distance from examples to their centroid)
        var ownCentroids = engine.TensorMatMul(membership, centroids);
        var exampleDistances = engine.TensorSqrt(engine.ReduceSum(
            Squared(engine, engine.TensorSubtract(features, ownCentroids)), new[] { 1 }, keepDims: false));
        T intraClassVar = NumOps.Divide(engine.TensorSum(exampleDistances), NumOps.FromDouble(numExamples));

        // 4. Difficulty = intra-class variance / (inter-class distance + epsilon)
        //    High intra / low inter = hard; low intra / high inter = easy
        T ratio = NumOps.Divide(intraClassVar, NumOps.Add(interClassDist, NumOps.FromDouble(1e-8)));

        // Map through sigmoid to [0, 1]: 1 / (1 + exp(-(ratio - 1) * 3))
        T exponent = NumOps.Multiply(NumOps.Subtract(NumOps.One, ratio), NumOps.FromDouble(3.0));
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(exponent)));
    }

    /// <summary>
    /// Estimates difficulty using the Fisher discriminant ratio:
    /// difficulty = 1 - (between-class variance / total variance).
    /// </summary>
    /// <param name="supportX">Support features (flat vector).</param>
    /// <param name="supportY">Support labels.</param>
    /// <param name="numWays">Number of classes.</param>
    /// <returns>Difficulty score in [0, 1].</returns>
    /// <remarks>
    /// When supportX does not fill numExamples x featureDim exactly, the missing trailing entries take no part and
    /// any surplus past numExamples * featureDim is ignored.
    /// </remarks>
    public static T EstimateFisherDifficulty(Vector<T> supportX, Vector<T> supportY, int numWays)
    {
        if (supportX.Length == 0 || supportY.Length == 0 || numWays < 2)
            return NumOps.FromDouble(0.5);

        int n = supportY.Length;
        int featureDim = Math.Max(1, supportX.Length / n);
        var engine = AiDotNetEngine.Current;

        // Lay the features out as [n, featureDim]; validity marks the entries supportX actually has (a short
        // vector leaves trailing entries missing; a long one has a surplus that is ignored).
        int cells = n * featureDim;
        int present = Math.Min(cells, supportX.Length);
        var flat = present == supportX.Length ? supportX : supportX.GetSubVector(0, present);
        var presentMask = new Vector<T>(present);
        presentMask.Fill(NumOps.One);
        if (present < cells)
        {
            flat = Vector<T>.Concatenate(flat, new Vector<T>(cells - present));
            presentMask = Vector<T>.Concatenate(presentMask, new Vector<T>(cells - present));
        }

        var features = Tensor<T>.FromVector(flat, new[] { n, featureDim });
        var validity = Tensor<T>.FromVector(presentMask, new[] { n, featureDim });

        // Grand mean
        var grandMean = engine.TensorDivideScalar(
            engine.ReduceSum(features, new[] { 0 }, keepDims: true), NumOps.FromDouble(n));

        // Total variance
        T totalVar = engine.TensorSum(engine.TensorMultiply(
            Squared(engine, engine.TensorSubtract(features, grandMean)), validity));

        // Between-class variance (same centroid computation as above)
        var classOf = RoundedLabels(supportY).Select(label =>
        {
            int c = label % numWays;
            if (c < 0) c += numWays;
            return c;
        }).ToArray();
        var (_, counts, centroids) = ClassCentroids(engine, features, classOf, numWays);
        var centroidSpread = engine.ReduceSum(
            Squared(engine, engine.TensorSubtract(centroids, grandMean)), new[] { 1 }, keepDims: false);
        T betweenVar = engine.TensorSum(engine.TensorMultiply(counts, centroidSpread));

        // Fisher ratio: higher between/total = easier
        T fisherRatio = NumOps.GreaterThan(totalVar, NumOps.FromDouble(1e-12))
            ? NumOps.Divide(betweenVar, totalVar)
            : NumOps.FromDouble(0.5);
        // Clamp to [0, 1]; NaN passes through, as Math.Max(0, Math.Min(1, NaN)) does.
        T clamped = NumOps.LessThan(fisherRatio, NumOps.Zero) ? NumOps.Zero
            : NumOps.GreaterThan(fisherRatio, NumOps.One) ? NumOps.One
            : fisherRatio;
        return NumOps.Subtract(NumOps.One, clamped);
    }

    /// <summary>Labels rounded half to even, as Math.Round does, read as integers.</summary>
    private static int[] RoundedLabels(Vector<T> labels)
        => AiDotNetEngine.Current.TensorRound(Tensor<T>.FromVector(labels)).Cast<int>().ToArray();

    /// <summary>Element-wise square.</summary>
    private static Tensor<T> Squared(IEngine engine, Tensor<T> values) => engine.TensorMultiply(values, values);

    /// <summary>
    /// The one-hot class membership of each row, [n, numWays], the class counts, [numWays], and the class
    /// centroids, [numWays, featureDim] (a class with no rows keeps a zero centroid).
    /// </summary>
    private static (Tensor<T> Membership, Tensor<T> Counts, Tensor<T> Centroids) ClassCentroids(
        IEngine engine, Tensor<T> features, int[] classOf, int numWays)
    {
        var membership = engine.TensorOneHot<T>(
            new Tensor<int>(new[] { classOf.Length }, new Vector<int>(classOf)), numWays);
        var counts = engine.ReduceSum(membership, new[] { 0 }, keepDims: false);
        var sums = engine.TensorMatMul(engine.TensorTranspose(membership), features);
        var centroids = engine.TensorDivide(
            sums, engine.Reshape(engine.TensorClampMin(counts, NumOps.One), new[] { numWays, 1 }));
        return (membership, counts, centroids);
    }
}
