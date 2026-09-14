using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Concerto's 2D-3D cross-modal joint-embedding objective (arXiv:2510.23607).
/// </summary>
/// <remarks>
/// <para>Aligns 3D point features with frozen 2D image-patch features. Correspondence is
/// established exactly as the paper describes it — a 3D-to-2D projection followed by
/// depth-based visibility verification — after which the point features falling inside each
/// patch are mean-pooled and matched to that patch's image embedding by cosine similarity.</para>
/// <para>Two details from the paper are load-bearing and easy to get wrong:</para>
/// <list type="bullet">
/// <item>Visibility is verified against a depth map, not assumed from projection alone. A point
/// behind a wall still projects onto a valid pixel; without the depth check it would be trained
/// to match whatever surface occludes it.</item>
/// <item>The paper reports that admitting FEWER visible points performed better than maximizing
/// matches, so a loose tolerance is not a free accuracy win.</item>
/// </list>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public static class ConcertoCrossModalObjective<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the mean cosine-similarity loss between point-derived patch features and the
    /// frozen image encoder's patch features, over every supplied view.
    /// </summary>
    /// <param name="pointFeatures">Point features, shape <c>[numPoints, featureDim]</c>.</param>
    /// <param name="pointCoordinates">World-space XYZ, shape <c>[numPoints, 3]</c>.</param>
    /// <param name="views">Calibrated views paired with this point cloud.</param>
    /// <param name="depthToleranceMeters">
    /// Visibility tolerance; a point counts only when <c>|d_camera - d_depthMap|</c> is below it.
    /// </param>
    /// <returns>
    /// <c>1 - mean cosine similarity</c> over all patches that received at least one visible
    /// point, so lower is better and a perfect match scores zero. Returns zero when no patch in
    /// any view received a visible point, which is the correct no-op rather than a spurious
    /// gradient.
    /// </returns>
    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    /// <summary>
    /// Point-to-patch matches for one view: which points landed in which patch, after projection
    /// and depth-based visibility.
    /// </summary>
    /// <remarks>
    /// Extracted so <see cref="ComputeLoss"/> and <see cref="ComputeTapeLoss"/> match points
    /// identically. Two copies of this geometry would let the loss a user monitors and the loss
    /// that trains disagree, with nothing reporting the divergence.
    /// </remarks>
    private static (List<int> PointIndices, List<int> PatchOfMatch, int[] PatchCounts) MatchPointsToPatches(
        Tensor<T> pointCoordinates,
        ConcertoPairedView<T> view,
        double depthToleranceMeters,
        int numPoints)
    {
        int grid = view.PatchGridSize;
        int patchCount = grid * grid;
        var pointIndices = new List<int>();
        var patchOfMatch = new List<int>();
        var counts = new int[patchCount];

        int depthHeight = view.DepthMap.Shape[0];
        int depthWidth = view.DepthMap.Shape[1];

        for (int p = 0; p < numPoints; p++)
        {
            double wx = NumOps.ToDouble(pointCoordinates[p, 0]);
            double wy = NumOps.ToDouble(pointCoordinates[p, 1]);
            double wz = NumOps.ToDouble(pointCoordinates[p, 2]);

            // World -> camera via the 4x4 extrinsic.
            double cx = Ext(view, 0, 0) * wx + Ext(view, 0, 1) * wy + Ext(view, 0, 2) * wz + Ext(view, 0, 3);
            double cy = Ext(view, 1, 0) * wx + Ext(view, 1, 1) * wy + Ext(view, 1, 2) * wz + Ext(view, 1, 3);
            double cz = Ext(view, 2, 0) * wx + Ext(view, 2, 1) * wy + Ext(view, 2, 2) * wz + Ext(view, 2, 3);

            // Behind the camera: no projection exists.
            if (cz <= 0) continue;

            // Camera -> pixel via the 3x3 intrinsic.
            double u = (In(view, 0, 0) * cx + In(view, 0, 1) * cy + In(view, 0, 2) * cz) / cz;
            double v = (In(view, 1, 0) * cx + In(view, 1, 1) * cy + In(view, 1, 2) * cz) / cz;

            int px = (int)Math.Floor(u);
            int py = (int)Math.Floor(v);
            if (px < 0 || py < 0 || px >= depthWidth || py >= depthHeight) continue;

            // Depth-based visibility verification -- the step that rejects occluded points.
            double observedDepth = NumOps.ToDouble(view.DepthMap[py, px]);
            if (Math.Abs(cz - observedDepth) >= depthToleranceMeters) continue;

            int patchX = Math.Min(grid - 1, px * grid / Math.Max(1, depthWidth));
            int patchY = Math.Min(grid - 1, py * grid / Math.Max(1, depthHeight));
            int patch = (patchY * grid) + patchX;

            pointIndices.Add(p);
            patchOfMatch.Add(patch);
            counts[patch]++;
        }

        return (pointIndices, patchOfMatch, counts);
    }

    /// <summary>
    /// The same objective as <see cref="ComputeLoss"/>, expressed in engine operations so a
    /// <c>GradientTape</c> can differentiate it with respect to the point features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="ComputeLoss"/> accumulates in raw <c>double</c> and returns a bare scalar, which
    /// carries no gradient graph -- it can report the objective but cannot train against it.
    /// </para>
    /// <para>
    /// Gradients flow to <paramref name="pointFeatures"/> only. The projection depends on the
    /// coordinates and the camera matrices, which are measurements rather than parameters, and the
    /// image patch features come from a frozen image encoder, so both are correctly constant here.
    /// </para>
    /// </remarks>
    /// <param name="pointFeatures">Student point features, shape [points, featureDim].</param>
    /// <param name="pointCoordinates">World coordinates, shape [points, 3].</param>
    /// <param name="views">Paired camera views for this point cloud.</param>
    /// <param name="depthToleranceMeters">Visibility tolerance between projected and observed depth.</param>
    /// <returns>A scalar tensor carrying the gradient graph.</returns>
    public static Tensor<T> ComputeTapeLoss(
        Tensor<T> pointFeatures,
        Tensor<T> pointCoordinates,
        IReadOnlyList<ConcertoPairedView<T>> views,
        double depthToleranceMeters)
    {
        if (pointFeatures is null) throw new ArgumentNullException(nameof(pointFeatures));
        if (pointCoordinates is null) throw new ArgumentNullException(nameof(pointCoordinates));
        if (views is null) throw new ArgumentNullException(nameof(views));

        var engine = Engine;
        int numPoints = pointCoordinates.Shape[0];
        int featureDim = pointFeatures.Shape[^1];

        var perViewSimilarities = new List<Tensor<T>>();

        foreach (var view in views)
        {
            var (pointIndices, patchOfMatch, counts) = MatchPointsToPatches(
                pointCoordinates, view, depthToleranceMeters, numPoints);

            if (pointIndices.Count == 0) continue;

            var occupiedPatches = new List<int>();
            for (int patch = 0; patch < counts.Length; patch++)
            {
                if (counts[patch] > 0) occupiedPatches.Add(patch);
            }

            if (occupiedPatches.Count == 0) continue;

            // Mean-pool each patch's contributing points as a matrix multiply, so the pooling is
            // differentiable. Row `r` of the pooling matrix holds 1/count for the matches belonging
            // to that patch, which is exactly the mean ComputeLoss takes with a running sum.
            var pooling = new Tensor<T>(new[] { occupiedPatches.Count, pointIndices.Count });
            var patchRow = new Dictionary<int, int>();
            for (int r = 0; r < occupiedPatches.Count; r++) patchRow[occupiedPatches[r]] = r;

            for (int m = 0; m < patchOfMatch.Count; m++)
            {
                int row = patchRow[patchOfMatch[m]];
                pooling[row, m] = NumOps.FromDouble(1.0 / counts[patchOfMatch[m]]);
            }

            var matchedIndices = new Tensor<int>(new[] { pointIndices.Count });
            for (int m = 0; m < pointIndices.Count; m++) matchedIndices[m] = pointIndices[m];

            var gathered = engine.TensorIndexSelect(pointFeatures, matchedIndices, axis: 0);
            var pooled = engine.TensorMatMul(pooling, gathered);

            var targetIndices = new Tensor<int>(new[] { occupiedPatches.Count });
            for (int r = 0; r < occupiedPatches.Count; r++) targetIndices[r] = occupiedPatches[r];

            var targets = engine.StopGradient(
                engine.TensorIndexSelect(view.ImagePatchFeatures, targetIndices, axis: 0));

            perViewSimilarities.Add(engine.TensorCosineSimilarity(pooled, targets, dim: -1));
        }

        if (perViewSimilarities.Count == 0)
        {
            return new Tensor<T>(Array.Empty<int>(), new Vector<T>(new[] { NumOps.Zero }));
        }

        var allSimilarities = perViewSimilarities.Count == 1
            ? perViewSimilarities[0]
            : engine.TensorConcatenate(perViewSimilarities.ToArray(), axis: 0);

        var meanSimilarity = engine.ReduceMean(allSimilarities, new[] { 0 }, keepDims: false);
        var one = new Tensor<T>(Array.Empty<int>(), new Vector<T>(new[] { NumOps.One }));
        return engine.TensorSubtract(one, meanSimilarity);
    }

    public static T ComputeLoss(
        Tensor<T> pointFeatures,
        Tensor<T> pointCoordinates,
        IReadOnlyList<ConcertoPairedView<T>> views,
        double depthToleranceMeters)
    {
        if (pointFeatures is null) throw new ArgumentNullException(nameof(pointFeatures));
        if (pointCoordinates is null) throw new ArgumentNullException(nameof(pointCoordinates));
        if (views is null) throw new ArgumentNullException(nameof(views));

        int numPoints = pointCoordinates.Shape[0];
        int featureDim = pointFeatures.Shape[^1];

        double similaritySum = 0;
        int matchedPatches = 0;

        foreach (var view in views)
        {
            int grid = view.PatchGridSize;
            int patchCount = grid * grid;

            // Shared with ComputeTapeLoss so the monitored objective and the trained one cannot
            // disagree about which points are visible in which patch.
            var (pointIndices, patchOfMatch, counts) = MatchPointsToPatches(
                pointCoordinates, view, depthToleranceMeters, numPoints);

            var accumulated = new double[patchCount, featureDim];
            for (int m = 0; m < pointIndices.Count; m++)
            {
                int p = pointIndices[m];
                int patch = patchOfMatch[m];
                for (int d = 0; d < featureDim; d++)
                    accumulated[patch, d] += NumOps.ToDouble(pointFeatures[p, d]);
            }

            // Mean-pool each patch's contributing points, then cosine-match to the image feature.
            for (int patch = 0; patch < patchCount; patch++)
            {
                if (counts[patch] == 0) continue;

                double dot = 0, normPredicted = 0, normTarget = 0;
                for (int d = 0; d < featureDim; d++)
                {
                    double predicted = accumulated[patch, d] / counts[patch];
                    double target = NumOps.ToDouble(view.ImagePatchFeatures[patch, d]);

                    dot += predicted * target;
                    normPredicted += predicted * predicted;
                    normTarget += target * target;
                }

                double denominator = Math.Sqrt(normPredicted) * Math.Sqrt(normTarget);
                if (denominator <= 1e-12) continue;

                similaritySum += dot / denominator;
                matchedPatches++;
            }
        }

        if (matchedPatches == 0) return NumOps.Zero;

        return NumOps.FromDouble(1.0 - (similaritySum / matchedPatches));
    }

    private static double Ext(ConcertoPairedView<T> view, int row, int col)
        => NumOps.ToDouble(view.Extrinsics[row, col]);

    private static double In(ConcertoPairedView<T> view, int row, int col)
        => NumOps.ToDouble(view.Intrinsics[row, col]);
}
