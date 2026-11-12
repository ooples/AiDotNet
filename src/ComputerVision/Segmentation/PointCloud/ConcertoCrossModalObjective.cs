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

    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    /// <summary>Lower bound on each vector norm in the cosine similarity (torch.nn.functional.cosine_similarity's eps).</summary>
    private const double CosineEpsilon = 1e-8;

    /// <summary>Half's smallest positive normal value, the floor for an epsilon Half would flush to zero.</summary>
    private const double SmallestHalfNormal = 6.103515625e-05;

    /// <summary>
    /// The patch every point is pooled into for one view: the patch it projects into when it is
    /// visible (in front of the camera, inside the image and passing the depth check), or
    /// <c>patchCount</c> -- a discard bin past the last patch -- when it is not.
    /// </summary>
    /// <remarks>
    /// <para>Shared by <see cref="ComputeLoss"/> and <see cref="ComputeTapeLoss"/> so the loss a user
    /// monitors and the loss that trains cannot disagree about which points are visible where.</para>
    /// <para>The projection and the depth test run in double, as they always did: they produce pixel
    /// indices, and a narrow <typeparamref name="T"/> would move points across pixel boundaries (in
    /// Half, u = 600.8 rounds to 601 and lands one pixel to the right).</para>
    /// <para>Invisible points are routed to the discard bin instead of being multiplied by a zero
    /// mask, so a NaN or infinite feature on a point behind the camera never reaches a patch.</para>
    /// </remarks>
    private static Tensor<int> AssignPointsToPatches(
        Tensor<T> pointCoordinates,
        ConcertoPairedView<T> view,
        double depthToleranceMeters,
        int numPoints)
    {
        var engine = Engine;
        int grid = view.PatchGridSize;
        int patchCount = grid * grid;
        int depthHeight = view.DepthMap.Shape[0];
        int depthWidth = view.DepthMap.Shape[1];

        // World -> camera via the top three rows of the 4x4 extrinsic, on homogeneous coordinates.
        var xyz = engine.TensorSlice(pointCoordinates, new[] { 0, 0 }, new[] { numPoints, 3 }).Cast<double>();
        var ones = engine.TensorFullLike(engine.TensorSlice(xyz, new[] { 0, 0 }, new[] { numPoints, 1 }), 1.0);
        var homogeneous = engine.TensorConcatenate(new[] { xyz, ones }, 1);
        var extrinsic = engine.TensorSlice(view.Extrinsics, new[] { 0, 0 }, new[] { 3, 4 }).Cast<double>();
        var camera = engine.TensorMatMul(homogeneous, engine.TensorTranspose(extrinsic));
        var cz = Column(camera, 2, numPoints);

        // Camera -> pixel via the 3x3 intrinsic.
        var pixel = engine.TensorMatMul(camera, engine.TensorTranspose(view.Intrinsics.Cast<double>()));
        var u = engine.TensorDivide(Column(pixel, 0, numPoints), cz);
        var v = engine.TensorDivide(Column(pixel, 1, numPoints), cz);

        // In front of the camera, and floor(u), floor(v) inside the depth map (NaN fails every test).
        var inFront = engine.TensorGreaterThan(cz, 0.0);
        var insideU = engine.TensorMultiply(
            engine.ScalarMinusTensor(1.0, engine.TensorLessThan(u, 0.0)),
            engine.TensorLessThan(u, (double)depthWidth));
        var insideV = engine.TensorMultiply(
            engine.ScalarMinusTensor(1.0, engine.TensorLessThan(v, 0.0)),
            engine.TensorLessThan(v, (double)depthHeight));
        var zeros = engine.TensorFullLike(u, 0.0);
        var px = engine.TensorFloor(engine.TensorWhere(insideU, u, zeros)).Cast<int>();
        var py = engine.TensorFloor(engine.TensorWhere(insideV, v, zeros)).Cast<int>();

        // Depth-based visibility verification -- the step that rejects occluded points. A point is
        // dropped only when |d_camera - d_depthMap| >= tolerance.
        var pixelIndex = engine.TensorAdd(engine.TensorMultiplyScalar(py, depthWidth), px);
        var observedDepth = engine.TensorIndexSelect(
            engine.Reshape(view.DepthMap, new[] { depthHeight * depthWidth }), pixelIndex, 0).Cast<double>();
        var depthError = engine.TensorAbs(engine.TensorSubtract(cz, observedDepth));
        var depthAgrees = engine.ScalarMinusTensor(1.0,
            engine.TensorGreaterEqual(depthError, engine.TensorFullLike(depthError, depthToleranceMeters)));

        // Every factor is a 0/1 comparison result, so the product is an exact 0/1 visibility flag.
        var visible = engine.TensorMultiply(engine.TensorMultiply(inFront, insideU),
            engine.TensorMultiply(insideV, depthAgrees)).Cast<int>();

        var patchX = engine.TensorClamp(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(px, grid), Math.Max(1, depthWidth)), 0, grid - 1);
        var patchY = engine.TensorClamp(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(py, grid), Math.Max(1, depthHeight)), 0, grid - 1);
        var patchIndex = engine.TensorAdd(engine.TensorMultiplyScalar(patchY, grid), patchX);

        // visible ? patchIndex : patchCount, in exact int arithmetic.
        return engine.TensorAddScalar(
            engine.TensorMultiply(visible, engine.TensorAddScalar(patchIndex, -patchCount)), patchCount);
    }

    /// <summary>
    /// The objective shared by both entry points: per view, the visible point features are
    /// mean-pooled into the patch they project into, and each occupied patch is matched to the
    /// frozen image feature by cosine similarity; the loss is one minus the mean over every
    /// occupied patch of every view, or zero when no patch received a visible point.
    /// </summary>
    /// <remarks>
    /// Only occupied patches are gathered (as the training loss always did), so an unoccupied patch's
    /// image feature never enters the loss. Point counts are integers, and the final mean scales each
    /// similarity by 1 / M before summing, so no Half count or sum rounds or overflows.
    /// </remarks>
    private static Tensor<T> CrossModalLoss(
        Tensor<T> pointFeatures,
        Tensor<T> pointCoordinates,
        IReadOnlyList<ConcertoPairedView<T>> views,
        double depthToleranceMeters)
    {
        var engine = Engine;
        int numPoints = pointCoordinates.Shape[0];
        int featureDim = pointFeatures.Shape[^1];
        var similarities = new List<Tensor<T>>();
        T epsilonSquared = RepresentableEpsilon(CosineEpsilon * CosineEpsilon);

        foreach (var view in views)
        {
            if (numPoints == 0) break;
            int patchCount = view.PatchGridSize * view.PatchGridSize;
            var patchOfPoint = AssignPointsToPatches(pointCoordinates, view, depthToleranceMeters, numPoints);

            // Visible points per patch, counted in int; the extra last bin holds the invisible points.
            var counts = engine.TensorBinCount(patchOfPoint, patchCount + 1);
            int[] countValues = counts.ToArray();
            int[] occupied = Enumerable.Range(0, patchCount).Where(p => countValues[p] > 0).ToArray();
            if (occupied.Length == 0) continue;

            // Mean-pool: each point carries 1 / (its patch's point count) and a scatter-add sums every
            // patch's share, which is exactly the per-patch mean and stays on the gradient tape.
            var weights = engine.TensorReciprocal(engine.TensorIndexSelect(counts, patchOfPoint, 0).Cast<T>());
            var weighted = engine.TensorMultiply(pointFeatures, engine.TensorBroadcastTo(
                engine.Reshape(weights, new[] { numPoints, 1 }), new[] { numPoints, featureDim }));
            var pooledAll = engine.TensorScatterAdd(
                new Tensor<T>(new[] { patchCount + 1, featureDim }), patchOfPoint, weighted, 0);

            var occupiedIndex = new Tensor<int>(new[] { occupied.Length }, new Vector<int>(occupied));
            var pooled = engine.TensorIndexSelect(pooledAll, occupiedIndex, 0);
            var targets = engine.StopGradient(engine.TensorIndexSelect(view.ImagePatchFeatures, occupiedIndex, 0));
            similarities.Add(CosineSimilarityRows(pooled, targets, epsilonSquared));
        }

        if (similarities.Count == 0)
        {
            return new Tensor<T>(Array.Empty<int>(), new Vector<T>(new[] { NumOps.Zero }));
        }

        var allSimilarities = similarities.Count == 1
            ? similarities[0]
            : engine.TensorConcatenate(similarities.ToArray(), 0);
        T inverseMatched = NumOps.FromDouble(1.0 / allSimilarities.Shape[0]);
        var meanSimilarity = engine.ReduceSum(
            engine.TensorMultiplyScalar(allSimilarities, inverseMatched), new[] { 0 }, false);
        return engine.ScalarMinusTensor(NumOps.One, meanSimilarity);
    }

    /// <summary>
    /// Row-wise cosine similarity of two <c>[M, D]</c> tensors, each row normalized by its norm clamped
    /// below at eps (torch.nn.functional.cosine_similarity), so an all-zero row scores zero.
    /// </summary>
    private static Tensor<T> CosineSimilarityRows(Tensor<T> a, Tensor<T> b, T epsilonSquared)
    {
        var engine = Engine;
        return engine.ReduceSum(
            engine.TensorMultiply(NormalizeRows(a, epsilonSquared), NormalizeRows(b, epsilonSquared)),
            new[] { 1 }, false);
    }

    /// <summary>
    /// Divides each row by its L2 norm, clamped below on the squared norm so an all-zero row keeps a
    /// finite gradient.
    /// </summary>
    /// <remarks>
    /// The row is first divided by its largest |entry| when that exceeds one -- the cosine is
    /// scale-invariant -- so the sum of squares cannot overflow a narrow <typeparamref name="T"/>
    /// (Half overflows once an entry passes 256). The scale carries no gradient: the result does not
    /// depend on it.
    /// </remarks>
    private static Tensor<T> NormalizeRows(Tensor<T> rows, T epsilonSquared)
    {
        var engine = Engine;
        int[] shape = rows.Shape.ToArray();
        var scale = engine.StopGradient(engine.TensorClampMin(
            engine.ReduceMax(engine.TensorAbs(rows), new[] { 1 }, true), NumOps.One));
        var scaled = engine.TensorDivide(rows, engine.TensorBroadcastTo(scale, shape));
        var norm = engine.TensorSqrt(engine.TensorClampMin(
            engine.ReduceSum(engine.TensorSquare(scaled), new[] { 1 }, true), epsilonSquared));
        return engine.TensorDivide(scaled, engine.TensorBroadcastTo(norm, shape));
    }

    /// <summary>
    /// <paramref name="epsilon"/> as a <typeparamref name="T"/>, raised to Half's smallest normal value
    /// when <typeparamref name="T"/> flushes it to zero (Half cannot hold 1e-16).
    /// </summary>
    private static T RepresentableEpsilon(double epsilon)
    {
        T value = NumOps.FromDouble(epsilon);
        return NumOps.GreaterThan(value, NumOps.Zero) ? value : NumOps.FromDouble(SmallestHalfNormal);
    }

    /// <summary>
    /// The same objective as <see cref="ComputeLoss"/>, expressed in engine operations so a
    /// <c>GradientTape</c> can differentiate it with respect to the point features.
    /// </summary>
    /// <remarks>
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

        return CrossModalLoss(pointFeatures, pointCoordinates, views, depthToleranceMeters);
    }

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
    /// any view received a visible point. Identical to <see cref="ComputeTapeLoss"/>.
    /// </returns>
    public static T ComputeLoss(
        Tensor<T> pointFeatures,
        Tensor<T> pointCoordinates,
        IReadOnlyList<ConcertoPairedView<T>> views,
        double depthToleranceMeters)
    {
        if (pointFeatures is null) throw new ArgumentNullException(nameof(pointFeatures));
        if (pointCoordinates is null) throw new ArgumentNullException(nameof(pointCoordinates));
        if (views is null) throw new ArgumentNullException(nameof(views));

        // A reported value, not a training signal: evaluated off the gradient tape.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Engine.TensorSum(CrossModalLoss(pointFeatures, pointCoordinates, views, depthToleranceMeters));
    }

    private static Tensor<TValue> Column<TValue>(Tensor<TValue> matrix, int column, int rows)
        => Engine.Reshape(Engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }), new[] { rows });
}
