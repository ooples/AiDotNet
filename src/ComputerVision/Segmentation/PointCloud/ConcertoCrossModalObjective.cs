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

    /// <summary>
    /// Point-to-patch matches for one view: the patch every point projects into, and a 0/1 mask of
    /// the points that project in front of the camera, inside the image and pass the depth check.
    /// </summary>
    /// <remarks>
    /// Shared by <see cref="ComputeLoss"/> and <see cref="ComputeTapeLoss"/> so the loss a user
    /// monitors and the loss that trains cannot disagree about which points are visible where.
    /// </remarks>
    private static (Tensor<int> PatchIndex, Tensor<T> Visible) MatchPointsToPatches(
        Tensor<T> pointCoordinates,
        ConcertoPairedView<T> view,
        double depthToleranceMeters,
        int numPoints)
    {
        var engine = Engine;
        int grid = view.PatchGridSize;
        int depthHeight = view.DepthMap.Shape[0];
        int depthWidth = view.DepthMap.Shape[1];

        // World -> camera via the top three rows of the 4x4 extrinsic, on homogeneous coordinates.
        var xyz = engine.TensorSlice(pointCoordinates, new[] { 0, 0 }, new[] { numPoints, 3 });
        var ones = engine.TensorFullLike(engine.TensorSlice(pointCoordinates, new[] { 0, 0 }, new[] { numPoints, 1 }), NumOps.One);
        var homogeneous = engine.TensorConcatenate(new[] { xyz, ones }, 1);
        var extrinsic = engine.TensorSlice(view.Extrinsics, new[] { 0, 0 }, new[] { 3, 4 });
        var camera = engine.TensorMatMul(homogeneous, engine.TensorTranspose(extrinsic));
        var cz = Column(camera, 2, numPoints);

        // Camera -> pixel via the 3x3 intrinsic.
        var pixel = engine.TensorMatMul(camera, engine.TensorTranspose(view.Intrinsics));
        var u = engine.TensorDivide(Column(pixel, 0, numPoints), cz);
        var v = engine.TensorDivide(Column(pixel, 1, numPoints), cz);

        // In front of the camera, and floor(u), floor(v) inside the depth map (NaN fails every test).
        var inFront = engine.TensorGreaterThan(cz, NumOps.Zero);
        var insideU = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(u, NumOps.Zero)),
            engine.TensorLessThan(u, NumOps.FromDouble(depthWidth)));
        var insideV = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(v, NumOps.Zero)),
            engine.TensorLessThan(v, NumOps.FromDouble(depthHeight)));
        var zeros = engine.TensorFullLike(u, NumOps.Zero);
        var px = engine.TensorFloor(engine.TensorWhere(insideU, u, zeros)).Cast<int>();
        var py = engine.TensorFloor(engine.TensorWhere(insideV, v, zeros)).Cast<int>();

        // Depth-based visibility verification -- the step that rejects occluded points. A point is
        // dropped only when |d_camera - d_depthMap| >= tolerance.
        var pixelIndex = engine.TensorAdd(engine.TensorMultiplyScalar(py, depthWidth), px);
        var observedDepth = engine.TensorIndexSelect(
            engine.Reshape(view.DepthMap, new[] { depthHeight * depthWidth }), pixelIndex, 0);
        var depthError = engine.TensorAbs(engine.TensorSubtract(cz, observedDepth));
        var depthAgrees = engine.ScalarMinusTensor(NumOps.One,
            engine.TensorGreaterEqual(depthError, engine.TensorFullLike(depthError, NumOps.FromDouble(depthToleranceMeters))));

        var visible = engine.TensorMultiply(engine.TensorMultiply(inFront, insideU),
            engine.TensorMultiply(insideV, depthAgrees));

        var patchX = engine.TensorClamp(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(px, grid), Math.Max(1, depthWidth)), 0, grid - 1);
        var patchY = engine.TensorClamp(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(py, grid), Math.Max(1, depthHeight)), 0, grid - 1);
        var patchIndex = engine.TensorAdd(engine.TensorMultiplyScalar(patchY, grid), patchX);

        return (patchIndex, visible);
    }

    /// <summary>
    /// The objective shared by both entry points: per view, the visible point features are
    /// mean-pooled into the patch they project into, and each occupied patch is matched to the
    /// frozen image feature by cosine similarity; the loss is one minus the mean over every
    /// occupied patch of every view, or zero when no patch received a visible point.
    /// </summary>
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
        var occupancies = new List<Tensor<T>>();
        T epsilonSquared = NumOps.FromDouble(CosineEpsilon * CosineEpsilon);
        T one = NumOps.One, zero = NumOps.Zero;

        foreach (var view in views)
        {
            if (numPoints == 0) break;
            int patchCount = view.PatchGridSize * view.PatchGridSize;
            var (patchIndex, visible) = MatchPointsToPatches(pointCoordinates, view, depthToleranceMeters, numPoints);

            // Mean-pool each patch's visible points (scatter-add of the masked features over the count).
            var counts = engine.TensorScatterAdd(new Tensor<T>(new[] { patchCount }), patchIndex, visible, 0);
            var maskedFeatures = engine.TensorMultiply(pointFeatures,
                engine.TensorBroadcastTo(engine.Reshape(visible, new[] { numPoints, 1 }), new[] { numPoints, featureDim }));
            var sums = engine.TensorScatterAdd(new Tensor<T>(new[] { patchCount, featureDim }), patchIndex, maskedFeatures, 0);
            var pooled = engine.TensorDivide(sums, engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorClampMin(counts, one), new[] { patchCount, 1 }), new[] { patchCount, featureDim }));

            // Cosine similarity to the frozen image feature; each norm is clamped below at eps (on the
            // squared norm, so an empty patch's zero vector keeps a finite gradient).
            var targets = engine.StopGradient(view.ImagePatchFeatures);
            var dot = engine.ReduceSum(engine.TensorMultiply(pooled, targets), new[] { 1 }, false);
            var pooledNorm = engine.TensorSqrt(engine.TensorClampMin(
                engine.ReduceSum(engine.TensorMultiply(pooled, pooled), new[] { 1 }, false), epsilonSquared));
            var targetNorm = engine.TensorSqrt(engine.TensorClampMin(
                engine.ReduceSum(engine.TensorMultiply(targets, targets), new[] { 1 }, false), epsilonSquared));
            var occupied = engine.TensorGreaterThan(counts, zero);
            similarities.Add(engine.TensorMultiply(
                engine.TensorDivide(dot, engine.TensorMultiply(pooledNorm, targetNorm)), occupied));
            occupancies.Add(occupied);
        }

        T matchedPatches = similarities.Count == 0
            ? NumOps.Zero
            : engine.TensorSum(engine.TensorConcatenate(occupancies.ToArray(), 0));
        if (!NumOps.GreaterThan(matchedPatches, NumOps.Zero))
        {
            return new Tensor<T>(Array.Empty<int>(), new Vector<T>(new[] { NumOps.Zero }));
        }

        var similaritySum = engine.ReduceSum(engine.TensorConcatenate(similarities.ToArray(), 0), new[] { 0 }, false);
        return engine.ScalarMinusTensor(NumOps.One, engine.TensorDivideScalar(similaritySum, matchedPatches));
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

    private static Tensor<T> Column(Tensor<T> matrix, int column, int rows)
        => Engine.Reshape(Engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }), new[] { rows });
}
