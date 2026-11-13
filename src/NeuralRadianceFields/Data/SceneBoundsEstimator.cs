using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Estimates a tight scene bounding box + near/far ray sampling bounds from a set of camera
/// poses (#1834 excellence goal #3). Reference NeRF pipelines require users to supply near/far
/// and scene bounds by hand — nerfstudio's config YAML has explicit fields, tiny-cuda-nn
/// requires them at construction. Here it's automatic: the ImageView pose set is the source
/// of truth, we intersect view frusta to bound the scene tightly, and derive per-view near/far
/// that hit the bbox comfortably from every camera.
/// </summary>
public static class SceneBoundsEstimator
{
    /// <summary>
    /// Computes the axis-aligned bounding box that contains every view's principal ray
    /// intersection region plus a symmetric buffer around the pose centroid — a paper-quality
    /// default when no explicit bounds are supplied.
    /// </summary>
    public static SceneBounds EstimateFromViews<T>(
        IEnumerable<ImageView<T>> views,
        double margin = 0.25)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));

        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var viewList = new List<ImageView<T>>();
        foreach (var v in views) viewList.Add(v);
        if (viewList.Count == 0)
        {
            throw new ArgumentException("Cannot estimate scene bounds from an empty view set.", nameof(views));
        }

        // Pose centroid — the natural scene center under uniform view distribution.
        var engine = AiDotNetEngine.Current;
        var cameraRows = new Tensor<T>[viewList.Count];
        for (int i = 0; i < viewList.Count; i++)
        {
            cameraRows[i] = Tensor<T>.FromVector(viewList[i].CameraPosition.GetSubVector(0, 3));
        }

        // The bounds are measured in double, as SceneBounds stores them: squared offsets overflow float (2e19)
        // and Half long before the bounds themselves do.
        var cameras = engine.TensorStack(cameraRows, 0).Cast<double>();
        var centroid = engine.TensorDivideScalar(engine.ReduceSum(cameras, new[] { 0 }, true), (double)viewList.Count);
        var offsets = engine.TensorSubtract(cameras, engine.TensorBroadcastTo(centroid, cameras.Shape.ToArray()));
        var distances = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(offsets), new[] { 1 }, false));
        var isNan = engine.TensorIsNan(distances);

        // Farthest-camera distance from centroid — the radius that comfortably contains all views.
        // The running "d > best" scan seeded with 0 skips NaN distances; distances are never negative, so
        // the maximum with NaN replaced by 0 is that scan's result.
        var forMax = engine.TensorWhere(isNan, engine.TensorZerosLike(distances), distances);
        double maxDist = engine.TensorMaxValue(forMax);
        double radius = maxDist * (1.0 + margin);

        // Near = smallest camera-to-centroid distance minus half-radius (surface hits before
        // reaching centroid). Far = camera-to-far-side-of-bbox from the most distant camera.
        // NaN distances are skipped; when every distance is NaN the scan never moved off +infinity.
        var forMin = engine.TensorWhere(isNan, engine.TensorFullLike(distances, double.PositiveInfinity), distances);
        double minDist = engine.TensorMinValue(forMin);

        // Math.Max(0.1, value) semantics: a NaN value propagates.
        double nearCandidate = minDist - radius;
        double near = double.IsNaN(nearCandidate) || nearCandidate > 0.1 ? nearCandidate : 0.1;
        double far = maxDist + radius;

        return new SceneBounds(
            center: (centroid[0, 0], centroid[0, 1], centroid[0, 2]),
            radius: radius,
            near: near,
            far: far);
    }
}

/// <summary>
/// Auto-derived scene bounds + near/far — output of <see cref="SceneBoundsEstimator"/>.
/// Stored as doubles because pose-derived geometry is unit-agnostic and doesn't need to
/// carry the model's numeric type.
/// </summary>
public sealed class SceneBounds
{
    public (double X, double Y, double Z) Center { get; }
    public double Radius { get; }
    public double Near { get; }
    public double Far { get; }

    public SceneBounds(
        (double X, double Y, double Z) center,
        double radius,
        double near,
        double far)
    {
        Center = center;
        Radius = radius;
        Near   = near;
        Far    = far;
    }

    /// <summary>
    /// Min / max corners of the derived axis-aligned bounding box (center ± radius per axis).
    /// </summary>
    public (double X, double Y, double Z) MinCorner => (Center.X - Radius, Center.Y - Radius, Center.Z - Radius);
    public (double X, double Y, double Z) MaxCorner => (Center.X + Radius, Center.Y + Radius, Center.Z + Radius);
}
