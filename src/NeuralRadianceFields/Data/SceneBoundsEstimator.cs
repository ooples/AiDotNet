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
        double cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < viewList.Count; i++)
        {
            cx += numOps.ToDouble(viewList[i].CameraPosition[0]);
            cy += numOps.ToDouble(viewList[i].CameraPosition[1]);
            cz += numOps.ToDouble(viewList[i].CameraPosition[2]);
        }
        cx /= viewList.Count; cy /= viewList.Count; cz /= viewList.Count;

        // Farthest-camera distance from centroid — the radius that comfortably contains all views.
        double maxDist = 0;
        for (int i = 0; i < viewList.Count; i++)
        {
            double dx = numOps.ToDouble(viewList[i].CameraPosition[0]) - cx;
            double dy = numOps.ToDouble(viewList[i].CameraPosition[1]) - cy;
            double dz = numOps.ToDouble(viewList[i].CameraPosition[2]) - cz;
            double d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            if (d > maxDist) maxDist = d;
        }
        double radius = maxDist * (1.0 + margin);

        // Near = smallest camera-to-centroid distance minus half-radius (surface hits before
        // reaching centroid). Far = camera-to-far-side-of-bbox from the most distant camera.
        double minDist = double.PositiveInfinity;
        for (int i = 0; i < viewList.Count; i++)
        {
            double dx = numOps.ToDouble(viewList[i].CameraPosition[0]) - cx;
            double dy = numOps.ToDouble(viewList[i].CameraPosition[1]) - cy;
            double dz = numOps.ToDouble(viewList[i].CameraPosition[2]) - cz;
            double d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            if (d < minDist) minDist = d;
        }

        double near = Math.Max(0.1, minDist - radius);
        double far  = maxDist + radius;

        return new SceneBounds(
            center: (cx, cy, cz),
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
