using AiDotNet.NeuralRadianceFields.NeuS;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>A camera pose relative to the input view (Zero123's conditioning) and in absolute terms.</summary>
internal readonly struct One2345ViewPose
{
    public One2345ViewPose(double polarDelta, double azimuthDelta, double polar, double azimuth)
    {
        PolarDelta = polarDelta;
        AzimuthDelta = azimuthDelta;
        Polar = polar;
        Azimuth = azimuth;
    }

    /// <summary>The polar change from the view this one is predicted from, in degrees.</summary>
    public double PolarDelta { get; }

    /// <summary>The azimuth change from the view this one is predicted from, in degrees.</summary>
    public double AzimuthDelta { get; }

    /// <summary>The absolute polar angle (from +z), in degrees.</summary>
    public double Polar { get; }

    /// <summary>The absolute azimuth, in degrees.</summary>
    public double Azimuth { get; }
}

/// <summary>
/// One-2-3-45's camera layout and scene normalization: <c>predict_stage1_gradio</c> and <c>zero123_infer</c> (the
/// Zero123 deltas), <c>get_poses</c> and <c>calc_pose</c> (absolute cameras), and the evaluation dataset's
/// <c>cal_scale_mat</c> / <c>get_boundingbox</c> (the reconstruction frame).
/// </summary>
internal static class One2345CameraLayout
{
    /// <summary>
    /// The 8 first-stage views: 4 at the input's elevation, then 4 on the offset ring, below the input when its polar
    /// angle is at most <see cref="One2345Options.OffsetBelowMaxPolar"/> and above it otherwise.
    /// </summary>
    public static One2345ViewPose[] Stage1(One2345Options options, double inputPolar)
    {
        double offset = inputPolar <= options.OffsetBelowMaxPolar ? options.Stage1OffsetPolarDelta : -options.Stage1OffsetPolarDelta;
        var poses = new One2345ViewPose[8];
        for (int i = 0; i < 4; i++)
        {
            double a = options.Stage1AzimuthDeltas[i];
            poses[i] = new One2345ViewPose(0.0, a, inputPolar, options.InputAzimuth + a);
            double b = options.Stage1OffsetAzimuthDeltas[i];
            poses[4 + i] = new One2345ViewPose(offset, b, inputPolar + offset, options.InputAzimuth + b);
        }

        return poses;
    }

    /// <summary>The 4 second-stage views around <paramref name="parent"/>.</summary>
    public static One2345ViewPose[] Stage2(One2345Options options, One2345ViewPose parent)
    {
        var poses = new One2345ViewPose[4];
        for (int j = 0; j < 4; j++)
        {
            double dp = options.Stage2PolarDeltas[j], da = options.Stage2AzimuthDeltas[j];
            poses[j] = new One2345ViewPose(dp, da, parent.Polar + dp, parent.Azimuth + da);
        }

        return poses;
    }

    /// <summary>
    /// <c>calc_pose</c>'s camera center: <c>(r sinθ sinφ, −r cosθ sinφ, r cosφ)</c> with φ the polar angle and θ the
    /// azimuth.
    /// </summary>
    public static double[] Center(double polarDegrees, double azimuthDegrees, double radius)
    {
        double phi = polarDegrees * Math.PI / 180.0, theta = azimuthDegrees * Math.PI / 180.0;
        return new[]
        {
            radius * Math.Sin(theta) * Math.Sin(phi),
            -radius * Math.Cos(theta) * Math.Sin(phi),
            radius * Math.Cos(phi),
        };
    }

    /// <summary>
    /// The OpenCV camera-to-world rotation (row-major 3×3) of a <c>calc_pose</c> camera at <paramref name="center"/>:
    /// <c>calc_pose</c>'s (right, up, forward) columns with up and forward negated (<c>blender2opencv</c>), so the
    /// camera looks at the origin with +y down.
    /// </summary>
    public static double[] OpenCvRotation(double[] center)
    {
        var forward = Normalize(center);
        var right = Normalize(Cross(new[] { 0.0, 0.0, 1.0 }, forward));
        if (right[0] * right[0] + right[1] * right[1] + right[2] * right[2] < 0.01) right = new[] { 0.0, 1.0, 0.0 };
        var up = Normalize(Cross(forward, right));
        return new[]
        {
            right[0], -up[0], -forward[0],
            right[1], -up[1], -forward[1],
            right[2], -up[2], -forward[2],
        };
    }

    /// <summary>The world-frame OpenCV camera of a pose, with the official intrinsics scaled to the image.</summary>
    public static NeuSCamera Camera(One2345Options options, One2345ViewPose pose, int width, int height)
    {
        var center = Center(pose.Polar, pose.Azimuth, options.CameraRadius);
        double focal = options.FocalLengthRatio * width;
        return new NeuSCamera(focal, focal, width / 2.0, height / 2.0, OpenCvRotation(center), center);
    }

    /// <summary>
    /// The reconstruction frame (the evaluation dataset's <c>__getitem__</c>): every camera is re-expressed in the
    /// reference camera's frame (<c>w2c · w2c_ref⁻¹</c>), then scaled so the bounding box of all view frusta between
    /// <see cref="One2345Options.Near"/> and <see cref="One2345Options.Far"/>, enlarged by
    /// <see cref="One2345Options.BoundingBoxFactor"/>, maps to the unit sphere (<c>cal_scale_mat</c>).
    /// </summary>
    public static One2345SceneFrame Normalize(One2345Options options, NeuSCamera reference, IReadOnlyList<NeuSCamera> cameras,
        int width, int height)
    {
        var rr = reference.Rotation;
        var relative = new List<(double[] R, double[] C)>();
        foreach (var camera in cameras)
            relative.Add((MultiplyTransposeLeft(rr, camera.Rotation), RotateInverse(rr, Subtract(camera.Center, reference.Center))));

        var min = new[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity };
        var max = new[] { double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity };
        for (int i = 0; i < relative.Count; i++)
        {
            var (r, c) = relative[i];
            var camera = cameras[i];
            foreach (double depth in new[] { options.Near, options.Far })
                foreach (double u in new[] { 0.0, width })
                    foreach (double v in new[] { 0.0, height })
                    {
                        var local = new[] { (u - camera.PrincipalX) * depth / camera.FocalX, (v - camera.PrincipalY) * depth / camera.FocalY, depth };
                        var world = Add(Rotate(r, local), c);
                        for (int k = 0; k < 3; k++)
                        {
                            min[k] = Math.Min(min[k], world[k]);
                            max[k] = Math.Max(max[k], world[k]);
                        }
                    }
        }

        var boxCenter = new[] { (min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2 };
        double radius = Math.Max(max[0] - min[0], Math.Max(max[1] - min[1], max[2] - min[2])) / 2 * options.BoundingBoxFactor;
        var normalized = new List<NeuSCamera>();
        for (int i = 0; i < cameras.Count; i++)
        {
            var (r, c) = relative[i];
            var center = new[] { (c[0] - boxCenter[0]) / radius, (c[1] - boxCenter[1]) / radius, (c[2] - boxCenter[2]) / radius };
            normalized.Add(new NeuSCamera(cameras[i].FocalX, cameras[i].FocalY, cameras[i].PrincipalX, cameras[i].PrincipalY, r, center));
        }

        var referenceCenter = new[] { -boxCenter[0] / radius, -boxCenter[1] / radius, -boxCenter[2] / radius };
        return new One2345SceneFrame(normalized, (double[])rr.Clone(), (double[])reference.Center.Clone(), boxCenter, radius, referenceCenter);
    }

    /// <summary>R v for a row-major 3×3 R.</summary>
    public static double[] Rotate(double[] r, double[] v)
        => new[]
        {
            r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
            r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
            r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
        };

    /// <summary>Rᵀ v for a row-major 3×3 R.</summary>
    public static double[] RotateInverse(double[] r, double[] v)
        => new[]
        {
            r[0] * v[0] + r[3] * v[1] + r[6] * v[2],
            r[1] * v[0] + r[4] * v[1] + r[7] * v[2],
            r[2] * v[0] + r[5] * v[1] + r[8] * v[2],
        };

    private static double[] MultiplyTransposeLeft(double[] a, double[] b)
    {
        var result = new double[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result[i * 3 + j] = a[i] * b[j] + a[3 + i] * b[3 + j] + a[6 + i] * b[6 + j];
        return result;
    }

    private static double[] Add(double[] a, double[] b) => new[] { a[0] + b[0], a[1] + b[1], a[2] + b[2] };

    private static double[] Subtract(double[] a, double[] b) => new[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] };

    private static double[] Cross(double[] a, double[] b)
        => new[] { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };

    // calc_pose's normalize: v / (|v| + 1e-10), so a degenerate cross product stays near zero and triggers the fallback.
    private static double[] Normalize(double[] v)
    {
        double n = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-10;
        return new[] { v[0] / n, v[1] / n, v[2] / n };
    }
}

/// <summary>The normalized reconstruction frame and the map from it back to the object (world) frame.</summary>
internal sealed class One2345SceneFrame
{
    public One2345SceneFrame(IReadOnlyList<NeuSCamera> cameras, double[] referenceRotation, double[] referenceCenter,
        double[] boxCenter, double radius, double[] queryCenter)
    {
        Cameras = cameras;
        ReferenceRotation = referenceRotation;
        ReferenceCenter = referenceCenter;
        BoxCenter = boxCenter;
        Radius = radius;
        QueryCenter = queryCenter;
    }

    /// <summary>The cameras in the normalized frame, in the input order.</summary>
    public IReadOnlyList<NeuSCamera> Cameras { get; }

    /// <summary>The reference camera's world rotation (row-major, OpenCV).</summary>
    public double[] ReferenceRotation { get; }

    /// <summary>The reference camera's world center.</summary>
    public double[] ReferenceCenter { get; }

    /// <summary>The bounding-box center in the reference camera frame (<c>scale_mat[:3, 3]</c>).</summary>
    public double[] BoxCenter { get; }

    /// <summary>The normalization radius (<c>scale_mat[0, 0]</c>).</summary>
    public double Radius { get; }

    /// <summary>The reference camera's center in the normalized frame (the colour query camera, <c>query_c2w</c>).</summary>
    public double[] QueryCenter { get; }

    /// <summary>
    /// The world point of a normalized point: <c>trans_mat · (scale_mat · p)</c>, i.e. <c>R_ref (r p + c) + C_ref</c>.
    /// </summary>
    public double[] ToWorld(double[] p)
    {
        var local = new[] { p[0] * Radius + BoxCenter[0], p[1] * Radius + BoxCenter[1], p[2] * Radius + BoxCenter[2] };
        var world = One2345CameraLayout.Rotate(ReferenceRotation, local);
        return new[] { world[0] + ReferenceCenter[0], world[1] + ReferenceCenter[1], world[2] + ReferenceCenter[2] };
    }
}

/// <summary>
/// One-2-3-45's elevation estimation (<c>elevation_estimate/utils/elev_est_api.py</c>): for each candidate polar
/// angle, the four nearby views of the first view get hypothesis cameras (<c>gen_pose_hypothesis</c>); matches of
/// one view pair are triangulated and reprojected into the other two views (<c>ba_error_general</c>), and the
/// confidence-weighted reprojection error is summed over three rotations of the view order
/// (<c>find_optim_elev</c>). A coarse search (30 to 150 by 10) is refined at 1° around its minimum.
/// </summary>
internal static class One2345ElevationEstimator
{
    /// <summary>Estimates the input's polar angle from the four nearby views' pairwise matches.</summary>
    /// <param name="options">The search configuration.</param>
    /// <param name="matches">Matches for each pair (i, j), i &lt; j, of the 4 views.</param>
    /// <param name="focal">The focal length in pixels.</param>
    /// <param name="principalX">The principal point x.</param>
    /// <param name="principalY">The principal point y.</param>
    /// <returns>The polar angle in degrees, or <see cref="One2345Options.FallbackPolar"/> when a pair has no match.</returns>
    public static double Estimate(One2345Options options, IReadOnlyDictionary<(int, int), IReadOnlyList<FeatureCorrespondence>> matches,
        double focal, double principalX, double principalY)
    {
        var all = new Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>>();
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
            {
                if (!matches.TryGetValue((i, j), out var forward) || forward is null || forward.Count == 0)
                    return options.FallbackPolar; // get_elev_est: "0 matches, could not estimate elevation"
                all[(i, j)] = forward;
                all[(j, i)] = forward.Select(m => m.Reversed()).ToList();
            }

        double coarse = Search(options, all, focal, principalX, principalY,
            Range(options.ElevationSearchMinPolar, options.ElevationSearchMaxPolar, options.ElevationCoarseStep));
        return Search(options, all, focal, principalX, principalY,
            Range(coarse - options.ElevationFineHalfWidth, coarse + options.ElevationFineHalfWidth, options.ElevationFineStep));
    }

    /// <summary><c>np.arange(start, stop, step)</c>.</summary>
    public static double[] Range(double start, double stop, double step)
    {
        int count = (int)Math.Ceiling((stop - start) / step - 1e-9);
        var values = new double[Math.Max(0, count)];
        for (int i = 0; i < values.Length; i++) values[i] = start + i * step;
        return values;
    }

    /// <summary>The four hypothesis cameras for a candidate polar angle (<c>gen_pose_hypothesis</c>).</summary>
    public static NeuSCamera[] Hypothesis(One2345Options options, double polar, double focal, double principalX, double principalY)
    {
        var parent = new One2345ViewPose(0.0, 0.0, polar, options.InputAzimuth);
        var poses = One2345CameraLayout.Stage2(options, parent);
        var cameras = new NeuSCamera[4];
        for (int k = 0; k < 4; k++)
        {
            var center = One2345CameraLayout.Center(poses[k].Polar, poses[k].Azimuth, options.CameraRadius);
            cameras[k] = new NeuSCamera(focal, focal, principalX, principalY, One2345CameraLayout.OpenCvRotation(center), center);
        }

        return cameras;
    }

    /// <summary>The total reprojection error of one polar hypothesis (one term of <c>find_optim_elev</c>).</summary>
    public static double Error(One2345Options options, IReadOnlyDictionary<(int, int), IReadOnlyList<FeatureCorrespondence>> matches,
        double polar, double focal, double principalX, double principalY)
    {
        var cameras = Hypothesis(options, polar, focal, principalX, principalY);
        const int views = 4;
        double error = 0;
        for (int start = 0; start < views - 1; start++)
        {
            var batchCameras = new NeuSCamera[views];
            for (int i = 0; i < views; i++) batchCameras[i] = cameras[(start + i) % views];
            var batchMatches = new IReadOnlyList<FeatureCorrespondence>[views - 1];
            for (int j = 0; j < views - 1; j++) batchMatches[j] = matches[(start, (start + j + 1) % views)];
            error += BundleError(options, batchMatches, batchCameras);
        }

        return error;
    }

    /// <summary><c>ba_error_general</c>: triangulate the first pair's matches and reproject them into the rest.</summary>
    public static double BundleError(One2345Options options, IReadOnlyList<FeatureCorrespondence>[] matches, NeuSCamera[] cameras)
    {
        var reference = matches[0];
        var points = new double[reference.Count][];
        for (int p = 0; p < reference.Count; p++)
            points[p] = Triangulate(cameras[0], cameras[1], reference[p].X0, reference[p].Y0, reference[p].X1, reference[p].Y1);

        double error = 0;
        for (int k = 1; k < matches.Length; k++)
        {
            var other = matches[k];
            if (other.Count == 0 || reference.Count == 0) continue;
            var camera = cameras[k + 1];
            double weighted = 0, confidence = 0;
            bool any = false;
            for (int p = 0; p < reference.Count; p++)
            {
                int nearest = -1;
                double best = double.PositiveInfinity;
                for (int q = 0; q < other.Count; q++)
                {
                    double dx = reference[p].X0 - other[q].X0, dy = reference[p].Y0 - other[q].Y0;
                    double d = Math.Sqrt(dx * dx + dy * dy);
                    if (d < best)
                    {
                        best = d;
                        nearest = q;
                    }
                }

                if (!(best < options.CorrespondenceTolerance)) continue;
                any = true;
                var (u, v, _) = camera.Project(points[p][0], points[p][1], points[p][2]);
                double ex = other[nearest].X1 - u, ey = other[nearest].Y1 - v;
                weighted += Math.Sqrt(ex * ex + ey * ey) * other[nearest].Confidence;
                confidence += other[nearest].Confidence;
            }

            if (any) error += weighted / confidence;
        }

        return error;
    }

    /// <summary>
    /// Linear (DLT) triangulation, as <c>cv2.triangulatePoints</c>: the null vector of the 4×4 system
    /// <c>[x P₃ − P₁; y P₃ − P₂]</c> over both views, dehomogenized.
    /// </summary>
    public static double[] Triangulate(NeuSCamera a, NeuSCamera b, double xa, double ya, double xb, double yb)
    {
        var pa = Projection(a);
        var pb = Projection(b);
        var rows = new double[4][];
        rows[0] = Row(pa, xa, 0);
        rows[1] = Row(pa, ya, 1);
        rows[2] = Row(pb, xb, 0);
        rows[3] = Row(pb, yb, 1);
        var ata = new double[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
            {
                double s = 0;
                for (int k = 0; k < 4; k++) s += rows[k][i] * rows[k][j];
                ata[i * 4 + j] = s;
            }

        var x = SmallestEigenvector(ata);
        return new[] { x[0] / x[3], x[1] / x[3], x[2] / x[3] };
    }

    /// <summary>The 3×4 projection <c>K [Rᵀ | −Rᵀ C]</c> (row-major).</summary>
    public static double[] Projection(NeuSCamera camera)
    {
        var r = camera.Rotation;
        var c = camera.Center;
        var p = new double[12];
        double[] k = { camera.FocalX, 0, camera.PrincipalX, 0, camera.FocalY, camera.PrincipalY, 0, 0, 1 };
        var rt = new double[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                rt[i * 3 + j] = r[j * 3 + i];
        var t = new double[3];
        for (int i = 0; i < 3; i++) t[i] = -(rt[i * 3] * c[0] + rt[i * 3 + 1] * c[1] + rt[i * 3 + 2] * c[2]);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                p[i * 4 + j] = k[i * 3] * rt[j] + k[i * 3 + 1] * rt[3 + j] + k[i * 3 + 2] * rt[6 + j];
            p[i * 4 + 3] = k[i * 3] * t[0] + k[i * 3 + 1] * t[1] + k[i * 3 + 2] * t[2];
        }

        return p;
    }

    private static double Search(One2345Options options, IReadOnlyDictionary<(int, int), IReadOnlyList<FeatureCorrespondence>> matches,
        double focal, double principalX, double principalY, double[] candidates)
    {
        if (candidates.Length == 0) throw new ArgumentException("The elevation search has no candidates.", nameof(candidates));
        double bestPolar = candidates[0];
        double best = double.PositiveInfinity;
        foreach (double polar in candidates)
        {
            double e = Error(options, matches, polar, focal, principalX, principalY);
            // torch.argmin keeps the first minimum; a NaN error (a degenerate triangulation) never wins.
            if (e < best)
            {
                best = e;
                bestPolar = polar;
            }
        }

        return bestPolar;
    }

    private static double[] Row(double[] p, double coordinate, int axis)
    {
        var row = new double[4];
        for (int j = 0; j < 4; j++) row[j] = coordinate * p[8 + j] - p[axis * 4 + j];
        return row;
    }

    /// <summary>The unit eigenvector of the smallest eigenvalue of a symmetric 4×4 matrix (cyclic Jacobi).</summary>
    public static double[] SmallestEigenvector(double[] symmetric)
    {
        const int n = 4;
        var a = (double[])symmetric.Clone();
        var v = new double[n * n];
        for (int i = 0; i < n; i++) v[i * n + i] = 1.0;
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0, scale = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    if (i != j) off += a[i * n + j] * a[i * n + j];
                    scale += a[i * n + j] * a[i * n + j];
                }

            if (off <= 1e-30 * Math.Max(scale, 1e-300)) break;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    if (apq == 0.0) continue;
                    double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    if (theta == 0.0) t = 1.0;
                    double c = 1.0 / Math.Sqrt(t * t + 1.0), s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k * n + p], akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }

                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p * n + k], aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }

                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k * n + p], vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
        }

        int smallest = 0;
        for (int i = 1; i < n; i++)
            if (a[i * n + i] < a[smallest * n + smallest]) smallest = i;
        var result = new double[n];
        for (int k = 0; k < n; k++) result[k] = v[k * n + smallest];
        return result;
    }
}
