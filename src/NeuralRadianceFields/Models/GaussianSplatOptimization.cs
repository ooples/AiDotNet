namespace AiDotNet.NeuralRadianceFields.Models;

/// <summary>
/// A pinhole camera for the differentiable Gaussian rasterizer, in the convention of the 3D Gaussian Splatting
/// rasterizer (Kerbl et al. 2023): camera-space x to the right, y down, z forward, pixel centres at integer
/// coordinates and the principal point at the image centre.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="CameraToWorld"/> is row-major 3x3 whose columns are the world-space directions of the camera's right,
/// down and forward axes, which is the convention <see cref="GaussianSplatting{T}.RenderImage"/> uses. A world point
/// p has camera coordinates <c>R^T (p - c)</c> and lands on pixel <c>(fx x / z + (W - 1) / 2, fy y / z + (H - 1) / 2)</c>,
/// exactly the <c>ndc2Pix</c> mapping of the reference rasterizer.
/// </para>
/// </remarks>
internal sealed class GaussianSplatCamera
{
    /// <summary>Creates a camera.</summary>
    /// <param name="position">Camera centre in world space, three values.</param>
    /// <param name="cameraToWorld">Row-major 3x3 rotation whose columns are the right, down and forward axes.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="tanHalfFovX">tan(fovx / 2).</param>
    /// <param name="tanHalfFovY">tan(fovy / 2).</param>
    public GaussianSplatCamera(double[] position, double[] cameraToWorld, int width, int height, double tanHalfFovX, double tanHalfFovY)
    {
        if (position is null || position.Length != 3)
            throw new ArgumentException("The camera position needs three values.", nameof(position));
        if (cameraToWorld is null || cameraToWorld.Length != 9)
            throw new ArgumentException("The camera rotation needs nine values (row-major 3x3).", nameof(cameraToWorld));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");
        if (!(tanHalfFovX > 0.0)) throw new ArgumentOutOfRangeException(nameof(tanHalfFovX), "tan(fovx / 2) must be positive.");
        if (!(tanHalfFovY > 0.0)) throw new ArgumentOutOfRangeException(nameof(tanHalfFovY), "tan(fovy / 2) must be positive.");

        Position = (double[])position.Clone();
        CameraToWorld = (double[])cameraToWorld.Clone();
        Width = width;
        Height = height;
        TanHalfFovX = tanHalfFovX;
        TanHalfFovY = tanHalfFovY;
    }

    /// <summary>Camera centre in world space.</summary>
    public double[] Position { get; }

    /// <summary>Row-major camera-to-world rotation; columns are the right, down and forward axes.</summary>
    public double[] CameraToWorld { get; }

    /// <summary>Image width in pixels.</summary>
    public int Width { get; }

    /// <summary>Image height in pixels.</summary>
    public int Height { get; }

    /// <summary>tan(fovx / 2).</summary>
    public double TanHalfFovX { get; }

    /// <summary>tan(fovy / 2).</summary>
    public double TanHalfFovY { get; }

    /// <summary>Horizontal focal length in pixels, W / (2 tan(fovx / 2)).</summary>
    public double FocalX => Width / (2.0 * TanHalfFovX);

    /// <summary>Vertical focal length in pixels, H / (2 tan(fovy / 2)).</summary>
    public double FocalY => Height / (2.0 * TanHalfFovY);

    /// <summary>
    /// The camera of DreamGaussian's <c>orbit_camera(elevation, azimuth, radius)</c> looking at the origin, with
    /// <c>fovx = 2 atan(tan(fovy / 2) W / H)</c> as its <c>OrbitCamera.fovx</c>.
    /// </summary>
    /// <param name="elevationDegrees">Elevation in degrees; negative looks from above (the reference's convention:
    /// "from +y to -y is (-90, 90)").</param>
    /// <param name="azimuthDegrees">Azimuth in degrees; 0 is on +z and 90 on +x.</param>
    /// <param name="radius">Distance from the origin.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="fovYDegrees">Vertical field of view in degrees.</param>
    /// <returns>The camera.</returns>
    /// <remarks>
    /// The reference builds an OpenGL camera-to-world pose (x right, y up, z backward) and the rasterizer flips y and
    /// z (<c>w2c[1:3, :3] *= -1</c>), so the right axis is kept and the down and forward axes are the negated up and
    /// backward axes.
    /// </remarks>
    public static GaussianSplatCamera FromOrbit(
        double elevationDegrees, double azimuthDegrees, double radius, int width, int height, double fovYDegrees)
    {
        double elevation = elevationDegrees * Math.PI / 180.0;
        double azimuth = azimuthDegrees * Math.PI / 180.0;
        double cx = radius * Math.Cos(elevation) * Math.Sin(azimuth);
        double cy = -radius * Math.Sin(elevation);
        double cz = radius * Math.Cos(elevation) * Math.Cos(azimuth);

        // look_at(campos, target = 0, opengl = True).
        var backward = SafeNormalize(cx, cy, cz);
        var right = SafeNormalize(Cross(0.0, 1.0, 0.0, backward[0], backward[1], backward[2]));
        var up = SafeNormalize(Cross(backward[0], backward[1], backward[2], right[0], right[1], right[2]));

        var rotation = new[]
        {
            right[0], -up[0], -backward[0],
            right[1], -up[1], -backward[1],
            right[2], -up[2], -backward[2],
        };

        double tanY = Math.Tan(0.5 * fovYDegrees * Math.PI / 180.0);
        double tanX = tanY * width / height;
        return new GaussianSplatCamera(new[] { cx, cy, cz }, rotation, width, height, tanX, tanY);
    }

    private static double[] Cross(double ax, double ay, double az, double bx, double by, double bz)
        => new[] { (ay * bz) - (az * by), (az * bx) - (ax * bz), (ax * by) - (ay * bx) };

    private static double[] SafeNormalize(double[] v) => SafeNormalize(v[0], v[1], v[2]);

    private static double[] SafeNormalize(double x, double y, double z)
    {
        // cam_utils.safe_normalize: x / sqrt(max(|x|^2, 1e-20)).
        double length = Math.Sqrt(Math.Max((x * x) + (y * y) + (z * z), 1e-20));
        return new[] { x / length, y / length, z / length };
    }
}

/// <summary>
/// The output of <see cref="GaussianSplatting{T}.RenderDifferentiable"/>: the image, its accumulated opacity, each
/// Gaussian's screen radius, and the per-pixel state the backward pass replays.
/// </summary>
internal sealed class GaussianSplatRender
{
    internal GaussianSplatRender(int width, int height, int gaussianCount, int stateVersion)
    {
        Width = width;
        Height = height;
        GaussianCount = gaussianCount;
        StateVersion = stateVersion;
        Image = new double[width * height * 3];
        UnclampedImage = new double[width * height * 3];
        Alpha = new double[width * height];
        FinalTransmittance = new double[width * height];
        Radii = new int[gaussianCount];
        MeanX = new double[gaussianCount];
        MeanY = new double[gaussianCount];
        ConicA = new double[gaussianCount];
        ConicB = new double[gaussianCount];
        ConicC = new double[gaussianCount];
        CovA = new double[gaussianCount];
        CovB = new double[gaussianCount];
        CovC = new double[gaussianCount];
        Opacity = new double[gaussianCount];
        ColorR = new double[gaussianCount];
        ColorG = new double[gaussianCount];
        ColorB = new double[gaussianCount];
        ColorClamped = new bool[3 * gaussianCount];
        ViewX = new double[gaussianCount];
        ViewY = new double[gaussianCount];
        ViewZ = new double[gaussianCount];
        ClampedX = new bool[gaussianCount];
        ClampedY = new bool[gaussianCount];
        Background = new double[3];
        TileLists = Array.Empty<int[]>();
    }

    /// <summary>Image width.</summary>
    public int Width { get; }

    /// <summary>Image height.</summary>
    public int Height { get; }

    /// <summary>The number of Gaussians rendered.</summary>
    public int GaussianCount { get; }

    /// <summary>The training-state version the render was made from; the backward pass checks it.</summary>
    internal int StateVersion { get; }

    /// <summary>Rendered colour [H, W, 3] row-major, clamped to [0, 1] as the reference renderer returns it.</summary>
    public double[] Image { get; }

    /// <summary>Colour before the clamp, C + T bg; the clamp's gradient mask is read from it.</summary>
    internal double[] UnclampedImage { get; }

    /// <summary>Accumulated opacity 1 - T_final, [H, W].</summary>
    public double[] Alpha { get; }

    /// <summary>Transmittance left after the last contributor, per pixel.</summary>
    internal double[] FinalTransmittance { get; }

    /// <summary>Screen-space radius per Gaussian in pixels; 0 means culled (the reference's visibility filter).</summary>
    public int[] Radii { get; }

    internal double[] MeanX { get; }
    internal double[] MeanY { get; }
    internal double[] ConicA { get; }
    internal double[] ConicB { get; }
    internal double[] ConicC { get; }
    internal double[] CovA { get; }
    internal double[] CovB { get; }
    internal double[] CovC { get; }
    internal double[] Opacity { get; }
    internal double[] ColorR { get; }
    internal double[] ColorG { get; }
    internal double[] ColorB { get; }
    internal bool[] ColorClamped { get; }
    internal double[] ViewX { get; }
    internal double[] ViewY { get; }
    internal double[] ViewZ { get; }
    internal bool[] ClampedX { get; }
    internal bool[] ClampedY { get; }
    internal double[] Background { get; }
    internal double FocalX { get; set; }
    internal double FocalY { get; set; }
    internal double[] CameraToWorld { get; set; } = new double[9];
    internal int TilesX { get; set; }
    internal int TilesY { get; set; }

    /// <summary>Per 16x16 tile, the Gaussians overlapping it in front-to-back order.</summary>
    internal int[][] TileLists { get; set; }
}

/// <summary>
/// Gradients of a loss with respect to every Gaussian's parameters, in the parameterisation the optimiser updates:
/// centre, log standard deviation, raw (unnormalised) quaternion, opacity logit and DC colour coefficient.
/// </summary>
internal sealed class GaussianSplatGradients
{
    /// <summary>Allocates zero gradients for <paramref name="count"/> Gaussians.</summary>
    public GaussianSplatGradients(int count)
    {
        Count = count;
        Position = new double[3 * count];
        LogScale = new double[3 * count];
        Rotation = new double[4 * count];
        OpacityLogit = new double[count];
        Color = new double[3 * count];
        MeanNdc = new double[2 * count];
    }

    /// <summary>The number of Gaussians.</summary>
    public int Count { get; }

    /// <summary>dL/d centre, [N, 3].</summary>
    public double[] Position { get; }

    /// <summary>dL/d log standard deviation, [N, 3].</summary>
    public double[] LogScale { get; }

    /// <summary>dL/d raw quaternion (w, x, y, z), [N, 4].</summary>
    public double[] Rotation { get; }

    /// <summary>dL/d opacity logit, [N].</summary>
    public double[] OpacityLogit { get; }

    /// <summary>dL/d DC colour coefficient, [N, 3].</summary>
    public double[] Color { get; }

    /// <summary>dL/d screen-space centre in normalised device coordinates (the reference's viewspace_points grad), [N, 2].</summary>
    public double[] MeanNdc { get; }

    /// <summary>Adds <paramref name="other"/> into this gradient.</summary>
    public void Add(GaussianSplatGradients other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (other.Count != Count) throw new ArgumentException("Gradient counts differ.", nameof(other));
        AddInto(Position, other.Position);
        AddInto(LogScale, other.LogScale);
        AddInto(Rotation, other.Rotation);
        AddInto(OpacityLogit, other.OpacityLogit);
        AddInto(Color, other.Color);
        AddInto(MeanNdc, other.MeanNdc);
    }

    private static void AddInto(double[] target, double[] source)
    {
        for (int i = 0; i < target.Length; i++)
            target[i] += source[i];
    }
}

/// <summary>
/// Adam settings per Gaussian attribute, as the reference's <c>training_setup</c> builds its optimiser
/// (<c>torch.optim.Adam(groups, lr=0.0, eps=1e-15)</c>) and its exponential centre schedule.
/// </summary>
internal sealed class GaussianAdamSettings
{
    /// <summary>Initial centre learning rate. DreamGaussian: 0.001.</summary>
    public double PositionLearningRateInit { get; set; } = 0.001;

    /// <summary>Final centre learning rate. DreamGaussian: 0.00002.</summary>
    public double PositionLearningRateFinal { get; set; } = 0.00002;

    /// <summary>Delay multiplier of the centre schedule. DreamGaussian: 0.02.</summary>
    public double PositionLearningRateDelayMultiplier { get; set; } = 0.02;

    /// <summary>Delay steps of the centre schedule. DreamGaussian: 0 (the multiplier has no effect).</summary>
    public int PositionLearningRateDelaySteps { get; set; }

    /// <summary>Steps over which the centre rate decays. DreamGaussian: 500 (image) or 300 (text).</summary>
    public int PositionLearningRateMaxSteps { get; set; } = 500;

    /// <summary>DC colour learning rate. DreamGaussian: 0.01.</summary>
    public double FeatureLearningRate { get; set; } = 0.01;

    /// <summary>Opacity learning rate. DreamGaussian: 0.05.</summary>
    public double OpacityLearningRate { get; set; } = 0.05;

    /// <summary>Log-scale learning rate. DreamGaussian: 0.005.</summary>
    public double ScalingLearningRate { get; set; } = 0.005;

    /// <summary>Quaternion learning rate. DreamGaussian: 0.005.</summary>
    public double RotationLearningRate { get; set; } = 0.005;

    /// <summary>Adam beta1 (PyTorch default 0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Adam beta2 (PyTorch default 0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Adam epsilon. The reference: 1e-15.</summary>
    public double Epsilon { get; set; } = 1e-15;

    /// <summary>
    /// The reference's <c>get_expon_lr_func</c>: a log-linear interpolation from the initial to the final rate over
    /// <see cref="PositionLearningRateMaxSteps"/>, scaled by the reverse-cosine delay when delay steps are set.
    /// </summary>
    public double PositionLearningRate(int step)
    {
        double init = PositionLearningRateInit, final = PositionLearningRateFinal;
        if (init == final) return init;
        if (step < 0 || (init == 0.0 && final == 0.0)) return 0.0;
        double delayRate = 1.0;
        if (PositionLearningRateDelaySteps > 0)
        {
            double ratio = Math.Min(1.0, Math.Max(0.0, (double)step / PositionLearningRateDelaySteps));
            delayRate = PositionLearningRateDelayMultiplier
                + ((1.0 - PositionLearningRateDelayMultiplier) * Math.Sin(0.5 * Math.PI * ratio));
        }

        double t = PositionLearningRateMaxSteps > 0
            ? Math.Min(1.0, Math.Max(0.0, (double)step / PositionLearningRateMaxSteps))
            : 1.0;
        return delayRate * Math.Exp((Math.Log(init) * (1.0 - t)) + (Math.Log(final) * t));
    }
}

/// <summary>
/// Adaptive density control settings, as DreamGaussian calls
/// <c>densify_and_prune(densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)</c>.
/// </summary>
internal sealed class GaussianDensificationSettings
{
    /// <summary>Average NDC gradient norm at or above which a Gaussian is cloned or split. DreamGaussian: 0.01.</summary>
    public double GradientThreshold { get; set; } = 0.01;

    /// <summary>Opacity below which a Gaussian is pruned. DreamGaussian: 0.01.</summary>
    public double MinOpacity { get; set; } = 0.01;

    /// <summary>Scene extent. DreamGaussian: 4.</summary>
    public double SceneExtent { get; set; } = 4.0;

    /// <summary>Screen-radius prune limit in pixels; 0 disables it. DreamGaussian: 1.</summary>
    public double MaxScreenSize { get; set; } = 1.0;

    /// <summary>Fraction of the extent separating clone (small) from split (large). DreamGaussian: 0.01.</summary>
    public double PercentDense { get; set; } = 0.01;

    /// <summary>Children per split Gaussian. Reference: 2.</summary>
    public int SplitCount { get; set; } = 2;

    /// <summary>Split children's scale is divided by this times <see cref="SplitCount"/>. Reference: 0.8.</summary>
    public double SplitScaleDivisor { get; set; } = 0.8;

    /// <summary>Gaussians whose largest scale exceeds this fraction of the extent are pruned. Reference: 0.1.</summary>
    public double WorldSizePruneFraction { get; set; } = 0.1;
}

/// <summary>What one adaptive density control pass did.</summary>
internal readonly struct GaussianDensificationReport
{
    public GaussianDensificationReport(int cloned, int split, int pruned, int count)
    {
        Cloned = cloned;
        Split = split;
        Pruned = pruned;
        Count = count;
    }

    /// <summary>Gaussians cloned.</summary>
    public int Cloned { get; }

    /// <summary>Gaussians split (each replaced by <see cref="GaussianDensificationSettings.SplitCount"/> children).</summary>
    public int Split { get; }

    /// <summary>Gaussians pruned after densification.</summary>
    public int Pruned { get; }

    /// <summary>Gaussians left.</summary>
    public int Count { get; }
}
