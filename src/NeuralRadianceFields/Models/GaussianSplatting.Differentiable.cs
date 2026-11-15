using AiDotNet.Attributes;

namespace AiDotNet.NeuralRadianceFields.Models;

/// <summary>
/// The differentiable tile rasterizer of 3D Gaussian Splatting (Kerbl et al. 2023, section 6 and the official
/// <c>diff-gaussian-rasterization</c> CUDA kernels, in the fork DreamGaussian uses that also returns the accumulated
/// opacity), with its analytic backward pass.
/// </summary>
/// <remarks>
/// <para>
/// Forward, per Gaussian: the camera-space centre t; frustum culling at <c>t_z &lt;= 0.2</c>; the EWA projection
/// <c>Sigma' = J W Sigma W^T J^T + 0.3 I</c> with <c>t_x / t_z</c> and <c>t_y / t_z</c> clamped to 1.3 tan(fov / 2);
/// the conic (inverse of Sigma'); the radius <c>ceil(3 sqrt(lambda_max))</c> with <c>lambda = mid +- sqrt(max(0.1,
/// mid^2 - det))</c>; the 16x16 tile rectangle (a Gaussian whose rectangle covers no tile is culled); the degree-0
/// colour clamped at 0. Per pixel, front to back over its tile's depth-sorted list:
/// <c>alpha = min(0.99, o exp(power))</c>, skipped when <c>power &gt; 0</c> or <c>alpha &lt; 1/255</c>, stopping before
/// the Gaussian that would take the transmittance below 1e-4; the colour is <c>sum c alpha T + T_final bg</c> and the
/// opacity <c>1 - T_final</c>. The colour is then clamped to [0, 1], as DreamGaussian's renderer returns it.
/// </para>
/// <para>
/// Backward follows the reference's <c>renderCUDA</c> / <c>preprocessCUDA</c> backward: the 0.99 alpha clamp passes
/// its gradient through, the conic is differentiated as the inverse of the dilated 2D covariance, the projection
/// Jacobian contributes to the centre gradient (with the reference's zero gradient on a clamped axis), and the
/// rotation gradient goes through the quaternion normalisation that the reference applies in Python. The screen-space
/// centre gradient is also returned in normalised device coordinates (<c>dL/dpixel * W / 2</c>), which is what the
/// reference accumulates for densification. The arithmetic is double precision on the host.
/// </para>
/// </remarks>
public partial class GaussianSplatting<T>
{
    private const int RasterTileSize = 16;
    private const double RasterNearPlane = 0.2;
    private const double RasterCovarianceDilation = 0.3;
    private const double RasterAlphaMax = 0.99;
    private const double RasterAlphaMin = 1.0 / 255.0;
    private const double RasterTransmittanceMin = 0.0001;
    private const double RasterFrustumSlack = 1.3;

    /// <summary>
    /// Renders the generative state from <paramref name="camera"/> over <paramref name="background"/>.
    /// </summary>
    /// <param name="camera">The camera.</param>
    /// <param name="background">The background colour, three values.</param>
    /// <returns>The render, including what <see cref="BackwardDifferentiable"/> needs.</returns>
    internal GaussianSplatRender RenderDifferentiable(GaussianSplatCamera camera, double[] background)
    {
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        if (background is null || background.Length != 3)
            throw new ArgumentException("The background needs three values.", nameof(background));

        var state = EnsureGenerativeState();
        int n = state.Count;
        int width = camera.Width, height = camera.Height;
        var render = new GaussianSplatRender(width, height, n, state.Version);
        Array.Copy(background, render.Background, 3);
        double fx = camera.FocalX, fy = camera.FocalY;
        render.FocalX = fx;
        render.FocalY = fy;
        Array.Copy(camera.CameraToWorld, render.CameraToWorld, 9);
        int tilesX = (width + RasterTileSize - 1) / RasterTileSize;
        int tilesY = (height + RasterTileSize - 1) / RasterTileSize;
        render.TilesX = tilesX;
        render.TilesY = tilesY;

        var rectangles = new int[4 * n];
        var depth = new double[n];
        ProjectGenerativeGaussians(state, camera, render, rectangles, depth);
        BuildRasterTiles(render, rectangles, depth);
        RasterizeForward(render);
        return render;
    }

    /// <summary>
    /// Back-propagates image and opacity gradients of <paramref name="render"/> to every Gaussian's parameters.
    /// </summary>
    /// <param name="render">A render of the current state.</param>
    /// <param name="imageGradient">dL/d image, [H, W, 3] row-major, for the clamped image.</param>
    /// <param name="alphaGradient">dL/d opacity image, [H, W], or null.</param>
    /// <returns>The parameter gradients.</returns>
    internal GaussianSplatGradients BackwardDifferentiable(
        GaussianSplatRender render, double[] imageGradient, double[]? alphaGradient)
    {
        if (render is null) throw new ArgumentNullException(nameof(render));
        if (imageGradient is null) throw new ArgumentNullException(nameof(imageGradient));
        var state = EnsureGenerativeState();
        if (render.StateVersion != state.Version || render.GaussianCount != state.Count)
            throw new InvalidOperationException("The render is stale: the Gaussians changed after it was made.");
        int pixels = render.Width * render.Height;
        if (imageGradient.Length != 3 * pixels)
            throw new ArgumentException($"The image gradient needs {3 * pixels} values.", nameof(imageGradient));
        if (alphaGradient is not null && alphaGradient.Length != pixels)
            throw new ArgumentException($"The opacity gradient needs {pixels} values.", nameof(alphaGradient));

        int n = state.Count;
        // Per Gaussian: d mean x, d mean y, d conic a, b, c, d opacity, d colour r, g, b.
        var screen = new double[9 * n];
        RasterizeBackward(render, imageGradient, alphaGradient, screen);

        var gradients = new GaussianSplatGradients(n);
        PropagateScreenGradients(state, render, screen, gradients);
        return gradients;
    }

    [ScalarKernel("Projects each Gaussian once with scalar 3x3 algebra, as the reference's per-Gaussian preprocess kernel does.")]
    private void ProjectGenerativeGaussians(
        GenerativeState state, GaussianSplatCamera camera, GaussianSplatRender render, int[] rectangles, double[] depth)
    {
        int n = state.Count;
        int width = camera.Width, height = camera.Height;
        double fx = render.FocalX, fy = render.FocalY;
        double principalX = 0.5 * (width - 1), principalY = 0.5 * (height - 1);
        double limitX = RasterFrustumSlack * camera.TanHalfFovX, limitY = RasterFrustumSlack * camera.TanHalfFovY;
        var r = camera.CameraToWorld;
        var c = camera.Position;
        double colorScale = GenerativeColorScale;

        System.Threading.Tasks.Parallel.For(0, n, i =>
        {
            double px = state.Position[3 * i] - c[0], py = state.Position[(3 * i) + 1] - c[1], pz = state.Position[(3 * i) + 2] - c[2];
            // t = R^T (p - c): the columns of R are the camera axes.
            double tx = (r[0] * px) + (r[3] * py) + (r[6] * pz);
            double ty = (r[1] * px) + (r[4] * py) + (r[7] * pz);
            double tz = (r[2] * px) + (r[5] * py) + (r[8] * pz);
            render.ViewX[i] = tx;
            render.ViewY[i] = ty;
            render.ViewZ[i] = tz;
            if (tz <= RasterNearPlane)
                return;

            var sigma = new double[9];
            WorldCovariance(state, i, sigma, null, null);
            var view = RotateCovarianceToCamera(sigma, r);

            double txtz = tx / tz, tytz = ty / tz;
            render.ClampedX[i] = txtz < -limitX || txtz > limitX;
            render.ClampedY[i] = tytz < -limitY || tytz > limitY;
            double clampedTx = Math.Min(limitX, Math.Max(-limitX, txtz)) * tz;
            double clampedTy = Math.Min(limitY, Math.Max(-limitY, tytz)) * tz;
            double j00 = fx / tz, j02 = -fx * clampedTx / (tz * tz);
            double j11 = fy / tz, j12 = -fy * clampedTy / (tz * tz);

            double a = (j00 * j00 * view[0]) + (2.0 * j00 * j02 * view[2]) + (j02 * j02 * view[8]) + RasterCovarianceDilation;
            double b = (j00 * ((view[1] * j11) + (view[2] * j12))) + (j02 * ((view[7] * j11) + (view[8] * j12)));
            double cc = (j11 * j11 * view[4]) + (2.0 * j11 * j12 * view[5]) + (j12 * j12 * view[8]) + RasterCovarianceDilation;
            double det = (a * cc) - (b * b);
            if (det == 0.0)
                return;

            double mid = 0.5 * (a + cc);
            double spread = Math.Sqrt(Math.Max(0.1, (mid * mid) - det));
            double lambda = Math.Max(mid + spread, mid - spread);
            int radius = (int)Math.Ceiling(3.0 * Math.Sqrt(lambda));
            double meanX = (fx * tx / tz) + principalX;
            double meanY = (fy * ty / tz) + principalY;

            int tilesX = render.TilesX, tilesY = render.TilesY;
            int minX = Math.Min(tilesX, Math.Max(0, (int)((meanX - radius) / RasterTileSize)));
            int minY = Math.Min(tilesY, Math.Max(0, (int)((meanY - radius) / RasterTileSize)));
            int maxX = Math.Min(tilesX, Math.Max(0, (int)((meanX + radius + RasterTileSize - 1) / RasterTileSize)));
            int maxY = Math.Min(tilesY, Math.Max(0, (int)((meanY + radius + RasterTileSize - 1) / RasterTileSize)));
            if ((maxX - minX) * (maxY - minY) == 0)
                return;

            rectangles[4 * i] = minX;
            rectangles[(4 * i) + 1] = minY;
            rectangles[(4 * i) + 2] = maxX;
            rectangles[(4 * i) + 3] = maxY;
            depth[i] = tz;
            render.Radii[i] = radius;
            render.MeanX[i] = meanX;
            render.MeanY[i] = meanY;
            render.CovA[i] = a;
            render.CovB[i] = b;
            render.CovC[i] = cc;
            render.ConicA[i] = cc / det;
            render.ConicB[i] = -b / det;
            render.ConicC[i] = a / det;
            render.Opacity[i] = 1.0 / (1.0 + Math.Exp(-state.OpacityLogit[i]));
            double red = colorScale * state.Color[3 * i], green = colorScale * state.Color[(3 * i) + 1], blue = colorScale * state.Color[(3 * i) + 2];
            render.ColorClamped[3 * i] = red < 0.0;
            render.ColorClamped[(3 * i) + 1] = green < 0.0;
            render.ColorClamped[(3 * i) + 2] = blue < 0.0;
            render.ColorR[i] = Math.Max(0.0, red);
            render.ColorG[i] = Math.Max(0.0, green);
            render.ColorB[i] = Math.Max(0.0, blue);
        });
    }

    private static void BuildRasterTiles(GaussianSplatRender render, int[] rectangles, double[] depth)
    {
        int n = render.GaussianCount;
        var visible = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (render.Radii[i] > 0)
                visible.Add(i);
        }

        // The reference sorts (tile, depth) keys with a radix sort; ties are broken by index here so the order is
        // deterministic.
        visible.Sort((x, y) =>
        {
            int byDepth = depth[x].CompareTo(depth[y]);
            return byDepth != 0 ? byDepth : x.CompareTo(y);
        });

        int tileCount = render.TilesX * render.TilesY;
        var lists = new List<int>[tileCount];
        foreach (int g in visible)
        {
            for (int ty = rectangles[(4 * g) + 1]; ty < rectangles[(4 * g) + 3]; ty++)
            {
                for (int tx = rectangles[4 * g]; tx < rectangles[(4 * g) + 2]; tx++)
                {
                    int tile = (ty * render.TilesX) + tx;
                    var list = lists[tile];
                    if (list is null)
                    {
                        list = new List<int>();
                        lists[tile] = list;
                    }

                    list.Add(g);
                }
            }
        }

        var tiles = new int[tileCount][];
        for (int t = 0; t < tileCount; t++)
            tiles[t] = lists[t] is null ? Array.Empty<int>() : lists[t].ToArray();
        render.TileLists = tiles;
    }

    private static void RasterizeForward(GaussianSplatRender render)
    {
        int width = render.Width, height = render.Height;
        var bg = render.Background;
        System.Threading.Tasks.Parallel.For(0, render.TilesX * render.TilesY, tile =>
        {
            var list = render.TileLists[tile];
            int x0 = (tile % render.TilesX) * RasterTileSize, y0 = (tile / render.TilesX) * RasterTileSize;
            int x1 = Math.Min(width, x0 + RasterTileSize), y1 = Math.Min(height, y0 + RasterTileSize);
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    double transmittance = 1.0, red = 0.0, green = 0.0, blue = 0.0;
                    for (int k = 0; k < list.Length; k++)
                    {
                        int g = list[k];
                        double alpha = SplatAlpha(render, g, x, y, out _, out _, out _);
                        if (alpha < RasterAlphaMin)
                            continue;
                        double next = transmittance * (1.0 - alpha);
                        if (next < RasterTransmittanceMin)
                            break;
                        double weight = alpha * transmittance;
                        red += render.ColorR[g] * weight;
                        green += render.ColorG[g] * weight;
                        blue += render.ColorB[g] * weight;
                        transmittance = next;
                    }

                    int pixel = (y * width) + x;
                    double outR = red + (transmittance * bg[0]);
                    double outG = green + (transmittance * bg[1]);
                    double outB = blue + (transmittance * bg[2]);
                    render.UnclampedImage[3 * pixel] = outR;
                    render.UnclampedImage[(3 * pixel) + 1] = outG;
                    render.UnclampedImage[(3 * pixel) + 2] = outB;
                    render.Image[3 * pixel] = Math.Min(1.0, Math.Max(0.0, outR));
                    render.Image[(3 * pixel) + 1] = Math.Min(1.0, Math.Max(0.0, outG));
                    render.Image[(3 * pixel) + 2] = Math.Min(1.0, Math.Max(0.0, outB));
                    render.Alpha[pixel] = 1.0 - transmittance;
                    render.FinalTransmittance[pixel] = transmittance;
                }
            }
        });
    }

    /// <summary>
    /// The opacity of Gaussian <paramref name="g"/> at pixel (x, y), 0 when the reference skips it
    /// (<c>power &gt; 0</c>), with the falloff G, the offset d = mean - pixel.
    /// </summary>
    private static double SplatAlpha(GaussianSplatRender render, int g, int x, int y, out double falloff, out double dx, out double dy)
    {
        dx = render.MeanX[g] - x;
        dy = render.MeanY[g] - y;
        double power = (-0.5 * ((render.ConicA[g] * dx * dx) + (render.ConicC[g] * dy * dy))) - (render.ConicB[g] * dx * dy);
        if (power > 0.0)
        {
            falloff = 0.0;
            return 0.0;
        }

        falloff = Math.Exp(power);
        return Math.Min(RasterAlphaMax, render.Opacity[g] * falloff);
    }

    private static void RasterizeBackward(
        GaussianSplatRender render, double[] imageGradient, double[]? alphaGradient, double[] screen)
    {
        int width = render.Width, height = render.Height;
        int tileCount = render.TilesX * render.TilesY;
        var partial = new double[tileCount][];
        var bg = render.Background;
        System.Threading.Tasks.Parallel.For(0, tileCount, tile =>
        {
            var list = render.TileLists[tile];
            if (list.Length == 0)
                return;
            var local = new double[9 * list.Length];
            partial[tile] = local;
            int x0 = (tile % render.TilesX) * RasterTileSize, y0 = (tile / render.TilesX) * RasterTileSize;
            int x1 = Math.Min(width, x0 + RasterTileSize), y1 = Math.Min(height, y0 + RasterTileSize);
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    int pixel = (y * width) + x;
                    // torch.clamp passes the gradient where min <= x <= max.
                    double gR = PassClamp(render.UnclampedImage[3 * pixel]) ? imageGradient[3 * pixel] : 0.0;
                    double gG = PassClamp(render.UnclampedImage[(3 * pixel) + 1]) ? imageGradient[(3 * pixel) + 1] : 0.0;
                    double gB = PassClamp(render.UnclampedImage[(3 * pixel) + 2]) ? imageGradient[(3 * pixel) + 2] : 0.0;
                    double gA = alphaGradient is null ? 0.0 : alphaGradient[pixel];
                    if (gR == 0.0 && gG == 0.0 && gB == 0.0 && gA == 0.0)
                        continue;

                    double totalR = render.UnclampedImage[3 * pixel];
                    double totalG = render.UnclampedImage[(3 * pixel) + 1];
                    double totalB = render.UnclampedImage[(3 * pixel) + 2];
                    double finalT = render.FinalTransmittance[pixel];
                    double transmittance = 1.0, accR = 0.0, accG = 0.0, accB = 0.0;
                    for (int k = 0; k < list.Length; k++)
                    {
                        int g = list[k];
                        double alpha = SplatAlpha(render, g, x, y, out double falloff, out double dx, out double dy);
                        if (alpha < RasterAlphaMin)
                            continue;
                        double next = transmittance * (1.0 - alpha);
                        if (next < RasterTransmittanceMin)
                            break;

                        double weight = alpha * transmittance;
                        double red = render.ColorR[g], green = render.ColorG[g], blue = render.ColorB[g];
                        int o = 9 * k;
                        local[o + 6] += weight * gR;
                        local[o + 7] += weight * gG;
                        local[o + 8] += weight * gB;

                        accR += red * weight;
                        accG += green * weight;
                        accB += blue * weight;
                        double inverse = 1.0 / (1.0 - alpha);
                        // dC/dalpha_i = T_i c_i - (everything behind i, background included) / (1 - alpha_i).
                        double dAlpha = (gR * ((transmittance * red) - ((totalR - accR) * inverse)))
                            + (gG * ((transmittance * green) - ((totalG - accG) * inverse)))
                            + (gB * ((transmittance * blue) - ((totalB - accB) * inverse)))
                            + (gA * finalT * inverse);
                        transmittance = next;

                        double opacity = render.Opacity[g];
                        local[o + 5] += falloff * dAlpha;
                        double dPower = opacity * dAlpha * falloff;
                        double conicA = render.ConicA[g], conicB = render.ConicB[g], conicC = render.ConicC[g];
                        local[o] += dPower * -((conicA * dx) + (conicB * dy));
                        local[o + 1] += dPower * -((conicC * dy) + (conicB * dx));
                        local[o + 2] += dPower * (-0.5 * dx * dx);
                        local[o + 3] += dPower * (-dx * dy);
                        local[o + 4] += dPower * (-0.5 * dy * dy);
                    }
                }
            }
        });

        // Deterministic reduction in tile order.
        for (int tile = 0; tile < tileCount; tile++)
        {
            var local = partial[tile];
            if (local is null)
                continue;
            var list = render.TileLists[tile];
            for (int k = 0; k < list.Length; k++)
            {
                int g = list[k];
                for (int j = 0; j < 9; j++)
                    screen[(9 * g) + j] += local[(9 * k) + j];
            }
        }
    }

    private static bool PassClamp(double value) => value >= 0.0 && value <= 1.0;

    [ScalarKernel("Chains each Gaussian's screen-space gradient through its 3x3 covariance algebra, as the reference's per-Gaussian backward kernel does.")]
    private void PropagateScreenGradients(
        GenerativeState state, GaussianSplatRender render, double[] screen, GaussianSplatGradients gradients)
    {
        var r = render.CameraToWorld;
        double fx = render.FocalX, fy = render.FocalY;
        double colorScale = GenerativeColorScale;
        System.Threading.Tasks.Parallel.For(0, state.Count, i =>
        {
            if (render.Radii[i] <= 0)
                return;
            int o = 9 * i;
            double dMeanX = screen[o], dMeanY = screen[o + 1];
            double gA = screen[o + 2], gB = screen[o + 3], gC = screen[o + 4];

            // Colour and opacity.
            for (int k = 0; k < 3; k++)
                gradients.Color[(3 * i) + k] = render.ColorClamped[(3 * i) + k] ? 0.0 : colorScale * screen[o + 6 + k];
            double opacity = render.Opacity[i];
            gradients.OpacityLogit[i] = screen[o + 5] * opacity * (1.0 - opacity);
            gradients.MeanNdc[2 * i] = dMeanX * 0.5 * render.Width;
            gradients.MeanNdc[(2 * i) + 1] = dMeanY * 0.5 * render.Height;

            // Conic K = inverse(X), X the dilated 2D covariance: dL/dX = -K G K with G the symmetric conic gradient
            // (the off-diagonal coefficient appears twice in the quadratic form, so each entry gets half).
            double kA = render.ConicA[i], kB = render.ConicB[i], kC = render.ConicC[i];
            double hB = 0.5 * gB;
            double p00 = (kA * gA) + (kB * hB), p01 = (kA * hB) + (kB * gC);
            double p10 = (kB * gA) + (kC * hB), p11 = (kB * hB) + (kC * gC);
            double g00 = -((p00 * kA) + (p01 * kB));
            double g01 = -((p00 * kB) + (p01 * kC));
            double g11 = -((p10 * kB) + (p11 * kC));

            double tx = render.ViewX[i], ty = render.ViewY[i], tz = render.ViewZ[i];
            double limitX = (render.Width / (2.0 * fx)) * RasterFrustumSlack;
            double limitY = (render.Height / (2.0 * fy)) * RasterFrustumSlack;
            double clampedTx = Math.Min(limitX, Math.Max(-limitX, tx / tz)) * tz;
            double clampedTy = Math.Min(limitY, Math.Max(-limitY, ty / tz)) * tz;
            double j00 = fx / tz, j02 = -fx * clampedTx / (tz * tz);
            double j11 = fy / tz, j12 = -fy * clampedTy / (tz * tz);

            var sigma = new double[9];
            var rotation = new double[9];
            var scale = new double[3];
            WorldCovariance(state, i, sigma, rotation, scale);
            var view = RotateCovarianceToCamera(sigma, r);

            // dL/dV = M^T G M with M = [[j00, 0, j02], [0, j11, j12]] (columns m0, m1, m2).
            double[] m0 = { j00, 0.0 }, m1 = { 0.0, j11 }, m2 = { j02, j12 };
            var columns = new[] { m0, m1, m2 };
            var dView = new double[9];
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    var u = columns[a];
                    var v = columns[b];
                    dView[(3 * a) + b] = (u[0] * ((g00 * v[0]) + (g01 * v[1]))) + (u[1] * ((g01 * v[0]) + (g11 * v[1])));
                }
            }

            // dL/dM = 2 G M V; only the Jacobian's non-zero entries matter.
            double a00 = (j00 * view[0]) + (j02 * view[6]);
            double a02 = (j00 * view[2]) + (j02 * view[8]);
            double a10 = (j11 * view[3]) + (j12 * view[6]);
            double a11 = (j11 * view[4]) + (j12 * view[7]);
            double a12 = (j11 * view[5]) + (j12 * view[8]);
            double a01 = (j00 * view[1]) + (j02 * view[7]);
            double dJ00 = 2.0 * ((g00 * a00) + (g01 * a10));
            double dJ02 = 2.0 * ((g00 * a02) + (g01 * a12));
            double dJ11 = 2.0 * ((g01 * a01) + (g11 * a11));
            double dJ12 = 2.0 * ((g01 * a02) + (g11 * a12));

            double tz2 = 1.0 / (tz * tz), tz3 = tz2 / tz;
            double dTx = render.ClampedX[i] ? 0.0 : -fx * tz2 * dJ02;
            double dTy = render.ClampedY[i] ? 0.0 : -fy * tz2 * dJ12;
            double dTz = (-fx * tz2 * dJ00) - (fy * tz2 * dJ11) + (2.0 * fx * clampedTx * tz3 * dJ02) + (2.0 * fy * clampedTy * tz3 * dJ12);

            // Screen-space centre u = fx tx / tz + cx, v = fy ty / tz + cy.
            dTx += dMeanX * fx / tz;
            dTy += dMeanY * fy / tz;
            dTz -= ((dMeanX * fx * tx) + (dMeanY * fy * ty)) * tz2;

            // t = R^T (p - c), so dL/dp = R dL/dt.
            for (int k = 0; k < 3; k++)
                gradients.Position[(3 * i) + k] = (r[3 * k] * dTx) + (r[(3 * k) + 1] * dTy) + (r[(3 * k) + 2] * dTz);

            // V = R^T Sigma R, so dL/dSigma = R dL/dV R^T.
            var dSigma = new double[9];
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                            sum += r[(3 * a) + k] * dView[(3 * k) + l] * r[(3 * b) + l];
                    }

                    dSigma[(3 * a) + b] = sum;
                }
            }

            // Sigma = M M^T with M = R_q S: dL/dM = 2 dL/dSigma M.
            var dRot = new double[9];
            for (int k = 0; k < 3; k++)
            {
                double dScale = 0.0;
                for (int a = 0; a < 3; a++)
                {
                    double dM = 0.0;
                    for (int b = 0; b < 3; b++)
                        dM += 2.0 * dSigma[(3 * a) + b] * rotation[(3 * b) + k] * scale[k];
                    dScale += rotation[(3 * a) + k] * dM;
                    dRot[(3 * a) + k] = scale[k] * dM;
                }

                // scale = exp(log scale).
                gradients.LogScale[(3 * i) + k] = dScale * scale[k];
            }

            QuaternionGradient(state, i, dRot, gradients.Rotation);
        });
    }

    /// <summary>
    /// The world covariance R S S^T R^T of Gaussian <paramref name="i"/> (row-major), optionally with its rotation
    /// matrix and scales.
    /// </summary>
    private static void WorldCovariance(GenerativeState state, int i, double[] sigma, double[]? rotationOut, double[]? scaleOut)
    {
        var rotation = rotationOut ?? new double[9];
        NormalizedRotation(state, i, out double w, out double x, out double y, out double z);
        rotation[0] = 1.0 - (2.0 * ((y * y) + (z * z)));
        rotation[1] = 2.0 * ((x * y) - (w * z));
        rotation[2] = 2.0 * ((x * z) + (w * y));
        rotation[3] = 2.0 * ((x * y) + (w * z));
        rotation[4] = 1.0 - (2.0 * ((x * x) + (z * z)));
        rotation[5] = 2.0 * ((y * z) - (w * x));
        rotation[6] = 2.0 * ((x * z) - (w * y));
        rotation[7] = 2.0 * ((y * z) + (w * x));
        rotation[8] = 1.0 - (2.0 * ((x * x) + (y * y)));
        var s = scaleOut ?? new double[3];
        for (int k = 0; k < 3; k++)
            s[k] = Math.Exp(state.LogScale[(3 * i) + k]);

        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                double sum = 0.0;
                for (int k = 0; k < 3; k++)
                    sum += rotation[(3 * a) + k] * s[k] * s[k] * rotation[(3 * b) + k];
                sigma[(3 * a) + b] = sum;
            }
        }
    }

    /// <summary>V = R^T Sigma R for the camera-to-world rotation R.</summary>
    private static double[] RotateCovarianceToCamera(double[] sigma, double[] r)
    {
        var view = new double[9];
        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                double sum = 0.0;
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                        sum += r[(3 * k) + a] * sigma[(3 * k) + l] * r[(3 * l) + b];
                }

                view[(3 * a) + b] = sum;
            }
        }

        return view;
    }

    private static void NormalizedRotation(GenerativeState state, int i, out double w, out double x, out double y, out double z)
    {
        w = state.Rotation[4 * i];
        x = state.Rotation[(4 * i) + 1];
        y = state.Rotation[(4 * i) + 2];
        z = state.Rotation[(4 * i) + 3];
        // F.normalize(q, eps=1e-12).
        double norm = Math.Max(1e-12, Math.Sqrt((w * w) + (x * x) + (y * y) + (z * z)));
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;
    }

    /// <summary>
    /// dL/dq for the raw quaternion from dL/dR, through the rotation matrix and the normalisation q / |q|.
    /// </summary>
    private static void QuaternionGradient(GenerativeState state, int i, double[] g, double[] output)
    {
        NormalizedRotation(state, i, out double w, out double x, out double y, out double z);
        double dw = 2.0 * ((-z * g[1]) + (y * g[2]) + (z * g[3]) - (x * g[5]) - (y * g[6]) + (x * g[7]));
        double dx = 2.0 * ((y * g[1]) + (z * g[2]) + (y * g[3]) - (2.0 * x * g[4]) - (w * g[5]) + (z * g[6]) + (w * g[7]) - (2.0 * x * g[8]));
        double dy = 2.0 * ((-2.0 * y * g[0]) + (x * g[1]) + (w * g[2]) + (x * g[3]) + (z * g[5]) - (w * g[6]) + (z * g[7]) - (2.0 * y * g[8]));
        double dz = 2.0 * ((-2.0 * z * g[0]) - (w * g[1]) + (x * g[2]) + (w * g[3]) - (2.0 * z * g[4]) + (y * g[5]) + (x * g[6]) + (y * g[7]));

        double rw = state.Rotation[4 * i], rx = state.Rotation[(4 * i) + 1], ry = state.Rotation[(4 * i) + 2], rz = state.Rotation[(4 * i) + 3];
        double rawNorm = Math.Sqrt((rw * rw) + (rx * rx) + (ry * ry) + (rz * rz));
        double norm = Math.Max(1e-12, rawNorm);
        double dot = (w * dw) + (x * dx) + (y * dy) + (z * dz);
        // d(q / |q|)/dq applied to the gradient: (g - q_hat (q_hat . g)) / |q|; below eps the norm is a constant.
        bool below = rawNorm < 1e-12;
        output[4 * i] = below ? dw / norm : (dw - (w * dot)) / norm;
        output[(4 * i) + 1] = below ? dx / norm : (dx - (x * dot)) / norm;
        output[(4 * i) + 2] = below ? dy / norm : (dy - (y * dot)) / norm;
        output[(4 * i) + 3] = below ? dz / norm : (dz - (z * dot)) / norm;
    }
}
