namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Finds pixel correspondences between two views of the same object, for camera-pose (elevation) estimation.
/// </summary>
/// <typeparam name="T">The numeric type of the images.</typeparam>
/// <remarks>
/// <para>
/// One-2-3-45 matches its four nearby Zero123 views with LoFTR (Sun et al. 2021, the <c>indoor_ds_new</c> checkpoint)
/// and estimates the input's elevation from the matches. Any matcher that returns sub-pixel correspondences with a
/// confidence can stand in; <see cref="NormalizedCrossCorrelationMatcher{T}"/> is the built-in default.
/// </para>
/// <para><b>For Beginners:</b> a correspondence says "this pixel in the first picture shows the same spot on the
/// object as that pixel in the second picture". Several of them pin down how the camera moved.</para>
/// </remarks>
public interface IViewFeatureMatcher<T>
{
    /// <summary>Matches two images.</summary>
    /// <param name="image0">The first image, <c>[H, W, 3]</c> in [0, 1].</param>
    /// <param name="image1">The second image, <c>[H, W, 3]</c> in [0, 1], the same size.</param>
    /// <returns>The correspondences, in pixel coordinates (x right, y down).</returns>
    IReadOnlyList<FeatureCorrespondence> Match(Tensor<T> image0, Tensor<T> image1);
}

/// <summary>One pixel correspondence between two images, with the matcher's confidence.</summary>
public readonly struct FeatureCorrespondence
{
    /// <summary>Creates a correspondence.</summary>
    /// <param name="x0">x in the first image.</param>
    /// <param name="y0">y in the first image.</param>
    /// <param name="x1">x in the second image.</param>
    /// <param name="y1">y in the second image.</param>
    /// <param name="confidence">The match confidence (LoFTR's <c>mconf</c>); weights the reprojection error.</param>
    public FeatureCorrespondence(double x0, double y0, double x1, double y1, double confidence)
    {
        X0 = x0;
        Y0 = y0;
        X1 = x1;
        Y1 = y1;
        Confidence = confidence;
    }

    /// <summary>x in the first image.</summary>
    public double X0 { get; }

    /// <summary>y in the first image.</summary>
    public double Y0 { get; }

    /// <summary>x in the second image.</summary>
    public double X1 { get; }

    /// <summary>y in the second image.</summary>
    public double Y1 { get; }

    /// <summary>The match confidence.</summary>
    public double Confidence { get; }

    /// <summary>The same correspondence seen from the second image (<c>match_ji</c>).</summary>
    public FeatureCorrespondence Reversed() => new FeatureCorrespondence(X1, Y1, X0, Y0, Confidence);
}

/// <summary>
/// A classical dense matcher: zero-mean normalized cross-correlation of grayscale patches on a keypoint grid,
/// kept only when the match is mutual (searching back from the best location returns to the start).
/// </summary>
/// <typeparam name="T">The numeric type of the images.</typeparam>
/// <remarks>
/// <para>
/// This is the default stand-in for LoFTR, which One-2-3-45 uses but which needs its own pretrained weights. The
/// nearby views differ by only 10°, so local patch correlation finds reliable matches on textured objects. Supply a
/// learned matcher through <see cref="IViewFeatureMatcher{T}"/> for textureless inputs.
/// </para>
/// <para>Grayscale is <c>0.299 R + 0.587 G + 0.114 B</c>, as <c>cv2.IMREAD_GRAYSCALE</c> converts.</para>
/// </remarks>
public sealed class NormalizedCrossCorrelationMatcher<T> : IViewFeatureMatcher<T>
{
    private readonly int _patchRadius;
    private readonly int _searchRadius;
    private readonly int _stride;
    private readonly double _minCorrelation;
    private readonly double _minTextureStd;

    /// <summary>Creates the matcher.</summary>
    /// <param name="patchRadius">Half the patch side; patches are <c>(2r + 1)²</c> pixels.</param>
    /// <param name="searchRadius">How far, in pixels, a match may move between the images.</param>
    /// <param name="stride">The keypoint grid spacing in pixels.</param>
    /// <param name="minCorrelation">The smallest correlation kept (and the reported confidence otherwise).</param>
    /// <param name="minTextureStd">Patches flatter than this standard deviation are skipped.</param>
    public NormalizedCrossCorrelationMatcher(int patchRadius = 3, int searchRadius = 12, int stride = 4,
        double minCorrelation = 0.8, double minTextureStd = 0.02)
    {
        if (patchRadius < 1) throw new ArgumentOutOfRangeException(nameof(patchRadius));
        if (searchRadius < 1) throw new ArgumentOutOfRangeException(nameof(searchRadius));
        if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride));
        if (!(minCorrelation > -1.0) || minCorrelation > 1.0) throw new ArgumentOutOfRangeException(nameof(minCorrelation));
        if (!(minTextureStd >= 0)) throw new ArgumentOutOfRangeException(nameof(minTextureStd));
        _patchRadius = patchRadius;
        _searchRadius = searchRadius;
        _stride = stride;
        _minCorrelation = minCorrelation;
        _minTextureStd = minTextureStd;
    }

    /// <inheritdoc />
    public IReadOnlyList<FeatureCorrespondence> Match(Tensor<T> image0, Tensor<T> image1)
    {
        var (gray0, h, w) = Grayscale(image0, nameof(image0));
        var (gray1, h1, w1) = Grayscale(image1, nameof(image1));
        if (h1 != h || w1 != w) throw new ArgumentException("Both images must have the same size.", nameof(image1));
        int r = _patchRadius;
        var result = new List<FeatureCorrespondence>();
        for (int y = r; y < h - r; y += _stride)
        {
            for (int x = r; x < w - r; x += _stride)
            {
                if (PatchStd(gray0, w, x, y) < _minTextureStd) continue;
                var (bx, by, score) = Best(gray0, gray1, w, h, x, y);
                if (bx < 0 || score < _minCorrelation) continue;
                var (rx, ry, _) = Best(gray1, gray0, w, h, bx, by);
                if (rx < 0 || Math.Abs(rx - x) > 1 || Math.Abs(ry - y) > 1) continue;
                result.Add(new FeatureCorrespondence(x, y, bx, by, score));
            }
        }

        return result;
    }

    private (int X, int Y, double Score) Best(double[] from, double[] to, int w, int h, int x, int y)
    {
        int r = _patchRadius;
        int bestX = -1, bestY = -1;
        double best = double.NegativeInfinity;
        for (int dy = -_searchRadius; dy <= _searchRadius; dy++)
        {
            int cy = y + dy;
            if (cy < r || cy >= h - r) continue;
            for (int dx = -_searchRadius; dx <= _searchRadius; dx++)
            {
                int cx = x + dx;
                if (cx < r || cx >= w - r) continue;
                double score = Correlation(from, to, w, x, y, cx, cy);
                if (score > best)
                {
                    best = score;
                    bestX = cx;
                    bestY = cy;
                }
            }
        }

        return (bestX, bestY, best);
    }

    private double Correlation(double[] a, double[] b, int w, int ax, int ay, int bx, int by)
    {
        int r = _patchRadius;
        int n = (2 * r + 1) * (2 * r + 1);
        double sa = 0, sb = 0;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
            {
                sa += a[(ay + dy) * w + ax + dx];
                sb += b[(by + dy) * w + bx + dx];
            }

        double ma = sa / n, mb = sb / n;
        double cross = 0, va = 0, vb = 0;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
            {
                double da = a[(ay + dy) * w + ax + dx] - ma;
                double db = b[(by + dy) * w + bx + dx] - mb;
                cross += da * db;
                va += da * da;
                vb += db * db;
            }

        double denominator = Math.Sqrt(va * vb);
        return denominator > 1e-12 ? cross / denominator : -1.0;
    }

    private double PatchStd(double[] g, int w, int x, int y)
    {
        int r = _patchRadius;
        int n = (2 * r + 1) * (2 * r + 1);
        double s = 0, s2 = 0;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
            {
                double v = g[(y + dy) * w + x + dx];
                s += v;
                s2 += v * v;
            }

        double mean = s / n;
        return Math.Sqrt(Math.Max(0.0, s2 / n - mean * mean));
    }

    private static (double[] Gray, int Height, int Width) Grayscale(Tensor<T> image, string name)
    {
        if (image is null) throw new ArgumentNullException(name);
        if (image.Shape.Length != 3 || image.Shape[2] != 3) throw new ArgumentException("The image must be [H, W, 3].", name);
        int h = image.Shape[0], w = image.Shape[1];
        var rgb = AiDotNet.NeuralRadianceFields.NeuS.NeuSHost<T>.ToDoubles(image);
        var gray = new double[h * w];
        for (int p = 0; p < gray.Length; p++)
            gray[p] = 0.299 * rgb[p * 3] + 0.587 * rgb[p * 3 + 1] + 0.114 * rgb[p * 3 + 2];
        return (gray, h, w);
    }
}
