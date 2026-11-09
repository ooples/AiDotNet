namespace AiDotNet.Video.Enhancement;

/// <summary>
/// NIQE (Mittal, Soundararajan and Bovik 2013), the no-reference quality score, as pyiqa computes it to match MATLAB: the
/// image's luma (rgb2yiq's float32 0.299, 0.587, 0.114, on the 0-255 scale, rounded) cropped to whole 96x96 blocks; at two scales (the
/// second from MATLAB's antialiased bicubic imresize by one half) the locally normalized luminance (a 7x7 Gaussian of
/// σ 7/6, replicate padding, (I − μ) / (σ + 1)) gives each block 18 asymmetric-generalized-Gaussian features; the
/// blocks' 36-feature mean and covariance are compared with the pristine model's by the Mahalanobis-like distance
/// √((ν₁ − ν₂)ᵀ ((Σ₁ + Σ₂) / 2)⁺ (ν₁ − ν₂)). Lower is better.
/// </summary>
internal static class Niqe
{
    private const int Block = 96;
    private const int Features = 36;
    private static readonly double[] Gammas = BuildGammas();
    private static readonly double[] GammaRatios = Gammas.Select(g => Math.Exp(2 * LogGamma(2 / g) - LogGamma(1 / g) - LogGamma(3 / g))).ToArray();
    private static readonly double[,] Window = Gaussian(7, 7.0 / 6.0);

    // rgb2yiq's luma row, a float32 tensor widened to the image's dtype.
    private const double YiqRed = (double)0.299f, YiqGreen = (double)0.587f, YiqBlue = (double)0.114f;

    // torch.arange(0.2, 10.001, 0.001) in its default float32, then widened: the grid's α carry float32 rounding. Each
    // point is the float32 nearest the exact grid value; torch's vectorized arange builds 8- or 16-lane chunks from a
    // rounded base and lands one ulp away on about a fifth of the points, an amount that depends on the CPU it runs on.
    private static double[] BuildGammas()
    {
        int n = (int)Math.Ceiling((10.0 + 0.001 - 0.2) / 0.001);
        return Enumerable.Range(0, n).Select(i => (double)(float)(0.2 + i * 0.001)).ToArray();
    }

    /// <summary>NIQE of an RGB image [3, H, W] in [0, 1] (row-major planes); NaN when it holds no whole 96x96 block.</summary>
    internal static double Score(double[] rgb, int height, int width)
    {
        var y = new double[height, width];
        int plane = height * width;
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                y[r, c] = Math.Round((rgb[i] * YiqRed + rgb[plane + i] * YiqGreen + rgb[2 * plane + i] * YiqBlue) * 255.0,
                    MidpointRounding.ToEven);
            }
        return Score(y);
    }

    /// <summary>NIQE of a luma image on the 0-255 scale.</summary>
    internal static double Score(double[,] luma)
    {
        int rows = luma.GetLength(0) / Block, cols = luma.GetLength(1) / Block;
        if (rows == 0 || cols == 0) return double.NaN;
        var img = new double[rows * Block, cols * Block];
        for (int r = 0; r < img.GetLength(0); r++)
            for (int c = 0; c < img.GetLength(1); c++) img[r, c] = luma[r, c];
        var features = new double[rows * cols, Features];
        for (int scale = 1; scale <= 2; scale++)
        {
            var normalized = Normalize(img);
            int size = Block / scale;
            for (int br = 0; br < rows; br++)
                for (int bc = 0; bc < cols; bc++)
                {
                    var block = new double[size, size];
                    for (int r = 0; r < size; r++)
                        for (int c = 0; c < size; c++) block[r, c] = normalized[br * size + r, bc * size + c];
                    var f = BlockFeatures(block);
                    for (int k = 0; k < f.Length; k++) features[br * cols + bc, (scale - 1) * 18 + k] = f[k];
                }
            if (scale == 1) img = HalveMatlab(img);
        }
        return Distance(features);
    }

    // The 18 features of a normalized block: AGGD α and mean β of the block, then α, mean, β_l and β_r of its products
    // with the block circularly rolled by (0, 1), (1, 0), (1, 1) and (1, −1).
    private static double[] BlockFeatures(double[,] block)
    {
        int n = block.GetLength(0), m = block.GetLength(1);
        var result = new List<double>(18);
        var (alpha, left, right) = Aggd(Flatten(block));
        result.Add(alpha);
        result.Add((left + right) / 2);
        foreach (var (dr, dc) in new[] { (0, 1), (1, 0), (1, 1), (1, -1) })
        {
            var product = new double[n * m];
            for (int r = 0; r < n; r++)
                for (int c = 0; c < m; c++)
                    product[r * m + c] = block[r, c] * block[((r - dr) % n + n) % n, ((c - dc) % m + m) % m];
            var (a, l, rr) = Aggd(product);
            result.Add(a);
            result.Add((rr - l) * Math.Exp(LogGamma(2 / a) - LogGamma(1 / a)));
            result.Add(l);
            result.Add(rr);
        }
        return result.ToArray();
    }

    private static double[] Flatten(double[,] x)
    {
        var flat = new double[x.Length];
        Buffer.BlockCopy(x, 0, flat, 0, x.Length * sizeof(double));
        return flat;
    }

    // estimate_aggd_param: the left and right standard deviations, then the α whose moment ratio best matches.
    private static (double Alpha, double Left, double Right) Aggd(double[] x)
    {
        double leftSum = 0, rightSum = 0, absSum = 0, squareSum = 0;
        int leftCount = 0, rightCount = 0;
        foreach (double v in x)
        {
            if (v < 0) { leftSum += v * v; leftCount++; }
            else if (v > 0) { rightSum += v * v; rightCount++; }
            absSum += Math.Abs(v);
            squareSum += v * v;
        }
        double leftStd = Math.Sqrt(leftSum / leftCount), rightStd = Math.Sqrt(rightSum / rightCount);
        double gammaHat = leftStd / rightStd;
        double meanAbs = absSum / x.Length;
        double rHat = meanAbs * meanAbs / (squareSum / x.Length);
        double rHatNorm = rHat * (Math.Pow(gammaHat, 3) + 1) * (gammaHat + 1) / Math.Pow(gammaHat * gammaHat + 1, 2);
        if (double.IsNaN(rHatNorm)) return (double.NaN, double.NaN, double.NaN);
        int best = 0;
        double bestDiff = double.PositiveInfinity;
        for (int i = 0; i < GammaRatios.Length; i++)
        {
            double d = Math.Abs(GammaRatios[i] - rHatNorm);
            if (d < bestDiff) { bestDiff = d; best = i; }
        }
        double alpha = Gammas[best];
        double scale = Math.Sqrt(Math.Exp(LogGamma(1 / alpha) - LogGamma(3 / alpha)));
        return (alpha, leftStd * scale, rightStd * scale);
    }

    // normalize_img_with_gauss: (I − μ) / (σ + 1) with μ and σ from the 7x7 Gaussian, replicate padding.
    private static double[,] Normalize(double[,] img)
    {
        int h = img.GetLength(0), w = img.GetLength(1), half = Window.GetLength(0) / 2;
        var result = new double[h, w];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
            {
                double mu = 0, second = 0;
                for (int i = -half; i <= half; i++)
                    for (int j = -half; j <= half; j++)
                    {
                        double v = img[Math.Min(Math.Max(r + i, 0), h - 1), Math.Min(Math.Max(c + j, 0), w - 1)];
                        double k = Window[i + half, j + half];
                        mu += k * v;
                        second += k * v * v;
                    }
                double sigma = Math.Sqrt(Math.Abs(second - mu * mu));
                result[r, c] = (img[r, c] - mu) / (sigma + 1);
            }
        return result;
    }

    // MATLAB's fspecial('gaussian', size, σ), as pyiqa computes it.
    private static double[,] Gaussian(int size, double sigma)
    {
        var kernel = new double[size, size];
        double m = (size - 1) / 2.0, sum = 0, max = 0;
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
            {
                double y = r - m, x = c - m;
                kernel[r, c] = Math.Exp(-(x * x + y * y) / (2 * sigma * sigma));
                max = Math.Max(max, kernel[r, c]);
            }
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
            {
                if (kernel[r, c] < 2.220446049250313e-16 * max) kernel[r, c] = 0;
                sum += kernel[r, c];
            }
        // pyiqa's fspecial returns the normalized window as float32.
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++) kernel[r, c] = (float)(kernel[r, c] / sum);
        return kernel;
    }

    // imresize(img / 255, 0.5) · 255 as the released pyiqa runs it: cast_input always converts to float32 (its dtype
    // guard is always true), then resize_1d along the height and then the width, each a sum of 10 taps whose weights
    // are its float32 cubic at (pos − base − k) / 2, normalized, over MATLAB's symmetric padding. Carried in float32
    // with sequential sums, as torch computes it, so the second scale's features are pyiqa's.
    private static double[,] HalveMatlab(double[,] img)
    {
        int h = img.GetLength(0), w = img.GetLength(1);
        var x = new float[h, w];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++) x[r, c] = (float)(img[r, c] / 255.0);
        var (rowWeights, rowBase) = HalfWeights32(h);
        var (colWeights, colBase) = HalfWeights32(w);
        int oh = rowBase.Length, ow = colBase.Length;
        var rows = new float[oh, w];
        for (int i = 0; i < oh; i++)
            for (int c = 0; c < w; c++)
            {
                float sum = 0f;
                for (int k = 0; k < 10; k++) sum += x[Mirror(rowBase[i] + k, h), c] * rowWeights[i, k];
                rows[i, c] = sum;
            }
        var result = new double[oh, ow];
        for (int r = 0; r < oh; r++)
            for (int j = 0; j < ow; j++)
            {
                float sum = 0f;
                for (int k = 0; k < 10; k++) sum += rows[r, Mirror(colBase[j] + k, w)] * colWeights[j, k];
                result[r, j] = sum * 255.0;
            }
        return result;
    }

    private static int Mirror(int j, int n) => j < 0 ? -j - 1 : j >= n ? 2 * n - 1 - j : j;

    // resize_1d's taps for a factor of one half in float32: 10 weights per output and the first input index.
    private static (float[,] Weights, int[] Bases) HalfWeights32(int n)
    {
        int size = (n + 1) / 2;
        var weights = new float[size, 10];
        var bases = new int[size];
        for (int i = 0; i < size; i++)
        {
            float pos = (i + 0.5f) / 0.5f - 0.5f;
            float b = (float)Math.Floor(pos) - 5 + 1;
            float dist = pos - b;
            float total = 0f;
            for (int k = 0; k < 10; k++)
            {
                weights[i, k] = Cubic32((dist - k) * 0.5f);
                total += weights[i, k];
            }
            for (int k = 0; k < 10; k++) weights[i, k] /= total;
            bases[i] = (int)b;
        }
        return (weights, bases);
    }

    // cubic_contribution (a = −0.5) in float32, term by term as torch evaluates it.
    private static float Cubic32(float x)
    {
        float ax = Math.Abs(x), ax2 = ax * ax, ax3 = ax * ax2;
        if (ax <= 1f) return 1.5f * ax3 - 2.5f * ax2 + 1f;
        if (ax <= 2f) return -0.5f * ax3 - -2.5f * ax2 + -4f * ax - -2f;
        return 0f;
    }

    /// <summary>MATLAB's imresize(img, 0.5): antialiased bicubic (a = −0.5), symmetric boundaries, rows first.</summary>
    internal static double[,] Halve(double[,] img)
    {
        int h = img.GetLength(0), w = img.GetLength(1);
        var rows = Resize1D(Enumerable.Range(0, w).Select(c => Enumerable.Range(0, h).Select(r => img[r, c]).ToArray()).ToArray());
        int oh = rows[0].Length;
        var cols = Resize1D(Enumerable.Range(0, oh).Select(r => Enumerable.Range(0, w).Select(c => rows[c][r]).ToArray()).ToArray());
        int ow = cols[0].Length;
        var result = new double[oh, ow];
        for (int r = 0; r < oh; r++)
            for (int c = 0; c < ow; c++) result[r, c] = cols[r][c];
        return result;
    }

    private static double[][] Resize1D(double[][] lines)
    {
        const double scale = 0.5;
        int n = lines[0].Length, size = (int)Math.Ceiling(n * scale);
        int kernel = (int)Math.Ceiling(4 / scale) + 2;
        var weights = new double[size, kernel];
        var bases = new int[size];
        for (int i = 0; i < size; i++)
        {
            double pos = (i + 0.5) / scale - 0.5;
            int b = (int)Math.Floor(pos) - kernel / 2 + 1;
            bases[i] = b;
            double total = 0;
            for (int k = 0; k < kernel; k++)
            {
                weights[i, k] = Cubic((pos - b - k) * scale);
                total += weights[i, k];
            }
            for (int k = 0; k < kernel; k++) weights[i, k] /= total;
        }
        return lines.Select(line =>
        {
            var output = new double[size];
            for (int i = 0; i < size; i++)
            {
                double s = 0;
                for (int k = 0; k < kernel; k++)
                {
                    int j = bases[i] + k;
                    if (j < 0) j = -j - 1;
                    else if (j >= n) j = 2 * n - 1 - j;
                    s += weights[i, k] * line[j];
                }
                output[i] = s;
            }
            return output;
        }).ToArray();
    }

    private static double Cubic(double x)
    {
        const double a = -0.5;
        double ax = Math.Abs(x), ax2 = ax * ax, ax3 = ax2 * ax;
        if (ax <= 1) return (a + 2) * ax3 - (a + 3) * ax2 + 1;
        if (ax <= 2) return a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a;
        return 0;
    }

    // The distorted image's feature mean and unbiased covariance over the blocks free of NaN, against the pristine model.
    private static double Distance(double[,] features)
    {
        var rows = Enumerable.Range(0, features.GetLength(0))
            .Where(r => Enumerable.Range(0, Features).All(k => !double.IsNaN(features[r, k]))).ToList();
        // The unbiased covariance has no estimate with fewer than two usable patches.
        // Preserve the metric's unscorable result so callers can rank it last, as for undersized images.
        if (rows.Count < 2) return double.NaN;
        var mean = new double[Features];
        for (int k = 0; k < Features; k++)
        {
            int count = 0;
            for (int r = 0; r < features.GetLength(0); r++)
                if (!double.IsNaN(features[r, k])) { mean[k] += features[r, k]; count++; }
            mean[k] /= count;
        }
        var covMean = new double[Features];
        foreach (int r in rows)
            for (int k = 0; k < Features; k++) covMean[k] += features[r, k] / rows.Count;
        var pooled = new double[Features, Features];
        for (int i = 0; i < Features; i++)
            for (int j = 0; j < Features; j++)
            {
                double s = 0;
                foreach (int r in rows) s += (features[r, i] - covMean[i]) * (features[r, j] - covMean[j]);
                double distorted = s / (rows.Count - 1);
                pooled[i, j] = (NiqePristineModel.Covariance[i * Features + j] + distorted) / 2;
            }
        var inverse = PseudoInverse(pooled);
        var diff = Enumerable.Range(0, Features).Select(k => NiqePristineModel.Mean[k] - mean[k]).ToArray();
        double q = 0;
        for (int i = 0; i < Features; i++)
            for (int j = 0; j < Features; j++) q += diff[i] * inverse[i, j] * diff[j];
        return Math.Sqrt(q);
    }

    // torch.linalg.pinv of a symmetric matrix: cyclic Jacobi eigenvectors, inverting eigenvalues above
    // max(n)·ε·|λ|max.
    internal static double[,] PseudoInverse(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var a = (double[,])matrix.Clone();
        var v = new double[n, n];
        for (int i = 0; i < n; i++) v[i, i] = 1;
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++) off += a[p, q] * a[p, q];
            if (off < 1e-30) break;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p, q]) < 1e-300) continue;
                    double theta = (a[q, q] - a[p, p]) / (2 * a[p, q]);
                    double t = Math.Sign(theta == 0 ? 1 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k, p], akq = a[k, q];
                        a[k, p] = c * akp - s * akq;
                        a[k, q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p, k], aqk = a[q, k];
                        a[p, k] = c * apk - s * aqk;
                        a[q, k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k, p], vkq = v[k, q];
                        v[k, p] = c * vkp - s * vkq;
                        v[k, q] = s * vkp + c * vkq;
                    }
                }
        }
        double largest = Enumerable.Range(0, n).Max(i => Math.Abs(a[i, i]));
        double cutoff = n * 2.220446049250313e-16 * largest;
        var inverse = new double[n, n];
        for (int k = 0; k < n; k++)
        {
            double lambda = a[k, k];
            if (Math.Abs(lambda) <= cutoff) continue;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) inverse[i, j] += v[i, k] * v[j, k] / lambda;
        }
        return inverse;
    }

    // ln Γ(x) for x > 0 (Lanczos, g = 7, nine coefficients).
    internal static double LogGamma(double x)
    {
        double[] g =
        {
            0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
        };
        if (x < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        x -= 1;
        double sum = g[0];
        for (int i = 1; i < g.Length; i++) sum += g[i] / (x + i);
        double t = x + 7.5;
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }
}
