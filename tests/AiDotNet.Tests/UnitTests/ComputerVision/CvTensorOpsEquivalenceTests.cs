using AiDotNet.ComputerVision;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Pins the detection and OCR tensor transforms that were rewritten from scalar element loops into
/// engine ops (#2152) - to the loops they replaced, and to the gradient tape.
/// </summary>
/// <remarks>
/// <para>
/// The <c>Ref*</c> methods are VERBATIM copies of the pre-rewrite helpers (only their signatures are
/// adapted to <c>double</c>). The rewrite had to change tape visibility and nothing else, so every
/// comparison is exact for pure data movement and within rounding for arithmetic.
/// </para>
/// <para>
/// The gradient checks compare the tape's gradient of a random linear functional of each op's output
/// with central finite differences. A scalar element loop fails them outright: it builds its result
/// outside the tape, so no gradient reaches the input at all.
/// </para>
/// <para>
/// The tolerances assume the CPU engine, which the test assembly's module initializer selects.
/// </para>
/// </remarks>
public class CvTensorOpsEquivalenceTests
{
    private readonly List<string> _failures = new();

    private static Tensor<double> Rand(int[] shape, int seed)
    {
        var r = new Random(seed); var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = r.NextDouble() * 4 - 2;
        return t;
    }

    private static int[] Dims(Tensor<double> t) => Enumerable.Range(0, t.Shape.Length).Select(i => t.Shape[i]).ToArray();

    private void Compare(string name, Tensor<double> expected, Tensor<double> actual, double tol)
    {
        if (!Dims(expected).SequenceEqual(Dims(actual)))
        {
            _failures.Add($"{name}: shape [{string.Join(",", Dims(expected))}] vs [{string.Join(",", Dims(actual))}]");
            return;
        }

        double max = 0;
        for (int i = 0; i < expected.Length; i++) max = Math.Max(max, Math.Abs(expected[i] - actual[i]));
        if (max > tol) _failures.Add($"{name}: max|diff| = {max:e2} (tolerance {tol:e0})");
    }

    private void AssertNoFailures()
        => Assert.True(_failures.Count == 0, $"{_failures.Count} mismatch(es):\n" + string.Join("\n", _failures.Take(20)));

    // ---- references: verbatim copies of the pre-rewrite helpers ----

    private static double At(Tensor<double> t, int n, int c, int h, int w)
        => t[((n * t.Shape[1] + c) * t.Shape[2] + h) * t.Shape[3] + w];

    private static void Set(Tensor<double> t, int n, int c, int h, int w, double v)
        => t[((n * t.Shape[1] + c) * t.Shape[2] + h) * t.Shape[3] + w] = v;

    // ---- references: copied from the original helpers ----

    private static Tensor<double> RefResizeToMatch(Tensor<double> source, int targetH, int targetW) // FPN/PANet/BiFPN
    {
        int batch = source.Shape[0], channels = source.Shape[1], sourceH = source.Shape[2], sourceW = source.Shape[3];
        var result = new Tensor<double>(new[] { batch, channels, targetH, targetW });
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < targetH; h++)
                    for (int w = 0; w < targetW; w++)
                    {
                        int srcH = Math.Min(h * sourceH / targetH, sourceH - 1);
                        int srcW = Math.Min(w * sourceW / targetW, sourceW - 1);
                        Set(result, n, c, h, w, At(source, n, c, srcH, srcW));
                    }
        return result;
    }

    private static Tensor<double> RefUpsample2x(Tensor<double> input) // NeckBase
    {
        int batch = input.Shape[0], channels = input.Shape[1], height = input.Shape[2], width = input.Shape[3];
        var output = new Tensor<double>(new[] { batch, channels, height * 2, width * 2 });
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height * 2; h++)
                    for (int w = 0; w < width * 2; w++)
                        Set(output, b, c, h, w, At(input, b, c, h / 2, w / 2));
        return output;
    }

    private static Tensor<double> RefDownsample2x(Tensor<double> input) // NeckBase, ceil mode
    {
        int batch = input.Shape[0], channels = input.Shape[1], height = input.Shape[2], width = input.Shape[3];
        int outHeight = (height + 1) / 2, outWidth = (width + 1) / 2;
        var output = new Tensor<double>(new[] { batch, channels, outHeight, outWidth });
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < outHeight; h++)
                    for (int w = 0; w < outWidth; w++)
                    {
                        int r = h * 2, col = w * 2;
                        double m = At(input, b, c, r, col);
                        if (col + 1 < width) m = Math.Max(m, At(input, b, c, r, col + 1));
                        if (r + 1 < height) m = Math.Max(m, At(input, b, c, r + 1, col));
                        if (r + 1 < height && col + 1 < width) m = Math.Max(m, At(input, b, c, r + 1, col + 1));
                        Set(output, b, c, h, w, m);
                    }
        return output;
    }

    private static Tensor<double> RefBilinear(Tensor<double> x, int targetH, int targetW) // CRAFT/DBNet/EAST
    {
        int batch = x.Shape[0], channels = x.Shape[1], srcH = x.Shape[2], srcW = x.Shape[3];
        var result = new Tensor<double>(new[] { batch, channels, targetH, targetW });
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < targetH; h++)
                    for (int w = 0; w < targetW; w++)
                    {
                        double srcY = (double)h / targetH * srcH;
                        double srcX = (double)w / targetW * srcW;
                        int y0 = (int)Math.Floor(srcY), x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, srcH - 1), x1 = Math.Min(x0 + 1, srcW - 1);
                        double wy1 = srcY - y0, wy0 = 1.0 - wy1, wx1 = srcX - x0, wx0 = 1.0 - wx1;
                        double v00 = At(x, b, c, y0, x0), v01 = At(x, b, c, y0, x1);
                        double v10 = At(x, b, c, y1, x0), v11 = At(x, b, c, y1, x1);
                        Set(result, b, c, h, w, wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11));
                    }
        return result;
    }

    private static Tensor<double> RefCrnnMaxPool(Tensor<double> x, int kernelH, int kernelW) // CRNN, floor
    {
        int batch = x.Shape[0], channels = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        int outH = height / kernelH, outW = width / kernelW;
        var result = new Tensor<double>(new[] { batch, channels, outH, outW });
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < outH; h++)
                    for (int w = 0; w < outW; w++)
                    {
                        double m = double.NegativeInfinity;
                        for (int kh = 0; kh < kernelH; kh++)
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int sh = h * kernelH + kh, sw = w * kernelW + kw;
                                if (sh < height && sw < width) m = Math.Max(m, At(x, b, c, sh, sw));
                            }
                        Set(result, b, c, h, w, m);
                    }
        return result;
    }

    private static Tensor<double> RefYoloMaxPool(Tensor<double> x, int kernelSize) // YOLOv11 SPPF, same
    {
        int padding = kernelSize / 2;
        int batch = x.Shape[0], channels = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        var output = new Tensor<double>(new[] { batch, channels, height, width });
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double m = double.NegativeInfinity;
                        for (int kh = 0; kh < kernelSize; kh++)
                            for (int kw = 0; kw < kernelSize; kw++)
                            {
                                int ih = h - padding + kh, iw = w - padding + kw;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) m = Math.Max(m, At(x, n, c, ih, iw));
                            }
                        Set(output, n, c, h, w, m == double.NegativeInfinity ? 0 : m);
                    }
        return output;
    }

    private static Tensor<double> RefResNetMaxPool(Tensor<double> x, int kernelSize, int stride, int padding) // ResNet stem (removed BackboneOps.MaxPool2D)
    {
        int batch = x.Shape[0], channels = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        int outH = (height + 2 * padding - kernelSize) / stride + 1, outW = (width + 2 * padding - kernelSize) / stride + 1;
        var output = new Tensor<double>(new[] { batch, channels, outH, outW });
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double m = double.NegativeInfinity;
                        for (int kh = 0; kh < kernelSize; kh++)
                            for (int kw = 0; kw < kernelSize; kw++)
                            {
                                int ih = oh * stride - padding + kh, iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) m = Math.Max(m, At(x, n, c, ih, iw));
                            }
                        Set(output, n, c, oh, ow, m == double.NegativeInfinity ? 0 : m);
                    }
        return output;
    }

    private static Tensor<double> RefCrnnBatchNorm(Tensor<double> x) // CRNN
    {
        int batch = x.Shape[0], channels = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        var result = new Tensor<double>(new[] { batch, channels, height, width });
        double epsilon = 1e-5;
        for (int c = 0; c < channels; c++)
        {
            double sum = 0, sumSq = 0; int count = batch * height * width;
            for (int b = 0; b < batch; b++) for (int h = 0; h < height; h++) for (int w = 0; w < width; w++)
            { double v = At(x, b, c, h, w); sum += v; sumSq += v * v; }
            double mean = sum / count, variance = (sumSq / count) - (mean * mean), std = Math.Sqrt(variance + epsilon);
            for (int b = 0; b < batch; b++) for (int h = 0; h < height; h++) for (int w = 0; w < width; w++)
                Set(result, b, c, h, w, (At(x, b, c, h, w) - mean) / std);
        }
        return result;
    }

    private static Tensor<double> RefLayerNorm(Tensor<double> x, Tensor<double> gamma, Tensor<double> beta, double eps) // DETR LayerNorm
    {
        int batch = x.Shape[0], seqLen = x.Shape[1], hiddenDim = x.Shape[2];
        var result = new Tensor<double>(new[] { batch, seqLen, hiddenDim });
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int row = (b * seqLen + s) * hiddenDim;
                double mean = 0;
                for (int d = 0; d < hiddenDim; d++) mean += x[row + d];
                mean /= hiddenDim;
                double variance = 0;
                for (int d = 0; d < hiddenDim; d++) { double diff = x[row + d] - mean; variance += diff * diff; }
                variance /= hiddenDim;
                double std = Math.Sqrt(variance + eps);
                for (int d = 0; d < hiddenDim; d++)
                    result[row + d] = gamma[d] * ((x[row + d] - mean) / std) + beta[d];
            }
        return result;
    }

    private static Tensor<double> RefAttention(Tensor<double> q, Tensor<double> k, Tensor<double> v, int numHeads, bool causal) // DETR/TrOCR
    {
        int batch = q.Shape[0], queryLen = q.Shape[1], hidden = q.Shape[2], keyLen = k.Shape[1];
        int headDim = hidden / numHeads; double scale = 1.0 / Math.Sqrt(headDim);
        var output = new Tensor<double>(new[] { batch, queryLen, hidden });
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                int off = h * headDim;
                var scores = new double[queryLen, keyLen];
                for (int i = 0; i < queryLen; i++)
                    for (int j = 0; j < keyLen; j++)
                    {
                        if (causal && j > i) { scores[i, j] = double.NegativeInfinity; continue; }
                        double sc = 0;
                        for (int d = 0; d < headDim; d++)
                            sc += q[(b * queryLen + i) * hidden + off + d] * k[(b * keyLen + j) * hidden + off + d];
                        scores[i, j] = sc * scale;
                    }
                for (int i = 0; i < queryLen; i++)
                {
                    double mx = double.NegativeInfinity;
                    for (int j = 0; j < keyLen; j++) mx = Math.Max(mx, scores[i, j]);
                    double sum = 0;
                    for (int j = 0; j < keyLen; j++) { scores[i, j] = Math.Exp(scores[i, j] - mx); sum += scores[i, j]; }
                    for (int j = 0; j < keyLen; j++) scores[i, j] /= sum;
                }
                for (int i = 0; i < queryLen; i++)
                    for (int d = 0; d < headDim; d++)
                    {
                        double val = 0;
                        for (int j = 0; j < keyLen; j++) val += scores[i, j] * v[(b * keyLen + j) * hidden + off + d];
                        output[(b * queryLen + i) * hidden + off + d] = val;
                    }
            }
        return output;
    }

    // Swin, verbatim modulo indexing helpers. Layout NHWC.
    private static double G4(Tensor<double> t, int a, int b, int c, int d) => t[((a * t.Shape[1] + b) * t.Shape[2] + c) * t.Shape[3] + d];
    private static void S4(Tensor<double> t, int a, int b, int c, int d, double v) => t[((a * t.Shape[1] + b) * t.Shape[2] + c) * t.Shape[3] + d] = v;
    private static double G3(Tensor<double> t, int a, int b, int c) => t[(a * t.Shape[1] + b) * t.Shape[2] + c];
    private static void S3(Tensor<double> t, int a, int b, int c, double v) => t[(a * t.Shape[1] + b) * t.Shape[2] + c] = v;

    private static Tensor<double> RefCyclicShift(Tensor<double> x, int shift)
    {
        int batch = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3];
        var shifted = new Tensor<double>(new[] { batch, h, w, c });
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    int srcI = (i - shift % h + h) % h;
                    int srcJ = (j - shift % w + w) % w;
                    for (int d = 0; d < c; d++) S4(shifted, b, i, j, d, G4(x, b, srcI, srcJ, d));
                }
        return shifted;
    }

    private static (Tensor<double>, int, int) RefWindowPartition(Tensor<double> x, int ws)
    {
        int batch = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3];
        int padH = (ws - h % ws) % ws, padW = (ws - w % ws) % ws, paddedH = h + padH, paddedW = w + padW;
        var padded = new Tensor<double>(new[] { batch, paddedH, paddedW, c });
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < paddedH; i++)
                for (int j = 0; j < paddedW; j++)
                    for (int d = 0; d < c; d++)
                        S4(padded, b, i, j, d, (i < h && j < w) ? G4(x, b, i, j, d) : 0.0);
        int nH = paddedH / ws, nW = paddedW / ws, nWin = nH * nW, area = ws * ws;
        var windows = new Tensor<double>(new[] { batch * nWin, area, c });
        for (int b = 0; b < batch; b++)
            for (int wh = 0; wh < nH; wh++)
                for (int ww = 0; ww < nW; ww++)
                {
                    int widx = b * nWin + wh * nW + ww;
                    for (int i = 0; i < ws; i++)
                        for (int j = 0; j < ws; j++)
                            for (int d = 0; d < c; d++)
                                S3(windows, widx, i * ws + j, d, G4(padded, b, wh * ws + i, ww * ws + j, d));
                }
        return (windows, nH, nW);
    }

    private static Tensor<double> RefWindowReverse(Tensor<double> windows, int nH, int nW, int batch, int h, int w, int ws)
    {
        int nWin = nH * nW, c = windows.Shape[2];
        var spatial = new Tensor<double>(new[] { batch, h, w, c });
        for (int b = 0; b < batch; b++)
            for (int wh = 0; wh < nH; wh++)
                for (int ww = 0; ww < nW; ww++)
                {
                    int widx = b * nWin + wh * nW + ww;
                    for (int i = 0; i < ws; i++)
                        for (int j = 0; j < ws; j++)
                        {
                            int oh = wh * ws + i, ow = ww * ws + j;
                            if (oh < h && ow < w)
                                for (int d = 0; d < c; d++) S4(spatial, b, oh, ow, d, G3(windows, widx, i * ws + j, d));
                        }
                }
        return spatial;
    }

    private static Tensor<double> RefBiasedAttention(Tensor<double> q, Tensor<double> k, Tensor<double> v, int heads, Tensor<double> table, int[,] idx)
    {
        int nw = q.Shape[0], area = q.Shape[1], c = q.Shape[2], hd = c / heads; double scale = 1.0 / Math.Sqrt(hd);
        var output = new Tensor<double>(new[] { nw, area, c });
        for (int wi = 0; wi < nw; wi++)
            for (int head = 0; head < heads; head++)
            {
                int off = head * hd; var sc = new double[area, area];
                for (int i = 0; i < area; i++)
                    for (int j = 0; j < area; j++)
                    {
                        double s = 0; for (int d = 0; d < hd; d++) s += G3(q, wi, i, off + d) * G3(k, wi, j, off + d);
                        s *= scale; s += table[idx[i, j] * table.Shape[1] + head]; sc[i, j] = s;
                    }
                for (int i = 0; i < area; i++)
                {
                    double mx = double.NegativeInfinity; for (int j = 0; j < area; j++) mx = Math.Max(mx, sc[i, j]);
                    double sum = 0; for (int j = 0; j < area; j++) { sc[i, j] = Math.Exp(sc[i, j] - mx); sum += sc[i, j]; }
                    for (int j = 0; j < area; j++) sc[i, j] /= sum;
                }
                for (int i = 0; i < area; i++)
                    for (int d = 0; d < hd; d++)
                    {
                        double val = 0; for (int j = 0; j < area; j++) val += sc[i, j] * G3(v, wi, j, off + d);
                        S3(output, wi, i, off + d, val);
                    }
            }
        return output;
    }

    private static Tensor<double> RefPatchMerge(Tensor<double> xs, int h, int w) // Swin PatchMergingBlock, seq layout
    {
        int batch = xs.Shape[0], dim = xs.Shape[2];
        int hPad = h + (h & 1), wPad = w + (w & 1);
        var src = xs;
        if (hPad != h || wPad != w)
        {
            src = new Tensor<double>(new[] { batch, hPad * wPad, dim });
            for (int n = 0; n < batch; n++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++)
                for (int d = 0; d < dim; d++) S3(src, n, i * wPad + j, d, G3(xs, n, i * w + j, d));
        }
        int newH = hPad / 2, newW = wPad / 2;
        var merged = new Tensor<double>(new[] { batch, newH * newW, dim * 4 });
        for (int n = 0; n < batch; n++)
            for (int i = 0; i < newH; i++)
                for (int j = 0; j < newW; j++)
                {
                    int ni = i * newW + j;
                    int i0 = (2 * i) * wPad + (2 * j), i1 = (2 * i) * wPad + (2 * j + 1);
                    int i2 = (2 * i + 1) * wPad + (2 * j), i3 = (2 * i + 1) * wPad + (2 * j + 1);
                    for (int d = 0; d < dim; d++)
                    {
                        S3(merged, n, ni, d, G3(src, n, i0, d)); S3(merged, n, ni, dim + d, G3(src, n, i1, d));
                        S3(merged, n, ni, 2 * dim + d, G3(src, n, i2, d)); S3(merged, n, ni, 3 * dim + d, G3(src, n, i3, d));
                    }
                }
        return merged;
    }

    private static Tensor<double> RefRoIAlign(Tensor<double> features, Tensor<double> rois, double spatialScale, int outputSize, int samplingRatio, int[]? batchIndices)
    {
        int batchSize = features.Shape[0], channels = features.Shape[1], featureH = features.Shape[2], featureW = features.Shape[3];
        int numRois = rois.Shape[0];
        var output = new Tensor<double>(new[] { numRois, channels, outputSize, outputSize });
        double Feat(int b, int c, int y, int x) => At(features, b, c, y, x);
        double Bilinear(int batch, int channel, double y, double x)
        {
            int y0 = (int)Math.Floor(y), x0 = (int)Math.Floor(x);
            int y1 = Math.Min(y0 + 1, featureH - 1), x1 = Math.Min(x0 + 1, featureW - 1);
            double wy1 = y - y0, wy0 = 1.0 - wy1, wx1 = x - x0, wx0 = 1.0 - wx1;
            return wy0 * (wx0 * Feat(batch, channel, y0, x0) + wx1 * Feat(batch, channel, y0, x1))
                 + wy1 * (wx0 * Feat(batch, channel, y1, x0) + wx1 * Feat(batch, channel, y1, x1));
        }
        for (int roiIdx = 0; roiIdx < numRois; roiIdx++)
        {
            int batchIdx = batchIndices is not null && roiIdx < batchIndices.Length ? Math.Min(batchIndices[roiIdx], batchSize - 1) : 0;
            double x1 = rois[roiIdx * 4 + 0] * spatialScale, y1 = rois[roiIdx * 4 + 1] * spatialScale;
            double x2 = rois[roiIdx * 4 + 2] * spatialScale, y2 = rois[roiIdx * 4 + 3] * spatialScale;
            double binW = (x2 - x1) / outputSize, binH = (y2 - y1) / outputSize;
            for (int c = 0; c < channels; c++)
                for (int ph = 0; ph < outputSize; ph++)
                    for (int pw = 0; pw < outputSize; pw++)
                    {
                        double binStartY = y1 + ph * binH, binStartX = x1 + pw * binW, sum = 0; int count = 0;
                        for (int iy = 0; iy < samplingRatio; iy++)
                            for (int ix = 0; ix < samplingRatio; ix++)
                            {
                                double y = binStartY + (iy + 0.5) * binH / samplingRatio;
                                double x = binStartX + (ix + 0.5) * binW / samplingRatio;
                                if (y >= 0 && y < featureH && x >= 0 && x < featureW) { sum += Bilinear(batchIdx, c, y, x); count++; }
                            }
                        Set(output, roiIdx, c, ph, pw, count > 0 ? sum / count : 0);
                    }
        }
        return output;
    }

    private static Tensor<double> RefReshapeRPN(Tensor<double> x, int outputDim) // RPN.ReshapeRPNOutput
    {
        int batch = x.Shape[0], channelDim = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        int numAnchors = channelDim / outputDim;
        var result = new Tensor<double>(new[] { batch, height * width * numAnchors, outputDim });
        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                    for (int a = 0; a < numAnchors; a++)
                    {
                        for (int d = 0; d < outputDim; d++) S3(result, b, idx, d, At(x, b, a * outputDim + d, h, w));
                        idx++;
                    }
        }
        return result;
    }

    /// <summary>Nearest and bilinear resizing, 2x upsampling, the four max-pool variants, batch-statistics normalisation and the spatial flatten, against the neck, text-detection, YOLO, ResNet and CRNN loops they replaced.</summary>
    [Fact]
    public async Task SpatialResamplingAndPooling_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 1000;
            int[] sizes = { 1, 2, 3, 4, 5, 7, 8, 13, 16 };
            foreach (int n in new[] { 1, 2 })
            foreach (int h in sizes)
            foreach (int w in sizes)
            {
                var x = Rand(new[] { n, 3, h, w }, ++seed);
                string tag = $"[{n},3,{h},{w}]";

                foreach (var (th, tw) in new[] { (h * 2, w * 2), (Math.Max(1, h / 2), Math.Max(1, w / 2)), (h + 3, w + 1), (5, 7) })
                {
                    Compare($"ResizeNearest {tag}->{th}x{tw}", RefResizeToMatch(x, th, tw), CvTensorOps<double>.ResizeNearest(x, th, tw), 0);
                    Compare($"Bilinear {tag}->{th}x{tw}", RefBilinear(x, th, tw), CvTensorOps<double>.ResizeBilinearAsymmetric(x, th, tw), 1e-13);
                }

                Compare($"Upsample2x {tag}", RefUpsample2x(x), CvTensorOps<double>.Upsample2xNearest(x), 0);
                Compare($"MaxPool2x2Ceil {tag}", RefDownsample2x(x), CvTensorOps<double>.MaxPool2x2Ceil(x), 0);
                Compare($"BatchStatsNorm {tag}", RefCrnnBatchNorm(x), CvTensorOps<double>.BatchStatisticsNorm(x, 1e-5), 1e-9);

                foreach (int k in new[] { 3, 5 })
                {
                    Compare($"MaxPoolSame k{k} {tag}", RefYoloMaxPool(x, k), CvTensorOps<double>.MaxPoolSame(x, k), 0);
                }

                foreach (var (k, s, p) in new[] { (3, 2, 1), (3, 2, 0), (2, 2, 1), (3, 1, 2), (5, 3, 2), (1, 1, 0) })
                {
                    if (h + 2 * p < k || w + 2 * p < k) continue;
                    Compare($"MaxPoolPadded k{k}s{s}p{p} {tag}", RefResNetMaxPool(x, k, s, p), CvTensorOps<double>.MaxPoolPadded(x, k, s, p), 0);
                }

                foreach (var (kh, kw) in new[] { (2, 2), (2, 1) })
                {
                    if (h < kh || w < kw) continue;
                    Compare($"MaxPoolFloor {kh}x{kw} {tag}", RefCrnnMaxPool(x, kh, kw), CvTensorOps<double>.MaxPoolFloor(x, kh, kw), 0);
                }

                var tokens = CvTensorOps<double>.FlattenSpatial(x);
                Compare($"Flatten/Unflatten {tag}", x, CvTensorOps<double>.UnflattenSpatial(tokens, h, w), 0);
            }
        AssertNoFailures();
    }

    /// <summary>Multi-head attention (plain and causal) and last-axis layer normalisation, against the DETR/TrOCR loops.</summary>
    [Fact]
    public async Task AttentionAndLayerNorm_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 2000;
            foreach (int n in new[] { 1, 2 })
            foreach (var (lq, lk, dd, heads) in new[] { (1, 1, 4, 1), (3, 5, 8, 2), (6, 6, 12, 3), (4, 9, 16, 8) })
            {
                var q = Rand(new[] { n, lq, dd }, ++seed);
                var k = Rand(new[] { n, lk, dd }, ++seed);
                var v = Rand(new[] { n, lk, dd }, ++seed);
                string tag = $"n{n} lq{lq} lk{lk} d{dd} h{heads}";
                double sc = 1.0 / Math.Sqrt(dd / heads);
                Compare($"Attention {tag}", RefAttention(q, k, v, heads, false), CvTensorOps<double>.MultiHeadAttention(q, k, v, heads, sc), 1e-12);
                if (lq == lk)
                {
                    Compare($"CausalAttention {tag}", RefAttention(q, k, v, heads, true), CvTensorOps<double>.MultiHeadAttention(q, k, v, heads, sc, causal: true), 1e-12);
                }
                var g = Rand(new[] { dd }, ++seed);
                var bb = Rand(new[] { dd }, ++seed);
                Compare($"LayerNorm {tag}", RefLayerNorm(q, g, bb, 1e-6), CvTensorOps<double>.LayerNormLastAxis(q, g, bb, 1e-6), 1e-12);
            }
        AssertNoFailures();
    }

    /// <summary>Swin's cyclic shift, window partition and window reverse, including padded and cropped grids.</summary>
    [Fact]
    public async Task SwinWindowOps_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 3000;
            foreach (int n in new[] { 1, 2 })
            foreach (var (h, w, ws) in new[] { (4, 4, 2), (7, 5, 3), (8, 8, 4), (5, 9, 4), (1, 3, 2), (6, 6, 7) })
            {
                var x = Rand(new[] { n, h, w, 5 }, ++seed);
                string tag = $"n{n} {h}x{w} ws{ws}";
                foreach (int s in new[] { -1, 1, -ws / 2, 3 })
                {
                    Compare($"CyclicShift {s} {tag}", RefCyclicShift(x, s), CvTensorOps<double>.CyclicShift(x, s), 0);
                }
                var (rw, rh, rwd) = RefWindowPartition(x, ws);
                var (cw, ch, cwd) = CvTensorOps<double>.WindowPartition(x, ws);
                if (rh != ch || rwd != cwd) { _failures.Add($"window counts {tag}"); }
                Compare($"WindowPartition {tag}", rw, cw, 0);
                Compare($"WindowReverse {tag}", RefWindowReverse(rw, rh, rwd, n, h, w, ws), CvTensorOps<double>.WindowReverse(rw, rh, rwd, n, h, w, ws), 0);
                Compare($"Partition/Reverse roundtrip {tag}", x, CvTensorOps<double>.WindowReverse(cw, ch, cwd, n, h, w, ws), 0);
            }
        AssertNoFailures();
    }

    /// <summary>Swin window attention with its relative-position bias table.</summary>
    [Fact]
    public async Task WindowAttentionWithRelativePositionBias_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 4000;
            foreach (var (nw, area, c, heads) in new[] { (1, 4, 8, 2), (3, 9, 12, 3), (2, 16, 16, 4) })
            {
                var q = Rand(new[] { nw, area, c }, ++seed); var k = Rand(new[] { nw, area, c }, ++seed); var v = Rand(new[] { nw, area, c }, ++seed);
                int R = 2 * area; var table = Rand(new[] { R, heads }, ++seed);
                var idx = new int[area, area]; var r = new Random(++seed);
                for (int i = 0; i < area; i++) for (int j = 0; j < area; j++) idx[i, j] = r.Next(R);
                var bias = CvTensorOps<double>.RelativePositionBias(table, idx);
                Compare($"BiasedAttention nw{nw} a{area} c{c} h{heads}", RefBiasedAttention(q, k, v, heads, table, idx),
                    CvTensorOps<double>.MultiHeadAttention(q, k, v, heads, 1.0 / Math.Sqrt(c / heads), scoreBias: bias), 1e-12);
            }
        AssertNoFailures();
    }

    /// <summary>Swin 2x2 patch merging, including odd grids.</summary>
    [Fact]
    public async Task PatchMerging_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 5000;
            foreach (int n in new[] { 1, 2 })
            foreach (var (h, w) in new[] { (4, 4), (7, 7), (4, 6), (5, 8), (1, 1), (3, 2) })
            {
                var seq = Rand(new[] { n, h * w, 3 }, ++seed);
                var nhwc = new Tensor<double>(new[] { n, h, w, 3 });
                for (int i = 0; i < seq.Length; i++) nhwc[i] = seq[i];
                Compare($"PatchMerge n{n} {h}x{w}", RefPatchMerge(seq, h, w), CvTensorOps<double>.PatchMerge2x2(nhwc), 0);
            }
        AssertNoFailures();
    }

    /// <summary>RoIAlign, including boxes partly or wholly outside the feature map.</summary>
    [Fact]
    public async Task RoIAlign_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 6000;
            {
                var rr = new Random(++seed);
                foreach (int nb in new[] { 1, 2 })
                foreach (var (fh, fw) in new[] { (8, 8), (5, 7), (16, 12) })
                foreach (var (ps, sr) in new[] { (2, 2), (7, 2), (3, 1) })
                {
                    var feat = Rand(new[] { nb, 3, fh, fw }, ++seed);
                    int R = 6; var rois = new Tensor<double>(new[] { R, 4 }); var bi = new int[R];
                    double imgW = fw * 16.0, imgH = fh * 16.0;
                    for (int r = 0; r < R; r++)
                    {
                        // Include boxes partly and wholly outside the map, and degenerate ones.
                        double ax = rr.NextDouble() * imgW * 1.3 - imgW * 0.15, ay = rr.NextDouble() * imgH * 1.3 - imgH * 0.15;
                        double bw = rr.NextDouble() * imgW * 0.8, bh = rr.NextDouble() * imgH * 0.8;
                        if (r == R - 1) { ax = imgW * 2; ay = imgH * 2; }
                        rois[r * 4] = ax; rois[r * 4 + 1] = ay; rois[r * 4 + 2] = ax + bw; rois[r * 4 + 3] = ay + bh;
                        bi[r] = rr.Next(nb);
                    }
                    var flat = new double[R * 4]; for (int i = 0; i < flat.Length; i++) flat[i] = rois[i];
                    Compare($"RoIAlign n{nb} {fh}x{fw} p{ps} s{sr}", RefRoIAlign(feat, rois, 1.0 / 16.0, ps, sr, bi),
                        CvTensorOps<double>.RoIAlign(feat, flat, bi, 1.0 / 16.0, ps, sr), 1e-12);
                }
            }
        AssertNoFailures();
    }

    /// <summary>The RPN head's [B, A*D, H, W] to [B, H*W*A, D] reshape.</summary>
    [Fact]
    public async Task RpnOutputReshape_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 7000;
            foreach (var (nb, a, dd, hh, ww) in new[] { (1, 3, 2, 4, 5), (2, 9, 4, 3, 3), (1, 1, 4, 1, 7) })
            {
                var x = Rand(new[] { nb, a * dd, hh, ww }, ++seed);
                var got = RPN<double>.ReshapeRPNOutput(x, nb, hh, ww, dd);
                Compare($"ReshapeRPNOutput n{nb} A{a} D{dd} {hh}x{ww}", RefReshapeRPN(x, dd), got, 0);
            }

    
        AssertNoFailures();
    }

    // ---- model-local rewrites ----

    // Verbatim copy of CascadeRCNN.RefineBoxes before the rewrite.
    private static Tensor<double> RefRefineBoxes(Tensor<double> boxes, Tensor<double> deltas, int imageWidth, int imageHeight)
    {
        int numBoxes = boxes.Shape[0];
        var refinedBoxes = new Tensor<double>(new[] { numBoxes, 4 });
        for (int i = 0; i < numBoxes; i++)
        {
            double px1 = boxes[i, 0];
            double py1 = boxes[i, 1];
            double px2 = boxes[i, 2];
            double py2 = boxes[i, 3];

            double pw = px2 - px1;
            double ph = py2 - py1;
            double pcx = px1 + pw / 2;
            double pcy = py1 + ph / 2;

            int deltaOffset = 4; // Skip background class
            double dx = deltas[i, deltaOffset];
            double dy = deltas[i, deltaOffset + 1];
            double dw = deltas[i, deltaOffset + 2];
            double dh = deltas[i, deltaOffset + 3];

            double predCx = pcx + dx * pw;
            double predCy = pcy + dy * ph;
            double predW = pw * Math.Exp(Math.Min(dw, 4.0));
            double predH = ph * Math.Exp(Math.Min(dh, 4.0));

            refinedBoxes[i, 0] = Math.Max(0, predCx - predW / 2);
            refinedBoxes[i, 1] = Math.Max(0, predCy - predH / 2);
            refinedBoxes[i, 2] = Math.Min(imageWidth, predCx + predW / 2);
            refinedBoxes[i, 3] = Math.Min(imageHeight, predCy + predH / 2);
        }

        return refinedBoxes;
    }

    /// <summary>
    /// Cascade R-CNN's between-stage box refinement, including scale deltas above the cap of 4 and
    /// boxes pushed past every image edge.
    /// </summary>
    [Fact]
    public async Task CascadeBoxRefinement_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        var r = new Random(7001);
        foreach (int numBoxes in new[] { 1, 5, 17 })
        foreach (int numClasses in new[] { 2, 4 })
        {
            const int imageWidth = 96, imageHeight = 64;
            var boxes = new Tensor<double>(new[] { numBoxes, 4 });
            for (int i = 0; i < numBoxes; i++)
            {
                double x1 = r.NextDouble() * imageWidth, y1 = r.NextDouble() * imageHeight;
                boxes[i, 0] = x1;
                boxes[i, 1] = y1;
                boxes[i, 2] = x1 + 1 + r.NextDouble() * 40;
                boxes[i, 3] = y1 + 1 + r.NextDouble() * 40;
            }

            // Wide enough to exceed the exp cap (4) and to push boxes off the image on every side.
            var deltas = Rand(new[] { numBoxes, 4 * numClasses }, r.Next());
            for (int i = 0; i < deltas.Length; i++) deltas[i] *= 3;

            Compare($"RefineBoxes n{numBoxes} c{numClasses}", RefRefineBoxes(boxes, deltas, imageWidth, imageHeight),
                CascadeRCNN<double>.RefineBoxes(boxes, deltas, imageWidth, imageHeight), 1e-12);
        }

        AssertNoFailures();
    }

    // Verbatim copy of DBNet.ApplyDifferentiableBinarization before the rewrite.
    private static Tensor<double> RefDbBinarization(Tensor<double> prob, Tensor<double> thresh, double k)
    {
        var result = new Tensor<double>(Dims(prob));
        for (int i = 0; i < prob.Length; i++)
        {
            double p = prob[i];
            double t = thresh[i];
            result[i] = 1.0 / (1.0 + Math.Exp(-k * (p - t)));
        }

        return result;
    }

    /// <summary>DBNet's differentiable binarization B = 1 / (1 + exp(-k (P - T))).</summary>
    [Fact]
    public async Task DbBinarization_MatchesTheLoopItReplaced()
    {
        await Task.Yield();
        int seed = 8001;
        foreach (double k in new[] { 1.0, 50.0 })
        foreach (var shape in new[] { new[] { 1, 1, 5, 7 }, new[] { 2, 1, 16, 16 } })
        {
            var prob = Rand(shape, ++seed);
            var thresh = Rand(shape, ++seed);
            Compare($"DB k{k} [{string.Join(",", shape)}]", RefDbBinarization(prob, thresh, k),
                DBNet<double>.ApplyDifferentiableBinarization(prob, thresh, k), 1e-12);
        }

        AssertNoFailures();
    }

    // ---- gradient tape ----

    private static Tensor<double> Const(int[] shape, int seed)
    {
        var r = new Random(seed); var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = r.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<double> CascadeInputBoxes()
    {
        var boxes = new Tensor<double>(new[] { 3, 4 });
        double[] values = { 10, 12, 40, 30, 5, 5, 20, 50, 30, 8, 60, 44 };
        for (int i = 0; i < values.Length; i++) boxes[i] = values[i];
        return boxes;
    }

    private static readonly Dictionary<string, (int[] Shape, Func<Tensor<double>, Tensor<double>> Op)> GradientCases = new()
    {
        ["ResizeNearest"] = (new[] { 2, 3, 5, 7 }, x => CvTensorOps<double>.ResizeNearest(x, 8, 3)),
        ["Upsample2xNearest"] = (new[] { 1, 2, 3, 5 }, x => CvTensorOps<double>.Upsample2xNearest(x)),
        ["BilinearUp"] = (new[] { 2, 2, 4, 5 }, x => CvTensorOps<double>.ResizeBilinearAsymmetric(x, 9, 7)),
        ["BilinearDown"] = (new[] { 1, 2, 8, 8 }, x => CvTensorOps<double>.ResizeBilinearAsymmetric(x, 3, 5)),
        ["MaxPool2x2Ceil"] = (new[] { 2, 2, 5, 7 }, x => CvTensorOps<double>.MaxPool2x2Ceil(x)),
        ["MaxPoolFloor"] = (new[] { 1, 2, 5, 4 }, x => CvTensorOps<double>.MaxPoolFloor(x, 2, 1)),
        ["MaxPoolSame"] = (new[] { 1, 2, 6, 5 }, x => CvTensorOps<double>.MaxPoolSame(x, 5)),
        ["MaxPoolPadded"] = (new[] { 2, 2, 7, 6 }, x => CvTensorOps<double>.MaxPoolPadded(x, 3, 2, 1)),
        ["BatchStatisticsNorm"] = (new[] { 2, 3, 4, 3 }, x => CvTensorOps<double>.BatchStatisticsNorm(x, 1e-5)),
        ["FlattenUnflatten"] = (new[] { 2, 3, 4, 5 }, x => CvTensorOps<double>.UnflattenSpatial(CvTensorOps<double>.FlattenSpatial(x), 4, 5)),
        ["AttentionQuery"] = (new[] { 2, 3, 8 }, x => CvTensorOps<double>.MultiHeadAttention(x, Const(new[] { 2, 5, 8 }, 11), Const(new[] { 2, 5, 8 }, 12), 2, 0.5)),
        ["AttentionKey"] = (new[] { 2, 5, 8 }, x => CvTensorOps<double>.MultiHeadAttention(Const(new[] { 2, 3, 8 }, 11), x, Const(new[] { 2, 5, 8 }, 12), 2, 0.5)),
        ["AttentionValue"] = (new[] { 2, 5, 8 }, x => CvTensorOps<double>.MultiHeadAttention(Const(new[] { 2, 3, 8 }, 11), Const(new[] { 2, 5, 8 }, 12), x, 2, 0.5)),
        ["CausalAttention"] = (new[] { 2, 4, 8 }, x => CvTensorOps<double>.MultiHeadAttention(x, x, x, 2, 0.5, causal: true)),
        ["LayerNorm"] = (new[] { 2, 3, 6 }, x => CvTensorOps<double>.LayerNormLastAxis(x, Const(new[] { 6 }, 7), Const(new[] { 6 }, 8), 1e-6)),
        ["CyclicShift"] = (new[] { 1, 5, 4, 3 }, x => CvTensorOps<double>.CyclicShift(x, -2)),
        ["WindowPartitionPadded"] = (new[] { 1, 5, 7, 3 }, x => CvTensorOps<double>.WindowPartition(x, 3).Windows),
        ["WindowReverseCropped"] = (new[] { 6, 9, 3 }, x => CvTensorOps<double>.WindowReverse(x, 2, 3, 1, 5, 7, 3)),
        ["RelativePositionBiasTable"] = (new[] { 8, 2 }, t => CvTensorOps<double>.MultiHeadAttention(
            Const(new[] { 1, 4, 8 }, 21), Const(new[] { 1, 4, 8 }, 22), Const(new[] { 1, 4, 8 }, 23), 2, 0.5,
            scoreBias: CvTensorOps<double>.RelativePositionBias(t, new int[,] { { 0, 1, 2, 3 }, { 4, 5, 6, 7 }, { 7, 6, 5, 4 }, { 3, 2, 1, 0 } }))),
        ["PatchMergeOdd"] = (new[] { 1, 5, 7, 3 }, x => CvTensorOps<double>.PatchMerge2x2(x)),
        ["RoIAlignFeatures"] = (new[] { 2, 3, 6, 5 }, x => CvTensorOps<double>.RoIAlign(
            x, new double[] { 5, 7, 60, 70, -10, 3, 40, 50, 20, 20, 21, 90 }, new[] { 0, 1, 1 }, 1.0 / 16.0, 3, 2)),
        ["RpnOutputReshape"] = (new[] { 2, 12, 3, 4 }, x => RPN<double>.ReshapeRPNOutput(x, 2, 3, 4, 4)),
        ["CascadeRefineBoxesDeltas"] = (new[] { 3, 8 }, d => CascadeRCNN<double>.RefineBoxes(CascadeInputBoxes(), d, 96, 64)),
        ["DbBinarizationProbability"] = (new[] { 1, 1, 4, 5 }, p => DBNet<double>.ApplyDifferentiableBinarization(p, Const(new[] { 1, 1, 4, 5 }, 41), 5.0)),
        ["DbBinarizationThreshold"] = (new[] { 1, 1, 4, 5 }, t => DBNet<double>.ApplyDifferentiableBinarization(Const(new[] { 1, 1, 4, 5 }, 42), t, 5.0)),
    };

    public static IEnumerable<object[]> GradientCaseNames => GradientCases.Keys.Select(k => new object[] { k });

    /// <summary>
    /// The tape's gradient of a random linear functional of the op's output matches central finite
    /// differences - and is not missing, which is what a tape-severing element loop produces.
    /// </summary>
    [Theory]
    [MemberData(nameof(GradientCaseNames))]
    public async Task TapeGradient_MatchesFiniteDifferences(string name)
    {
        await Task.Yield();
        var (shape, op) = GradientCases[name];
        var engine = AiDotNetEngine.Current;
        var x = Const(shape, 1);
        var weights = Const(Dims(op(x)), 2);

        double Loss(Tensor<double> input)
        {
            var output = op(input);
            double sum = 0;
            for (int i = 0; i < output.Length; i++) sum += output[i] * weights[i];
            return sum;
        }

        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var loss = engine.ReduceSum(engine.TensorMultiply(op(x), weights), null);
            gradients = tape.ComputeGradients(loss, new[] { x });
        }

        Assert.True(gradients.TryGetValue(x, out var gradient) && gradient is not null,
            $"{name}: no gradient reached the input - the op is not on the tape.");

        const double step = 1e-6, tolerance = 1e-5;
        double maxError = 0, maxGradient = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double original = x[i];
            x[i] = original + step;
            double plus = Loss(x);
            x[i] = original - step;
            double minus = Loss(x);
            x[i] = original;
            double finiteDifference = (plus - minus) / (2 * step);
            maxError = Math.Max(maxError, Math.Abs(finiteDifference - gradient[i]));
            maxGradient = Math.Max(maxGradient, Math.Abs(gradient[i]));
        }

        Assert.True(maxGradient > 0, $"{name}: the gradient is identically zero.");
        Assert.True(maxError < tolerance, $"{name}: max |tape - finite difference| = {maxError:e2} (tolerance {tolerance:e0}).");
    }
}
