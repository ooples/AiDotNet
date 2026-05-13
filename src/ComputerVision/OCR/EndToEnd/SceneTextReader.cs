using AiDotNet.Attributes;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.ComputerVision.OCR.Recognition;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.OCR.EndToEnd;

/// <summary>
/// End-to-end scene text reader that combines detection and recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SceneTextReader is a complete OCR pipeline that first
/// detects text regions in images (using CRAFT, EAST, or DBNet), then recognizes
/// the text in each region (using CRNN or TrOCR). It's designed for reading text
/// in natural images like photos of signs, billboards, and product labels.</para>
///
/// <para>Key features:
/// - Two-stage pipeline: detection + recognition
/// - Handles arbitrary text orientations
/// - Works with curved and rotated text
/// - Configurable detection and recognition models
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("ABCNet: Real-time Scene Text Spotting", "https://arxiv.org/abs/1911.09941")]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class SceneTextReader<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{
    private readonly TextDetectorBase<T> _detector;
    private readonly OCRBase<T> _recognizer;
    private readonly OCROptions<T> _options;

    /// <summary>
    /// Name of this scene text reader.
    /// </summary>
    public string Name => $"SceneTextReader-{_detector.Name}-{_recognizer.Name}";

    /// <summary>
    /// Creates a new scene text reader with default models.
    /// </summary>
    public SceneTextReader(OCROptions<T> options)
    {
        _options = options;

        // Initialize detector based on options
        var detectionOptions = new TextDetectionOptions<T>
        {
            Architecture = options.DetectionModel switch
            {
                TextDetectionModel.CRAFT => TextDetectionArchitecture.CRAFT,
                TextDetectionModel.EAST => TextDetectionArchitecture.EAST,
                TextDetectionModel.DBNet => TextDetectionArchitecture.DBNet,
                _ => TextDetectionArchitecture.DBNet
            },
            ConfidenceThreshold = options.ConfidenceThreshold
        };

        _detector = options.DetectionModel switch
        {
            TextDetectionModel.CRAFT => new CRAFT<T>(detectionOptions),
            TextDetectionModel.EAST => new EAST<T>(detectionOptions),
            TextDetectionModel.DBNet => new DBNet<T>(detectionOptions),
            _ => new DBNet<T>(detectionOptions)
        };

        // Initialize recognizer based on options
        _recognizer = options.RecognitionModel switch
        {
            TextRecognitionModel.CRNN => new CRNN<T>(options),
            TextRecognitionModel.TrOCR => new TrOCR<T>(options),
            _ => new CRNN<T>(options)
        };
    }

    /// <summary>
    /// Reads all text in an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>OCR result with all recognized text.</returns>
    public OCRResult<T> ReadText(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageWidth = image.Shape[3];
        int imageHeight = image.Shape[2];

        // Step 1: Detect text regions
        var detectionResult = _detector.Detect(image);

        // Step 2: Recognize text in each region
        var recognizedTexts = new List<RecognizedText<T>>();

        foreach (var region in detectionResult.TextRegions)
        {
            // Crop text region from image
            var crop = CropRegion(image, region);

            // Recognize text
            var (text, confidence) = _recognizer.RecognizeText(crop);

            if (!string.IsNullOrWhiteSpace(text) &&
                NumOps.ToDouble(confidence) >= NumOps.ToDouble(_options.ConfidenceThreshold))
            {
                var recognized = new RecognizedText<T>(text, confidence)
                {
                    Box = region.Box,
                    Polygon = region.Polygon
                };

                recognizedTexts.Add(recognized);
            }
        }

        // Sort text regions by reading order (top-to-bottom, left-to-right)
        if (_options.GroupTextLines)
        {
            recognizedTexts = SortByReadingOrder(recognizedTexts);
        }

        // Build full text
        string fullText = string.Join(" ", recognizedTexts.Select(r => r.Text));

        return new OCRResult<T>
        {
            TextRegions = recognizedTexts,
            FullText = fullText,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };
    }

    /// <summary>
    /// Reads text from a pre-detected region.
    /// </summary>
    public (string text, T confidence) ReadRegion(Tensor<T> croppedRegion)
    {
        return _recognizer.RecognizeText(croppedRegion);
    }

    private Tensor<T> CropRegion(Tensor<T> image, TextRegion<T> region)
    {
        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Get bounding box
        double x1 = NumOps.ToDouble(region.Box.X1);
        double y1 = NumOps.ToDouble(region.Box.Y1);
        double x2 = NumOps.ToDouble(region.Box.X2);
        double y2 = NumOps.ToDouble(region.Box.Y2);

        // Clamp to image bounds
        int cropX1 = Math.Max(0, (int)x1);
        int cropY1 = Math.Max(0, (int)y1);
        int cropX2 = Math.Min(width, (int)x2);
        int cropY2 = Math.Min(height, (int)y2);

        int cropW = Math.Max(1, cropX2 - cropX1);
        int cropH = Math.Max(1, cropY2 - cropY1);

        var crop = new Tensor<T>(new[] { batch, channels, cropH, cropW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < cropH; h++)
                {
                    for (int w = 0; w < cropW; w++)
                    {
                        int srcH = cropY1 + h;
                        int srcW = cropX1 + w;

                        if (srcH < height && srcW < width)
                        {
                            crop[b, c, h, w] = image[b, c, srcH, srcW];
                        }
                    }
                }
            }
        }

        // If polygon available and text is rotated, apply perspective correction.
        // Pass the crop's origin in full-image coordinates so the warp can
        // translate the polygon's source corners into crop-local space —
        // without this, `sx`/`sy` would reference positions outside the
        // crop's bounds and the bilinear sampler would return black for
        // every region whose box doesn't start at (0, 0).
        if (region.Polygon.Count >= 4 && Math.Abs(region.RotationAngle) > 5)
        {
            crop = CorrectPerspective(crop, region, cropX1, cropY1);
        }

        return crop;
    }

    /// <summary>
    /// Removes perspective distortion from a text crop by warping the four
    /// polygon corners onto a canonical axis-aligned rectangle via a planar
    /// homography (DLT, Hartley &amp; Zisserman 2003 §4.1). Output is sampled
    /// with bilinear interpolation. For quads with rotation > 45° we keep
    /// the existing 90° rotate fast-path because the bilinear sampler is
    /// dominated by the corner correspondence; for the typical 5°-45°
    /// regime this is the implementation that actually undoes the warp.
    /// </summary>
    private Tensor<T> CorrectPerspective(Tensor<T> crop, TextRegion<T> region, int cropX, int cropY)
    {
        if (region.Polygon.Count < 4) return crop;

        int batch = crop.Shape[0];
        int channels = crop.Shape[1];
        int srcH = crop.Shape[2];
        int srcW = crop.Shape[3];

        // Reduce arbitrary N-gon contours (N >= 4) to 4 quad corners using the
        // OpenCV-canonical extreme-points approach: tl = argmin(x+y),
        // br = argmax(x+y), tr = argmax(x-y), bl = argmin(x-y). For N == 4
        // this is a no-op pass-through to OrderQuadCorners. For N > 4 it
        // selects a stable quad bounded by the contour's extremes instead of
        // letting OrderQuadCorners silently consume the first 4 indices.
        var quad = ReducePolygonToQuad(region.Polygon);
        var (tl, tr, br, bl) = OrderQuadCorners(quad);

        // Target rectangle dimensions: preserve aspect ratio from the quad.
        double topEdge    = Math.Sqrt(SqDist(tl, tr));
        double bottomEdge = Math.Sqrt(SqDist(bl, br));
        double leftEdge   = Math.Sqrt(SqDist(tl, bl));
        double rightEdge  = Math.Sqrt(SqDist(tr, br));
        int dstW = Math.Max(1, (int)Math.Round(Math.Max(topEdge, bottomEdge)));
        int dstH = Math.Max(1, (int)Math.Round(Math.Max(leftEdge, rightEdge)));

        // Solve for the inverse homography H^-1 mapping destination → source.
        // Target corners in destination image space:
        var (dx, dy) = (
            new[] { 0.0,         dstW - 1.0,  dstW - 1.0,  0.0         },
            new[] { 0.0,         0.0,         dstH - 1.0,  dstH - 1.0  });
        // Translate the polygon's source corners into CROP-LOCAL coordinates
        // (subtract the crop origin) because `crop` is indexed from (0, 0).
        // Using region.Polygon's full-image coordinates directly would make
        // the bilinear sampler read mostly out-of-bounds and return a
        // black/garbled rectified crop for any region whose box doesn't
        // start at the image origin.
        var (sx, sy) = (
            new[] { tl.X - cropX, tr.X - cropX, br.X - cropX, bl.X - cropX },
            new[] { tl.Y - cropY, tr.Y - cropY, br.Y - cropY, bl.Y - cropY });
        if (!SolveHomographyDLT(dx, dy, sx, sy, out double[] h))
        {
            // Degenerate corners (collinear / coincident) — fall back to crop.
            return Math.Abs(region.RotationAngle) > 45 ? Rotate90(crop) : crop;
        }

        // Warp.
        var output = new Tensor<T>(new[] { batch, channels, dstH, dstW });
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < dstH; y++)
                {
                    for (int x = 0; x < dstW; x++)
                    {
                        // (x, y) in dst → (xs, ys) in src via the projective map.
                        double w = h[6] * x + h[7] * y + 1.0;
                        if (Math.Abs(w) < 1e-12) continue;
                        double xs = (h[0] * x + h[1] * y + h[2]) / w;
                        double ys = (h[3] * x + h[4] * y + h[5]) / w;
                        output[b, c, y, x] = BilinearSample(crop, b, c, xs, ys, srcW, srcH);
                    }
                }
            }
        }
        return output;
    }

    /// <summary>
    /// Reduces a polygon with N >= 4 vertices to a stable 4-corner quad using
    /// the canonical extreme-point selection from OpenCV's perspective-correct
    /// pipeline:
    ///   top-left     = argmin(x + y)
    ///   bottom-right = argmax(x + y)
    ///   top-right    = argmax(x - y)
    ///   bottom-left  = argmin(x - y)
    /// When N == 4 the result is the same 4 points the detector produced
    /// (ordering may differ; OrderQuadCorners re-canonicalises afterwards).
    /// When N > 4 it picks the 4 extremes instead of silently truncating to
    /// poly[0..3], which OrderQuadCorners would do.
    /// </summary>
    private static List<(T X, T Y)> ReducePolygonToQuad(List<(T X, T Y)> poly)
    {
        if (poly.Count == 4) return poly;
        int n = poly.Count;
        int tlIdx = 0, brIdx = 0, trIdx = 0, blIdx = 0;
        double minSum = double.PositiveInfinity, maxSum = double.NegativeInfinity;
        double maxDiff = double.NegativeInfinity, minDiff = double.PositiveInfinity;
        for (int i = 0; i < n; i++)
        {
            double x = NumOps.ToDouble(poly[i].X);
            double y = NumOps.ToDouble(poly[i].Y);
            double s = x + y;
            double d = x - y;
            if (s < minSum) { minSum = s; tlIdx = i; }
            if (s > maxSum) { maxSum = s; brIdx = i; }
            if (d > maxDiff) { maxDiff = d; trIdx = i; }
            if (d < minDiff) { minDiff = d; blIdx = i; }
        }
        return new List<(T X, T Y)>
        {
            poly[tlIdx],
            poly[trIdx],
            poly[brIdx],
            poly[blIdx],
        };
    }

    /// <summary>
    /// Orders 4 polygon corners as (top-left, top-right, bottom-right,
    /// bottom-left) so the homography target rectangle has a consistent
    /// orientation. Uses centroid-angle sort, which is robust to corner
    /// permutations from different detectors.
    /// </summary>
    private static ((double X, double Y) TL, (double X, double Y) TR, (double X, double Y) BR, (double X, double Y) BL)
        OrderQuadCorners(List<(T X, T Y)> poly)
    {
        // Use the base class's protected static NumOps instead of
        // Convert.ToDouble — Convert.ToDouble throws on non-IConvertible
        // generic T types and bypasses the library's numeric abstraction.
        var pts = new (double X, double Y)[4];
        for (int i = 0; i < 4; i++)
        {
            pts[i] = (NumOps.ToDouble(poly[i].X), NumOps.ToDouble(poly[i].Y));
        }
        // Centroid.
        double cx = (pts[0].X + pts[1].X + pts[2].X + pts[3].X) / 4.0;
        double cy = (pts[0].Y + pts[1].Y + pts[2].Y + pts[3].Y) / 4.0;
        // Sort by angle from centroid (ascending — Atan2 default ordering),
        // then re-anchor on the top-left corner (smallest x+y) below. The
        // re-anchoring step handles any starting position, so functional
        // correctness doesn't depend on sort direction.
        Array.Sort(pts, (a, b) => Math.Atan2(a.Y - cy, a.X - cx).CompareTo(Math.Atan2(b.Y - cy, b.X - cx)));
        int startIdx = 0;
        double minSum = double.PositiveInfinity;
        for (int i = 0; i < 4; i++)
        {
            double s = pts[i].X + pts[i].Y;
            if (s < minSum) { minSum = s; startIdx = i; }
        }
        return (
            pts[startIdx],
            pts[(startIdx + 1) % 4],
            pts[(startIdx + 2) % 4],
            pts[(startIdx + 3) % 4]);
    }

    private static double SqDist((double X, double Y) a, (double X, double Y) b)
    {
        double dx = a.X - b.X, dy = a.Y - b.Y;
        return dx * dx + dy * dy;
    }

    /// <summary>
    /// Direct Linear Transform: solves for the 8 parameters of a planar
    /// homography mapping (dx_i, dy_i) → (sx_i, sy_i) for i ∈ {0..3}.
    /// Returns false when the 8×8 normal-equation matrix is singular
    /// (collinear corners).
    /// </summary>
    private static bool SolveHomographyDLT(double[] dx, double[] dy, double[] sx, double[] sy, out double[] h)
    {
        // 8 equations, 8 unknowns. Rows alternate:
        //   x' = h0·x + h1·y + h2 − h6·x·x' − h7·y·x'
        //   y' = h3·x + h4·y + h5 − h6·x·y' − h7·y·y'
        var A = new double[8, 8];
        var b = new double[8];
        for (int i = 0; i < 4; i++)
        {
            A[2 * i, 0] = dx[i]; A[2 * i, 1] = dy[i]; A[2 * i, 2] = 1;
            A[2 * i, 6] = -dx[i] * sx[i]; A[2 * i, 7] = -dy[i] * sx[i];
            b[2 * i] = sx[i];

            A[2 * i + 1, 3] = dx[i]; A[2 * i + 1, 4] = dy[i]; A[2 * i + 1, 5] = 1;
            A[2 * i + 1, 6] = -dx[i] * sy[i]; A[2 * i + 1, 7] = -dy[i] * sy[i];
            b[2 * i + 1] = sy[i];
        }
        h = new double[8];
        return GaussianEliminate(A, b, h);
    }

    /// <summary>
    /// Gaussian elimination with partial pivoting on an 8×8 system. Returns
    /// false if the matrix is singular (a pivot ≈ 0).
    /// </summary>
    private static bool GaussianEliminate(double[,] A, double[] b, double[] x)
    {
        int n = b.Length;
        for (int i = 0; i < n; i++)
        {
            int pivot = i;
            double pivAbs = Math.Abs(A[i, i]);
            for (int r = i + 1; r < n; r++)
            {
                double a = Math.Abs(A[r, i]);
                if (a > pivAbs) { pivAbs = a; pivot = r; }
            }
            if (pivAbs < 1e-12) return false;
            if (pivot != i)
            {
                for (int c = 0; c < n; c++) (A[i, c], A[pivot, c]) = (A[pivot, c], A[i, c]);
                (b[i], b[pivot]) = (b[pivot], b[i]);
            }
            for (int r = i + 1; r < n; r++)
            {
                double m = A[r, i] / A[i, i];
                for (int c = i; c < n; c++) A[r, c] -= m * A[i, c];
                b[r] -= m * b[i];
            }
        }
        for (int i = n - 1; i >= 0; i--)
        {
            double s = b[i];
            for (int c = i + 1; c < n; c++) s -= A[i, c] * x[c];
            x[i] = s / A[i, i];
        }
        return true;
    }

    /// <summary>
    /// Bilinear-interpolated sample from a [batch, channel, H, W] tensor at
    /// a fractional position. Returns 0 for out-of-bounds reads.
    /// </summary>
    private T BilinearSample(Tensor<T> src, int b, int c, double xs, double ys, int srcW, int srcH)
    {
        if (xs < 0 || xs > srcW - 1 || ys < 0 || ys > srcH - 1) return NumOps.Zero;
        int x0 = (int)Math.Floor(xs);
        int y0 = (int)Math.Floor(ys);
        int x1 = Math.Min(x0 + 1, srcW - 1);
        int y1 = Math.Min(y0 + 1, srcH - 1);
        double dxw = xs - x0, dyw = ys - y0;
        // Route every T → double via the library's NumericOperations rather
        // than Convert.ToDouble, which throws on non-IConvertible T types
        // (e.g. when a custom NumericOperations<T> is installed).
        double v00 = NumOps.ToDouble(src[b, c, y0, x0]);
        double v01 = NumOps.ToDouble(src[b, c, y0, x1]);
        double v10 = NumOps.ToDouble(src[b, c, y1, x0]);
        double v11 = NumOps.ToDouble(src[b, c, y1, x1]);
        double v = v00 * (1 - dxw) * (1 - dyw)
                 + v01 * dxw * (1 - dyw)
                 + v10 * (1 - dxw) * dyw
                 + v11 * dxw * dyw;
        return NumOps.FromDouble(v);
    }

    private Tensor<T> Rotate90(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>(new[] { batch, channels, width, height });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        output[b, c, w, height - 1 - h] = input[b, c, h, w];
                    }
                }
            }
        }

        return output;
    }

    private List<RecognizedText<T>> SortByReadingOrder(List<RecognizedText<T>> texts)
    {
        if (texts.Count <= 1)
            return texts;

        // Group into lines based on Y-coordinate overlap
        var lines = new List<List<RecognizedText<T>>>();

        foreach (var text in texts.OrderBy(t => t.Box != null ? NumOps.ToDouble(t.Box.Y1) : 0))
        {
            bool addedToLine = false;

            foreach (var line in lines)
            {
                // Check if this text overlaps vertically with the line
                var lineTop = line.Min(t => t.Box != null ? NumOps.ToDouble(t.Box.Y1) : 0);
                var lineBottom = line.Max(t => t.Box != null ? NumOps.ToDouble(t.Box.Y2) : 0);
                var textTop = text.Box != null ? NumOps.ToDouble(text.Box.Y1) : 0;
                var textBottom = text.Box != null ? NumOps.ToDouble(text.Box.Y2) : 0;

                double overlapThreshold = (lineBottom - lineTop) * 0.5;

                if (textTop < lineBottom && textBottom > lineTop &&
                    Math.Min(textBottom, lineBottom) - Math.Max(textTop, lineTop) > overlapThreshold)
                {
                    line.Add(text);
                    addedToLine = true;
                    break;
                }
            }

            if (!addedToLine)
            {
                lines.Add(new List<RecognizedText<T>> { text });
            }
        }

        // Sort each line left-to-right, then concatenate lines
        var result = new List<RecognizedText<T>>();

        foreach (var line in lines.OrderBy(l => l.Min(t => t.Box != null ? NumOps.ToDouble(t.Box.Y1) : 0)))
        {
            var sortedLine = line.OrderBy(t => t.Box != null ? NumOps.ToDouble(t.Box.X1) : 0);
            result.AddRange(sortedLine);
        }

        return result;
    }

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long GetParameterCount()
    {
        return _detector.GetParameterCount() + _recognizer.GetParameterCount();
    }

    /// <summary>
    /// Loads pretrained weights.
    /// </summary>
    public async Task LoadWeightsAsync(string detectorPath, string recognizerPath, CancellationToken cancellationToken = default)
    {
        await _detector.LoadWeightsAsync(detectorPath, cancellationToken);
        await _recognizer.LoadWeightsAsync(recognizerPath, cancellationToken);
    }

    #region ModelBase Overrides

    /// <summary>
    /// Runs end-to-end OCR and returns region info as a tensor [numRegions, 6].
    /// Columns: confidence, textLength, x1, y1, x2, y2.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var result = ReadText(input);
        int regions = result.TextRegions.Count;
        if (regions == 0)
            return new Tensor<T>([0, 6]);

        var output = new Tensor<T>([regions, 6]);
        for (int i = 0; i < regions; i++)
        {
            var region = result.TextRegions[i];
            output[i, 0] = region.Confidence;
            output[i, 1] = NumOps.FromDouble(region.Text.Length);
            if (region.Box is not null)
            {
                output[i, 2] = region.Box.X1;
                output[i, 3] = region.Box.Y1;
                output[i, 4] = region.Box.X2;
                output[i, 5] = region.Box.Y2;
            }
        }
        return output;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) { }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters) { }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        InterfaceGuard.Parameterizable(copy).SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        => (SceneTextReader<T>)MemberwiseClone();

    #endregion
}
