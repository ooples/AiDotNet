namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Draws segmentation masks over an image, producing an overlay image tensor.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A segmentation model tells you which pixels belong to which object, but
/// that answer is a grid of numbers. This turns it into something you can actually look at: the
/// original image with each detected region tinted a distinct colour, optionally outlined.
/// </para>
/// <para>
/// <b>Shape of the API.</b> The primary entry point mirrors
/// <c>torchvision.utils.draw_segmentation_masks(image, masks, alpha, colors)</c> — a free function
/// taking the image, the masks and the render settings, returning a new image tensor. Visualization
/// is an evaluation-time concern, so nothing here holds state or needs to be configured in advance.
/// The <see cref="SegmentationVisualizationConfig"/> overload exists because this library's config
/// carries knobs torchvision has no equivalent for (contours, palettes, score thresholds).
/// </para>
/// <para>
/// <b>Value range is preserved.</b> The image may be in [0, 1] or [0, 255]; the range is detected and
/// the output is returned in the SAME range as the input. Silently changing an image's scale is the
/// kind of defect that only surfaces as a washed-out or saturated picture much later.
/// </para>
/// <para>
/// This is post-processing over a produced result, not a training path, so it indexes tensors
/// directly rather than going through Engine ops — consistent with <see cref="SegmentationTensorOps"/>.
/// No gradient ever flows through a visualization.
/// </para>
/// </remarks>
public static class SegmentationRenderer
{
    /// <summary>Default alpha, matching torchvision's <c>draw_segmentation_masks</c>.</summary>
    public const double DefaultAlpha = 0.8;

    /// <summary>
    /// Draws boolean masks over an image. Direct analogue of
    /// <c>torchvision.utils.draw_segmentation_masks</c>.
    /// </summary>
    /// <param name="image">RGB image [3, H, W], or a single-channel [H, W] / [1, H, W] image which is
    /// replicated to three channels. Values in [0, 1] or [0, 255].</param>
    /// <param name="masks">Masks [N, H, W] (or a single [H, W] mask). Any value &gt; 0.5 is inside.</param>
    /// <param name="alpha">Blend factor. 0 leaves the image untouched, 1 replaces it with solid colour.</param>
    /// <param name="colors">Optional palette [numColors, 3] with components in [0, 255]. When null a
    /// deterministic palette is generated, so the same masks always render the same colours.</param>
    /// <returns>A new image tensor [3, H, W] in the same value range as <paramref name="image"/>.</returns>
    public static Tensor<T> DrawSegmentationMasks<T>(
        Tensor<T> image,
        Tensor<T> masks,
        double alpha = DefaultAlpha,
        byte[,]? colors = null)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (masks is null) throw new ArgumentNullException(nameof(masks));
        if (alpha < 0.0 || alpha > 1.0)
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "alpha must be in [0, 1].");

        return Draw(image, masks, alpha, colors, drawContours: false, contourThickness: 0).Image;
    }

    /// <summary>
    /// Draws masks honouring a full <see cref="SegmentationVisualizationConfig"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// Thrown when <see cref="SegmentationVisualizationConfig.ShowLabels"/> or
    /// <see cref="SegmentationVisualizationConfig.ShowScores"/> is set. A bare mask stack has no
    /// per-instance classes or scores to render; use <see cref="Render{T}"/> when labels are needed.
    /// Throwing is the point: a renderer that quietly ignored those flags would be the same
    /// silently-dropped-configuration defect AIDN096 exists to prevent.
    /// </exception>
    public static Tensor<T> DrawSegmentationMasks<T>(
        Tensor<T> image,
        Tensor<T> masks,
        SegmentationVisualizationConfig config)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        RejectLabelsWithoutInstanceContext(config);

        return Draw(
            image ?? throw new ArgumentNullException(nameof(image)),
            masks ?? throw new ArgumentNullException(nameof(masks)),
            config.Alpha,
            ResolvePalette(config),
            config.DrawContours,
            config.ContourThickness).Image;
    }

    /// <summary>
    /// Renders a whole <see cref="SegmentationOutput{T}"/> over an image, choosing the right source
    /// automatically: instance masks when present, otherwise the per-pixel class map.
    /// </summary>
    /// <remarks>
    /// Instances below <see cref="SegmentationVisualizationConfig.MinDisplayConfidence"/> are skipped
    /// when scores are available, and boxes are outlined when
    /// <see cref="SegmentationVisualizationConfig.ShowBoundingBoxes"/> is set.
    /// </remarks>
    public static Tensor<T> Render<T>(
        Tensor<T> image,
        SegmentationOutput<T> output,
        SegmentationVisualizationConfig? config = null)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (output is null) throw new ArgumentNullException(nameof(output));

        config ??= new SegmentationVisualizationConfig();

        var numOps = MathHelper.GetNumericOperations<T>();
        byte[,] palette = ResolvePalette(config) ?? BuildPalette(Math.Max(output.NumClasses, 1));

        Tensor<T> masks;
        // Kept instance indices, so labels can be looked up against the ORIGINAL arrays after
        // low-scoring instances are filtered out. Null when drawing a class map instead.
        List<int>? keptInstances = null;

        if (output.InstanceMasks is not null && output.NumInstances > 0)
        {
            keptInstances = SelectInstancesToDraw(output, config, numOps);
            masks = SelectMasks(output.InstanceMasks, keptInstances);
        }
        else if (output.ClassMap is not null)
        {
            if (config.ShowScores)
            {
                throw new NotSupportedException(
                    "ShowScores cannot be derived from a class map alone because it has no " +
                    "per-instance confidence value. Supply InstanceMasks with InstanceScores, " +
                    "or set ShowScores = false. The setting is rejected rather than ignored.");
            }
            masks = ClassMapToMasks(output.ClassMap, Math.Max(output.NumClasses, 1), numOps);
        }
        else
        {
            throw new ArgumentException(
                "SegmentationOutput carries neither InstanceMasks nor ClassMap, so there is nothing to draw.",
                nameof(output));
        }

        var drawResult = Draw(
            image,
            masks,
            config.Alpha,
            palette,
            config.DrawContours,
            config.ContourThickness);
        var rendered = drawResult.Image;

        if (config.ShowBoundingBoxes && output.InstanceBoxes is not null && keptInstances is not null)
            DrawBoxes(rendered, output.InstanceBoxes, keptInstances, palette, numOps, drawResult.Scale);

        if ((config.ShowLabels || config.ShowScores) && keptInstances is not null)
            DrawInstanceLabels(
                rendered,
                drawResult.MaskAnchors,
                output,
                keptInstances,
                config,
                palette,
                numOps,
                drawResult.Scale);
        else if (config.ShowLabels && output.ClassMap is not null)
            DrawClassLabels(
                rendered,
                drawResult.MaskAnchors,
                output,
                palette,
                numOps,
                drawResult.Scale);

        return rendered;
    }

    /// <summary>
    /// Labels need a per-instance class and position, which only <see cref="Render{T}"/> has. The
    /// mask-only overloads therefore reject the flags rather than ignoring them: a silently dropped
    /// setting is the defect AIDN096 exists to prevent, and here the caller simply needs the overload
    /// that carries the information labels require.
    /// </summary>
    private static void RejectLabelsWithoutInstanceContext(SegmentationVisualizationConfig config)
    {
        if (config.ShowLabels || config.ShowScores)
        {
            throw new NotSupportedException(
                "ShowLabels/ShowScores need per-instance classes and scores, which a bare mask stack " +
                "does not carry. Call Render(image, SegmentationOutput, config) instead — it has the " +
                "class ids, names and scores to label. The flags are rejected rather than ignored so " +
                "the setting cannot be silently dropped.");
        }
    }

    /// <summary>
    /// Resolves which palette to use. An explicit <see cref="SegmentationVisualizationConfig.ColorPalette"/>
    /// always wins; otherwise <see cref="SegmentationVisualizationConfig.UseFixedPalette"/> selects
    /// between the standard ADE20K palette and a palette generated from the mask count.
    /// </summary>
    /// <remarks>
    /// Returning null for the generated case defers to <see cref="BuildPalette"/>, which needs the mask
    /// count that is only known further in. Both branches are genuinely distinct — quietly collapsing
    /// them would ignore UseFixedPalette, which is the same silently-dropped-configuration defect
    /// AIDN096 exists to catch.
    /// </remarks>
    private static byte[,]? ResolvePalette(SegmentationVisualizationConfig config)
    {
        if (config.ColorPalette is not null) return config.ColorPalette;
        return config.UseFixedPalette ? SegmentationVisualizationConfig.GetADE20KPalette() : null;
    }

    /// <summary>
    /// Builds a deterministic, visually distinct palette by walking hue with the golden-ratio
    /// conjugate — successive entries land far apart on the colour wheel, and the sequence is fixed,
    /// so a given mask index always renders the same colour across runs and machines.
    /// </summary>
    internal static byte[,] BuildPalette(int count)
    {
        if (count <= 0) count = 1;
        var palette = new byte[count, 3];
        const double goldenRatioConjugate = 0.618033988749895;
        double hue = 0.0;
        for (int i = 0; i < count; i++)
        {
            hue = (hue + goldenRatioConjugate) % 1.0;
            var (r, g, b) = HsvToRgb(hue, 0.75, 0.95);
            palette[i, 0] = r;
            palette[i, 1] = g;
            palette[i, 2] = b;
        }
        return palette;
    }

    /// <summary>
    /// Detects whether an image tensor uses normalized [0,1] values or 8-bit-style [0,255] values.
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="image">An image tensor in any supported image shape.</param>
    /// <returns><c>1.0</c> for normalized values or <c>255.0</c> when any value exceeds 1.</returns>
    /// <remarks>
    /// This is the single value-range rule used by both the core renderer and dashboard encoders.
    /// Keeping the decision public prevents presentation layers from silently drifting to a
    /// different scale heuristic.
    /// </remarks>
    public static double DetectImageScale<T>(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        return DetectImageScale(image, MathHelper.GetNumericOperations<T>());
    }

    private static (byte R, byte G, byte B) HsvToRgb(double h, double s, double v)
    {
        int sector = (int)(h * 6.0) % 6;
        double f = h * 6.0 - Math.Floor(h * 6.0);
        double p = v * (1.0 - s);
        double q = v * (1.0 - f * s);
        double t = v * (1.0 - (1.0 - f) * s);
        (double r, double g, double b) = sector switch
        {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };
        return ((byte)Math.Round(r * 255), (byte)Math.Round(g * 255), (byte)Math.Round(b * 255));
    }

    private static DrawResult<T> Draw<T>(
        Tensor<T> image,
        Tensor<T> masks,
        double alpha,
        byte[,]? colors,
        bool drawContours,
        int contourThickness)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var rgb = ToRgb(image, numOps);
        double scale = DetectImageScale(rgb, numOps);
        int height = rgb.Shape[1];
        int width = rgb.Shape[2];

        var maskStack = ToMaskStack(masks, height, width);
        int numMasks = maskStack.Shape[0];
        byte[,] palette = colors ?? BuildPalette(numMasks);
        int paletteSize = palette.GetLength(0);
        if (paletteSize == 0)
            throw new ArgumentException("Colour palette is empty.", nameof(colors));

        var half = numOps.FromDouble(0.5);
        var anchors = new MaskAnchor[numMasks];

        for (int m = 0; m < numMasks; m++)
        {
            double cr = palette[m % paletteSize, 0] * scale / 255.0;
            double cg = palette[m % paletteSize, 1] * scale / 255.0;
            double cb = palette[m % paletteSize, 2] * scale / 255.0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (numOps.LessThanOrEquals(maskStack[m, y, x], half)) continue;
                    if (!anchors[m].HasValue) anchors[m] = new MaskAnchor(y, x);
                    Blend(rgb, numOps, y, x, cr, cg, cb, alpha);
                }
            }

            if (drawContours && contourThickness > 0)
                DrawContour(rgb, maskStack, numOps, m, height, width, contourThickness, cr, cg, cb);
        }

        return new DrawResult<T>(rgb, scale, anchors);
    }

    private static void Blend<T>(
        Tensor<T> rgb, INumericOperations<T> numOps, int y, int x,
        double cr, double cg, double cb, double alpha)
    {
        double keep = 1.0 - alpha;
        rgb[0, y, x] = numOps.FromDouble(numOps.ToDouble(rgb[0, y, x]) * keep + cr * alpha);
        rgb[1, y, x] = numOps.FromDouble(numOps.ToDouble(rgb[1, y, x]) * keep + cg * alpha);
        rgb[2, y, x] = numOps.FromDouble(numOps.ToDouble(rgb[2, y, x]) * keep + cb * alpha);
    }

    /// <summary>
    /// Paints the boundary of a mask fully opaque. A pixel is on the boundary when it is inside the
    /// mask and at least one 4-neighbour is outside it (or off the edge of the image), then dilated
    /// to the requested thickness.
    /// </summary>
    private static void DrawContour<T>(
        Tensor<T> rgb, Tensor<T> maskStack, INumericOperations<T> numOps,
        int maskIndex, int height, int width, int thickness,
        double cr, double cg, double cb)
    {
        var half = numOps.FromDouble(0.5);
        bool Inside(int y, int x) =>
            y >= 0 && y < height && x >= 0 && x < width
            && numOps.GreaterThan(maskStack[maskIndex, y, x], half);

        var boundary = new List<(int Y, int X)>();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (!Inside(y, x)) continue;
                if (!Inside(y - 1, x) || !Inside(y + 1, x) || !Inside(y, x - 1) || !Inside(y, x + 1))
                    boundary.Add((y, x));
            }
        }

        int radius = Math.Max(0, thickness - 1);
        foreach (var (by, bx) in boundary)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int y = by + dy, x = bx + dx;
                    if (y < 0 || y >= height || x < 0 || x >= width) continue;
                    rgb[0, y, x] = numOps.FromDouble(cr);
                    rgb[1, y, x] = numOps.FromDouble(cg);
                    rgb[2, y, x] = numOps.FromDouble(cb);
                }
            }
        }
    }

    private static void DrawBoxes<T>(
        Tensor<T> rgb,
        Tensor<T> boxes,
        List<int> keptInstances,
        byte[,] palette,
        INumericOperations<T> numOps,
        double scale)
    {
        if (boxes.Rank != 2 || boxes.Shape[1] < 4)
        {
            throw new ArgumentException(
                $"InstanceBoxes must be rank 2 [N,4] with at least four coordinate columns; got rank {boxes.Rank}" +
                (boxes.Rank == 2 ? $" and shape [{boxes.Shape[0]},{boxes.Shape[1]}]." : "."),
                nameof(boxes));
        }

        int height = rgb.Shape[1];
        int width = rgb.Shape[2];
        int count = boxes.Shape[0];
        int paletteSize = palette.GetLength(0);

        for (int drawn = 0; drawn < keptInstances.Count; drawn++)
        {
            int original = keptInstances[drawn];
            if (original < 0 || original >= count)
            {
                throw new ArgumentException(
                    $"InstanceBoxes has {count} row(s), but kept instance index {original} requires a matching row.",
                    nameof(boxes));
            }

            int x1 = (int)Math.Round(numOps.ToDouble(boxes[original, 0]));
            int y1 = (int)Math.Round(numOps.ToDouble(boxes[original, 1]));
            int x2 = (int)Math.Round(numOps.ToDouble(boxes[original, 2]));
            int y2 = (int)Math.Round(numOps.ToDouble(boxes[original, 3]));

            double cr = palette[drawn % paletteSize, 0] * scale / 255.0;
            double cg = palette[drawn % paletteSize, 1] * scale / 255.0;
            double cb = palette[drawn % paletteSize, 2] * scale / 255.0;

            void Plot(int y, int x)
            {
                if (y < 0 || y >= height || x < 0 || x >= width) return;
                rgb[0, y, x] = numOps.FromDouble(cr);
                rgb[1, y, x] = numOps.FromDouble(cg);
                rgb[2, y, x] = numOps.FromDouble(cb);
            }

            for (int x = Math.Min(x1, x2); x <= Math.Max(x1, x2); x++) { Plot(y1, x); Plot(y2, x); }
            for (int y = Math.Min(y1, y2); y <= Math.Max(y1, y2); y++) { Plot(y, x1); Plot(y, x2); }
        }
    }

    /// <summary>
    /// Normalizes the input image to a fresh [3, H, W] tensor and reports the value scale (1.0 for
    /// [0,1] images, 255.0 for [0,255]) so colours are mixed in the caller's own range.
    /// </summary>
    private static Tensor<T> ToRgb<T>(Tensor<T> image, INumericOperations<T> numOps)
    {
        var unbatched = SegmentationTensorOps.EnsureUnbatched(image);
        int height, width, channels;

        if (unbatched.Rank == 2) { channels = 1; height = unbatched.Shape[0]; width = unbatched.Shape[1]; }
        else if (unbatched.Rank == 3) { channels = unbatched.Shape[0]; height = unbatched.Shape[1]; width = unbatched.Shape[2]; }
        else throw new ArgumentException($"Image must be [H,W], [C,H,W] or [B,C,H,W]; got rank {unbatched.Rank}.", nameof(image));

        if (channels != 1 && channels != 3)
            throw new ArgumentException($"Image must have 1 or 3 channels; got {channels}.", nameof(image));

        var rgb = new Tensor<T>([3, height, width]);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    rgb[c, y, x] = unbatched.Rank == 2
                        ? unbatched[y, x]
                        : unbatched[channels == 1 ? 0 : c, y, x];
                }
            }
        }

        return rgb;
    }

    /// <summary>
    /// Returns 255.0 when the image looks like 8-bit pixel values and 1.0 when it looks normalized.
    /// A value above 1 can only occur in the [0,255] convention.
    /// </summary>
    private static double DetectImageScale<T>(Tensor<T> image, INumericOperations<T> numOps)
    {
        var one = numOps.One;
        for (int index = 0; index < image.Length; index++)
            if (numOps.GreaterThan(image[index], one)) return 255.0;
        return 1.0;
    }

    private static Tensor<T> ToMaskStack<T>(Tensor<T> masks, int height, int width)
    {
        var unbatched = SegmentationTensorOps.EnsureUnbatched(masks);
        if (unbatched.Rank == 2)
        {
            if (unbatched.Shape[0] != height || unbatched.Shape[1] != width)
                throw new ArgumentException(
                    $"Mask [{unbatched.Shape[0]},{unbatched.Shape[1]}] does not match image [{height},{width}].",
                    nameof(masks));

            var single = new Tensor<T>([1, height, width]);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    single[0, y, x] = unbatched[y, x];
            return single;
        }

        if (unbatched.Rank != 3)
            throw new ArgumentException($"Masks must be [H,W] or [N,H,W]; got rank {unbatched.Rank}.", nameof(masks));

        if (unbatched.Shape[1] != height || unbatched.Shape[2] != width)
            throw new ArgumentException(
                $"Masks [{unbatched.Shape[1]},{unbatched.Shape[2]}] do not match image [{height},{width}].",
                nameof(masks));

        return unbatched;
    }

    /// <summary>
    /// Chooses which instances to draw, dropping any scoring below
    /// <see cref="SegmentationVisualizationConfig.MinDisplayConfidence"/>. Instances with no score are
    /// kept, since an absent score is not evidence of a weak detection.
    /// </summary>
    private static List<int> SelectInstancesToDraw<T>(
        SegmentationOutput<T> output, SegmentationVisualizationConfig config, INumericOperations<T> numOps)
    {
        var scores = output.InstanceScores;
        int count = output.InstanceMasks!.Shape[0];
        var keep = new List<int>(count);
        for (int i = 0; i < count; i++)
        {
            if (scores is null || i >= scores.Length
                || numOps.ToDouble(scores[i]) >= config.MinDisplayConfidence)
            {
                keep.Add(i);
            }
        }
        return keep;
    }

    private static Tensor<T> SelectMasks<T>(Tensor<T> masks, List<int> keep)
    {
        int count = masks.Shape[0];
        if (keep.Count == count) return masks;

        int height = masks.Shape[1];
        int width = masks.Shape[2];
        var filtered = new Tensor<T>([keep.Count, height, width]);
        for (int k = 0; k < keep.Count; k++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    filtered[k, y, x] = masks[keep[k], y, x];
        return filtered;
    }

    /// <summary>
    /// Draws a text label for each drawn instance, anchored just above its mask's top-left extent so
    /// the label sits outside the region it names wherever there is room.
    /// </summary>
    private static void DrawInstanceLabels<T>(
        Tensor<T> rgb,
        MaskAnchor[] anchors,
        SegmentationOutput<T> output,
        List<int> keptInstances,
        SegmentationVisualizationConfig config,
        byte[,] palette,
        INumericOperations<T> numOps,
        double scaleRange)
    {
        int paletteSize = palette.GetLength(0);

        for (int drawn = 0; drawn < keptInstances.Count && drawn < anchors.Length; drawn++)
        {
            int original = keptInstances[drawn];
            MaskAnchor anchor = anchors[drawn];
            if (!anchor.HasValue) continue;

            string text = BuildLabelText(output, original, config);
            if (text.Length == 0) continue;

            // Prefer just above the mask; fall back to inside it when there is no room.
            int textHeight = BitmapFont5x7.MeasureHeight(1);
            int originY = anchor.Y - textHeight - 1;
            if (originY < 0) originY = anchor.Y + 1;

            double r = palette[drawn % paletteSize, 0] * scaleRange / 255.0;
            double g = palette[drawn % paletteSize, 1] * scaleRange / 255.0;
            double b = palette[drawn % paletteSize, 2] * scaleRange / 255.0;

            BitmapFont5x7.DrawText(rgb, numOps, text, anchor.X, originY, r, g, b);
        }
    }

    private static string BuildLabelText<T>(
        SegmentationOutput<T> output, int instanceIndex, SegmentationVisualizationConfig config)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        string label = string.Empty;

        if (config.ShowLabels)
        {
            string? name = output.Segments is not null && instanceIndex < output.Segments.Count
                ? output.Segments[instanceIndex].ClassName
                : null;

            if (!string.IsNullOrEmpty(name))
            {
                label = name!;
            }
            else if (output.InstanceClasses is not null && instanceIndex < output.InstanceClasses.Length)
            {
                label = $"class {output.InstanceClasses[instanceIndex]}";
            }
            else
            {
                label = $"#{instanceIndex}";
            }
        }

        if (config.ShowScores && output.InstanceScores is not null
            && instanceIndex < output.InstanceScores.Length)
        {
            double score = numOps.ToDouble(output.InstanceScores[instanceIndex]);
            string scoreText = score.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);
            label = label.Length == 0 ? scoreText : $"{label} {scoreText}";
        }

        return label;
    }

    /// <summary>
    /// Labels semantic regions by class ID. Class-map mask channels intentionally share their index
    /// with the source class ID, so anchors and palette entries remain aligned without a lookup table.
    /// </summary>
    private static void DrawClassLabels<T>(
        Tensor<T> rgb,
        MaskAnchor[] anchors,
        SegmentationOutput<T> output,
        byte[,] palette,
        INumericOperations<T> numOps,
        double scaleRange)
    {
        int paletteSize = palette.GetLength(0);
        for (int classId = 1; classId < anchors.Length; classId++)
        {
            MaskAnchor anchor = anchors[classId];
            if (!anchor.HasValue) continue;

            string text = output.ClassNames is not null
                && classId < output.ClassNames.Length
                && !string.IsNullOrWhiteSpace(output.ClassNames[classId])
                    ? output.ClassNames[classId]
                    : $"class {classId}";

            int originY = anchor.Y - BitmapFont5x7.MeasureHeight(1) - 1;
            if (originY < 0) originY = anchor.Y + 1;

            double r = palette[classId % paletteSize, 0] * scaleRange / 255.0;
            double g = palette[classId % paletteSize, 1] * scaleRange / 255.0;
            double b = palette[classId % paletteSize, 2] * scaleRange / 255.0;
            BitmapFont5x7.DrawText(rgb, numOps, text, anchor.X, originY, r, g, b);
        }
    }

    /// <summary>
    /// Expands a per-pixel class map [H, W] into one binary mask per class, skipping class 0, which
    /// is background by the usual convention and is left showing the original image.
    /// </summary>
    private static Tensor<T> ClassMapToMasks<T>(Tensor<T> classMap, int numClasses, INumericOperations<T> numOps)
    {
        var unbatched = SegmentationTensorOps.EnsureUnbatchedClassMap(classMap);

        int height = unbatched.Shape[0];
        int width = unbatched.Shape[1];
        // Channel indices deliberately equal class IDs. Channel 0 stays empty as background, so
        // class N is rendered with palette entry N rather than being shifted to N-1.
        int channels = Math.Max(1, numClasses);
        var masks = new Tensor<T>([channels, height, width]);
        var one = numOps.One;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int cls = (int)Math.Round(numOps.ToDouble(unbatched[y, x]));
                if (cls <= 0 || cls >= numClasses) continue;
                masks[cls, y, x] = one;
            }
        }
        return masks;
    }

    private readonly struct DrawResult<T>
    {
        internal DrawResult(Tensor<T> image, double scale, MaskAnchor[] maskAnchors)
        {
            Image = image;
            Scale = scale;
            MaskAnchors = maskAnchors;
        }

        internal Tensor<T> Image { get; }
        internal double Scale { get; }
        internal MaskAnchor[] MaskAnchors { get; }
    }

    private readonly struct MaskAnchor
    {
        internal MaskAnchor(int y, int x)
        {
            Y = y;
            X = x;
            HasValue = true;
        }

        internal bool HasValue { get; }
        internal int Y { get; }
        internal int X { get; }
    }
}
