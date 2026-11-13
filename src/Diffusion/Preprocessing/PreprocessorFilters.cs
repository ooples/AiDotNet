namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Whole-tensor image filters shared by the ControlNet preprocessors. Images are [batch, channels, height, width].
/// </summary>
internal static class PreprocessorFilters
{
    /// <summary>3x3 Sobel x then y kernels, as [2, 1, 3, 3].</summary>
    internal static readonly int[] Sobel =
    {
        -1, 0, 1, -2, 0, 2, -1, 0, 1,
        -1, -2, -1, 0, 0, 0, 1, 2, 1,
    };

    /// <summary>4-neighbour Laplacian 4 c - (n + s + w + e), as [1, 1, 3, 3].</summary>
    internal static readonly int[] Laplacian = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };

    /// <summary>Central differences I(h, w + 1) - I(h, w - 1) then I(h + 1, w) - I(h - 1, w), as [2, 1, 3, 3].</summary>
    internal static readonly int[] CentralDifferences =
    {
        0, 0, 0, -1, 0, 1, 0, 0, 0,
        0, -1, 0, 0, 0, 0, 0, 1, 0,
    };

    /// <summary>Channel <paramref name="channel"/> of the image as [batch, 1, height, width].</summary>
    internal static Tensor<T> Channel<T>(Tensor<T> image, int channel)
        => AiDotNetEngine.Current.TensorNarrow(image, 1, channel, 1);

    /// <summary>
    /// The image as three channels, output channel c reading input channel min(c, channels - 1).
    /// </summary>
    internal static Tensor<T> ToRgb<T>(Tensor<T> image)
    {
        int channels = Math.Min(image._shape[1], 3);
        if (channels == 3 && image._shape[1] == 3)
            return image;
        return AiDotNetEngine.Current.TensorConcatenate(new[]
        {
            Channel(image, 0),
            Channel(image, Math.Min(1, channels - 1)),
            Channel(image, Math.Min(2, channels - 1)),
        }, 1);
    }

    /// <summary>
    /// ITU-R BT.601 luma 0.299 R + 0.587 G + 0.114 B as [batch, 1, height, width]; a missing green or blue channel
    /// reuses red.
    /// </summary>
    internal static Tensor<T> Grayscale<T>(Tensor<T> image)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int channels = image._shape[1];
        var r = Channel(image, 0);
        var g = channels > 1 ? Channel(image, 1) : r;
        var b = channels > 2 ? Channel(image, 2) : r;
        return engine.TensorAdd(
            engine.TensorAdd(engine.TensorMultiplyScalar(r, ops.FromDouble(0.299)), engine.TensorMultiplyScalar(g, ops.FromDouble(0.587))),
            engine.TensorMultiplyScalar(b, ops.FromDouble(0.114)));
    }

    /// <summary>True when the image has interior pixels (a 3x3 neighbourhood fits).</summary>
    internal static bool HasInterior(int height, int width) => height > 2 && width > 2;

    /// <summary>
    /// Correlates a [batch, 1, height, width] map with the given 3x3 integer kernels (9 coefficients each) over the
    /// interior, returning [batch, kernels, height - 2, width - 2].
    /// </summary>
    internal static Tensor<T> Interior3x3<T>(Tensor<T> map, int[] kernels)
    {
        var engine = AiDotNetEngine.Current;
        var typedKernels = new Tensor<int>(kernels, new[] { kernels.Length }).Cast<T>().ToArray();
        int count = kernels.Length / 9;
        int height = map._shape[2] - 2, width = map._shape[3] - 2;
        var responses = new Tensor<T>[count];
        for (int filter = 0; filter < count; filter++)
        {
            var response = new Tensor<T>(new[] { map._shape[0], 1, height, width });
            for (int position = 0; position < 9; position++)
            {
                int coefficient = kernels[filter * 9 + position];
                // A zero coefficient does not read that pixel in the original stencil.
                // Dense convolution would let 0 * NaN/Infinity poison unrelated derivatives.
                if (coefficient == 0) continue;
                var pixels = engine.TensorNarrow(engine.TensorNarrow(map, 2, position / 3, height),
                    3, position % 3, width);
                response = engine.TensorAdd(response, engine.TensorMultiplyScalar(pixels, typedKernels[filter * 9 + position]));
            }
            responses[filter] = response;
        }
        return engine.TensorConcatenate(responses, 1);
    }

    /// <summary>Zero-pads an interior [batch, c, height - 2, width - 2] result back to [batch, c, height, width].</summary>
    internal static Tensor<T> PadBorder<T>(Tensor<T> interior)
        => AiDotNetEngine.Current.Pad(interior, 1, 1, 1, 1, MathHelper.GetNumericOperations<T>().Zero);

    /// <summary>sqrt(gx^2 + gy^2) of a [batch, 2, h, w] gradient pair, as [batch, 1, h, w].</summary>
    internal static Tensor<T> Magnitude<T>(Tensor<T> gradients)
    {
        var engine = AiDotNetEngine.Current;
        var squared = engine.TensorSquare(gradients);
        return engine.TensorSqrt(engine.TensorAdd(Channel(squared, 0), Channel(squared, 1)));
    }

    /// <summary>
    /// Stacks <paramref name="map"/> [batch, 1, h, w] scaled by the red, green and blue factors along the channel
    /// axis (a colour-coded map), as [batch, 3, h, w].
    /// </summary>
    internal static Tensor<T> ColorChannels<T>(Tensor<T> map, double red, double green, double blue)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorConcatenate(new[]
        {
            red == 1.0 ? map : engine.TensorMultiplyScalar(map, ops.FromDouble(red)),
            green == 1.0 ? map : engine.TensorMultiplyScalar(map, ops.FromDouble(green)),
            blue == 1.0 ? map : engine.TensorMultiplyScalar(map, ops.FromDouble(blue)),
        }, 1);
    }

    /// <summary>
    /// Central-difference gradient magnitude sqrt(dx^2 + dy^2) of each of the first <paramref name="channels"/>
    /// channels over the interior, as [batch, channels, height - 2, width - 2].
    /// </summary>
    internal static Tensor<T> ChannelGradientMagnitudes<T>(Tensor<T> image, int channels)
    {
        var engine = AiDotNetEngine.Current;
        int[] shape = image._shape;
        int batch = shape[0], height = shape[2], width = shape[3];
        var planes = engine.Reshape(engine.TensorNarrow(image, 1, 0, channels), new[] { batch * channels, 1, height, width });
        var magnitude = Magnitude(Interior3x3(planes, CentralDifferences));
        return engine.Reshape(magnitude, new[] { batch, channels, height - 2, width - 2 });
    }

    /// <summary>
    /// Mean of each pixel's (2 radius + 1)^2 neighbourhood clipped to the image (pixels outside the image are not
    /// counted), per channel of a [batch, channels, height, width] image.
    /// </summary>
    internal static Tensor<T> ClippedBoxMean<T>(Tensor<T> image, int radius)
    {
        // Half convolution accumulates in Half, which stops counting (and summing) past 2048, so a window of more
        // than 2048 pixels (radius >= 23) gives the wrong mean. Reduce in double like the loops this replaced.
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
            return ClippedBoxMean(image.Cast<double>(), radius).Cast<T>();
#endif
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int[] shape = image._shape;
        int planes = shape[0] * shape[1];
        int size = 2 * radius + 1;
        var box = Tensor<T>.CreateDefault(new[] { 1, 1, size, size }, ops.One);
        var planeShape = new[] { planes, 1, shape[2], shape[3] };
        var sum = engine.Conv2D(engine.Reshape(image, planeShape), box, 1, radius, 1);
        var count = engine.Conv2D(Tensor<T>.CreateDefault(new[] { 1, 1, shape[2], shape[3] }, ops.One), box, 1, radius, 1);
        return engine.Reshape(engine.TensorDivide(sum, engine.TensorBroadcastTo(count, planeShape)), shape);
    }

    /// <summary>
    /// Rearranges the patchSize x patchSize patches of a [1, channels, height, width] image so destination patch p
    /// (row-major over the complete patch grid) is source patch <paramref name="order"/>[p]; pixels outside the
    /// complete grid are zero.
    /// </summary>
    internal static Tensor<T> PermutePatches<T>(Tensor<T> image, int patchSize, int[] order)
    {
        var engine = AiDotNetEngine.Current;
        int[] shape = image._shape;
        int channels = shape[1], height = shape[2], width = shape[3];
        int rows = height / patchSize, cols = width / patchSize;
        if (rows == 0 || cols == 0)
            return new Tensor<T>(shape);

        var grid = engine.TensorNarrow(engine.TensorNarrow(image, 2, 0, rows * patchSize), 3, 0, cols * patchSize);
        var patches = engine.Reshape(
            engine.TensorPermute(engine.Reshape(grid, new[] { channels, rows, patchSize, cols, patchSize }), new[] { 1, 3, 0, 2, 4 }),
            new[] { rows * cols, channels, patchSize, patchSize });
        var index = new Tensor<int>(new[] { order.Length }, new Vector<int>(order));
        var moved = engine.Reshape(engine.TensorIndexSelect(patches, index, 0), new[] { rows, cols, channels, patchSize, patchSize });
        var assembled = engine.Reshape(engine.TensorPermute(moved, new[] { 2, 0, 3, 1, 4 }),
            new[] { 1, channels, rows * patchSize, cols * patchSize });
        return engine.Pad(assembled, 0, height - rows * patchSize, 0, width - cols * patchSize,
            MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>
    /// Mean of each patchSize x patchSize patch over the first <paramref name="channels"/> channels of a
    /// [1, c, height, width] image, row-major over the complete patch grid, as [rows * cols].
    /// </summary>
    internal static Tensor<T> PatchMeans<T>(Tensor<T> image, int patchSize, int channels)
    {
        var engine = AiDotNetEngine.Current;
        int rows = image._shape[2] / patchSize, cols = image._shape[3] / patchSize;
        var grid = engine.TensorNarrow(engine.TensorNarrow(engine.TensorNarrow(image, 1, 0, channels), 2, 0, rows * patchSize), 3, 0, cols * patchSize);
        var blocks = engine.Reshape(grid, new[] { channels, rows, patchSize, cols, patchSize });
        return engine.Reshape(engine.ReduceMean(blocks, new[] { 0, 2, 4 }, false), new[] { rows * cols });
    }

    /// <summary>
    /// Stable ascending argsort with linear auxiliary memory and O(n log n) comparisons.
    /// NaNs sort first, matching Double.CompareTo; equal values keep their source order.
    /// </summary>
    internal static int[] StableArgsort<T>(Tensor<T> values)
    {
        // The engine argsort does not promise stable ties. Sort native wide keys after one bulk cast
        // rather than building quadratic pairwise comparison grids (gigabytes for ordinary image grids).
        return values.Cast<double>().ToArray()
            .Select((value, index) => (Value: value, Index: index))
            .OrderBy(entry => entry.Value)
            .Select(entry => entry.Index)
            .ToArray();
    }

    /// <summary>C#-style integer remainder (truncated, sign of the dividend) of an integral-valued tensor.</summary>
    internal static Tensor<T> Mod<T>(Tensor<T> values, int divisor)
        => AiDotNetEngine.Current.TensorFmod(values,
            Tensor<T>.CreateDefault(values._shape, MathHelper.GetNumericOperations<T>().FromDouble(divisor)));

    /// <summary>
    /// Deterministic label colours ((label * multiplier) mod 256) / 255 for three multipliers, as
    /// [batch, 3, h, w] from integral labels [batch, 1, h, w].
    /// </summary>
    internal static Tensor<T> LabelColors<T>(Tensor<T> labels, int red, int green, int blue)
    {
        // Half rounds integer color hashes before the remainder, changing ordinary label colors.
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
            return LabelColors(labels.Cast<double>(), red, green, blue).Cast<T>();
#endif
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var scale = ops.FromDouble(1.0 / 255.0);
        return engine.TensorConcatenate(new[]
        {
            engine.TensorMultiplyScalar(Mod(engine.TensorMultiplyScalar(labels, ops.FromDouble(red)), 256), scale),
            engine.TensorMultiplyScalar(Mod(engine.TensorMultiplyScalar(labels, ops.FromDouble(green)), 256), scale),
            engine.TensorMultiplyScalar(Mod(engine.TensorMultiplyScalar(labels, ops.FromDouble(blue)), 256), scale),
        }, 1);
    }

    /// <summary>One where <paramref name="values"/> &gt; <paramref name="threshold"/>, zero elsewhere.</summary>
    internal static Tensor<T> Above<T>(Tensor<T> values, double threshold)
        => AiDotNetEngine.Current.TensorGreaterThan(values.Cast<double>(), threshold).Cast<T>();
}
