namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>Grounding DINO's predictions for a batch.</summary>
/// <param name="Logits">Per decoder layer, the query-to-token similarities <c>[B, Q, MaxTextLength]</c>; masked tokens and
/// tokens past the prompt hold <see cref="GroundingDinoNetwork{T}.Masked"/>.</param>
/// <param name="Boxes">Per decoder layer, the boxes <c>[B, Q, 4]</c> as (cx, cy, w, h) normalized to the image.</param>
/// <param name="EncoderLogits">The selected queries' similarities before decoding (two-stage auxiliary output).</param>
/// <param name="EncoderBoxes">The selected proposals' boxes <c>[B, Q, 4]</c> (two-stage auxiliary output).</param>
/// <param name="TokenMask">The prompt tokens the logits score (after truncation to MaxTextLength).</param>
internal sealed record GroundingDinoOutput<T>(Tensor<T>[] Logits, Tensor<T>[] Boxes, Tensor<T> EncoderLogits, Tensor<T> EncoderBoxes, bool[,] TokenMask);

internal sealed partial class GroundingDinoNetwork<T>
{
    /// <summary>The logit of a token a query cannot match (the reference's −inf, finite so losses stay defined).</summary>
    public const double Masked = -1e9;

    // An invalid proposal's coordinate logit (the reference's +inf; its sigmoid is exactly 1 either way).
    private const double InvalidProposal = 1e30;

    // BERT's [CLS], [SEP], "." and "?": each phrase of a prompt attends only within itself.
    private static readonly int[] SpecialTokens = { 101, 102, 1012, 1029 };

    /// <summary>
    /// The reference's <c>generate_masks_with_special_tokens_and_transfer_map</c>: tokens attend within their phrase
    /// (the span ending at the next special token, except the leading [CLS] and trailing [SEP] spans), and each phrase's
    /// positions restart at 0.
    /// </summary>
    internal static (bool[,,] SelfMask, int[,] Positions) PhraseMasks(int[,] ids)
    {
        int b = ids.GetLength(0), l = ids.GetLength(1);
        var mask = new bool[b, l, l];
        var positions = new int[b, l];
        for (int i = 0; i < b; i++)
        {
            var previous = new int[l];
            var next = new int[l];
            int last = -1;
            for (int j = 0; j < l; j++)
            {
                if (Array.IndexOf(SpecialTokens, ids[i, j]) >= 0) last = j;
                previous[j] = last;
            }
            int following = l;
            for (int j = l - 1; j >= 0; j--)
            {
                if (Array.IndexOf(SpecialTokens, ids[i, j]) >= 0) following = j;
                next[j] = following;
            }
            for (int q = 0; q < l; q++)
            {
                bool valid = next[q] != 0 && next[q] != l - 1 && next[q] != l;
                positions[i, q] = valid ? Math.Max(0, q - previous[q] - 1) : 0;
                for (int k = 0; k < l; k++)
                {
                    bool validKey = next[k] != 0 && next[k] != l - 1 && next[k] != l;
                    mask[i, q, k] = q == k || (next[q] == next[k] && validKey);
                }
            }
        }
        return (mask, positions);
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder)
        => holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));

    private Tensor<T> Mlp(DenseLayer<T>[] layers, Tensor<T> x)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            x = Rows(layers[i], x);
            if (i < layers.Length - 1) x = _engine.ReLU(x);
        }
        return x;
    }

    private Tensor<T> Dropout(Tensor<T> x, double rate)
    {
        if (!Training || rate <= 0) return x;
        var keep = new Tensor<T>(x.Shape.ToArray());
        var scale = NumOps.FromDouble(1.0 / (1 - rate));
        for (int i = 0; i < keep.Length; i++) keep[i] = _dropRandom.NextDouble() < rate ? NumOps.Zero : scale;
        return _engine.TensorMultiply(x, keep);
    }

    private Tensor<T> DropPath(Tensor<T> x, double rate)
    {
        if (!Training || rate <= 0) return x;
        int b = x.Shape[0];
        var keep = new Tensor<T>(new[] { b, 1, 1 });
        for (int i = 0; i < b; i++) keep[i, 0, 0] = NumOps.FromDouble(_dropRandom.NextDouble() < 1 - rate ? 1.0 / (1 - rate) : 0.0);
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(keep, x.Shape.ToArray()));
    }

    /// <summary>The reference's <c>encode_sinusoidal_position_embedding</c> of <c>[..., coords]</c> values: per coordinate,
    /// <paramref name="features"/> interleaved sin/cos of value·2π/10000^(2⌊i/2⌋/features); x and y swap places.</summary>
    private static double[] SineFeatures(double[] coordinates, int features)
    {
        int n = coordinates.Length;
        var output = new double[n * features];
        var order = Enumerable.Range(0, n).ToArray();
        if (n >= 2) (order[0], order[1]) = (1, 0);
        for (int slot = 0; slot < n; slot++)
        {
            double v = coordinates[order[slot]] * 2 * Math.PI;
            for (int i = 0; i < features; i++)
            {
                int pair = i / 2;   // sin and cos of one frequency share an index
                double a = v / Math.Pow(10000, 2.0 * pair / features);
                output[slot * features + i] = i % 2 == 0 ? Math.Sin(a) : Math.Cos(a);
            }
        }
        return output;
    }

    /// <summary>The sine position embedding of an unpadded <c>h × w</c> level, <c>[D, h, w]</c> (y features, then x).</summary>
    private double[,,] LevelPositions(int h, int w)
    {
        int half = Configuration.Dim / 2;
        double temperature = Configuration.PositionTemperature, scale = 2 * Math.PI;
        var pos = new double[Configuration.Dim, h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double ye = (y + 1) / (h + 1e-6) * scale, xe = (x + 1) / (w + 1e-6) * scale;
                for (int i = 0; i < half; i++)
                {
                    int pair = i / 2;   // sin and cos of one frequency share an index
                    double t = Math.Pow(temperature, 2.0 * pair / half);
                    pos[i, y, x] = i % 2 == 0 ? Math.Sin(ye / t) : Math.Cos(ye / t);
                    pos[half + i, y, x] = i % 2 == 0 ? Math.Sin(xe / t) : Math.Cos(xe / t);
                }
            }
        return pos;
    }

    /// <summary>Multi-head attention <c>softmax(QKᵀ/√d + mask)V</c> over <c>[B, N, D]</c> queries and <c>[B, M, D]</c> keys.</summary>
    private Tensor<T> Attend(Attention a, Tensor<T> queries, Tensor<T> keys, Tensor<T> values, Tensor<T>? mask)
    {
        int b = queries.Shape[0], n = queries.Shape[1], m = keys.Shape[1], d = queries.Shape[2], heads = a.Heads, hd = d / heads;
        Tensor<T> Heads(DenseLayer<T> projection, Tensor<T> x, int len) => _engine.Reshape(
            _engine.TensorPermute(_engine.Reshape(Rows(projection, x), new[] { b, len, heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b * heads, len, hd });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(a.Query, queries, n), _engine.TensorPermute(Heads(a.Key, keys, m), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        if (mask is not null) scores = _engine.TensorAdd(scores, mask);
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(a.Value, values, m));
        return Rows(a.Output, _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { b, heads, n, hd }), new[] { 0, 2, 1, 3 }), new[] { b, n, d }));
    }

    /// <summary>A per-key additive mask <c>[B·heads, N, M]</c>: <paramref name="blocked"/> where a key may not be attended.</summary>
    private Tensor<T> KeyMask(bool[,] allowed, int heads, int n, double blocked)
    {
        int b = allowed.GetLength(0), m = allowed.GetLength(1);
        var mask = new Tensor<T>(new[] { b * heads, n, m });
        var value = NumOps.FromDouble(blocked);
        for (int i = 0; i < b; i++)
            for (int k = 0; k < m; k++)
                if (!allowed[i, k])
                    for (int h = 0; h < heads; h++)
                        for (int q = 0; q < n; q++) mask[i * heads + h, q, k] = value;
        return mask;
    }

    private static double Finfo => typeof(T) == typeof(float) ? float.MinValue : double.MinValue;

    /// <summary>
    /// Multi-scale deformable attention (Zhu et al. 2021) over the flattened levels <paramref name="memory"/>
    /// <c>[B, S, D]</c>: per head, level and point, an offset from the reference (scaled by the level size for 2-D
    /// references, by half the box over the point count for boxes) and a softmax weight, sampled bilinearly
    /// (zero padding, align_corners false).
    /// </summary>
    private Tensor<T> Sample(Deformable a, Tensor<T> queries, Tensor<T> memory, Tensor<T> reference, (int H, int W)[] shapes, int[] starts)
    {
        var c = Configuration;
        int b = queries.Shape[0], n = queries.Shape[1], s = memory.Shape[1], heads = a.Heads, levels = c.Levels, points = a.Points, hd = c.Dim / heads;
        var value = _engine.Reshape(Rows(a.Value, memory), new[] { b, s, heads, hd });
        var offsets = _engine.Reshape(Rows(a.Offsets, queries), new[] { b, n, heads, levels, points, 2 });
        var weights = _engine.Reshape(_engine.Softmax(_engine.Reshape(Rows(a.Weights, queries), new[] { b, n, heads, levels * points }), 3),
            new[] { b, n, heads, levels, points });
        var locationShape = new[] { b, n, heads, levels, points, 2 };
        Tensor<T> locations;
        if (reference.Shape[3] == 2)
        {
            var normalizer = new Tensor<T>(new[] { 1, 1, 1, levels, 1, 2 });
            for (int l = 0; l < levels; l++)
            {
                normalizer[l * 2] = NumOps.FromDouble(1.0 / shapes[l].W);
                normalizer[l * 2 + 1] = NumOps.FromDouble(1.0 / shapes[l].H);
            }
            locations = _engine.TensorAdd(_engine.TensorBroadcastTo(_engine.Reshape(reference, new[] { b, n, 1, levels, 1, 2 }), locationShape),
                _engine.TensorMultiply(offsets, _engine.TensorBroadcastTo(normalizer, locationShape)));
        }
        else
        {
            var centers = _engine.TensorSlice(reference, new[] { 0, 0, 0, 0 }, new[] { b, n, levels, 2 });
            var sizes = _engine.TensorSlice(reference, new[] { 0, 0, 0, 2 }, new[] { b, n, levels, 2 });
            locations = _engine.TensorAdd(_engine.TensorBroadcastTo(_engine.Reshape(centers, new[] { b, n, 1, levels, 1, 2 }), locationShape),
                _engine.TensorMultiply(offsets, _engine.TensorMultiplyScalar(
                    _engine.TensorBroadcastTo(_engine.Reshape(sizes, new[] { b, n, 1, levels, 1, 2 }), locationShape), NumOps.FromDouble(0.5 / points))));
        }
        var grids = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(locations, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var sampled = new Tensor<T>[levels];
        for (int l = 0; l < levels; l++)
        {
            int h = shapes[l].H, w = shapes[l].W;
            var levelValue = _engine.TensorSlice(value, new[] { 0, starts[l], 0, 0 }, new[] { b, h * w, heads, hd });
            levelValue = _engine.Reshape(_engine.TensorPermute(levelValue, new[] { 0, 2, 3, 1 }), new[] { b * heads, hd, h, w });
            var levelGrid = _engine.TensorSlice(grids, new[] { 0, 0, 0, l, 0, 0 }, new[] { b, n, heads, 1, points, 2 });
            levelGrid = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(levelGrid, new[] { b, n, heads, points, 2 }), new[] { 0, 2, 1, 3, 4 }),
                new[] { b * heads, n, points, 2 });
            sampled[l] = _engine.GridSample(levelValue, levelGrid);                                            // [B·h, hd, N, P]
        }
        var stacked = _engine.Reshape(_engine.TensorStack(sampled, 3), new[] { b * heads, hd, n, levels * points });
        var w8 = _engine.Reshape(_engine.TensorPermute(weights, new[] { 0, 2, 1, 3, 4 }), new[] { b * heads, 1, n, levels * points });
        var summed = _engine.ReduceSum(_engine.TensorMultiply(stacked, _engine.TensorBroadcastTo(w8, stacked._shape)), new[] { 3 }, keepDims: false);
        return Rows(a.Output, _engine.TensorPermute(_engine.Reshape(summed, new[] { b, heads * hd, n }), new[] { 0, 2, 1 }));
    }

    /// <summary>The contrastive head: query·token similarities, <see cref="Masked"/> for masked tokens, padded to
    /// MaxTextLength.</summary>
    private Tensor<T> Contrast(Tensor<T> queries, Tensor<T> text, bool[,] tokenMask)
    {
        int b = queries.Shape[0], n = queries.Shape[1], l = text.Shape[1], max = Configuration.MaxTextLength;
        var scores = _engine.BatchMatMul(queries, _engine.TensorPermute(text, new[] { 0, 2, 1 }));                 // [B, N, L]
        var bias = new Tensor<T>(new[] { b, n, l });
        for (int i = 0; i < b; i++)
            for (int k = 0; k < l; k++)
                if (!tokenMask[i, k])
                    for (int q = 0; q < n; q++) bias[i, q, k] = NumOps.FromDouble(Masked);
        // A masked token's score is the constant alone (masked_fill), not the similarity plus it.
        var keep = new Tensor<T>(new[] { b, n, l });
        for (int i = 0; i < b; i++)
            for (int k = 0; k < l; k++)
                for (int q = 0; q < n; q++) keep[i, q, k] = tokenMask[i, k] ? NumOps.One : NumOps.Zero;
        scores = _engine.TensorAdd(_engine.TensorMultiply(scores, keep), bias);
        if (l == max) return scores;
        var pad = new Tensor<T>(new[] { b, n, max - l });
        pad.AsWritableSpan().Fill(NumOps.FromDouble(Masked));
        return _engine.TensorConcatenate(new[] { scores, pad }, 2);
    }

    private static Tensor<T> Logit(IEngine engine, Tensor<T> p)
    {
        // torch.special.logit(x, eps=1e-5): x clamped to [eps, 1 - eps], then log(x / (1 - x)).
        var clamped = engine.TensorClamp(p, NumOps.FromDouble(1e-5), NumOps.FromDouble(1 - 1e-5));
        return engine.TensorLog(engine.TensorDivide(clamped, engine.ScalarMinusTensor(NumOps.One, clamped)));
    }
}
