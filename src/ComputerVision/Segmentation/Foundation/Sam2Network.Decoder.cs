using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

internal sealed partial class Sam2Network<T>
{
    /// <summary>The prompt position encoding of (x, y) in [0, 1]: sin and cos of 2π·(2·xy − 1)·G.</summary>
    private double[] Fourier(double x, double y)
    {
        int half = Configuration.Dim / 2;
        var f = new double[2 * half];
        for (int j = 0; j < half; j++)
        {
            double v = 2 * Math.PI * ((2 * x - 1) * _gaussian[0, j] + (2 * y - 1) * _gaussian[1, j]);
            f[j] = Math.Sin(v);
            f[half + j] = Math.Cos(v);
        }
        return f;
    }

    /// <summary>The dense position encoding <c>[1, Dim, Grid, Grid]</c> of the image embedding's cell centres.</summary>
    public Tensor<T> DensePositional()
    {
        int g = Configuration.Grid, p = Configuration.Dim;
        var pe = new Tensor<T>(new[] { 1, p, g, g });
        for (int i = 0; i < g; i++)
            for (int j = 0; j < g; j++)
            {
                var f = Fourier((j + 0.5) / g, (i + 0.5) / g);
                for (int k = 0; k < p; k++) pe[(k * g + i) * g + j] = NumOps.FromDouble(f[k]);
            }
        return pe;
    }

    /// <summary>
    /// The sparse prompt embeddings <c>[N, tokens, Dim]</c> and dense ones <c>[N, Dim, Grid, Grid]</c> for N prompts:
    /// points <c>[N, k, 2]</c> (x, y in the ImageSize frame) with labels <c>[N, k]</c> (1 positive, 0 negative,
    /// −1 padding), boxes <c>[N, 4]</c> (x0, y0, x1, y1), and low-resolution mask logits <c>[N, 1, 4·Grid, 4·Grid]</c>.
    /// </summary>
    public (Tensor<T> Sparse, Tensor<T> Dense) EncodePrompts(Tensor<T>? points, Tensor<T>? labels, Tensor<T>? boxes, Tensor<T>? masks)
    {
        var c = Configuration;
        int p = c.Dim, g = c.Grid;
        int n = points?.Shape[0] ?? boxes?.Shape[0] ?? masks?.Shape[0] ?? 1;
        var parts = new List<Tensor<T>>();
        if (points is null && boxes is null)
        {
            // The model's no-prompt input: one padding point.
            points = new Tensor<T>(new[] { n, 1, 2 });
            labels = new Tensor<T>(new[] { n, 1 });
            labels.AsWritableSpan().Fill(NumOps.FromDouble(-1));
        }
        if (points is not null && labels is not null)
        {
            int k = points.Shape[1];
            bool pad = boxes is null;
            int count = k + (pad ? 1 : 0);
            var fourier = new Tensor<T>(new[] { n, count, p });
            var kinds = new Tensor<T>[3];
            for (int i = 0; i < 3; i++) kinds[i] = new Tensor<T>(new[] { n, count, 1 });
            for (int b = 0; b < n; b++)
                for (int i = 0; i < count; i++)
                {
                    int label = i < k ? (int)Math.Round(NumOps.ToDouble(labels[b, i])) : -1;
                    if (label == -1) { kinds[0][b, i, 0] = NumOps.One; continue; }
                    double x = (NumOps.ToDouble(points[b, i, 0]) + 0.5) / c.ImageSize, y = (NumOps.ToDouble(points[b, i, 1]) + 0.5) / c.ImageSize;
                    var f = Fourier(x, y);
                    for (int j = 0; j < p; j++) fourier[(b * count + i) * p + j] = NumOps.FromDouble(f[j]);
                    kinds[label == 1 ? 2 : 1][b, i, 0] = NumOps.One;
                }
            var shape = new[] { n, count, p };
            Tensor<T> Kind(Tensor<T> mask, BiasFreeLinearLayer<T> embedding) => _engine.TensorMultiply(
                _engine.TensorBroadcastTo(mask, shape), _engine.TensorBroadcastTo(_engine.Reshape(Value(embedding), new[] { 1, 1, p }), shape));
            var embedded = _engine.TensorAdd(fourier, Kind(kinds[0], NotAPoint));
            embedded = _engine.TensorAdd(embedded, Kind(kinds[1], PointEmbeddings[0]));
            parts.Add(_engine.TensorAdd(embedded, Kind(kinds[2], PointEmbeddings[1])));
        }
        if (boxes is not null)
        {
            // Two corner tokens (with the corner embeddings) and SAM 2's third, padding token.
            var corners = new Tensor<T>(new[] { n, 2, p });
            for (int b = 0; b < n; b++)
                for (int corner = 0; corner < 2; corner++)
                {
                    var f = Fourier((NumOps.ToDouble(boxes[b, 2 * corner]) + 0.5) / c.ImageSize, (NumOps.ToDouble(boxes[b, 2 * corner + 1]) + 0.5) / c.ImageSize);
                    for (int j = 0; j < p; j++) corners[(b * 2 + corner) * p + j] = NumOps.FromDouble(f[j]);
                }
            var cornerEmbeddings = _engine.TensorConcatenate(new[]
            {
                _engine.Reshape(Value(PointEmbeddings[2]), new[] { 1, 1, p }), _engine.Reshape(Value(PointEmbeddings[3]), new[] { 1, 1, p }),
            }, 1);
            parts.Add(_engine.TensorAdd(corners, _engine.TensorBroadcastTo(cornerEmbeddings, new[] { n, 2, p })));
            parts.Add(_engine.TensorBroadcastTo(_engine.Reshape(Value(NotAPoint), new[] { 1, 1, p }), new[] { n, 1, p }));
        }
        var sparse = parts.Count == 1 ? parts[0] : _engine.TensorConcatenate(parts.ToArray(), 1);
        Tensor<T> dense;
        if (masks is not null)
        {
            var m = Gelu(ChannelNorm(MaskNorm1, Convolve(MaskConv1, masks, 2, 2, 0)));
            m = Gelu(ChannelNorm(MaskNorm2, Convolve(MaskConv2, m, 2, 2, 0)));
            dense = Convolve(MaskConv3, m, 1, 1, 0);
        }
        else
        {
            dense = _engine.TensorBroadcastTo(_engine.Reshape(Value(NoMask), new[] { 1, p, 1, 1 }), new[] { n, p, g, g });
        }
        return (sparse, dense);
    }

    private Tensor<T> Attend(Attention attention, Tensor<T> q, Tensor<T> k, Tensor<T> v)
    {
        int n = q.Shape[0], tq = q.Shape[1], tk = k.Shape[1];
        var qp = Rows(attention.Query, q);
        int inner = qp.Shape[2], heads = attention.Heads, hd = inner / heads;
        Tensor<T> Heads(Tensor<T> x, int t) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { n, t, heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { n * heads, t, hd });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(qp, tq), _engine.TensorPermute(Heads(Rows(attention.Key, k), tk), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(Rows(attention.Value, v), tk));
        context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { n, heads, tq, hd }), new[] { 0, 2, 1, 3 }), new[] { n, tq, inner });
        return Rows(attention.Output, context);
    }

    private Tensor<T> Mlp(DenseLayer<T>[] layers, Tensor<T> x)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            x = layers[i].Forward(x);
            if (i < layers.Length - 1) x = _engine.ReLU(x);
        }
        return x;
    }

    /// <summary>
    /// The mask logits <c>[N, masks, 4·Grid, 4·Grid]</c>, IoU predictions <c>[N, masks]</c> and object-score logits
    /// <c>[N, 1]</c> for N prompts on one image's features: the three ambiguity-aware masks with
    /// <paramref name="multimask"/>; otherwise one mask, chosen as the image predictor does when not training (the
    /// single-mask output if it is stable, else the best of the three).
    /// </summary>
    public (Tensor<T> Logits, Tensor<T> Iou, Tensor<T> ObjectScore) DecodeMasks(Tensor<T> embedding, Tensor<T> highRes0, Tensor<T> highRes1,
        Tensor<T> positional, Tensor<T> sparse, Tensor<T> dense, bool multimask)
    {
        var c = Configuration;
        int n = sparse.Shape[0], p = c.Dim, g = c.Grid, hw = g * g, m = c.MaskTokens;
        var output = _engine.TensorConcatenate(new[]
        {
            _engine.Reshape(Value(ObjectToken), new[] { 1, 1, p }), _engine.Reshape(Value(IouToken), new[] { 1, 1, p }), _engine.Reshape(Value(MaskTokenTable), new[] { 1, m, p }),
        }, 1);
        var tokens = _engine.TensorConcatenate(new[] { _engine.TensorBroadcastTo(output, new[] { n, 2 + m, p }), sparse }, 1);
        var src = _engine.TensorAdd(_engine.TensorBroadcastTo(embedding, new[] { n, p, g, g }), dense);
        Tensor<T> Sequence(Tensor<T> x) => _engine.TensorPermute(_engine.Reshape(x, new[] { x.Shape[0], p, hw }), new[] { 0, 2, 1 });
        var keys = Sequence(src);
        var keyPositional = Sequence(_engine.TensorBroadcastTo(positional, new[] { n, p, g, g }));
        var queries = tokens;
        var queryPositional = tokens;
        foreach (var block in TwoWayBlocks)
        {
            if (block.SkipFirstPositional)
            {
                queries = Attend(block.SelfAttention, queries, queries, queries);
            }
            else
            {
                var qp = _engine.TensorAdd(queries, queryPositional);
                queries = _engine.TensorAdd(queries, Attend(block.SelfAttention, qp, qp, queries));
            }
            queries = Rows(block.Norm1, queries);
            queries = Rows(block.Norm2, _engine.TensorAdd(queries,
                Attend(block.TokenToImage, _engine.TensorAdd(queries, queryPositional), _engine.TensorAdd(keys, keyPositional), keys)));
            queries = Rows(block.Norm3, _engine.TensorAdd(queries, Rows(block.Mlp2, _engine.ReLU(Rows(block.Mlp1, queries)))));
            keys = Rows(block.Norm4, _engine.TensorAdd(keys,
                Attend(block.ImageToToken, _engine.TensorAdd(keys, keyPositional), _engine.TensorAdd(queries, queryPositional), queries)));
        }
        queries = Rows(FinalNorm, _engine.TensorAdd(queries,
            Attend(FinalAttention, _engine.TensorAdd(queries, queryPositional), _engine.TensorAdd(keys, keyPositional), keys)));

        var image = _engine.Reshape(_engine.TensorPermute(keys, new[] { 0, 2, 1 }), new[] { n, p, g, g });
        var s1 = _engine.TensorBroadcastTo(highRes1, new[] { n, highRes1.Shape[1], highRes1.Shape[2], highRes1.Shape[3] });
        var s0 = _engine.TensorBroadcastTo(highRes0, new[] { n, highRes0.Shape[1], highRes0.Shape[2], highRes0.Shape[3] });
        var up = Gelu(ChannelNorm(UpscaleNorm, _engine.TensorAdd(Upscale1.Forward(image), s1)));
        up = Gelu(_engine.TensorAdd(Upscale2.Forward(up), s0));                                            // [N, p/8, 4g, 4g]
        int side = 4 * g, low = p / 8;
        Tensor<T> Token(int index) => _engine.Reshape(_engine.TensorSlice(queries, new[] { 0, index, 0 }, new[] { n, 1, p }), new[] { n, p });
        var weights = new Tensor<T>[m];
        for (int i = 0; i < m; i++) weights[i] = _engine.Reshape(Mlp(Hypernetworks[i], Token(2 + i)), new[] { n, 1, low });
        var masks = _engine.Reshape(_engine.BatchMatMul(_engine.TensorConcatenate(weights, 1), _engine.Reshape(up, new[] { n, low, side * side })),
            new[] { n, m, side, side });
        var iou = _engine.Sigmoid(Mlp(IouHead, Token(1)));
        var objectScore = Mlp(ObjectHead, Token(0));
        if (multimask)
            return (_engine.TensorSlice(masks, new[] { 0, 1, 0, 0 }, new[] { n, m - 1, side, side }),
                _engine.TensorSlice(iou, new[] { 0, 1 }, new[] { n, m - 1 }), objectScore);
        if (Training)
            return (_engine.TensorSlice(masks, new[] { 0, 0, 0, 0 }, new[] { n, 1, side, side }),
                _engine.TensorSlice(iou, new[] { 0, 0 }, new[] { n, 1 }), objectScore);

        // Dynamic multimask via stability: keep the single-mask output where its stability (the area above +δ over the
        // area above −δ) reaches the threshold, else the multimask output with the highest predicted IoU.
        var chosenMasks = new Tensor<T>[n];
        var chosenIou = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            int index = 0;
            var single = masks.ToArray().Skip(i * m * side * side).Take(side * side).Select(NumOps.ToDouble).ToArray();
            double above = single.Count(v => v > c.StabilityDelta), within = single.Count(v => v > -c.StabilityDelta);
            double stability = within > 0 ? above / within : 1.0;
            if (stability < c.StabilityThreshold)
            {
                double best = double.NegativeInfinity;
                for (int k = 1; k < m; k++)
                {
                    double score = NumOps.ToDouble(iou[i, k]);
                    if (score > best) { best = score; index = k; }
                }
            }
            chosenMasks[i] = _engine.TensorSlice(masks, new[] { i, index, 0, 0 }, new[] { 1, 1, side, side });
            chosenIou[i] = _engine.TensorSlice(iou, new[] { i, index }, new[] { 1, 1 });
        }
        return (n == 1 ? chosenMasks[0] : _engine.TensorConcatenate(chosenMasks, 0), n == 1 ? chosenIou[0] : _engine.TensorConcatenate(chosenIou, 0), objectScore);
    }

    /// <summary>The image predictor's postprocessing: low-resolution logits bilinearly resized (align_corners false) to
    /// the original image size.</summary>
    public Tensor<T> Postprocess(Tensor<T> logits, int originalHeight, int originalWidth)
        => SpatialResampling.Resize(_engine, logits, originalHeight, originalWidth);

    /// <summary>The official preprocessing of images <c>[B, 3, ImageSize, ImageSize]</c> in [0, 1]: normalized by the
    /// ImageNet mean and std.</summary>
    public Tensor<T> Normalize(Tensor<T> images)
    {
        var c = Configuration;
        var mean = new Tensor<T>(new[] { 1, 3, 1, 1 }, new Vector<T>(c.PixelMean.Select(NumOps.FromDouble).ToArray()));
        var inverse = new Tensor<T>(new[] { 1, 3, 1, 1 }, new Vector<T>(c.PixelStd.Select(s => NumOps.FromDouble(1 / s)).ToArray()));
        var shape = images.Shape.ToArray();
        return _engine.TensorMultiply(_engine.TensorSubtract(images, _engine.TensorBroadcastTo(mean, shape)), _engine.TensorBroadcastTo(inverse, shape));
    }

    /// <summary>
    /// The reference initialization (Hugging Face <c>Sam2PreTrainedModel._init_weights</c>): linear and convolution
    /// weights and embeddings N(0, 0.02²) with zero biases, layer norms unit, the Hiera position embeddings zero, and the
    /// prompt encoder's random Fourier matrix N(0, 1) (scale 1), shared with the image positional encoding.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double[]? fourier = null;
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            if (name.EndsWith("positional_embedding", StringComparison.Ordinal))
                return (double[])(fourier ??= Enumerable.Range(0, count).Select(_ => Normal()).ToArray()).Clone();
            if (name.Contains("pos_embed") || name == "no_memory_embedding") return values;
            bool weight = name.EndsWith(".weight", StringComparison.Ordinal);
            if (name.Contains("layer_norm"))
            {
                if (weight) for (int i = 0; i < count; i++) values[i] = 1.0;
                return values;
            }
            if (weight) for (int i = 0; i < count; i++) values[i] = 0.02 * Normal();
            return values;
        });
    }

    /// <summary>Loads a Hugging Face <c>Sam2Model</c> state dict (as converted from the official checkpoints).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int p = c.Dim, d0 = c.EmbedDims[0];
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        void Conv(DenseLayer<T> layer, string name, int inputs, int outputs, int kernel)
            => TorchParameters.Linear(_engine, layer, inputs * kernel * kernel, outputs, read($"{name}.weight", new[] { outputs, inputs, kernel, kernel }),
                read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim)
            => TorchParameters.LayerNorm(_engine, norm, read($"{name}.weight", new[] { dim }), read($"{name}.bias", new[] { dim }));
        void Values(BiasFreeLinearLayer<T> holder, string name, int[] shape)
            => TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, b) => a * b), read(name, shape));
        void Attention(Attention a, string name, int inner)
        {
            Linear(a.Query, $"{name}.q_proj", p, inner);
            Linear(a.Key, $"{name}.k_proj", p, inner);
            Linear(a.Value, $"{name}.v_proj", p, inner);
            Linear(a.Output, $"{name}.o_proj", inner, p);
        }
        void FeedForward(DenseLayer<T>[] layers, string name, int inputs, int hidden, int outputs)
        {
            Linear(layers[0], $"{name}.proj_in", inputs, hidden);
            for (int i = 1; i < layers.Length - 1; i++) Linear(layers[i], $"{name}.layers.{i - 1}", hidden, hidden);
            Linear(layers[^1], $"{name}.proj_out", hidden, outputs);
        }

        const string backbone = "vision_encoder.backbone";
        Conv(PatchProjection, $"{backbone}.patch_embed.projection", c.Channels, d0, c.PatchKernel);
        Values(BackgroundPosition, $"{backbone}.pos_embed", new[] { 1, d0, c.BackgroundPositionSize, c.BackgroundPositionSize });
        Values(WindowPosition, $"{backbone}.pos_embed_window", new[] { 1, d0, c.WindowSizes[0], c.WindowSizes[0] });
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string n = $"{backbone}.blocks.{i}";
            Norm(b.Norm1, $"{n}.layer_norm1", b.Dim);
            Linear(b.Qkv, $"{n}.attn.qkv", b.Dim, 3 * b.DimOut);
            Linear(b.Projection, $"{n}.attn.proj", b.DimOut, b.DimOut);
            if (b.Shortcut is not null) Linear(b.Shortcut, $"{n}.proj", b.Dim, b.DimOut);
            Norm(b.Norm2, $"{n}.layer_norm2", b.DimOut);
            Linear(b.Mlp1, $"{n}.mlp.proj_in", b.DimOut, (int)(b.DimOut * c.MlpRatio));
            Linear(b.Mlp2, $"{n}.mlp.proj_out", (int)(b.DimOut * c.MlpRatio), b.DimOut);
        }
        for (int i = 0; i < NeckConvs.Count; i++)
            Conv(NeckConvs[i], $"vision_encoder.neck.convs.{i}", c.EmbedDims[c.EmbedDims.Length - 1 - i], p, 1);
        Conv(HighRes0, "mask_decoder.conv_s0", p, p / 8, 1);
        Conv(HighRes1, "mask_decoder.conv_s1", p, p / 4, 1);
        Values(NoMemory, "no_memory_embedding", new[] { 1, 1, p });

        var gaussian = read("prompt_encoder.shared_embedding.positional_embedding", new[] { 2, p / 2 });
        for (int i = 0; i < 2; i++) for (int j = 0; j < p / 2; j++) _gaussian[i, j] = gaussian[i * (p / 2) + j];
        var points = read("prompt_encoder.point_embed.weight", new[] { 4, p });
        for (int i = 0; i < 4; i++) TorchParameters.Linear(PointEmbeddings[i], 1, p, points.Skip(i * p).Take(p).ToArray());
        Values(NotAPoint, "prompt_encoder.not_a_point_embed.weight", new[] { 1, p });
        Values(NoMask, "prompt_encoder.no_mask_embed.weight", new[] { 1, p });
        Conv(MaskConv1, "prompt_encoder.mask_embed.conv1", 1, c.MaskInChannels / 4, 2);
        Norm(MaskNorm1, "prompt_encoder.mask_embed.layer_norm1", c.MaskInChannels / 4);
        Conv(MaskConv2, "prompt_encoder.mask_embed.conv2", c.MaskInChannels / 4, c.MaskInChannels, 2);
        Norm(MaskNorm2, "prompt_encoder.mask_embed.layer_norm2", c.MaskInChannels);
        Conv(MaskConv3, "prompt_encoder.mask_embed.conv3", c.MaskInChannels, p, 1);

        Values(ObjectToken, "mask_decoder.obj_score_token.weight", new[] { 1, p });
        Values(IouToken, "mask_decoder.iou_token.weight", new[] { 1, p });
        Values(MaskTokenTable, "mask_decoder.mask_tokens.weight", new[] { c.MaskTokens, p });
        int inner = p / c.AttentionDownsample;
        for (int i = 0; i < TwoWayBlocks.Count; i++)
        {
            var b = TwoWayBlocks[i];
            string n = $"mask_decoder.transformer.layers.{i}";
            Attention(b.SelfAttention, $"{n}.self_attn", p);
            Norm(b.Norm1, $"{n}.layer_norm1", p);
            Attention(b.TokenToImage, $"{n}.cross_attn_token_to_image", inner);
            Norm(b.Norm2, $"{n}.layer_norm2", p);
            Linear(b.Mlp1, $"{n}.mlp.proj_in", p, c.DecoderMlp);
            Linear(b.Mlp2, $"{n}.mlp.proj_out", c.DecoderMlp, p);
            Norm(b.Norm3, $"{n}.layer_norm3", p);
            Norm(b.Norm4, $"{n}.layer_norm4", p);
            Attention(b.ImageToToken, $"{n}.cross_attn_image_to_token", inner);
        }
        Attention(FinalAttention, "mask_decoder.transformer.final_attn_token_to_image", inner);
        Norm(FinalNorm, "mask_decoder.transformer.layer_norm_final_attn", p);
        Upscale1.Load(read("mask_decoder.upscale_conv1.weight", new[] { p, p / 4, 2, 2 }), read("mask_decoder.upscale_conv1.bias", new[] { p / 4 }));
        Norm(UpscaleNorm, "mask_decoder.upscale_layer_norm", p / 4);
        Upscale2.Load(read("mask_decoder.upscale_conv2.weight", new[] { p / 4, p / 8, 2, 2 }), read("mask_decoder.upscale_conv2.bias", new[] { p / 8 }));
        for (int i = 0; i < Hypernetworks.Count; i++)
            FeedForward(Hypernetworks[i], $"mask_decoder.output_hypernetworks_mlps.{i}", p, p, p / 8);
        FeedForward(IouHead, "mask_decoder.iou_prediction_head", p, c.IouHidden, c.MaskTokens);
        FeedForward(ObjectHead, "mask_decoder.pred_obj_score_head", p, p, 1);
    }
}
