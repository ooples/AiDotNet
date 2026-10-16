namespace AiDotNet.VisionLanguage.Grounding;

internal sealed partial class GroundingDinoNetwork<T>
{
    /// <summary>
    /// Grounding DINO on normalized images <c>[B, 3, H, W]</c> (no padding) and BERT token ids <c>[B, L]</c> with their
    /// mask (false for padding).
    /// </summary>
    public GroundingDinoOutput<T> Forward(Tensor<T> images, int[,] inputIds, bool[,] tokenMask, int[,]? tokenTypeIds = null)
    {
        var c = Configuration;
        int b = images.Shape[0];
        if (inputIds.GetLength(0) != b || tokenMask.GetLength(0) != b || tokenMask.GetLength(1) != inputIds.GetLength(1))
            throw new ArgumentException("One id row and one mask row per image.", nameof(inputIds));

        // Text: phrase masks and positions, truncated to the head's width, through BERT and the projection.
        var (selfMask, positions) = PhraseMasks(inputIds);
        int l = Math.Min(inputIds.GetLength(1), c.MaxTextLength);
        int[,] Cut(int[,] x) { var r = new int[b, l]; for (int i = 0; i < b; i++) for (int j = 0; j < l; j++) r[i, j] = x[i, j]; return r; }
        var ids = Cut(inputIds);
        var types = Cut(tokenTypeIds ?? new int[b, inputIds.GetLength(1)]);
        var pos = Cut(positions);
        var mask = new bool[b, l];
        var self = new bool[b, l, l];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                mask[i, j] = tokenMask[i, j];
                for (int k = 0; k < l; k++) self[i, j, k] = selfMask[i, j, k];
            }
        var bert = Text.Encode(ids, types, pos, self);
        Record("bert", bert);
        var text = Rows(TextProjection, bert);                                                                    // [B, L, D]
        Record("text_projected", text);

        // Image: backbone levels and stride-2 extra levels, each projected to D and group-normalized.
        var features = Backbone.Forward(images);
        for (int j = 0; j < features.Count; j++) Record($"swin{j}", features[j]);
        var maps = new List<Tensor<T>>();
        for (int level = 0; level < c.Levels; level++)
        {
            bool fromBackbone = level < features.Count;
            var source = fromBackbone ? features[level] : level == features.Count ? features[^1] : maps[^1];
            int k = InputKernel[level], stride = fromBackbone ? 1 : 2, pad = fromBackbone ? 0 : 1;
            int h = (source.Shape[2] + 2 * pad - k) / stride + 1, w = (source.Shape[3] + 2 * pad - k) / stride + 1;
            var columns = _engine.Unfold(source, new[] { k, k }, new[] { stride, stride }, new[] { pad, pad });           // [B, C·k·k, h·w]
            var projected = Rows(InputProjections[level], _engine.TensorPermute(columns, new[] { 0, 2, 1 }));          // [B, h·w, D]
            maps.Add(InputNorms[level].Forward(_engine.Reshape(_engine.TensorPermute(projected, new[] { 0, 2, 1 }), new[] { b, c.Dim, h, w })));
            Record($"projected{level}", maps[^1]);
        }

        // Flatten the levels, with each level's sine position plus its learned level embedding.
        var shapes = maps.Select(m => (m.Shape[2], m.Shape[3])).ToArray();
        var starts = new int[c.Levels];
        int tokens = 0;
        for (int level = 0; level < c.Levels; level++) { starts[level] = tokens; tokens += shapes[level].Item1 * shapes[level].Item2; }
        var levelTable = _engine.Reshape(Value(LevelEmbedding), new[] { c.Levels, c.Dim });
        var sources = new List<Tensor<T>>();
        var positionParts = new List<Tensor<T>>();
        for (int level = 0; level < c.Levels; level++)
        {
            var (h, w) = shapes[level];
            sources.Add(_engine.TensorPermute(_engine.Reshape(maps[level], new[] { b, c.Dim, h * w }), new[] { 0, 2, 1 }));
            var sine = LevelPositions(h, w);
            var table = new Tensor<T>(new[] { 1, h * w, c.Dim });
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) for (int d = 0; d < c.Dim; d++) table[0, y * w + x, d] = NumOps.FromDouble(sine[d, y, x]);
            var levelRow = _engine.Reshape(_engine.TensorSlice(levelTable, new[] { level, 0 }, new[] { 1, c.Dim }), new[] { 1, 1, c.Dim });
            positionParts.Add(_engine.TensorBroadcastTo(_engine.TensorAdd(table, _engine.TensorBroadcastTo(levelRow, new[] { 1, h * w, c.Dim })), new[] { b, h * w, c.Dim }));
        }
        var vision = _engine.TensorConcatenate(sources.ToArray(), 1);                                              // [B, S, D]
        var visionPosition = _engine.TensorConcatenate(positionParts.ToArray(), 1);

        // Encoder reference points: every token's own normalized centre, on every level.
        var encoderReference = new Tensor<T>(new[] { b, tokens, c.Levels, 2 });
        for (int level = 0; level < c.Levels; level++)
        {
            var (h, w) = shapes[level];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int i = 0; i < b; i++)
                        for (int r = 0; r < c.Levels; r++)
                        {
                            encoderReference[i, starts[level] + y * w + x, r, 0] = NumOps.FromDouble((x + 0.5) / w);
                            encoderReference[i, starts[level] + y * w + x, r, 1] = NumOps.FromDouble((y + 0.5) / h);
                        }
        }
        var textPosition = new Tensor<T>(new[] { b, l, c.Dim });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                var sine = SineFeatures(new[] { (double)pos[i, j] }, c.Dim);
                for (int d = 0; d < c.Dim; d++) textPosition[i, j, d] = NumOps.FromDouble(sine[d]);
            }
        var textSelfBias = new Tensor<T>(new[] { b * (c.EncoderHeads / 2), l, l });
        for (int i = 0; i < b; i++)
            for (int h = 0; h < c.EncoderHeads / 2; h++)
                for (int q = 0; q < l; q++)
                    for (int k = 0; k < l; k++)
                        if (!self[i, q, k]) textSelfBias[i * (c.EncoderHeads / 2) + h, q, k] = NumOps.FromDouble(Finfo);

        foreach (var layer in EncoderLayers)
        {
            (vision, text) = Fuse(layer.Fusion, vision, text, mask);
            bool first = layer == EncoderLayers[0];
            if (first) { Record("fusion0_vision", vision); Record("fusion0_text", text); }
            var tq = _engine.TensorAdd(text, textPosition);
            text = Rows(layer.TextNormBefore, _engine.TensorAdd(text, Attend(layer.TextAttention, tq, tq, text, textSelfBias)));
            text = Rows(layer.TextNormAfter, _engine.TensorAdd(text, Rows(layer.TextFc2, _engine.ReLU(Rows(layer.TextFc1, text)))));
            if (first) Record("enhancer0", text);
            var attended = Sample(layer.ImageAttention, _engine.TensorAdd(vision, visionPosition), vision, encoderReference, shapes, starts);
            vision = Rows(layer.ImageNorm, _engine.TensorAdd(vision, Dropout(attended, c.Dropout)));
            var ffn = Rows(layer.ImageFc2, _engine.ReLU(Rows(layer.ImageFc1, vision)));
            vision = Rows(layer.ImageFinalNorm, _engine.TensorAdd(vision, Dropout(ffn, c.Dropout)));
            if (first) Record("deformable0", vision);
        }

        Record("encoder_vision", vision);
        Record("encoder_text", text);
        // Language-guided query selection (two-stage): score every encoder token against the text and keep the best.
        var (objectQuery, proposals) = Proposals(vision, shapes, starts);
        var encoderClass = Contrast(objectQuery, text, mask);
        var coordinateLogits = _engine.TensorAdd(Mlp(EncoderBoxHead, objectQuery), proposals);
        int queries = Math.Min(c.Queries, tokens);
        var selected = new int[b, queries];
        var maxima = _engine.ReduceMax(encoderClass, new[] { 2 }, keepDims: false, out _);
        for (int i = 0; i < b; i++)
        {
            var order = Enumerable.Range(0, tokens).OrderByDescending(t => NumOps.ToDouble(maxima[i, t])).ThenBy(t => t).Take(queries).ToArray();
            for (int q = 0; q < queries; q++) selected[i, q] = order[q];
        }
        var reference = new Tensor<T>(new[] { b, queries, 4 });
        for (int i = 0; i < b; i++)
            for (int q = 0; q < queries; q++)
                for (int k = 0; k < 4; k++)
                    reference[i, q, k] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(coordinateLogits[i, selected[i, q], k]))));
        var target = _engine.TensorBroadcastTo(_engine.Reshape(_engine.TensorSlice(_engine.Reshape(Value(QueryEmbedding), new[] { c.Queries, c.Dim }),
            new[] { 0, 0 }, new[] { queries, c.Dim }), new[] { 1, queries, c.Dim }), new[] { b, queries, c.Dim });
        // The encoder's outputs for its auxiliary loss, as the official code takes them (hs_enc and refpoint_embed_undetach):
        // the selected tokens' scores against the encoded text, and their boxes, on the tape.
        var rows = new int[b * queries];
        for (int i = 0; i < b; i++)
            for (int q = 0; q < queries; q++) rows[i * queries + q] = i * tokens + selected[i, q];
        int width = encoderClass.Shape[2];
        var encoderLogits = _engine.Reshape(AiDotNet.ComputerVision.CvTensorOps<T>.Select(
            _engine.Reshape(encoderClass, new[] { b * tokens, width }), rows, 0), new[] { b, queries, width });
        var encoderBoxes = _engine.Sigmoid(_engine.Reshape(AiDotNet.ComputerVision.CvTensorOps<T>.Select(
            _engine.Reshape(coordinateLogits, new[] { b * tokens, 4 }), rows, 0), new[] { b, queries, 4 }));
        var initialReference = reference;
        Record("init_reference", reference);

        // Decoder with iterative box refinement.
        var textBias = KeyMask(mask, c.DecoderHeads, queries, Finfo);
        var hidden = target;
        var intermediate = new List<Tensor<T>>();
        var references = new List<Tensor<T>> { reference };
        foreach (var layer in DecoderLayers)
        {
            var referenceInput = _engine.TensorBroadcastTo(_engine.Reshape(reference, new[] { b, queries, 1, 4 }), new[] { b, queries, c.Levels, 4 });
            var sine = new Tensor<T>(new[] { b, queries, 2 * c.Dim });
            for (int i = 0; i < b; i++)
                for (int q = 0; q < queries; q++)
                {
                    var sineValues = SineFeatures(Enumerable.Range(0, 4).Select(k => NumOps.ToDouble(reference[i, q, k])).ToArray(), c.Dim / 2);
                    for (int d = 0; d < sineValues.Length; d++) sine[i, q, d] = NumOps.FromDouble(sineValues[d]);
                }
            var queryPosition = Mlp(ReferenceHead, sine);
            var qk = _engine.TensorAdd(hidden, queryPosition);
            hidden = Rows(layer.SelfNorm, _engine.TensorAdd(hidden, Dropout(Attend(layer.SelfAttention, qk, qk, hidden, null), c.Dropout)));
            hidden = Rows(layer.TextNorm, _engine.TensorAdd(hidden,
                Dropout(Attend(layer.TextAttention, _engine.TensorAdd(hidden, queryPosition), text, text, textBias), c.Dropout)));
            hidden = Rows(layer.ImageNorm, _engine.TensorAdd(hidden,
                Dropout(Sample(layer.ImageAttention, _engine.TensorAdd(hidden, queryPosition), vision, referenceInput, shapes, starts), c.Dropout)));
            var ffn = Rows(layer.Fc2, _engine.ReLU(Rows(layer.Fc1, hidden)));
            hidden = Rows(layer.FinalNorm, _engine.TensorAdd(hidden, Dropout(ffn, c.Dropout)));
            // The refined reference feeds the next layer as data (detached).
            var refined = _engine.Sigmoid(_engine.TensorAdd(Mlp(BoxHead, hidden), Logit(_engine, reference)));
            reference = new Tensor<T>(refined.Shape.ToArray(), new Vector<T>(refined.ToArray()));
            intermediate.Add(Rows(DecoderNorm, hidden));
            references.Add(reference);
        }

        // Heads: each layer's normalized output against the encoded text, boxes relative to the layer's input reference.
        var logits = new Tensor<T>[intermediate.Count];
        var boxes = new Tensor<T>[intermediate.Count];
        for (int layer = 0; layer < intermediate.Count; layer++)
        {
            logits[layer] = Contrast(intermediate[layer], text, mask);
            boxes[layer] = _engine.Sigmoid(_engine.TensorAdd(Mlp(BoxHead, intermediate[layer]), Logit(_engine, references[layer])));
        }
        if (Trace is not null)
            Record("intermediate", _engine.TensorStack(intermediate.ToArray(), 1));
        return new GroundingDinoOutput<T>(logits, boxes, encoderLogits, encoderBoxes, mask);
    }

    /// <summary>When set, the forward records its intermediate tensors here by name (for reference comparisons).</summary>
    internal Dictionary<string, Tensor<T>>? Trace { get; set; }

    private void Record(string name, Tensor<T> value)
    {
        if (Trace is not null) Trace[name] = value;
    }

    /// <summary>The vision-text bi-attention fusion layer: both sides layer-normalized, then each attends to the other and
    /// adds the result scaled by a learned per-channel gate (initialized at 1e-4).</summary>
    private (Tensor<T> Vision, Tensor<T> Text) Fuse(Fusion f, Tensor<T> vision, Tensor<T> text, bool[,] tokenMask)
    {
        var c = Configuration;
        int b = vision.Shape[0], nv = vision.Shape[1], nt = text.Shape[1], heads = c.EncoderHeads / 2, embed = c.EncoderFfn / 2, hd = embed / heads;
        var v = Rows(f.VisionNorm, vision);
        var t = Rows(f.TextNorm, text);
        Tensor<T> Heads(Tensor<T> x, int len) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { b, len, heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b * heads, len, hd });
        var query = Heads(_engine.TensorMultiplyScalar(Rows(f.VisionProj, v), NumOps.FromDouble(1.0 / Math.Sqrt(hd))), nv);
        var key = Heads(Rows(f.TextProj, t), nt);
        var visionValue = Heads(Rows(f.VisionValue, v), nv);
        var textValue = Heads(Rows(f.TextValue, t), nt);
        var scores = _engine.BatchMatMul(query, _engine.TensorPermute(key, new[] { 0, 2, 1 }));                       // [B·h, Nv, Nt]
        // attn − attn.max() (one global maximum), clamped to ±50000; the text side subtracts each row's maximum. Both
        // shifts are constants under the softmaxes that follow, so they are taken as data.
        double globalMax = scores.ToArray().Max(NumOps.ToDouble);
        var shifted = _engine.TensorClamp(_engine.TensorAddScalar(scores, NumOps.FromDouble(-globalMax)), NumOps.FromDouble(-50000), NumOps.FromDouble(50000));
        var transposed = _engine.TensorPermute(shifted, new[] { 0, 2, 1 });                                       // [B·h, Nt, Nv]
        var rowMax = _engine.ReduceMax(transposed, new[] { 2 }, keepDims: true, out _);
        var rowMaxData = new Tensor<T>(rowMax.Shape.ToArray(), new Vector<T>(rowMax.ToArray()));
        var textScores = _engine.TensorClamp(_engine.TensorSubtract(transposed, _engine.TensorBroadcastTo(rowMaxData, transposed._shape)),
            NumOps.FromDouble(-50000), NumOps.FromDouble(50000));
        var textAttention = _engine.Softmax(textScores, axis: 2);
        var visionAttention = _engine.Softmax(_engine.TensorAdd(shifted, KeyMask(tokenMask, heads, nv, Masked)), axis: 2);
        var visionOut = Rows(f.VisionOut, _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(visionAttention, textValue),
            new[] { b, heads, nv, hd }), new[] { 0, 2, 1, 3 }), new[] { b, nv, embed }));
        var textOut = Rows(f.TextOut, _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(textAttention, visionValue),
            new[] { b, heads, nt, hd }), new[] { 0, 2, 1, 3 }), new[] { b, nt, embed }));
        var gateV = _engine.TensorBroadcastTo(_engine.Reshape(Value(f.VisionGamma), new[] { 1, 1, c.Dim }), new[] { b, nv, c.Dim });
        var gateT = _engine.TensorBroadcastTo(_engine.Reshape(Value(f.TextGamma), new[] { 1, 1, c.Dim }), new[] { b, nt, c.Dim });
        return (_engine.TensorAdd(v, DropPath(_engine.TensorMultiply(gateV, visionOut), c.FusionDropPath)),
                _engine.TensorAdd(t, DropPath(_engine.TensorMultiply(gateT, textOut), c.FusionDropPath)));
    }

    /// <summary>The reference's <c>generate_encoder_output_proposals</c>: per token a box at its centre with side
    /// 0.05·2^level, as logits (+∞ when outside (0.01, 0.99)), and the token's features (zeroed for those) through
    /// the encoder output projection and norm.</summary>
    private (Tensor<T> Query, Tensor<T> Proposals) Proposals(Tensor<T> memory, (int H, int W)[] shapes, int[] starts)
    {
        int b = memory.Shape[0], s = memory.Shape[1], d = memory.Shape[2];
        var proposals = new Tensor<T>(new[] { b, s, 4 });
        var valid = new Tensor<T>(new[] { b, s, d });
        for (int level = 0; level < shapes.Length; level++)
        {
            var (h, w) = shapes[level];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    var box = new[] { (x + 0.5) / w, (y + 0.5) / h, 0.05 * Math.Pow(2, level), 0.05 * Math.Pow(2, level) };
                    bool ok = box.All(v => v > 0.01 && v < 0.99);
                    for (int i = 0; i < b; i++)
                    {
                        for (int k = 0; k < 4; k++)
                            proposals[i, starts[level] + y * w + x, k] = NumOps.FromDouble(ok ? Math.Log(box[k] / (1 - box[k])) : InvalidProposal);
                        if (ok) for (int k = 0; k < d; k++) valid[i, starts[level] + y * w + x, k] = NumOps.One;
                    }
                }
        }
        var query = Rows(EncoderOutputNorm, Rows(EncoderOutput, _engine.TensorMultiply(memory, valid)));
        return (query, proposals);
    }
}
