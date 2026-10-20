using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>A tracked frame's outputs: the selected mask logits (low resolution and image size), the object pointer, the
/// object-score logit and the frame's memory.</summary>
internal sealed class Sam2FrameOutput<T>
{
    public required Tensor<T> LowRes { get; init; }

    public required Tensor<T> HighRes { get; init; }

    public required Tensor<T> Pointer { get; init; }

    public required double ObjectScore { get; init; }

    /// <summary>The memory <c>[1, MemoryDim, Grid, Grid]</c>.</summary>
    public Tensor<T>? Memory { get; set; }
}

/// <summary>The outputs of SAM 2's heads for one prompt: every returned mask (<c>[1, M, 4g, 4g]</c> and image size), their
/// predicted IoUs <c>[1, M]</c>, the selected mask, the object pointer and the object-score logit <c>[1, 1]</c>.</summary>
internal sealed record Sam2Heads<T>(Tensor<T> MultiLow, Tensor<T> MultiHigh, Tensor<T> Iou, Tensor<T> Low, Tensor<T> High, Tensor<T> Pointer,
    Tensor<T> ObjectScore)
{
    public double Score(INumericOperations<T> ops) => ops.ToDouble(ObjectScore[0, 0]);
}

/// <summary>A frame's prompt: points <c>[1, k, 2]</c> in the ImageSize frame with labels <c>[1, k]</c>, or a binary mask
/// <c>[1, 1, ImageSize, ImageSize]</c>.</summary>
internal sealed record Sam2Prompt<T>(Tensor<T>? Points, Tensor<T>? Labels, Tensor<T>? Mask);

/// <summary>
/// SAM 2's tracking of one object through a video (Ravi et al. 2024; facebookresearch/sam2's <c>SAM2Base.track_step</c>
/// and <c>SAM2VideoPredictor</c>). A prompted frame is decoded from its own features with the no-memory embedding (a
/// mask prompt is taken as the output, its pointer decoded from the mask); every other frame's features attend to the
/// memory bank (the prompted frames, the closest first, and the last <c>MaskMemories − 1</c> frames, each with its slot
/// encoding, plus the object pointers of the prompted frames and of the last <c>MaxObjectPointers − 1</c> frames). Each
/// output mask is then encoded into the frame's memory.
/// </summary>
internal sealed class Sam2VideoTracker<T>
{
    private const double NoObjectScore = -1024.0;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Sam2Network<T> _net;

    public Sam2VideoTracker(IEngine engine, Sam2Network<T> network)
    {
        _engine = engine;
        _net = network;
        if (network.Configuration.Memory is null) throw new ArgumentException("SAM 2 tracking needs the video modules.", nameof(network));
    }

    /// <summary>The prompted frames' outputs.</summary>
    public Dictionary<int, Sam2FrameOutput<T>> Conditioning { get; } = new();

    /// <summary>The other frames' outputs.</summary>
    public Dictionary<int, Sam2FrameOutput<T>> Tracked { get; } = new();

    /// <summary>The video's length (the pointer window is capped by it).</summary>
    public int Frames { get; set; }

    private Sam2MemoryConfiguration Memory => _net.Configuration.Memory!;

    public void Reset()
    {
        Conditioning.Clear();
        Tracked.Clear();
    }

    /// <summary>The official rule for multimask output: with at most one point (or none), on prompted and tracked frames
    /// alike.</summary>
    public static bool UseMultimask(Tensor<T>? labels) => (labels?.Shape[1] ?? 0) <= 1;

    /// <summary>
    /// SAM 2's heads on (memory-conditioned) features <c>[1, Dim, g, g]</c>: the prompt encoder and mask decoder, masks
    /// set to −1024 when the object score says the object is absent, upsampled to the image, the best by predicted IoU
    /// selected from a multimask output, and the object pointer of its mask token (the no-object pointer when absent).
    /// </summary>
    public Sam2Heads<T> Heads(Tensor<T> features, Tensor<T> highRes0, Tensor<T> highRes1, Tensor<T>? points, Tensor<T>? labels, Tensor<T>? maskInput,
        bool multimask)
    {
        var c = _net.Configuration;
        int size = c.ImageSize, side = 4 * c.Grid;
        Tensor<T>? lowMask = null;
        if (maskInput is not null)
            lowMask = maskInput.Shape[2] == side && maskInput.Shape[3] == side ? maskInput : SpatialResampling.ResizeAntialias(_engine, maskInput, side, side);
        var (sparse, dense) = _net.EncodePrompts(points, labels, null, lowMask);
        var (masks, iou, tokens, objectScore) = _net.DecodeAll(features, highRes0, highRes1, _net.DensePositional(), sparse, dense);
        var (multi, ious, firstToken) = _net.SelectOutputs(masks, iou, multimask);
        int count = multi.Shape[1];
        bool present = NumOps.ToDouble(objectScore[0, 0]) > 0;
        if (!present)
        {
            multi = new Tensor<T>(multi.Shape.ToArray());
            multi.AsWritableSpan().Fill(NumOps.FromDouble(NoObjectScore));
        }
        var multiHigh = SpatialResampling.Resize(_engine, multi, size, size);
        int best = 0;
        if (multimask)
        {
            double top = double.NegativeInfinity;
            for (int k = 0; k < count; k++)
            {
                double v = NumOps.ToDouble(ious[0, k]);
                if (v > top) { top = v; best = k; }
            }
        }
        var low = count == 1 ? multi : _engine.TensorSlice(multi, new[] { 0, best, 0, 0 }, new[] { 1, 1, side, side });
        var high = count == 1 ? multiHigh : _engine.TensorSlice(multiHigh, new[] { 0, best, 0, 0 }, new[] { 1, 1, size, size });
        var token = _engine.Reshape(_engine.TensorSlice(tokens, new[] { 0, firstToken + best, 0 }, new[] { 1, 1, c.Dim }), new[] { 1, c.Dim });
        var pointer = present ? _net.ObjectPointer(token) : _net.NoObjectPointerValue();
        return new Sam2Heads<T>(multi, multiHigh, ious, low, high, pointer, objectScore);
    }

    /// <summary>A mask prompt <c>[1, 1, S, S]</c> taken as the output (logits ±10), its pointer decoded from the raw features
    /// with the mask (downsampled by the 4×4 convolution) as the mask input, and its object score ±10 by whether it is
    /// empty.</summary>
    public Sam2Heads<T> MaskAsOutput(Tensor<T> features, Tensor<T> highRes0, Tensor<T> highRes1, Tensor<T> mask)
    {
        int size = _net.Configuration.ImageSize;
        var high = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(mask, NumOps.FromDouble(20.0)), NumOps.FromDouble(-10.0));
        var low = SpatialResampling.ResizeAntialias(_engine, high, size / 4, size / 4);
        var iou = new Tensor<T>(new[] { 1, 1 });
        iou[0, 0] = NumOps.One;
        bool present = mask.ToArray().Any(v => NumOps.ToDouble(v) > 0);
        var pointer = present
            ? Heads(features, highRes0, highRes1, null, null, _net.DownsampleMaskPrompt(mask), multimask: false).Pointer
            : _net.NoObjectPointerValue();
        var score = new Tensor<T>(new[] { 1, 1 });
        score[0, 0] = NumOps.FromDouble(present ? 10.0 : -10.0);
        return new Sam2Heads<T>(low, high, iou, low, high, pointer, score);
    }

    /// <summary>A frame's features <c>[1, Dim, g, g]</c> conditioned on the memory bank, or with the no-memory embedding on a
    /// prompted frame.</summary>
    public Tensor<T> Condition(int frame, bool prompted, Tensor<T> features, bool reverse)
    {
        if (prompted) return _net.AddNoMemory(features);
        var m = Memory;
        int p = _net.Configuration.Dim, g = _net.Configuration.Grid, md = m.MemoryDim, slots = m.MaskMemories;
        if (Conditioning.Count == 0) throw new InvalidOperationException("Tracking needs a prompted frame first.");
        var (selected, unselected) = SelectClosest(frame);
        var picked = selected.Values.Select(o => (Slot: slots - 1, Output: (Sam2FrameOutput<T>?)o)).ToList();
        for (int position = 1; position < slots; position++)
        {
            int distance = slots - position, previous = reverse ? frame + distance : frame - distance;
            picked.Add((slots - position - 1, Tracked.TryGetValue(previous, out var o) ? o : unselected.TryGetValue(previous, out var u) ? u : null));
        }
        Tensor<T> Sequence(Tensor<T> map) => _engine.TensorPermute(_engine.Reshape(map, new[] { 1, md, g * g }), new[] { 0, 2, 1 });
        var spatialPosition = Sequence(_net.SinePositional(md, g, g));
        var memories = new List<Tensor<T>>();
        var positions = new List<Tensor<T>>();
        foreach (var (slot, output) in picked)
        {
            if (output?.Memory is null) continue;
            memories.Add(Sequence(output.Memory));
            positions.Add(_engine.TensorAdd(spatialPosition, _engine.TensorBroadcastTo(_net.MemorySlotEncoding(slot), spatialPosition.Shape.ToArray())));
        }
        int spatial = memories.Count;

        // Object pointers: the prompted frames' (only those not after the frame when not training), then the last frames'.
        int window = Math.Min(Frames, m.MaxObjectPointers);
        var offsets = new List<int>();
        var pointers = new List<Tensor<T>>();
        foreach (var (t, output) in selected)
        {
            if (!_net.Training && (reverse ? t < frame : t > frame)) continue;
            offsets.Add((frame - t) * (reverse ? -1 : 1));
            pointers.Add(output.Pointer);
        }
        for (int difference = 1; difference < window; difference++)
        {
            int t = reverse ? frame + difference : frame - difference;
            if (t < 0 || t >= Frames) break;
            var output = Tracked.TryGetValue(t, out var o) ? o : unselected.TryGetValue(t, out var u) ? u : null;
            if (output is null) continue;
            offsets.Add(difference);
            pointers.Add(output.Pointer);
        }
        int pointerTokens = 0;
        if (pointers.Count > 0)
        {
            int splits = p / md;
            var stacked = _engine.Reshape(_engine.TensorConcatenate(pointers.ToArray(), 0), new[] { 1, pointers.Count * splits, md });
            double largest = window - 1;
            var encoding = _net.PointerPositional(offsets.Select(o => o / largest).ToArray());                   // [P, md]
            var repeated = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(encoding, new[] { pointers.Count, 1, md }), new[] { pointers.Count, splits, md }),
                new[] { 1, pointers.Count * splits, md });
            memories.Add(stacked);
            positions.Add(repeated);
            pointerTokens = pointers.Count * splits;
        }
        var memory = memories.Count == 1 ? memories[0] : _engine.TensorConcatenate(memories.ToArray(), 1);
        var memoryPositional = positions.Count == 1 ? positions[0] : _engine.TensorConcatenate(positions.ToArray(), 1);
        return _net.AttendMemory(features, memory, memoryPositional, spatial, pointerTokens);
    }

    /// <summary>The official selection of prompted frames for the memory: all of them, or the closest before and after the
    /// frame and then the closest others up to the limit.</summary>
    private (SortedDictionary<int, Sam2FrameOutput<T>> Selected, Dictionary<int, Sam2FrameOutput<T>> Unselected) SelectClosest(int frame)
    {
        int limit = Memory.MaxConditioningFrames;
        var selected = new SortedDictionary<int, Sam2FrameOutput<T>>();
        var unselected = new Dictionary<int, Sam2FrameOutput<T>>();
        if (limit == -1 || Conditioning.Count <= limit)
        {
            foreach (var kv in Conditioning) selected[kv.Key] = kv.Value;
            return (selected, unselected);
        }
        var before = Conditioning.Keys.Where(t => t < frame).DefaultIfEmpty(int.MinValue).Max();
        if (before != int.MinValue) selected[before] = Conditioning[before];
        var after = Conditioning.Keys.Where(t => t >= frame).DefaultIfEmpty(int.MaxValue).Min();
        if (after != int.MaxValue) selected[after] = Conditioning[after];
        foreach (var t in Conditioning.Keys.Where(t => !selected.ContainsKey(t)).OrderBy(t => Math.Abs(t - frame)).Take(limit - selected.Count).ToList())
            selected[t] = Conditioning[t];
        foreach (var kv in Conditioning.Where(kv => !selected.ContainsKey(kv.Key))) unselected[kv.Key] = kv.Value;
        return (selected, unselected);
    }

    /// <summary>
    /// One tracking step: a mask prompt taken as the output, otherwise the heads on the memory-conditioned features (with
    /// <paramref name="previousLogits"/>, a previous low-resolution prediction, as the mask input). Returns the heads and
    /// the conditioned features (the training's corrective clicks reuse them).
    /// </summary>
    public (Sam2Heads<T> Heads, Tensor<T> Features) Step(int frame, bool prompted, (Tensor<T> Features, Tensor<T> HighRes0, Tensor<T> HighRes1) frameFeatures,
        Sam2Prompt<T>? prompt, bool reverse, Tensor<T>? previousLogits = null)
    {
        var (raw, s0, s1) = frameFeatures;
        if (prompt?.Mask is { } mask) return (MaskAsOutput(raw, s0, s1, mask), raw);
        var conditioned = Condition(frame, prompted, raw, reverse);
        return (Heads(conditioned, s0, s1, prompt?.Points, prompt?.Labels, previousLogits, UseMultimask(prompt?.Labels)), conditioned);
    }

    /// <summary>Encodes a frame's output into its memory: the mask logits (binarized when they come from clicks, outside
    /// training) as probabilities scaled to the memory encoder's input, the occlusion embedding added when the object
    /// score is not positive.</summary>
    public Tensor<T> EncodeMemory(Tensor<T> features, Tensor<T> highRes, double objectScore, bool fromPoints)
    {
        var m = Memory;
        Tensor<T> probability;
        if (fromPoints && !_net.Training)
        {
            probability = new Tensor<T>(highRes.Shape.ToArray());
            var values = highRes.ToArray();
            for (int i = 0; i < values.Length; i++) probability[i] = NumOps.ToDouble(values[i]) > 0 ? NumOps.One : NumOps.Zero;
        }
        else
        {
            probability = _engine.Sigmoid(highRes);
        }
        var input = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(probability, NumOps.FromDouble(m.SigmoidScale)), NumOps.FromDouble(m.SigmoidBias));
        return _net.EncodeMemory(features, input, occluded: objectScore <= 0);
    }

    /// <summary>
    /// Segments one object through a video as the official video predictor does: each prompted frame is decoded on its own,
    /// then the object is propagated forward from the first prompted frame to the end and backward from it to the start.
    /// With <paramref name="freshReverse"/> the backward pass starts again from the prompted frames alone, without the
    /// forward pass's memories (MedSAM2's inference resets the predictor between the two directions). Returns every frame's
    /// output.
    /// </summary>
    public Sam2FrameOutput<T>[] Propagate(int frames, Func<int, (Tensor<T> Features, Tensor<T> HighRes0, Tensor<T> HighRes1)> features,
        IReadOnlyDictionary<int, Sam2Prompt<T>> prompts, bool freshReverse = false)
    {
        if (prompts.Count == 0) throw new ArgumentException("SAM 2 needs at least one prompted frame.", nameof(prompts));
        Reset();
        Frames = frames;
        var cache = new Dictionary<int, (Tensor<T>, Tensor<T>, Tensor<T>)>();
        (Tensor<T>, Tensor<T>, Tensor<T>) Features(int t) => cache.TryGetValue(t, out var f) ? f : cache[t] = features(t);
        var outputs = new Sam2FrameOutput<T>[frames];
        void Run(int t, bool prompted, Sam2Prompt<T>? prompt, bool reverse)
        {
            var frameFeatures = Features(t);
            var (heads, _) = Step(t, prompted, frameFeatures, prompt, reverse);
            var output = new Sam2FrameOutput<T> { LowRes = heads.Low, HighRes = heads.High, Pointer = heads.Pointer, ObjectScore = heads.Score(NumOps) };
            output.Memory = EncodeMemory(frameFeatures.Item1, heads.High, output.ObjectScore, prompt?.Points is not null);
            (prompted ? Conditioning : Tracked)[t] = output;
            outputs[t] = output;
        }
        void Prompted()
        {
            foreach (var kv in prompts.OrderBy(kv => kv.Key))
            {
                if (kv.Key < 0 || kv.Key >= frames) throw new ArgumentOutOfRangeException(nameof(prompts), $"Frame {kv.Key} is outside the video.");
                Run(kv.Key, true, kv.Value, reverse: false);
            }
        }
        Prompted();
        int start = prompts.Keys.Min();
        for (int t = start + 1; t < frames; t++)
            if (!Conditioning.ContainsKey(t)) Run(t, false, null, reverse: false);
        if (freshReverse && start > 0)
        {
            var forward = (Sam2FrameOutput<T>[])outputs.Clone();
            Reset();
            Prompted();
            for (int t = start + 1; t < frames; t++) outputs[t] = forward[t];
        }
        for (int t = start - 1; t >= 0; t--)
            if (!Conditioning.ContainsKey(t)) Run(t, false, null, reverse: true);
        return outputs;
    }
}
