using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of GLaMM; the defaults are the released model's (CLIP ViT-L/14-336, Vicuna-7B with LoRA, SAM ViT-H).</summary>
internal sealed record GlammConfiguration
{
    public ClipVisionConfiguration Clip { get; init; } = new();

    /// <summary>Vicuna-7B with GLaMM's seven added tokens and LoRA (rank 8, alpha 16, dropout 0.05) on q_proj and v_proj.</summary>
    public LlamaConfiguration Llm { get; init; } = new() { VocabSize = 32007, LoraRank = 8, LoraAlpha = 16, LoraDropout = 0.05 };

    public GlammRegionConfiguration Region { get; init; } = new();

    public SamConfiguration Sam { get; init; } = SamConfiguration.For(Enums.SAMModelSize.ViTHuge);

    /// <summary>The [SEG] projection's output width, SAM's prompt width (256).</summary>
    public int OutDim { get; init; } = 256;

    /// <summary>The CLIP hidden state LLaVA reads, counted from the end (-2: the second-to-last layer).</summary>
    public int SelectLayer { get; init; } = -2;

    /// <summary>The CLIP levels the region encoder reads (4).</summary>
    public int RegionLevels { get; init; } = 4;
}

/// <summary>
/// GLaMM (Rasheed et al. 2024) as the official code (mbzuai-oryx/groundingLMM) builds it: LLaVA-1.5 (CLIP ViT-L/14-336,
/// a two-layer GELU projector and Vicuna) with a region encoder whose features replace the <c>&lt;bbox&gt;</c> tokens, and
/// SAM, whose mask decoder takes each <c>[SEG]</c> token's final hidden state, projected to SAM's prompt width, as its
/// prompt.
/// </summary>
/// <remarks>
/// <para>The multimodal sequence is the official <c>prepare_inputs_labels_for_multimodal</c> with
/// <c>mm_use_im_start_end</c>: the image placeholder replaced by the projected CLIP patch features (the class token
/// dropped), its label positions ignored, and every <c>&lt;bbox&gt;</c> position's embedding replaced by the next region
/// feature. The region encoder reads CLIP hidden states SelectLayer, SelectLayer − 3, … (the last RegionLevels, finest
/// read first). A <c>[SEG]</c> at sequence position t (in the expanded sequence) is read from the hidden state at t − 1,
/// the state that predicts it, as the official mask built from <c>input_ids[:, 1:]</c> with 575 leading zeros does.</para>
/// </remarks>
internal sealed class GlammNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public GlammNetwork(IEngine engine, List<LayerBase<T>> layers, GlammConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Region.Embed != c.Clip.Hidden || c.Region.OutDims != c.Llm.Hidden)
            throw new ArgumentException("The region encoder reads the CLIP width and writes the language model's.", nameof(configuration));
        if (c.Region.Grid * c.Clip.PatchSize != c.Clip.ImageSize)
            throw new ArgumentException("The region encoder's grid is the CLIP patch grid.", nameof(configuration));
        DenseLayer<T> Linear(List<LayerBase<T>> group, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            group.Add(layer);
            return layer;
        }
        Clip = new ClipVisionNetwork<T>(engine, ClipLayers, c.Clip);
        Projector1 = Linear(ProjectorLayers, c.Llm.Hidden);
        Projector2 = Linear(ProjectorLayers, c.Llm.Hidden);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, c.Llm);
        Region = new GlammRegionEncoder<T>(engine, RegionLayers, c.Region);
        Sam = new SamNetwork<T>(engine, SamLayers, c.Sam);
        TextFc1 = Linear(TextFcLayers, c.Llm.Hidden);
        TextFc2 = Linear(TextFcLayers, c.OutDim);
        foreach (var group in new[] { ClipLayers, ProjectorLayers, LlmLayers, RegionLayers, SamLayers, TextFcLayers }) layers.AddRange(group);
    }

    public GlammConfiguration Configuration { get; }

    public ClipVisionNetwork<T> Clip { get; }

    public DenseLayer<T> Projector1 { get; }

    public DenseLayer<T> Projector2 { get; }

    public LlamaNetwork<T> Llm { get; }

    public GlammRegionEncoder<T> Region { get; }

    public SamNetwork<T> Sam { get; }

    public DenseLayer<T> TextFc1 { get; }

    public DenseLayer<T> TextFc2 { get; }

    public List<LayerBase<T>> ClipLayers { get; } = new();

    public List<LayerBase<T>> ProjectorLayers { get; } = new();

    public List<LayerBase<T>> LlmLayers { get; } = new();

    public List<LayerBase<T>> RegionLayers { get; } = new();

    public List<LayerBase<T>> SamLayers { get; } = new();

    public List<LayerBase<T>> TextFcLayers { get; } = new();

    /// <summary>
    /// The layers the official training updates (<c>set_trainable_modules</c> with LoRA): the token embeddings and head,
    /// the LoRA adapters, the region encoder, the [SEG] projection and SAM's mask decoder. CLIP, the projector, the
    /// language model's other weights, and SAM's image and prompt encoders stay frozen.
    /// </summary>
    public IEnumerable<LayerBase<T>> TrainableLayers()
    {
        yield return Llm.Embeddings;
        yield return Llm.LmHead;
        foreach (var layer in Llm.Layers)
            foreach (var lora in new[] { layer.QueryLora, layer.ValueLora })
                if (lora is not null)
                {
                    yield return lora.A;
                    yield return lora.B;
                }
        if (Configuration.Llm.LoraRank <= 0)   // no LoRA: the official code trains every decoder layer
            foreach (var layer in LlmLayers) yield return layer;
        foreach (var layer in RegionLayers) yield return layer;
        foreach (var layer in TextFcLayers) yield return layer;
        int decoderStart = SamLayers.IndexOf(Sam.IouToken);
        for (int i = decoderStart; i < SamLayers.Count; i++) yield return SamLayers[i];
    }

    public bool Training
    {
        get => Llm.Training;
        set => Llm.Training = value;
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    /// <summary>
    /// The global image encoding of CLIP-normalized images <c>[B, 3, 336, 336]</c>: the projected patch features
    /// <c>[B, Patches, LlmHidden]</c> and the CLIP levels for the region encoder (finest read first, class token dropped).
    /// </summary>
    public (Tensor<T> Features, List<Tensor<T>> RegionLevels) EncodeGlobal(Tensor<T> images)
    {
        var c = Configuration;
        var states = Clip.Forward(images);
        int select = states.Count + c.SelectLayer, n = Clip.Patches, b = images.Shape[0], h = c.Clip.Hidden;
        Tensor<T> Patches(Tensor<T> state) => _engine.TensorSlice(state, new[] { 0, 1, 0 }, new[] { b, n, h });
        var features = Rows(Projector2, ActivationHelper.ExactGelu(Rows(Projector1, Patches(states[select])), _engine));
        // hidden_states[select::-3][::-1][-levels:]
        var picked = new List<int>();
        for (int i = select; i >= 0; i -= 3) picked.Add(i);
        picked.Reverse();
        var levels = picked.Skip(Math.Max(0, picked.Count - c.RegionLevels)).Select(i => Patches(states[i])).ToList();
        return (features, levels);
    }

    /// <summary>
    /// One sample's multimodal input embeddings <c>[1, L', Hidden]</c> and its labels and full ids (the image positions
    /// carry the placeholder's index): the official sequence with <c>mm_use_im_start_end</c>.
    /// </summary>
    /// <param name="ids">The token ids with one image placeholder (-200).</param>
    /// <param name="labels">Labels aligned with <paramref name="ids"/>, or null.</param>
    /// <param name="imageFeatures">The image's projected patch features <c>[Patches, Hidden]</c>.</param>
    /// <param name="regions">The region features <c>[boxes, Hidden]</c> for the <c>&lt;bbox&gt;</c> tokens, or null.</param>
    /// <param name="bboxId">The <c>&lt;bbox&gt;</c> token id.</param>
    public (Tensor<T> Embeddings, int[] FullIds, int[]? Labels) Multimodal(IReadOnlyList<int> ids, IReadOnlyList<int>? labels,
        Tensor<T> imageFeatures, Tensor<T>? regions, int bboxId)
    {
        int image = -1;
        for (int i = 0; i < ids.Count; i++)
            if (ids[i] == GlammTokenizer.ImageTokenIndex)
            {
                if (image >= 0) throw new ArgumentException("GLaMM reads one image per sequence.", nameof(ids));
                image = i;
            }
        if (image < 0) throw new ArgumentException("The sequence has no image placeholder.", nameof(ids));
        if (image + 1 >= ids.Count) throw new ArgumentException("The image placeholder must be followed by <im_end>.", nameof(ids));
        int patches = imageFeatures.Shape[0];
        int[] Slice(IReadOnlyList<int> x, int start, int count) => Enumerable.Range(start, count).Select(i => x[i]).ToArray();
        Tensor<T> Embed(int[] tokens) => _engine.Reshape(Llm.Embed(To2D(tokens)), new[] { tokens.Length, Configuration.Llm.Hidden });
        var before = Slice(ids, 0, image);
        var end = Slice(ids, image + 1, 1);
        var after = Slice(ids, image + 2, ids.Count - image - 2);
        var parts = new List<Tensor<T>>();
        if (before.Length > 0) parts.Add(Embed(before));
        parts.Add(imageFeatures);
        parts.Add(Embed(end));
        if (after.Length > 0) parts.Add(Embed(after));
        var sequence = _engine.TensorConcatenate(parts.ToArray(), 0);                                    // [L', H]
        var fullIds = before.Concat(Enumerable.Repeat(image, patches)).Concat(end).Concat(after).ToArray();
        int[]? fullLabels = null;
        if (labels is not null)
        {
            if (labels.Count != ids.Count) throw new ArgumentException("One label per id.", nameof(labels));
            fullLabels = Slice(labels, 0, image).Concat(Enumerable.Repeat(GlammTokenizer.IgnoreIndex, patches))
                .Concat(Slice(labels, image + 1, 1)).Concat(Slice(labels, image + 2, labels.Count - image - 2)).ToArray();
        }
        if (regions is not null)
        {
            // reg_embeds[reg_mask] = reg_feat (truncated to the <bbox> count); embeds * ~mask + reg_embeds.
            var positions = Enumerable.Range(0, fullIds.Length).Where(i => fullIds[i] == bboxId).ToArray();
            int used = Math.Min(positions.Length, regions.Shape[0]);
            if (used > 0)
            {
                var keep = new Tensor<T>(new[] { fullIds.Length, 1 });
                for (int i = 0; i < fullIds.Length; i++) keep[i, 0] = NumOps.One;
                var index = new Tensor<int>(new[] { fullIds.Length });
                var scatter = new Tensor<T>(new[] { fullIds.Length, 1 });
                for (int k = 0; k < used; k++)
                {
                    keep[positions[k], 0] = NumOps.Zero;
                    scatter[positions[k], 0] = NumOps.One;
                    index[positions[k]] = k;
                }
                var spread = _engine.TensorMultiply(_engine.TensorIndexSelect(regions, index, 0),
                    _engine.TensorBroadcastTo(scatter, new[] { fullIds.Length, regions.Shape[1] }));
                sequence = _engine.TensorAdd(_engine.TensorMultiply(sequence, _engine.TensorBroadcastTo(keep, sequence._shape)), spread);
            }
        }
        return (_engine.Reshape(sequence, new[] { 1, fullIds.Length, Configuration.Llm.Hidden }), fullIds, fullLabels);
    }

    private static int[,] To2D(int[] tokens)
    {
        var m = new int[1, tokens.Length];
        for (int i = 0; i < tokens.Length; i++) m[0, i] = tokens[i];
        return m;
    }

    /// <summary>The [SEG] embeddings <c>[n, OutDim]</c>: the projected hidden states before each [SEG] token of the
    /// original ids (positions shifted by the image expansion, as the official 575-zero mask).</summary>
    public Tensor<T>? SegEmbeddings(Tensor<T> hidden, IReadOnlyList<int> ids, int segId, int patches)
    {
        var rows = new List<int>();
        int image = Enumerable.Range(0, ids.Count).First(i => ids[i] == GlammTokenizer.ImageTokenIndex);
        for (int i = 1; i < ids.Count; i++)
            if (ids[i] == segId)
            {
                // mask = [zeros(patches - 1), input_ids[:, 1:] == seg]: original token i sits at i + patches - 1 when it
                // follows the image, so the mask marks i - 1 + (patches - 1).
                int position = i - 1 + patches - 1;
                if (i <= image) throw new ArgumentException("A [SEG] before the image is not read by the official code.", nameof(ids));
                rows.Add(position);
            }
        if (rows.Count == 0) return null;
        int length = hidden.Shape[1], h = hidden.Shape[2];
        var flat = _engine.Reshape(hidden, new[] { length, h });
        var picked = _engine.TensorIndexSelect(flat, new Tensor<int>(new[] { rows.Count }, new Vector<int>(rows.ToArray())), 0);
        return Rows(TextFc2, _engine.ReLU(Rows(TextFc1, picked)));
    }

    /// <summary>
    /// The mask logits <c>[n, originalHeight, originalWidth]</c> for [SEG] embeddings <c>[n, OutDim]</c> on one SAM image
    /// embedding: each embedding the only sparse prompt, no-mask dense prompt, single-mask output, then
    /// <c>postprocess_masks</c>.
    /// </summary>
    public Tensor<T> DecodeMasks(Tensor<T> segEmbeddings, Tensor<T> samEmbedding, int inputHeight, int inputWidth, int originalHeight, int originalWidth)
    {
        int n = segEmbeddings.Shape[0], p = Configuration.Sam.PromptDim;
        var sparse = _engine.Reshape(segEmbeddings, new[] { n, 1, p });
        var (_, dense) = Sam.EncodePrompts(null, null, null, null);
        dense = _engine.TensorBroadcastTo(dense, new[] { n, dense.Shape[1], dense.Shape[2], dense.Shape[3] });
        var (logits, _) = Sam.DecodeMasks(samEmbedding, Sam.DensePositional(), sparse, dense, false);
        var masks = Sam.Postprocess(logits, inputHeight, inputWidth, originalHeight, originalWidth);
        return _engine.Reshape(masks, new[] { n, originalHeight, originalWidth });
    }

    /// <summary>
    /// GLaMM's mask losses (LISA's): per mask the mean sigmoid cross-entropy and the DICE loss
    /// (1 − (2Σ p/1000·g + 1e-6) / (Σ p/1000 + Σ g/1000 + 1e-6)), each summed over masks with the official normalizations.
    /// </summary>
    public (Tensor<T> Bce, Tensor<T> Dice) MaskLosses(Tensor<T> logits, Tensor<T> targets)
    {
        int n = logits.Shape[0], pixels = logits.Shape[1] * logits.Shape[2];
        var flatLogits = _engine.Reshape(logits, new[] { n, pixels });
        var flatTargets = _engine.Reshape(targets, new[] { n, pixels });
        var oneMinus = _engine.ScalarMinusTensor(NumOps.One, flatTargets);
        var crossEntropy = _engine.TensorAdd(_engine.TensorMultiply(flatTargets, _engine.Softplus(_engine.TensorNegate(flatLogits))),
            _engine.TensorMultiply(oneMinus, _engine.Softplus(flatLogits)));
        double count = n;
        // compute_sigmoid_cross_entropy(pred, gt, n) * n, then / (num_masks + 1e-8).
        var bce = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.ReduceMean(crossEntropy, new[] { 1 }, false), null),
            NumOps.FromDouble(1.0 / (count + 1e-8) * count / (count + 1e-8)));
        var p = _engine.TensorMultiplyScalar(_engine.Sigmoid(flatLogits), NumOps.FromDouble(1.0 / 1000));
        var g = _engine.TensorMultiplyScalar(flatTargets, NumOps.FromDouble(1.0 / 1000));
        // The official intersection scales the probabilities, not the targets: 2 Σ (p / 1000 · g).
        var intersection = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(p, flatTargets), new[] { 1 }, false), NumOps.FromDouble(2));
        var union = _engine.TensorAdd(_engine.ReduceSum(p, new[] { 1 }, false), _engine.ReduceSum(g, new[] { 1 }, false));
        var ratio = _engine.TensorDivide(_engine.TensorAddScalar(intersection, NumOps.FromDouble(1e-6)), _engine.TensorAddScalar(union, NumOps.FromDouble(1e-6)));
        var dice = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.ScalarMinusTensor(NumOps.One, ratio), null),
            NumOps.FromDouble(1.0 / (count + 1e-8) * count / (count + 1e-8)));
        return (bce, dice);
    }

    /// <summary>Loads GLaMM's state dict (the merged released checkpoint's names: <c>model.vision_tower.*</c>,
    /// <c>model.mm_projector.*</c>, <c>model.layers.*</c>, <c>model.region_encoder.*</c>,
    /// <c>model.grounding_encoder.*</c>, <c>model.text_hidden_fcs.0.*</c>, <c>lm_head.weight</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        Clip.LoadTorchWeights(read, "model.vision_tower.vision_tower.vision_model.");
        TorchParameters.Linear(_engine, Projector1, c.Clip.Hidden, c.Llm.Hidden, read("model.mm_projector.0.weight", new[] { c.Llm.Hidden, c.Clip.Hidden }),
            read("model.mm_projector.0.bias", new[] { c.Llm.Hidden }));
        TorchParameters.Linear(_engine, Projector2, c.Llm.Hidden, c.Llm.Hidden, read("model.mm_projector.2.weight", new[] { c.Llm.Hidden, c.Llm.Hidden }),
            read("model.mm_projector.2.bias", new[] { c.Llm.Hidden }));
        Llm.LoadTorchWeights(read, "model.", "lm_head.weight");
        Region.LoadTorchWeights(read, "model.region_encoder.");
        Sam.LoadTorchWeights((name, shape) => read("model.grounding_encoder." + name, shape));
        TorchParameters.Linear(_engine, TextFc1, c.Llm.Hidden, c.Llm.Hidden, read("model.text_hidden_fcs.0.0.weight", new[] { c.Llm.Hidden, c.Llm.Hidden }),
            read("model.text_hidden_fcs.0.0.bias", new[] { c.Llm.Hidden }));
        TorchParameters.Linear(_engine, TextFc2, c.Llm.Hidden, c.OutDim, read("model.text_hidden_fcs.0.2.weight", new[] { c.OutDim, c.Llm.Hidden }),
            read("model.text_hidden_fcs.0.2.bias", new[] { c.OutDim }));
    }

    /// <summary>
    /// A fresh GLaMM: each part with its own reference initialization, and the projector and [SEG] projection with
    /// PyTorch's linear default (uniform in ±1/√fan_in).
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        Clip.InitializeLikeReference(random);
        Llm.InitializeLikeReference(random);
        Region.InitializeLikeReference(random);
        Sam.InitializeLikeReference(random);
        var c = Configuration;
        double[] Uniform(int count, int fanIn) => Enumerable.Range(0, count).Select(_ => (2 * random.NextDouble() - 1) / Math.Sqrt(fanIn)).ToArray();
        void Default(DenseLayer<T> layer, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, Uniform(inputs * outputs, inputs), Uniform(outputs, inputs));
        Default(Projector1, c.Clip.Hidden, c.Llm.Hidden);
        Default(Projector2, c.Llm.Hidden, c.Llm.Hidden);
        Default(TextFc1, c.Llm.Hidden, c.Llm.Hidden);
        Default(TextFc2, c.Llm.Hidden, c.OutDim);
    }
}
