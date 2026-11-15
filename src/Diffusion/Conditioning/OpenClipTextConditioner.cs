using AiDotNet.Diffusion.VAE;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>The shape of an open_clip text transformer.</summary>
public sealed class OpenClipTextEncoderOptions
{
    /// <summary>The vocabulary size (49408, CLIP's BPE vocabulary).</summary>
    public int VocabularySize { get; set; } = 49408;

    /// <summary>The context length in tokens (77).</summary>
    public int ContextLength { get; set; } = 77;

    /// <summary>The transformer width (1024 for ViT-H/14).</summary>
    public int Width { get; set; } = 1024;

    /// <summary>The transformer depth (24 for ViT-H/14).</summary>
    public int Layers { get; set; } = 24;

    /// <summary>The attention head count (16 for ViT-H/14).</summary>
    public int Heads { get; set; } = 16;

    /// <summary>
    /// How many final transformer blocks the embedding skips: 1 is LDM's <c>layer="penultimate"</c> (Stable Diffusion 2
    /// and MVDream), 0 its <c>"last"</c>.
    /// </summary>
    public int SkippedFinalLayers { get; set; } = 1;

    /// <summary>open_clip ViT-H/14 (<c>laion2b_s32b_b79k</c>), the text encoder of Stable Diffusion 2 and MVDream.</summary>
    public static OpenClipTextEncoderOptions ViTH14() => new OpenClipTextEncoderOptions();
}

/// <summary>
/// The open_clip text encoder as LDM's <c>FrozenOpenCLIPEmbedder</c> runs it (Stable Diffusion 2, MVDream): token and
/// position embeddings, the causal pre-norm transformer up to the penultimate block, then <c>ln_final</c>, giving
/// <c>[B, 77, Width]</c> context tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reference: <c>ldm/modules/encoders/modules.py</c> <c>FrozenOpenCLIPEmbedder.encode_with_transformer</c> and
/// open_clip's <c>TextTransformer</c> / <c>ResidualAttentionBlock</c> (Cherti et al., 2023; Ilharco et al., open_clip).
/// Each block is <c>x + attn(ln_1(x))</c> with a causal mask, then <c>x + c_proj(GELU(c_fc(ln_2(x))))</c>.
/// </para>
/// <para>
/// Weights load under the official names (<see cref="LoadTorchWeights"/>): <c>token_embedding.weight</c>,
/// <c>positional_embedding</c>, <c>transformer.resblocks.{i}.{ln_1, attn.in_proj_weight, attn.in_proj_bias,
/// attn.out_proj, ln_2, mlp.c_fc, mlp.c_proj}</c> and <c>ln_final</c>. No weight is allocated until the encoder is
/// first used or loaded.
/// </para>
/// <para>
/// Tokens follow <c>open_clip.tokenize</c>: <c>&lt;start_of_text&gt;</c> (49406), the BPE tokens, <c>&lt;end_of_text&gt;</c>
/// (49407), zeros to 77. Encoding text needs CLIP's BPE tokenizer (pass one, for example
/// <c>AutoTokenizer.FromPretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")</c>); the empty prompt, which is the
/// unconditional embedding, needs none.
/// </para>
/// <para><b>For Beginners:</b> this turns a text prompt into the numbers the diffusion U-Net reads to know what to
/// draw.</para>
/// </remarks>
public sealed class OpenClipTextConditioner<T> : IConditioningModule<T>, IParameterSource<T>, ICloneable<OpenClipTextConditioner<T>>
{
    /// <summary>open_clip's <c>&lt;start_of_text&gt;</c> token id.</summary>
    public const int StartOfText = 49406;

    /// <summary>open_clip's <c>&lt;end_of_text&gt;</c> token id.</summary>
    public const int EndOfText = 49407;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly OpenClipTextEncoderOptions _options;
    private readonly ITokenizer? _tokenizer;
    private readonly OpenClipTextTransformer<T> _transformer;

    /// <summary>Creates the encoder; ViT-H/14 by default.</summary>
    /// <param name="options">The transformer's shape; <see cref="OpenClipTextEncoderOptions.ViTH14"/> when null.</param>
    /// <param name="tokenizer">CLIP's BPE tokenizer, needed to encode non-empty text.</param>
    public OpenClipTextConditioner(OpenClipTextEncoderOptions? options = null, ITokenizer? tokenizer = null)
    {
        _options = options ?? OpenClipTextEncoderOptions.ViTH14();
        var o = _options;
        if (o.VocabularySize <= EndOfText)
            throw new ArgumentException($"The vocabulary must hold the special tokens (ids up to {EndOfText}).", nameof(options));
        if (o.ContextLength < 2 || o.Width <= 0 || o.Layers <= 0 || o.Heads <= 0 || o.Width % o.Heads != 0)
            throw new ArgumentException("The open_clip sizes must be positive, with the width divisible by the heads.", nameof(options));
        if (o.SkippedFinalLayers < 0 || o.SkippedFinalLayers >= o.Layers)
            throw new ArgumentOutOfRangeException(nameof(options), "SkippedFinalLayers must leave at least one block.");
        _tokenizer = tokenizer;
        _transformer = new OpenClipTextTransformer<T>(o.VocabularySize, o.ContextLength, o.Width, o.Layers, o.Heads);
    }

    /// <summary>The encoder's options.</summary>
    public OpenClipTextEncoderOptions Options => _options;

    /// <summary>Whether any weight has been allocated.</summary>
    internal bool IsMaterialized => _transformer.AnyMaterialized;

    /// <inheritdoc />
    public int EmbeddingDimension => _options.Width;

    /// <inheritdoc />
    public ConditioningType ConditioningType => ConditioningType.Text;

    /// <inheritdoc />
    /// <remarks>LDM's embedder returns the token sequence only.</remarks>
    public bool ProducesPooledOutput => false;

    /// <inheritdoc />
    public int MaxSequenceLength => _options.ContextLength;

    /// <inheritdoc />
    public Tensor<T> Encode(Tensor<T> input) => EncodeText(input);

    /// <summary>
    /// The context tokens <c>[B, ContextLength, Width]</c> of token ids <c>[B, ContextLength]</c>
    /// (<c>encode_with_transformer</c>). The causal mask makes padding irrelevant, so <paramref name="attentionMask"/> is
    /// not used, as in the reference.
    /// </summary>
    public Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        var ids = tokenIds.Rank == 1 ? tokenIds.Reshape(new[] { 1, tokenIds.Shape[0] }) : tokenIds;
        if (ids.Rank != 2 || ids.Shape[1] != _options.ContextLength)
            throw new ArgumentException($"Token ids must be [B, {_options.ContextLength}].", nameof(tokenIds));
        return _transformer.Forward(ids, _options.Layers - _options.SkippedFinalLayers);
    }

    /// <inheritdoc />
    /// <remarks>Mean over the tokens; LDM's embedder has no pooled output.</remarks>
    public Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        if (sequenceEmbeddings is null) throw new ArgumentNullException(nameof(sequenceEmbeddings));
        return AiDotNetEngine.Current.ReduceMean(sequenceEmbeddings, new[] { 1 }, keepDims: false);
    }

    /// <summary>The empty prompt's context tokens, repeated (MVDream's <c>get_learned_conditioning([""])</c>).</summary>
    public Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        var texts = new string[batchSize];
        for (int i = 0; i < batchSize; i++) texts[i] = string.Empty;
        return EncodeText(TokenizeBatch(texts));
    }

    /// <inheritdoc />
    public Tensor<T> Tokenize(string text) => TokenizeBatch(new[] { text ?? string.Empty });

    /// <summary>
    /// <c>open_clip.tokenize</c>: <c>[StartOfText] + bpe(text) + [EndOfText]</c>, cut to the context length with the last
    /// token kept as <c>EndOfText</c>, zero-padded.
    /// </summary>
    public Tensor<T> TokenizeBatch(string[] texts)
    {
        if (texts is null) throw new ArgumentNullException(nameof(texts));
        int length = _options.ContextLength;
        var data = new Vector<T>(texts.Length * length);
        for (int b = 0; b < texts.Length; b++)
        {
            var tokens = new List<int> { StartOfText };
            tokens.AddRange(Bpe(texts[b] ?? string.Empty));
            tokens.Add(EndOfText);
            if (tokens.Count > length)
            {
                tokens.RemoveRange(length, tokens.Count - length);
                tokens[length - 1] = EndOfText;
            }
            for (int i = 0; i < tokens.Count; i++) data[(b * length) + i] = NumOps.FromDouble(tokens[i]);
        }
        return new Tensor<T>(new[] { texts.Length, length }, data);
    }

    // open_clip's whitespace_clean(basic_clean(text)).lower(), then the BPE ids without special tokens.
    private IReadOnlyList<int> Bpe(string text)
    {
        var cleaned = string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToLowerInvariant();
        if (cleaned.Length == 0) return Array.Empty<int>();
        if (_tokenizer is null)
            throw new InvalidOperationException(
                "Encoding text needs CLIP's BPE tokenizer: pass one to the OpenClipTextConditioner, for example " +
                "AutoTokenizer.FromPretrained(\"laion/CLIP-ViT-H-14-laion2B-s32B-b79K\").");
        var result = _tokenizer.Encode(cleaned, new EncodingOptions { AddSpecialTokens = false, Padding = false, Truncation = false });
        return result.TokenIds;
    }

    /// <summary>
    /// Loads the official open_clip weights (names in the remarks) under <paramref name="prefix"/>;
    /// <c>cond_stage_model.model.</c> in a Stable Diffusion 2 or MVDream checkpoint.
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and expected shape.</param>
    /// <param name="prefix">The state-dict prefix.</param>
    /// <returns>The names read, in order.</returns>
    public IReadOnlyCollection<string> LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "cond_stage_model.model.")
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        if (prefix is null) throw new ArgumentNullException(nameof(prefix));
        return KlModule<T>.Load(_transformer.NamedTensors(prefix.TrimEnd('.')), read);
    }

    /// <inheritdoc />
    public long ParameterCount => _transformer.ParameterCount;

    /// <inheritdoc />
    public Vector<T> GetParameters() => _transformer.GetParameters();

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters) => _transformer.SetParameters(parameters);

    /// <summary>
    /// A deep copy with storage of its own (the tokenizer, which holds no weights, is shared). Weights not yet
    /// materialized stay unmaterialized in the copy.
    /// </summary>
    public OpenClipTextConditioner<T> Clone()
    {
        var o = _options;
        var copy = new OpenClipTextConditioner<T>(new OpenClipTextEncoderOptions
        {
            VocabularySize = o.VocabularySize,
            ContextLength = o.ContextLength,
            Width = o.Width,
            Layers = o.Layers,
            Heads = o.Heads,
            SkippedFinalLayers = o.SkippedFinalLayers
        }, _tokenizer);
        if (_transformer.AnyMaterialized) copy.SetParameters(new Vector<T>(GetParameters().ToArray()));
        return copy;
    }

    /// <inheritdoc cref="Clone"/>
    public OpenClipTextConditioner<T> DeepCopy() => Clone();
}

/// <summary>open_clip's text tower without its projection: embeddings, causal resblocks, <c>ln_final</c>.</summary>
internal sealed class OpenClipTextTransformer<T> : KlModule<T>
{
    private readonly int _vocabulary;
    private readonly int _context;
    private readonly int _width;
    private readonly LazyWeight<T> _tokenEmbedding;
    private readonly LazyWeight<T> _positionalEmbedding;
    private readonly OpenClipTextBlock<T>[] _blocks;
    private readonly OpenClipLayerNorm<T> _finalNorm;

    public OpenClipTextTransformer(int vocabulary, int context, int width, int layers, int heads)
    {
        _vocabulary = vocabulary;
        _context = context;
        _width = width;
        // open_clip TextTransformer.init_parameters: token N(0, 0.02^2), positions N(0, 0.01^2); in_proj
        // N(0, W^-1), out_proj and c_proj N(0, W^-1 (2L)^-1), c_fc N(0, (2W)^-1).
        _tokenEmbedding = OpenClipInit<T>.Normal(new[] { vocabulary, width }, 0.02);
        _positionalEmbedding = OpenClipInit<T>.Normal(new[] { context, width }, 0.01);
        double attentionStd = Math.Pow(width, -0.5);
        double projectionStd = attentionStd * Math.Pow(2.0 * layers, -0.5);
        double feedForwardStd = Math.Pow(2.0 * width, -0.5);
        _blocks = new OpenClipTextBlock<T>[layers];
        for (int i = 0; i < layers; i++)
            _blocks[i] = new OpenClipTextBlock<T>(width, heads, attentionStd, projectionStd, feedForwardStd);
        _finalNorm = new OpenClipLayerNorm<T>(width);
    }

    /// <summary>Whether any weight is allocated.</summary>
    public bool AnyMaterialized
        => _tokenEmbedding.IsMaterialized || _positionalEmbedding.IsMaterialized || _finalNorm.AnyMaterialized
           || _blocks.Any(block => block.AnyMaterialized);

    /// <param name="ids">Token ids [B, context].</param>
    /// <param name="blocks">How many resblocks run (all but the skipped final ones).</param>
    public Tensor<T> Forward(Tensor<T> ids, int blocks)
    {
        int batch = ids.Shape[0];
        var table = _tokenEmbedding.Value;
        var x = new Tensor<T>(new[] { batch, _context, _width });
        var source = table.AsSpan();
        var destination = x.AsWritableSpan();
        var idValues = ids.ToArray();
        for (int t = 0; t < batch * _context; t++)
        {
            int id = checked((int)Math.Round(NumOps.ToDouble(idValues[t])));
            if (id < 0 || id >= _vocabulary)
                throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {id} is outside the vocabulary of {_vocabulary}.");
            source.Slice(id * _width, _width).CopyTo(destination.Slice(t * _width, _width));
        }
        x = Engine.TensorAdd(x, Engine.TensorBroadcastTo(
            Engine.Reshape(_positionalEmbedding.Value, new[] { 1, _context, _width }), new[] { batch, _context, _width }));
        var mask = CausalMask(_context);
        for (int i = 0; i < blocks; i++) x = _blocks[i].Forward(x, mask);
        return _finalNorm.Forward(x);
    }

    // open_clip build_causal_mask: -inf above the diagonal, zero on and below it.
    private static Tensor<T> CausalMask(int n)
    {
        var mask = new Tensor<T>(new[] { n, n });
        var span = mask.AsWritableSpan();
        T negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        for (int r = 0; r < n; r++)
            for (int c = r + 1; c < n; c++) span[(r * n) + c] = negativeInfinity;
        return mask;
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "token_embedding.weight"), _tokenEmbedding.Value);
        yield return (Join(prefix, "positional_embedding"), _positionalEmbedding.Value);
        for (int i = 0; i < _blocks.Length; i++)
            foreach (var t in _blocks[i].NamedTensors(Join(prefix, "transformer.resblocks." + i))) yield return t;
        foreach (var t in _finalNorm.NamedTensors(Join(prefix, "ln_final"))) yield return t;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _tokenEmbedding;
        yield return _positionalEmbedding;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        foreach (var block in _blocks) yield return block;
        yield return _finalNorm;
    }
}

/// <summary>open_clip's <c>ResidualAttentionBlock</c> with a causal mask and an exact-GELU MLP of width 4x.</summary>
internal sealed class OpenClipTextBlock<T> : KlModule<T>
{
    private readonly int _width;
    private readonly int _heads;
    private readonly OpenClipLayerNorm<T> _ln1;
    private readonly LazyWeight<T> _inProjWeight;
    private readonly LazyWeight<T> _inProjBias;
    private readonly OpenClipLinear<T> _outProj;
    private readonly OpenClipLayerNorm<T> _ln2;
    private readonly OpenClipLinear<T> _fc;
    private readonly OpenClipLinear<T> _proj;

    public OpenClipTextBlock(int width, int heads, double attentionStd, double projectionStd, double feedForwardStd)
    {
        _width = width;
        _heads = heads;
        _ln1 = new OpenClipLayerNorm<T>(width);
        _inProjWeight = OpenClipInit<T>.Normal(new[] { 3 * width, width }, attentionStd);
        _inProjBias = LazyWeight<T>.Constant(new[] { 3 * width }, 0.0);
        _outProj = new OpenClipLinear<T>(width, width, OpenClipInit<T>.Normal(new[] { width, width }, projectionStd), zeroBias: true);
        _ln2 = new OpenClipLayerNorm<T>(width);
        _fc = new OpenClipLinear<T>(width, 4 * width, OpenClipInit<T>.Normal(new[] { 4 * width, width }, feedForwardStd), zeroBias: false);
        _proj = new OpenClipLinear<T>(4 * width, width, OpenClipInit<T>.Normal(new[] { width, 4 * width }, projectionStd), zeroBias: false);
    }

    public bool AnyMaterialized => _inProjWeight.IsMaterialized || _inProjBias.IsMaterialized || _ln1.AnyMaterialized
        || _outProj.AnyMaterialized || _ln2.AnyMaterialized || _fc.AnyMaterialized || _proj.AnyMaterialized;

    /// <param name="x">[B, N, width].</param>
    /// <param name="mask">The additive attention mask [N, N].</param>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> mask)
    {
        int b = x.Shape[0], n = x.Shape[1], hd = _width / _heads;
        var normed = Engine.Reshape(_ln1.Forward(x), new[] { b * n, _width });
        var qkv = Engine.TensorAdd(Engine.TensorMatMul(normed, Engine.TensorTranspose(_inProjWeight.Value)),
            Engine.TensorBroadcastTo(Engine.Reshape(_inProjBias.Value, new[] { 1, 3 * _width }), new[] { b * n, 3 * _width }));
        var split = Engine.TensorPermute(Engine.Reshape(qkv, new[] { b, n, 3, _heads, hd }), new[] { 2, 0, 3, 1, 4 });
        Tensor<T> Part(int index) => Engine.Reshape(
            Engine.TensorSlice(split, new[] { index, 0, 0, 0, 0 }, new[] { 1, b, _heads, n, hd }), new[] { b * _heads, n, hd });
        var q = Engine.TensorMultiplyScalar(Part(0), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var scores = Engine.BatchMatMul(q, Engine.TensorPermute(Part(1), new[] { 0, 2, 1 }));
        scores = Engine.TensorAdd(scores, Engine.TensorBroadcastTo(Engine.Reshape(mask, new[] { 1, n, n }), new[] { b * _heads, n, n }));
        var probabilities = Engine.Softmax(scores, 2);
        var context = Engine.Reshape(Engine.TensorPermute(
            Engine.Reshape(Engine.BatchMatMul(probabilities, Part(2)), new[] { b, _heads, n, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b, n, _width });
        x = Engine.TensorAdd(x, _outProj.Forward(context));
        var hidden = ActivationHelper.ExactGelu(_fc.Forward(_ln2.Forward(x)), Engine);
        return Engine.TensorAdd(x, _proj.Forward(hidden));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var t in _ln1.NamedTensors(Join(prefix, "ln_1"))) yield return t;
        yield return (Join(prefix, "attn.in_proj_weight"), _inProjWeight.Value);
        yield return (Join(prefix, "attn.in_proj_bias"), _inProjBias.Value);
        foreach (var t in _outProj.NamedTensors(Join(prefix, "attn.out_proj"))) yield return t;
        foreach (var t in _ln2.NamedTensors(Join(prefix, "ln_2"))) yield return t;
        foreach (var t in _fc.NamedTensors(Join(prefix, "mlp.c_fc"))) yield return t;
        foreach (var t in _proj.NamedTensors(Join(prefix, "mlp.c_proj"))) yield return t;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _inProjWeight;
        yield return _inProjBias;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _ln1;
        yield return _outProj;
        yield return _ln2;
        yield return _fc;
        yield return _proj;
    }
}

/// <summary>nn.Linear over the last axis, weight [out, in], with a bias.</summary>
internal sealed class OpenClipLinear<T> : KlModule<T>
{
    private readonly int _in;
    private readonly int _out;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    public OpenClipLinear(int inFeatures, int outFeatures, LazyWeight<T> weight, bool zeroBias)
    {
        _in = inFeatures;
        _out = outFeatures;
        _weight = weight;
        // nn.MultiheadAttention zeroes out_proj's bias; nn.Linear keeps torch's default U(+-1/sqrt(in)).
        _bias = zeroBias ? LazyWeight<T>.Constant(new[] { outFeatures }, 0.0) : LazyWeight<T>.Uniform(new[] { outFeatures }, inFeatures);
    }

    public bool AnyMaterialized => _weight.IsMaterialized || _bias.IsMaterialized;

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int rows = x.Length / _in;
        var y = Engine.TensorMatMul(Engine.Reshape(x, new[] { rows, _in }), Engine.TensorTranspose(_weight.Value));
        y = Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(_bias.Value, new[] { 1, _out }), new[] { rows, _out }));
        shape[shape.Length - 1] = _out;
        return Engine.Reshape(y, shape);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _weight.Value);
        yield return (Join(prefix, "bias"), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>nn.LayerNorm over the last axis, epsilon 1e-5.</summary>
internal sealed class OpenClipLayerNorm<T> : KlModule<T>
{
    private readonly LazyWeight<T> _gamma;
    private readonly LazyWeight<T> _beta;

    public OpenClipLayerNorm(int width)
    {
        _gamma = LazyWeight<T>.Constant(new[] { width }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { width }, 0.0);
    }

    public bool AnyMaterialized => _gamma.IsMaterialized || _beta.IsMaterialized;

    public Tensor<T> Forward(Tensor<T> x) => Engine.LayerNorm(x, _gamma.Value, _beta.Value, 1e-5, out _, out _);

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _gamma.Value);
        yield return (Join(prefix, "bias"), _beta.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
        yield return _beta;
    }
}

/// <summary>Normal-distributed lazy weights.</summary>
internal static class OpenClipInit<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static LazyWeight<T> Normal(int[] shape, double std) => new(shape, (tensor, random) =>
    {
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            double normal = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
            span[i] = NumOps.FromDouble(std * normal);
        }
    });
}
