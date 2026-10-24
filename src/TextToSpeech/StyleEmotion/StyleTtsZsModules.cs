using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Shared pieces of StyleTTS-ZS's modules: learned rows (separator and null tokens), and sequence helpers.</summary>
internal static class StyleTtsZsOps
{
    /// <summary>A learned row <c>[1, d]</c> (a one-entry table), standard-normal initialized as <c>nn.Parameter(randn)</c>.</summary>
    public static TiedEmbeddingLayer<T> Row<T>(List<LayerBase<T>> layers, int dim, Random random)
    {
        var row = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(1, dim));
        row.Reinitialize(() => StyleTts2StyleDiffusion<T>.Gaussian(random));
        return row;
    }

    public static Tensor<T> Read<T>(TiedEmbeddingLayer<T> row) => row.Forward(new Tensor<T>(new[] { 1 }));

    /// <summary>The segments joined along time, a separator row between each pair (the &lt;SEP&gt; tokens of Fig. 2).</summary>
    public static Tensor<T> Join<T>(IEngine engine, TiedEmbeddingLayer<T>? separator, params Tensor<T>[] segments)
    {
        if (separator is null) return engine.TensorConcatenate(segments, 0);
        var sep = Read(separator);
        var parts = new List<Tensor<T>>();
        for (int i = 0; i < segments.Length; i++)
        {
            if (i > 0) parts.Add(sep);
            parts.Add(segments[i]);
        }
        return engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>The rows <c>[start, start + count)</c> of <c>[L, d]</c>.</summary>
    public static Tensor<T> Rows<T>(IEngine engine, Tensor<T> x, int start, int count) =>
        engine.TensorSlice(x, new[] { start, 0 }, new[] { count, x.Shape[1] });

    /// <summary>The row offset of segment <paramref name="index"/> after <paramref name="lengths"/> (with separators).</summary>
    public static int Offset(int[] lengths, int index, bool separators)
    {
        int offset = 0;
        for (int i = 0; i < index; i++) offset += lengths[i] + (separators ? 1 : 0);
        return offset;
    }
}

/// <summary>
/// Multi-head cross-attention as PyTorch's <c>nn.MultiheadAttention</c> computes it (packed q/k/v projections with
/// biases, scaled dot product, output projection): the prosody encoder's compression of the variable-length prosody
/// into K fixed-length rows (Table VIII).
/// </summary>
internal sealed class StyleTtsZsCrossAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTtsZsCrossAttention(IEngine engine, List<LayerBase<T>> layers, int dim, int heads)
    {
        _engine = engine;
        Dim = dim;
        Heads = heads;
        Query = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        Key = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        Value = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        Output = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
    }

    public int Dim { get; }
    public int Heads { get; }
    public DenseLayer<T> Query { get; }
    public DenseLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>The attended rows <c>[Lq, d]</c> of queries <c>[Lq, d]</c> over keys/values <c>[Lk, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> query, Tensor<T> memory)
    {
        int lq = query.Shape[0], lk = memory.Shape[0], dk = Dim / Heads;
        Tensor<T> Split(Tensor<T> t, int rows) => _engine.TensorPermute(_engine.Reshape(t, new[] { rows, Heads, dk }), new[] { 1, 0, 2 });
        var q = Split(Query.Forward(query), lq);
        var k = Split(Key.Forward(memory), lk);
        var v = Split(Value.Forward(memory), lk);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        return Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { lq, Dim }));
    }

    /// <summary>Loads <c>nn.MultiheadAttention</c>'s <c>in_proj_weight</c> / <c>in_proj_bias</c> / <c>out_proj</c>.</summary>
    public void Load(TorchReader read, string name)
    {
        int d = Dim;
        string Name(string part) => name.Length == 0 ? part : name + "." + part;
        var weight = read(Name("in_proj_weight"), new[] { 3 * d, d });
        var bias = read(Name("in_proj_bias"), new[] { 3 * d });
        for (int part = 0; part < 3; part++)
        {
            var layer = part == 0 ? Query : part == 1 ? Key : Value;
            TorchParameters.Linear(_engine, layer, d, d, weight.Skip(part * d * d).Take(d * d).ToArray(), bias.Skip(part * d).Take(d).ToArray());
        }
        StyleTtsOps.LoadLinear(_engine, Output, read, Name("out_proj"), d, d);
    }
}

/// <summary>
/// StyleTTS-ZS's joint prompt-text encoder (§III-A, Fig. 2c, Table VII): the phoneme embeddings and the prompt
/// mel-spectrogram's frames projected to the model width, joined (a separator between), through one conformer block of
/// kernel 31 and five of kernel 15; the text part projected again is the prompt-aligned text embedding, the prompt part
/// average-pooled is the global style.
/// </summary>
internal sealed class StyleTtsZsPromptTextEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTtsZsPromptTextEncoder(IEngine engine, List<LayerBase<T>> layers, int symbols, int mels, int dim, int heads, int feedForward,
        int[] kernels, double dropout, bool separators, Random random)
    {
        _engine = engine;
        Dim = dim;
        Mels = mels;
        Embedding = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(symbols, dim));
        Embedding.Reinitialize(() => StyleTts2StyleDiffusion<T>.Gaussian(random));
        PromptProjection = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        if (separators) Separator = StyleTtsZsOps.Row(layers, dim, random);
        Conformer = new StyleTtsZsConformer<T>(engine, layers, dim, heads, feedForward, kernels, dropout, random);
        TextProjection = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
    }

    public int Dim { get; }
    public int Mels { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public DenseLayer<T> PromptProjection { get; }
    public TiedEmbeddingLayer<T>? Separator { get; }
    public StyleTtsZsConformer<T> Conformer { get; }
    public DenseLayer<T> TextProjection { get; }

    /// <summary>The prompt-aligned text embeddings <c>[N, d]</c> and the global style <c>[1, d]</c> of phoneme ids and a
    /// prompt mel-spectrogram <c>[mels, T]</c>.</summary>
    public (Tensor<T> Text, Tensor<T> Style) Forward(int[] ids, Tensor<T> promptMel, bool training, Random random)
    {
        int n = ids.Length, t = promptMel.Shape[1];
        var idTensor = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++) idTensor[i] = NumOps.FromDouble(ids[i]);
        var text = Embedding.Forward(idTensor);
        var prompt = PromptProjection.Forward(_engine.TensorTranspose(promptMel));
        var h = Conformer.Forward(StyleTtsZsOps.Join(_engine, Separator, text, prompt), training, random);
        var textOut = TextProjection.Forward(StyleTtsZsOps.Rows(_engine, h, 0, n));
        var promptOut = StyleTtsZsOps.Rows(_engine, h, StyleTtsZsOps.Offset(new[] { n, t }, 1, Separator is not null), t);
        return (textOut, _engine.ReduceMean(promptOut, new[] { 0 }, keepDims: true));
    }
}

/// <summary>
/// StyleTTS-ZS's prosody encoder (§III-B, Fig. 2a, Table VIII): each frame's phoneme position embedded (the duration as
/// upsampled position embeddings), stacked with the pitch and energy, projected, through the conformer stack; K learned
/// queries cross-attend to it, giving the fixed-length time-varying style <c>[K, d]</c>. Masked frames read a learned mask
/// embedding and zero pitch and energy.
/// </summary>
internal sealed class StyleTtsZsProsodyEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTtsZsProsodyEncoder(IEngine engine, List<LayerBase<T>> layers, int maxPhonemes, int dim, int heads, int feedForward, int[] kernels,
        int styleLength, double dropout, Random random)
    {
        _engine = engine;
        Dim = dim;
        StyleLength = styleLength;
        Positions = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(maxPhonemes + 1, dim));
        Positions.Reinitialize(() => StyleTts2StyleDiffusion<T>.Gaussian(random));
        Input = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        Conformer = new StyleTtsZsConformer<T>(engine, layers, dim, heads, feedForward, kernels, dropout, random);
        Queries = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(styleLength, dim));
        Queries.Reinitialize(() => StyleTts2StyleDiffusion<T>.Gaussian(random));
        Attention = new StyleTtsZsCrossAttention<T>(engine, layers, dim, heads);
        MaxPhonemes = maxPhonemes;
    }

    public int Dim { get; }
    public int StyleLength { get; }
    public int MaxPhonemes { get; }

    /// <summary>Position embeddings: row i for phoneme i, the last row the mask embedding.</summary>
    public TiedEmbeddingLayer<T> Positions { get; }

    public DenseLayer<T> Input { get; }
    public StyleTtsZsConformer<T> Conformer { get; }
    public TiedEmbeddingLayer<T> Queries { get; }
    public StyleTtsZsCrossAttention<T> Attention { get; }

    /// <summary>The time-varying style <c>[K, d]</c> of per-frame phoneme indices (−1 where masked), pitch and energy
    /// <c>[T]</c> (zero where masked).</summary>
    public Tensor<T> Forward(int[] phonemeOfFrame, Tensor<T> pitch, Tensor<T> energy, bool training, Random random)
    {
        int frames = phonemeOfFrame.Length;
        if (pitch.Length != frames || energy.Length != frames)
            throw new ArgumentException("Pitch, energy and the upsampled durations must have one value per frame.");
        var index = new Tensor<T>(new[] { frames });
        for (int f = 0; f < frames; f++)
        {
            int i = phonemeOfFrame[f];
            if (i >= MaxPhonemes) throw new ArgumentException($"At most {MaxPhonemes} phonemes.", nameof(phonemeOfFrame));
            index[f] = NumOps.FromDouble(i < 0 ? MaxPhonemes : i);
        }
        var stacked = _engine.TensorConcatenate(new[] { Positions.Forward(index), _engine.Reshape(pitch, new[] { frames, 1 }),
            _engine.Reshape(energy, new[] { frames, 1 }) }, 1);                                                    // [T, d + 2]
        var variable = Conformer.Forward(Input.Forward(stacked), training, random);
        var positions = new Tensor<T>(new[] { StyleLength });
        for (int k = 0; k < StyleLength; k++) positions[k] = NumOps.FromDouble(k);
        return Attention.Forward(Queries.Forward(positions), variable);
    }
}

/// <summary>
/// StyleTTS-ZS's prosody decoder (§III-B, Fig. 2b, Table IX): text embeddings (per phoneme for the duration decoder,
/// upsampled to frames for the pitch-and-energy decoder) and the time-varying style joined, through six conformer blocks
/// of kernel 31; the text part projected to k outputs (1 duration, or pitch and energy).
/// </summary>
internal sealed class StyleTtsZsProsodyDecoder<T>
{
    private readonly IEngine _engine;

    public StyleTtsZsProsodyDecoder(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int feedForward, int[] kernels, int outputs,
        double dropout, bool separators, Random random)
    {
        _engine = engine;
        Outputs = outputs;
        if (separators) Separator = StyleTtsZsOps.Row(layers, dim, random);
        Conformer = new StyleTtsZsConformer<T>(engine, layers, dim, heads, feedForward, kernels, dropout, random);
        Head = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(outputs));
    }

    public int Outputs { get; }
    public TiedEmbeddingLayer<T>? Separator { get; }
    public StyleTtsZsConformer<T> Conformer { get; }
    public DenseLayer<T> Head { get; }

    /// <summary>The decoded values <c>[L, k]</c> of text rows <c>[L, d]</c> and the time-varying style <c>[K, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> text, Tensor<T> style, bool training, Random random)
    {
        var h = Conformer.Forward(StyleTtsZsOps.Join(_engine, Separator, text, style), training, random);
        return Head.Forward(StyleTtsZsOps.Rows(_engine, h, 0, text.Shape[0]));
    }
}

/// <summary>
/// StyleTTS-ZS's style denoiser (§III-C, Fig. 2d, Table X) and, with the same architecture, the distilled student: the
/// prompt mel frames projected, the text embeddings and the noisy time-varying style joined, plus a sinusoidal embedding
/// of the noise level (the guidance scale for the student) at every position, through two conformer blocks of kernel 31
/// and ten of kernel 15; the style part is the output. Classifier-free guidance replaces the prompt and text by learned
/// null rows.
/// </summary>
internal sealed class StyleTtsZsDenoiser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTtsZsDenoiser(IEngine engine, List<LayerBase<T>> layers, int mels, int dim, int heads, int feedForward, int[] kernels,
        double dropout, bool separators, Random random)
    {
        _engine = engine;
        Dim = dim;
        Prompt = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        if (separators) Separator = StyleTtsZsOps.Row(layers, dim, random);
        NullPrompt = StyleTtsZsOps.Row(layers, dim, random);
        NullText = StyleTtsZsOps.Row(layers, dim, random);
        Conformer = new StyleTtsZsConformer<T>(engine, layers, dim, heads, feedForward, kernels, dropout, random);
    }

    public int Dim { get; }
    public DenseLayer<T> Prompt { get; }
    public TiedEmbeddingLayer<T>? Separator { get; }
    public TiedEmbeddingLayer<T> NullPrompt { get; }
    public TiedEmbeddingLayer<T> NullText { get; }
    public StyleTtsZsConformer<T> Conformer { get; }

    /// <summary>The sinusoidal embedding <c>[1, d]</c> of a scalar (StyleTTS 2's <c>SinusoidalEmbedding</c>: sin then cos at
    /// frequencies exp(−k ln 10000 / (d/2 − 1))).</summary>
    public Tensor<T> Embed(double value)
    {
        int half = Dim / 2;
        var e = new Tensor<T>(new[] { 1, Dim });
        double scale = Math.Log(10000) / (half - 1);
        for (int k = 0; k < half; k++)
        {
            double a = value * Math.Exp(-k * scale);
            e[0, k] = NumOps.FromDouble(Math.Sin(a));
            e[0, half + k] = NumOps.FromDouble(Math.Cos(a));
        }
        return e;
    }

    /// <summary>The output <c>[K, d]</c> for the style input <c>[K, d]</c>, level <paramref name="level"/>, text rows
    /// <c>[N, d]</c> and prompt mel <c>[mels, T]</c>; <paramref name="unconditional"/> reads the null rows instead.</summary>
    public Tensor<T> Forward(Tensor<T> style, double level, Tensor<T> text, Tensor<T> promptMel, bool unconditional, bool training, Random random)
    {
        var prompt = unconditional ? StyleTtsZsOps.Read(NullPrompt) : Prompt.Forward(_engine.TensorTranspose(promptMel));
        var words = unconditional ? StyleTtsZsOps.Read(NullText) : text;
        var x = StyleTtsZsOps.Join(_engine, Separator, prompt, words, style);
        int length = x.Shape[0];
        x = _engine.TensorAdd(x, _engine.TensorTile(Embed(level), new[] { length, 1 }));
        var h = Conformer.Forward(x, training, random);
        return StyleTtsZsOps.Rows(_engine, h, length - style.Shape[0], style.Shape[0]);
    }

    /// <summary>The classifier-free-guided output (Eq. 4): K(∅) + ω (K(t, x′) − K(∅)); at ω = 1 just the conditional.</summary>
    public Tensor<T> Guided(Tensor<T> style, double level, Tensor<T> text, Tensor<T> promptMel, double guidance, bool training, Random random)
    {
        var conditional = Forward(style, level, text, promptMel, false, training, random);
        if (guidance == 1.0) return conditional;
        var unconditional = Forward(style, level, text, promptMel, true, training, random);
        return _engine.TensorAdd(unconditional, _engine.TensorMultiplyScalar(_engine.TensorSubtract(conditional, unconditional),
            NumOps.FromDouble(guidance)));
    }
}

/// <summary>
/// A StyleTTS-ZS multimodal discriminator (§III-D, Fig. 2e, Table XI): the judged output <c>[N, d_x]</c> and its
/// conditions <c>[T, k]</c> projected to the model width and joined, through six conformer blocks of kernel 15, and a
/// linear score per position; the blocks' outputs are its feature maps.
/// </summary>
internal sealed class StyleTtsZsMultimodalDiscriminator<T>
{
    private readonly IEngine _engine;

    public StyleTtsZsMultimodalDiscriminator(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int feedForward, int[] kernels,
        double dropout, Random random)
    {
        _engine = engine;
        Output = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        Condition = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(dim));
        Conformer = new StyleTtsZsConformer<T>(engine, layers, dim, heads, feedForward, kernels, dropout, random);
        Score = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(1));
    }

    public DenseLayer<T> Output { get; }
    public DenseLayer<T> Condition { get; }
    public StyleTtsZsConformer<T> Conformer { get; }
    public DenseLayer<T> Score { get; }

    /// <summary>The scores <c>[1, N + T]</c> and the feature map of every block.</summary>
    public (Tensor<T> Score, List<Tensor<T>> Features) Forward(Tensor<T> output, Tensor<T> conditions, bool training, Random random)
    {
        var x = _engine.TensorConcatenate(new[] { Output.Forward(output), Condition.Forward(conditions) }, 0);
        var features = new List<Tensor<T>>();
        x = Conformer.Forward(x, training, random, features);
        var score = Score.Forward(x);
        return (_engine.Reshape(score, new[] { 1, score.Shape[0] }), features);
    }
}
