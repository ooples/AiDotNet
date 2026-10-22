using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of CosyVoice's text-to-speech-token language model (Du et al. 2024, Table 4).</summary>
internal sealed record CosyVoiceLanguageModelConfiguration(
    int TextVocabulary, int SpeechTokens, int TextEncoderInput, WenetEncoderConfiguration TextEncoder,
    WenetEncoderConfiguration Language, int SpeakerEmbeddingDim = 192);

/// <summary>
/// CosyVoice's language model (Du et al. 2024, §2.2; the reference's <c>TransformerLM</c>): a causal WeNet text encoder
/// over the BPE tokens, then a causal Transformer over <c>[S, v, Ȳ, T, μ₁…μ_L]</c> that predicts the speech tokens and
/// the end token E (eq. 6), trained on their cross-entropy only (eq. 8).
/// </summary>
/// <remarks>
/// S and T are the two rows of a learned table, v the projected L2-normalized x-vector, Ȳ the text encodings projected
/// to the model's width, and μ the speech-token embeddings. Generation samples with the reference's repetition-aware
/// sampling (nucleus sampling, top-p 0.8 within the top 25, resampled from the full distribution when the token occurs
/// at least once in the last 10; <see cref="Sample"/>).
/// </remarks>
internal sealed class CosyVoiceLanguageModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int StartRow = 0, TaskRow = 1;
    private readonly IEngine _engine;

    public CosyVoiceLanguageModel(IEngine engine, List<LayerBase<T>> layers, CosyVoiceLanguageModelConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int width = c.Language.InputSize;
        TextEmbedding = WenetEncoderBlock<T>.Own(layers, new TiedEmbeddingLayer<T>(c.TextVocabulary, c.TextEncoderInput));
        TextEncoder = new WenetEncoder<T>(engine, layers, c.TextEncoder);
        TextAffine = WenetEncoderBlock<T>.Own(layers, new DenseLayer<T>(width));
        SequenceEmbedding = WenetEncoderBlock<T>.Own(layers, new TiedEmbeddingLayer<T>(2, width));
        Language = new WenetEncoder<T>(engine, layers, c.Language);
        Decoder = WenetEncoderBlock<T>.Own(layers, new DenseLayer<T>(c.SpeechTokens + 1));
        SpeechEmbedding = WenetEncoderBlock<T>.Own(layers, new TiedEmbeddingLayer<T>(c.SpeechTokens, width));
        SpeakerAffine = WenetEncoderBlock<T>.Own(layers, new DenseLayer<T>(width));
    }

    public CosyVoiceLanguageModelConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public WenetEncoder<T> TextEncoder { get; }
    public DenseLayer<T> TextAffine { get; }
    public TiedEmbeddingLayer<T> SequenceEmbedding { get; }
    public WenetEncoder<T> Language { get; }
    public DenseLayer<T> Decoder { get; }
    public TiedEmbeddingLayer<T> SpeechEmbedding { get; }
    public DenseLayer<T> SpeakerAffine { get; }

    /// <summary>The end-of-sequence token E: the decoder's last output.</summary>
    public int EndToken => Configuration.SpeechTokens;

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private Tensor<T> Row(TiedEmbeddingLayer<T> table, int row) =>
        _engine.TensorSlice(table.Table, new[] { row, 0 }, new[] { 1, table.Dimension });

    /// <summary>F.normalize(v, dim=1) then the speaker projection: <c>[1, width]</c>, or null without an x-vector.</summary>
    private Tensor<T>? Speaker(IReadOnlyList<double>? xvector)
    {
        if (xvector is null || xvector.Count == 0) return null;
        if (xvector.Count != Configuration.SpeakerEmbeddingDim)
            throw new ArgumentException($"Expected a {Configuration.SpeakerEmbeddingDim}-dimensional speaker embedding, got {xvector.Count}.");
        double norm = Math.Sqrt(xvector.Sum(x => x * x));
        double divisor = Math.Max(norm, 1e-12);
        var v = new Tensor<T>(new[] { 1, xvector.Count });
        for (int i = 0; i < xvector.Count; i++) v[0, i] = NumOps.FromDouble(xvector[i] / divisor);
        return SpeakerAffine.Forward(v);
    }

    // [S, v?, Ȳ, T, μ] → the language model's hidden states.
    private Tensor<T> Hidden(IReadOnlyList<int> text, IReadOnlyList<double>? xvector, IReadOnlyList<int> speech, bool training, Random random)
    {
        var encoded = TextAffine.Forward(TextEncoder.Forward(TextEmbedding.Forward(Ids(text)), training, random));
        var parts = new List<Tensor<T>> { Row(SequenceEmbedding, StartRow) };
        var speaker = Speaker(xvector);
        if (speaker is not null) parts.Add(speaker);
        parts.Add(encoded);
        parts.Add(Row(SequenceEmbedding, TaskRow));
        if (speech.Count > 0) parts.Add(SpeechEmbedding.Forward(Ids(speech)));
        return Language.Forward(_engine.TensorConcatenate(parts.ToArray(), 0), training, random);
    }

    /// <summary>
    /// The summed cross-entropy of the speech tokens and E (eq. 8 before its 1/(L + 1)) and the number of targets, for
    /// one utterance: <paramref name="text"/> BPE ids, its speech tokens and the speaker's x-vector.
    /// </summary>
    public (Tensor<T> Loss, int Targets) Loss(IReadOnlyList<int> text, IReadOnlyList<int> speech, IReadOnlyList<double>? xvector,
        bool training, Random random)
    {
        if (text.Count == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        var hidden = Hidden(text, xvector, speech, training, random);
        int prefix = hidden.Shape[0] - speech.Count - 1;                  // the position of T predicts μ₁
        var targetRows = _engine.TensorSlice(hidden, new[] { prefix, 0 }, new[] { speech.Count + 1, hidden.Shape[1] });
        var logProbabilities = _engine.TensorLogSoftmax(Decoder.Forward(targetRows), axis: -1);
        int vocabulary = Configuration.SpeechTokens + 1;
        var oneHot = new Tensor<T>(new[] { speech.Count + 1, vocabulary });
        for (int l = 0; l < speech.Count; l++) oneHot[l, speech[l]] = NumOps.One;
        oneHot[speech.Count, EndToken] = NumOps.One;
        var picked = _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, oneHot), new[] { 0, 1 }, keepDims: false);
        return (_engine.TensorNegate(picked), speech.Count + 1);
    }

    /// <summary>
    /// Generates speech tokens for <paramref name="text"/> after an optional prompt (its text ids and speech tokens),
    /// between <paramref name="minRatio"/> and <paramref name="maxRatio"/> tokens per text token (reference: 2 and 20).
    /// </summary>
    public List<int> Generate(IReadOnlyList<int> promptText, IReadOnlyList<int> text, IReadOnlyList<int> promptSpeech,
        IReadOnlyList<double>? xvector, Random random, CosyVoiceSampling sampling, double minRatio = 2, double maxRatio = 20)
    {
        var allText = promptText.Concat(text).ToList();
        int minLength = (int)(text.Count * minRatio), maxLength = (int)(text.Count * maxRatio);
        var generated = new List<int>();
        using var _ = new NoGradScope<T>();
        for (int i = 0; i < maxLength; i++)
        {
            var hidden = Hidden(allText, xvector, promptSpeech.Concat(generated).ToList(), false, random);
            var last = _engine.TensorSlice(hidden, new[] { hidden.Shape[0] - 1, 0 }, new[] { 1, hidden.Shape[1] });
            var logp = _engine.TensorLogSoftmax(Decoder.Forward(last), axis: -1);
            var scores = new double[logp.Length];
            for (int k = 0; k < scores.Length; k++) scores[k] = NumOps.ToDouble(logp[0, k]);
            if (i < minLength) scores[EndToken] = double.NegativeInfinity;
            int token = sampling.Sample(scores, generated, random);
            if (token == EndToken) break;
            generated.Add(token);
        }
        return generated;
    }

    public void InitializeLikeTorch(Random random)
    {
        var c = Configuration;
        int width = c.Language.InputSize;
        CosyVoiceInit.Embedding(random, TextEmbedding);
        TextEncoder.InitializeLikeTorch(random);
        CosyVoiceInit.Linear(_engine, random, TextAffine, c.TextEncoder.OutputSize, width);
        CosyVoiceInit.Embedding(random, SequenceEmbedding);
        Language.InitializeLikeTorch(random);
        CosyVoiceInit.Linear(_engine, random, Decoder, c.Language.OutputSize, c.SpeechTokens + 1);
        CosyVoiceInit.Embedding(random, SpeechEmbedding);
        CosyVoiceInit.Linear(_engine, random, SpeakerAffine, c.SpeakerEmbeddingDim, width);
    }

    /// <summary>Loads the reference's <c>llm.pt</c> state dict (PyTorch names, row-major).</summary>
    public void LoadTorch(IReadOnlyDictionary<string, double[]> state)
    {
        var c = Configuration;
        int width = c.Language.InputSize;
        TextEmbedding.LoadTable(state["text_embedding.weight"]);
        TextEncoder.LoadTorch(state, "text_encoder.");
        TorchParameters.Linear(_engine, TextAffine, c.TextEncoder.OutputSize, width, state["text_encoder_affine_layer.weight"], state["text_encoder_affine_layer.bias"]);
        SequenceEmbedding.LoadTable(state["llm_embedding.weight"]);
        Language.LoadTorch(state, "llm.");
        TorchParameters.Linear(_engine, Decoder, c.Language.OutputSize, c.SpeechTokens + 1, state["llm_decoder.weight"], state["llm_decoder.bias"]);
        SpeechEmbedding.LoadTable(state["speech_embedding.weight"]);
        TorchParameters.Linear(_engine, SpeakerAffine, c.SpeakerEmbeddingDim, width, state["spk_embed_affine_layer.weight"], state["spk_embed_affine_layer.bias"]);
    }
}

/// <summary>
/// CosyVoice's repetition-aware sampling (<c>ras_sampling</c>, after VALL-E 2): a nucleus draw from the top-p mass within
/// the top-k tokens; when that token already appears at least <c>window × τ_r</c> times in the last <c>window</c>
/// tokens, it is excluded and the token is drawn from the whole distribution instead.
/// </summary>
public sealed class CosyVoiceSampling
{
    /// <summary>The nucleus mass (reference: 0.8).</summary>
    public double TopP { get; init; } = 0.8;

    /// <summary>The most tokens the nucleus holds (reference: 25).</summary>
    public int TopK { get; init; } = 25;

    /// <summary>The window of recent tokens checked for repetition (reference: 10).</summary>
    public int Window { get; init; } = 10;

    /// <summary>The repetition ratio τ_r (reference: 0.1).</summary>
    public double RepetitionRatio { get; init; } = 0.1;

    /// <summary>Draws a token from log-probabilities <paramref name="scores"/>, given the tokens decoded so far.</summary>
    public int Sample(double[] scores, IReadOnlyList<int> decoded, Random random)
    {
        int top = Nucleus(scores, random);
        int repeats = decoded.Skip(Math.Max(0, decoded.Count - Window)).Count(id => id == top);
        if (repeats >= Window * RepetitionRatio)
        {
            var rest = (double[])scores.Clone();
            rest[top] = double.NegativeInfinity;
            top = Draw(Softmax(rest), random);
        }
        return top;
    }

    private int Nucleus(double[] scores, Random random)
    {
        var p = Softmax(scores);
        // torch.sort(descending=True, stable=True): ties keep index order.
        var order = Enumerable.Range(0, p.Length).OrderByDescending(i => p[i]).ThenBy(i => i).ToArray();
        var probabilities = new List<double>();
        var indices = new List<int>();
        double cumulative = 0;
        foreach (int i in order)
        {
            if (!(cumulative < TopP && probabilities.Count < TopK)) break;
            cumulative += p[i];
            probabilities.Add(p[i]);
            indices.Add(i);
        }
        return indices[Draw(probabilities.ToArray(), random)];
    }

    private static double[] Softmax(double[] scores)
    {
        double max = scores.Where(s => !double.IsNegativeInfinity(s)).DefaultIfEmpty(0).Max();
        var e = scores.Select(s => double.IsNegativeInfinity(s) ? 0.0 : Math.Exp(s - max)).ToArray();
        double sum = e.Sum();
        return e.Select(x => x / sum).ToArray();
    }

    // multinomial(1): an index with probability proportional to its weight.
    private static int Draw(double[] weights, Random random)
    {
        double total = weights.Sum(), u = random.NextDouble() * total, cumulative = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            cumulative += weights[i];
            if (u < cumulative) return i;
        }
        for (int i = weights.Length - 1; i >= 0; i--) if (weights[i] > 0) return i;
        return weights.Length - 1;
    }
}
