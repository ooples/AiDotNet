using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// Seed-ASR's networks (Bai et al. 2024 §3.2–3.3, Figure 2): the LUISE Conformer audio encoder at a 40 ms frame rate, the
/// converter (four frames spliced along the features, then a linear projection, for 160 ms) and the decoder-only LLM that
/// reads the converter's output ahead of the transcript.
/// </summary>
internal sealed class SeedAsrNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _llmWidth, _vocabulary;

    public SeedAsrNetwork(IEngine engine, ConformerAsrConfiguration encoder, int spliceFrames, LlamaConfiguration llm,
        List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        if (spliceFrames < 1) throw new ArgumentOutOfRangeException(nameof(spliceFrames), "The converter splices at least one frame.");
        SpliceFrames = spliceFrames;
        _llmWidth = llm.Hidden;
        _vocabulary = llm.VocabSize;
        Encoder = new ConformerAsrNetwork<T>(engine, EncoderLayers, encoder with { Decoder = ConformerAsrDecoder.None });
        EncoderWidth = encoder.EncoderDim;
        // The converter's linear projection, with PyTorch's default initialization.
        int inputs = EncoderWidth * spliceFrames;
        Converter = new DenseLayer<T>(_llmWidth, (IActivationFunction<T>)new IdentityActivation<T>());
        double bound = 1.0 / Math.Sqrt(inputs);
        TorchParameters.Linear(engine, Converter, inputs, _llmWidth,
            Enumerable.Range(0, inputs * _llmWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
            Enumerable.Range(0, _llmWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
        ConverterLayers.Add(Converter);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(ConverterLayers);
        layers.AddRange(LlmLayers);
    }

    public ConformerAsrNetwork<T> Encoder { get; }
    public int EncoderWidth { get; }
    public int SpliceFrames { get; }
    public DenseLayer<T> Converter { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> ConverterLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    /// <summary>The token that closes the answer: the vocabulary's last.</summary>
    public int EndToken => _vocabulary - 1;

    /// <summary>The speech representation <c>[⌊frames / k⌋, LLM width]</c> of a 16 kHz waveform: LUISE's 40 ms frames,
    /// the remainder past a multiple of k discarded, each k spliced along the features, then the linear projection.</summary>
    public Tensor<T> Speech(IReadOnlyList<double> waveform, bool training)
    {
        var encoded = Encoder.Encode(Encoder.Features(waveform), training);
        int frames = encoded.Shape[0] / SpliceFrames * SpliceFrames;
        if (frames == 0) throw new ArgumentException($"The recording is shorter than {SpliceFrames} encoder frames.", nameof(waveform));
        var kept = frames == encoded.Shape[0] ? encoded : _engine.TensorSlice(encoded, new[] { 0, 0 }, new[] { frames, EncoderWidth });
        return Converter.Forward(_engine.Reshape(kept, new[] { frames / SpliceFrames, EncoderWidth * SpliceFrames }));
    }

    /// <summary>The prompt embeddings: the context's tokens (§3.4's context SFT; none for plain recognition), then the
    /// speech representation.</summary>
    public Tensor<T> Prompt(Tensor<T> speech, IReadOnlyList<int> context)
        => context.Count == 0 ? speech : _engine.TensorConcatenate(new[] { TokenEmbeddings(context), speech }, 0);

    public Tensor<T> TokenEmbeddings(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        return _engine.Reshape(Llm.Embed(grid), new[] { ids.Count, _llmWidth });
    }

    /// <summary>The logits <c>[1, vocabulary]</c> of the answer's first token.</summary>
    public Tensor<T> FirstLogits(Tensor<T> prompt)
    {
        int length = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(prompt, new[] { 1, length, _llmWidth }));
        var all = logits ?? throw new InvalidOperationException("The decoder has no head.");
        return _engine.Reshape(_engine.TensorSlice(all, new[] { 0, length - 1, 0 }, new[] { 1, 1, _vocabulary }), new[] { 1, _vocabulary });
    }

    /// <summary>The training loss: the cross-entropy over the transcript's positions and the end token only (§3.3).</summary>
    public Tensor<T> Loss(Tensor<T> prompt, IReadOnlyList<int> transcript)
    {
        var target = transcript.Concat(new[] { EndToken }).ToArray();
        var input = _engine.TensorConcatenate(new[] { prompt, TokenEmbeddings(target) }, 0);
        int length = input.Shape[0], promptLength = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, length, _llmWidth }));
        var labels = new int[1, length];
        for (int i = 0; i < promptLength; i++) labels[0, i] = -100;
        for (int i = 0; i < target.Length; i++) labels[0, promptLength + i] = target[i];
        return Llm.CausalLmLoss(logits ?? throw new InvalidOperationException("The decoder has no head."), labels);
    }

    /// <summary>The transcript by greedy search after the prompt, without the end token.</summary>
    public int[] Greedy(Tensor<T> prompt, int maxTokens)
    {
        var cache = new LlamaCache<T>();
        var answer = new List<int>();
        var input = prompt;
        for (int step = 0; step < maxTokens; step++)
        {
            var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], _llmWidth }), null, null, cache, lastLogitsOnly: true);
            var row = logits ?? throw new InvalidOperationException("The decoder has no head.");
            int best = 0;
            for (int v = 1; v < _vocabulary; v++) if (NumOps.GreaterThan(row.GetFlat(v), row.GetFlat(best))) best = v;
            if (best == EndToken) break;
            answer.Add(best);
            input = TokenEmbeddings(new[] { best });
        }
        return answer.ToArray();
    }
}
