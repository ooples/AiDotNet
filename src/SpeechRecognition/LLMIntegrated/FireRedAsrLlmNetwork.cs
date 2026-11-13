using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// FireRedASR-LLM's networks (FireRed Team 2025 §2.2, and the release's <c>fireredasr_llm.py</c> and <c>adapter.py</c>):
/// the FireRedASR-AED Conformer encoder, the adapter (frame splicing, linear with ReLU, linear) and the Qwen2 decoder that
/// reads the adapter's output in place of the <c>&lt;speech&gt;</c> token.
/// </summary>
internal sealed class FireRedAsrLlmNetwork<T> : ILayerAliasOwner<T>
{
    /// <inheritdoc />
    public void RebindLayerAliases(IReadOnlyList<ILayer<T>> previousLayers, IReadOnlyList<ILayer<T>> replacementLayers)
        => LayerAliasGraph.Rebind(this, previousLayers, replacementLayers);

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    // Qwen2-7B-Instruct's tokenizer with the release's added <speech> token (id 151646).
    private static readonly Lazy<QwenTokenizer> QwenText = new(() => QwenTokenizer.Qwen2(new[] { SpeechToken }));
    private const string SpeechToken = "<speech>";
    private readonly IEngine _engine;
    private readonly int _llmWidth, _vocabulary;

    public FireRedAsrLlmNetwork(IEngine engine, ConformerAsrConfiguration encoder, int downsampleRate, LlamaConfiguration llm,
        List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        if (downsampleRate < 1) throw new ArgumentOutOfRangeException(nameof(downsampleRate), "The adapter splices at least one frame.");
        DownsampleRate = downsampleRate;
        _llmWidth = llm.Hidden;
        _vocabulary = llm.VocabSize;
        Encoder = new ConformerAsrNetwork<T>(engine, EncoderLayers, encoder with { Decoder = ConformerAsrDecoder.None });
        EncoderWidth = encoder.EncoderDim;
        // PyTorch's default initialization: uniform in ±1/√fan-in for weights and biases.
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
            AdapterLayers.Add(layer);
            return layer;
        }
        Adapter1 = Linear(EncoderWidth * downsampleRate, _llmWidth);
        Adapter2 = Linear(_llmWidth, _llmWidth);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(AdapterLayers);
        layers.AddRange(LlmLayers);
    }

    public ConformerAsrNetwork<T> Encoder { get; }
    public int EncoderWidth { get; }
    public int DownsampleRate { get; }
    public DenseLayer<T> Adapter1 { get; }
    public DenseLayer<T> Adapter2 { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> AdapterLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    /// <summary>Whether the vocabulary covers the release tokenizer's, so the prompt is ChatML text; a smaller vocabulary
    /// has no prompt text, and its last token closes the answer.</summary>
    public bool UsesQwenVocabulary => _vocabulary >= QwenText.Value.VocabularySize;

    public QwenTokenizer Tokenizer => QwenText.Value;

    /// <summary>The token that closes the answer: <c>&lt;|im_end|&gt;</c> (the release's end of sequence), or the last
    /// token without Qwen's vocabulary.</summary>
    public int EndToken => UsesQwenVocabulary ? Tokenizer.AddedTokenId("<|im_end|>") : _vocabulary - 1;

    /// <summary>The adapter's output <c>[⌊frames / k⌋, LLM width]</c> of a 16 kHz waveform: the encoder's frames, the
    /// remainder past a multiple of k discarded, each k concatenated, then linear, ReLU, linear.</summary>
    public Tensor<T> Speech(IReadOnlyList<double> waveform, bool training)
    {
        var encoded = Encoder.Encode(Encoder.Features(waveform), training);
        int frames = encoded.Shape[0] / DownsampleRate * DownsampleRate;
        if (frames == 0) throw new ArgumentException($"The recording is shorter than {DownsampleRate} encoder frames.", nameof(waveform));
        var kept = frames == encoded.Shape[0] ? encoded : _engine.TensorSlice(encoded, new[] { 0, 0 }, new[] { frames, EncoderWidth });
        var spliced = _engine.Reshape(kept, new[] { frames / DownsampleRate, EncoderWidth * DownsampleRate });
        return Adapter2.Forward(_engine.ReLU(Adapter1.Forward(spliced)));
    }

    /// <summary>
    /// The prompt embeddings <c>[L, LLM width]</c> (the release's <c>preprocess_texts</c> template, no system message):
    /// <c>&lt;|im_start|&gt;user\n</c>, the adapter's output in place of <c>&lt;speech&gt;</c>, then
    /// <c>{instruction}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>. Without Qwen's vocabulary the prompt is the
    /// speech alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> speech, string instruction)
    {
        if (!UsesQwenVocabulary) return speech;
        var head = Tokenizer.Encode("<|im_start|>user\n");
        var tail = Tokenizer.Encode($"{instruction}<|im_end|>\n<|im_start|>assistant\n");
        return _engine.TensorConcatenate(new[] { TokenEmbeddings(head), speech, TokenEmbeddings(tail) }, 0);
    }

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

    /// <summary>The training loss: the cross-entropy over the transcript and <c>&lt;|im_end|&gt;</c> only (§2.2; the
    /// release masks the prompt with the ignore index).</summary>
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

    /// <summary>The transcript by greedy decoding after the prompt, without the end token.</summary>
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
