using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>
/// Canary-Qwen-2.5B's networks (NeMo <c>speechlm2</c> SALM; the release's <c>config.json</c>): canary-1b-flash's Fast
/// Conformer encoder, the identity modality adapter, the linear projection to the LLM's width, and Qwen3-1.7B, which
/// reads the projection's output in place of <c>&lt;|audioplaceholder|&gt;</c>.
/// </summary>
internal sealed class CanaryQwenNetwork<T> : ILayerAliasOwner<T>
{
    /// <inheritdoc />
    public void RebindLayerAliases(IReadOnlyList<ILayer<T>> previousLayers, IReadOnlyList<ILayer<T>> replacementLayers)
        => LayerAliasGraph.Rebind(this, previousLayers, replacementLayers);

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    // Qwen3's tokenizer (Qwen2.5's added tokens, then <tool_response>, </tool_response>, <think>, </think>) with the
    // release's audio locator tag added.
    private static readonly Lazy<QwenTokenizer> QwenText = new(() => new QwenTokenizer(
        new[] { "<tool_response>", "</tool_response>", "<think>", "</think>", AudioLocator }));
    /// <summary>The release's <c>audio_locator_tag</c>.</summary>
    public const string AudioLocator = "<|audioplaceholder|>";
    private readonly IEngine _engine;
    private readonly int _llmWidth, _vocabulary;

    public CanaryQwenNetwork(IEngine engine, ConformerAsrConfiguration encoder, LlamaConfiguration llm, List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        _llmWidth = llm.Hidden;
        _vocabulary = llm.VocabSize;
        Encoder = new ConformerAsrNetwork<T>(engine, EncoderLayers, encoder with { Decoder = ConformerAsrDecoder.None });
        EncoderWidth = encoder.EncoderDim;
        // AudioPerceptionModule.proj: nn.Linear(d_model, output_dim) with PyTorch's default initialization.
        Projection = new DenseLayer<T>(_llmWidth, (IActivationFunction<T>)new IdentityActivation<T>());
        double bound = 1.0 / Math.Sqrt(EncoderWidth);
        TorchParameters.Linear(engine, Projection, EncoderWidth, _llmWidth,
            Enumerable.Range(0, EncoderWidth * _llmWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
            Enumerable.Range(0, _llmWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
        ProjectionLayers.Add(Projection);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(ProjectionLayers);
        layers.AddRange(LlmLayers);
    }

    public ConformerAsrNetwork<T> Encoder { get; }
    public int EncoderWidth { get; }
    public DenseLayer<T> Projection { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> ProjectionLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    /// <summary>Whether the vocabulary covers Qwen3's tokenizer, so the prompt is chat text; a smaller vocabulary has no
    /// prompt text, and its last token closes the answer.</summary>
    public bool UsesQwenVocabulary => _vocabulary >= QwenText.Value.VocabularySize;

    public QwenTokenizer Tokenizer => QwenText.Value;

    /// <summary>The token that closes the answer: <c>&lt;|im_end|&gt;</c>, or the last token without Qwen's vocabulary.</summary>
    public int EndToken => UsesQwenVocabulary ? Tokenizer.AddedTokenId("<|im_end|>") : _vocabulary - 1;

    /// <summary>The projection's output <c>[frames, LLM width]</c> of a 16 kHz waveform: the encoder's 80 ms frames, the
    /// identity adapter, then the linear projection.</summary>
    public Tensor<T> Speech(IReadOnlyList<double> waveform, bool training)
        => Projection.Forward(Encoder.Encode(Encoder.Features(waveform), training));

    /// <summary>
    /// The prompt embeddings <c>[L, LLM width]</c> in NeMo's <c>qwen</c> prompt format: <c>&lt;|im_start|&gt;user\n</c>,
    /// the instruction, the speech in place of the audio locator, <c>&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>.
    /// Without Qwen's vocabulary the prompt is the speech alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> speech, string instruction)
    {
        if (!UsesQwenVocabulary) return speech;
        var head = Tokenizer.Encode($"<|im_start|>user\n{instruction} ");
        var tail = Tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n");
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

    /// <summary>The training loss: the cross-entropy over the answer and <c>&lt;|im_end|&gt;</c> only.</summary>
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
