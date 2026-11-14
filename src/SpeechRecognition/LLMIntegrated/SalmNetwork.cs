using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// SALM's networks (Chen et al. 2023 §3.1, and NeMo's <c>modular_speechllm</c> <c>speechllm_perception.py</c>): the Fast
/// Conformer audio encoder, the modality adapter (Conformer layers with their own 4x subsampling), the projection to the
/// LLM's width and the Megatron GPT decoder that reads the speech ahead of the text prompt.
/// </summary>
internal sealed class SalmNetwork<T> : ILayerAliasOwner<T>
{
    /// <inheritdoc />
    public void RebindLayerAliases(IReadOnlyList<ILayer<T>> previousLayers, IReadOnlyList<ILayer<T>> replacementLayers)
        => LayerAliasGraph.Rebind(this, previousLayers, replacementLayers);

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    // Megatron GPT-2B-001's SentencePiece BPE model (256,000 pieces).
    private static readonly Lazy<SentencePieceModel> MegatronText = new(LoadTokenizer);
    private const string TokenizerResource = "AiDotNet.Tokenization.megatron-gpt-256k.model.gz";
    private readonly IEngine _engine;
    private readonly int _llmWidth, _vocabulary;

    public SalmNetwork(IEngine engine, ConformerAsrConfiguration encoder, ConformerAsrConfiguration adapter, LlamaConfiguration llm,
        List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        _llmWidth = llm.Hidden;
        _vocabulary = llm.VocabSize;
        Encoder = new ConformerAsrNetwork<T>(engine, EncoderLayers, encoder with { Decoder = ConformerAsrDecoder.None });
        // NeMo's ConformerEncoder as the modality adapter: its "features" are the encoder's frames.
        Adapter = new ConformerAsrNetwork<T>(engine, AdapterLayers, adapter with { NumMels = encoder.EncoderDim, Decoder = ConformerAsrDecoder.None });
        AdapterWidth = adapter.EncoderDim;
        // speechllm_perception.proj: nn.Linear(adapter width, LLM width) with PyTorch's default initialization.
        Projection = new DenseLayer<T>(_llmWidth, (IActivationFunction<T>)new IdentityActivation<T>());
        double bound = 1.0 / Math.Sqrt(AdapterWidth);
        TorchParameters.Linear(engine, Projection, AdapterWidth, _llmWidth,
            Enumerable.Range(0, AdapterWidth * _llmWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
            Enumerable.Range(0, _llmWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
        AdapterLayers.Add(Projection);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(AdapterLayers);
        layers.AddRange(LlmLayers);
    }

    public ConformerAsrNetwork<T> Encoder { get; }
    public ConformerAsrNetwork<T> Adapter { get; }
    public int AdapterWidth { get; }
    public DenseLayer<T> Projection { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    /// <summary>The modality adapter's layers and the projection.</summary>
    public List<LayerBase<T>> AdapterLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    /// <summary>Whether the vocabulary covers Megatron's, so the prompt is text; a smaller vocabulary has no prompt text,
    /// and its last token closes the answer.</summary>
    public bool UsesMegatronVocabulary => _vocabulary >= MegatronText.Value.Count;

    public SentencePieceModel Tokenizer => MegatronText.Value;

    /// <summary>The token that closes the answer: <c>&lt;/s&gt;</c> (the dataset's <c>add_eos</c>), or the last token
    /// without Megatron's vocabulary.</summary>
    public int EndToken => UsesMegatronVocabulary ? Tokenizer.EosId : _vocabulary - 1;

    private static SentencePieceModel LoadTokenizer()
    {
        using var stream = typeof(SalmNetwork<T>).Assembly.GetManifestResourceStream(TokenizerResource)
            ?? throw new InvalidDataException($"The embedded resource {TokenizerResource} is missing.");
        return SentencePieceModel.FromGzip(stream);
    }

    /// <summary>The speech prompt <c>[L, LLM width]</c> of a 16 kHz waveform: the encoder's 80 ms frames, the adapter's
    /// 4x subsampling to 320 ms (§3.1), then the projection to the LLM's width.</summary>
    public Tensor<T> Speech(IReadOnlyList<double> waveform, bool training)
    {
        var encoded = Encoder.Encode(Encoder.Features(waveform), training);
        return Projection.Forward(Adapter.Encode(encoded, training));
    }

    /// <summary>
    /// The prompt embeddings <c>[L, LLM width]</c>: the speech prompt prepended to the text instruction (§3.1, Figure 1;
    /// NeMo's <c>_concat_features</c>) in the release's template <c>Q: {input}\nA: {output}</c>, without a beginning of
    /// sequence token (<c>add_bos: false</c>). Without Megatron's vocabulary the prompt is the speech alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> speech, string instruction)
    {
        if (!UsesMegatronVocabulary) return speech;
        return _engine.TensorConcatenate(new[] { speech, TokenEmbeddings(PromptTokens(instruction)) }, 0);
    }

    /// <summary>The text after the speech: <c>Q: {instruction}\nA: </c>'s pieces.</summary>
    public int[] PromptTokens(string instruction) => Tokenizer.Encode($"Q: {instruction}\nA: ");

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

    /// <summary>The training loss: the cross-entropy over the answer and <c>&lt;/s&gt;</c> only (the release's
    /// <c>answer_only_loss</c>).</summary>
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

    /// <summary>The transcript by greedy decoding after the prompt (the paper's default), without the end token.</summary>
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
