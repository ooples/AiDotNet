using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.MultiModal;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// Fun-ASR-nano's networks (Fun-ASR report §3, Figure 2, and the release's <c>FunASRNano</c> model): the SenseVoice-Small
/// audio encoder; the audio adaptor, a linear-ReLU-linear projection to the LLM's width and two Transformer encoder layers;
/// the Qwen3 decoder reading the adaptor's output inside a ChatML prompt; and the CTC decoder, five Transformer encoder
/// layers and a linear CTC head on the detached encoder output.
/// </summary>
internal sealed class FunAsrNanoNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly Lazy<QwenTokenizer> QwenText = new(() => new QwenTokenizer(Array.Empty<string>()));
    private readonly IEngine _engine;
    private readonly FunASRNanoOptions _o;

    public FunAsrNanoNetwork(IEngine engine, FunASRNanoOptions options, List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        _o = options;
        var o = options;
        if (o.VocabSize < 2) throw new ArgumentException("The vocabulary needs a text token and the end token.", nameof(options));
        if (o.CtcVocabSize < 2) throw new ArgumentException("The CTC vocabulary needs a token and the blank.", nameof(options));
        Encoder = new SanmEncoder<T>(engine, EncoderLayers, new SanmEncoderConfiguration(o.NumMels, o.LfrM, o.LfrN, o.EncoderDim,
            o.NumAttentionHeads, o.EncoderFeedForwardDim, o.NumEncoderLayers, o.NumTpEncoderLayers, o.FsmnKernelSize), random);
        Adaptor = new MinMoInputProjector<T>(engine, AdaptorLayers, o.EncoderDim, o.AdaptorDownsampleRate, o.AdaptorFeedForwardDim,
            o.LlmHiddenDim, o.NumAdaptorLayers, o.AdaptorHeads, random);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.LlmHiddenDim, Layers = o.NumLlmLayers, Heads = o.LlmHeads, KeyValueHeads = o.LlmKeyValueHeads,
            HeadDim = o.LlmHeadDim, Intermediate = o.LlmIntermediateDim, RmsEpsilon = o.LlmRmsEpsilon, RopeTheta = o.LlmRopeTheta,
            QueryKeyNorm = true, TieWordEmbeddings = true, LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
        });
        Llm.InitializeLikeReference(random);
        CtcDecoder = new MinMoInputProjector<T>(engine, CtcLayers, o.EncoderDim, 1, o.CtcFeedForwardDim, o.CtcDecoderDim,
            o.NumCtcDecoderLayers, o.AdaptorHeads, random);
        CtcOutput = new DenseLayer<T>(o.CtcVocabSize, (IActivationFunction<T>)new IdentityActivation<T>());
        double bound = 1.0 / Math.Sqrt(o.CtcDecoderDim);
        TorchParameters.Linear(engine, CtcOutput, o.CtcDecoderDim, o.CtcVocabSize,
            Enumerable.Range(0, o.CtcDecoderDim * o.CtcVocabSize).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
            Enumerable.Range(0, o.CtcVocabSize).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
        CtcLayers.Add(CtcOutput);
        layers.AddRange(EncoderLayers);
        layers.AddRange(AdaptorLayers);
        layers.AddRange(LlmLayers);
        layers.AddRange(CtcLayers);
    }

    public SanmEncoder<T> Encoder { get; }
    public MinMoInputProjector<T> Adaptor { get; }
    public LlamaNetwork<T> Llm { get; }
    public MinMoInputProjector<T> CtcDecoder { get; }
    public DenseLayer<T> CtcOutput { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> AdaptorLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();
    public List<LayerBase<T>> CtcLayers { get; } = new();

    /// <summary>The LoRA branches' layers on the LLM's query and value projections.</summary>
    public IEnumerable<LayerBase<T>> LoraLayers => Llm.Layers
        .SelectMany(l => new[] { l.QueryLora, l.ValueLora })
        .Where(a => a is not null)
        .SelectMany(a => a is null ? Array.Empty<LayerBase<T>>() : new LayerBase<T>[] { a.A, a.B });

    /// <summary>Whether the vocabulary covers Qwen3's, so the prompt is ChatML text; a smaller vocabulary has no prompt
    /// text, and its last token closes the answer.</summary>
    public bool UsesQwenVocabulary => _o.VocabSize >= QwenText.Value.VocabularySize;

    public QwenTokenizer Tokenizer => QwenText.Value;

    /// <summary>The token that closes the answer: <c>&lt;|im_end|&gt;</c>, or the last token without Qwen's vocabulary.</summary>
    public int EndToken => UsesQwenVocabulary ? Tokenizer.AddedTokenId("<|im_end|>") : _o.VocabSize - 1;

    /// <summary>The CTC blank: the CTC vocabulary's last token.</summary>
    public int Blank => _o.CtcVocabSize - 1;

    /// <summary>The audio encoder's output <c>[frames, encoder width]</c> of a 16 kHz waveform.</summary>
    public Tensor<T> Encode(Tensor<T> audio) => Encoder.Encode(audio);

    /// <summary>
    /// The instruction (the release's <c>get_prompt</c>): "语音转写" (transcribe), "语音转写成{language}" with a language,
    /// "，不进行文本规整" without inverse text normalization, closed by "："; hotwords come first under a context header.
    /// </summary>
    public static string Instruction(IReadOnlyList<string>? hotwords, string? language, bool inverseTextNormalization)
    {
        var prompt = new System.Text.StringBuilder();
        if (hotwords is not null && hotwords.Count > 0)
        {
            prompt.Append("请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n");
            prompt.Append("热词列表：[").Append(string.Join(", ", hotwords)).Append("]\n");
        }
        prompt.Append(string.IsNullOrEmpty(language) ? "语音转写" : "语音转写成" + language);
        if (!inverseTextNormalization) prompt.Append("，不进行文本规整");
        return prompt.Append('：').ToString();
    }

    /// <summary>
    /// The prompt embeddings <c>[L, LLM width]</c> (the release's <c>generate_chatml</c> and <c>data_load_speech</c>):
    /// <c>&lt;|im_start|&gt;system\nYou are a helpful assistant.&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{instruction}</c>,
    /// the adaptor's output in place of the speech, then <c>&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>. Without
    /// Qwen's vocabulary the prompt is the speech alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> encoded, string instruction)
    {
        var speech = Adaptor.Forward(encoded);
        if (!UsesQwenVocabulary) return speech;
        var head = Tokenizer.Encode($"<|im_start|>system\n{_o.SystemPrompt}<|im_end|>\n<|im_start|>user\n{instruction}");
        var tail = Tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n");
        return _engine.TensorConcatenate(new[] { TokenEmbeddings(head), speech, TokenEmbeddings(tail) }, 0);
    }

    public Tensor<T> TokenEmbeddings(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        return _engine.Reshape(Llm.Embed(grid), new[] { ids.Count, _o.LlmHiddenDim });
    }

    /// <summary>The logits <c>[1, vocabulary]</c> of the answer's first token.</summary>
    public Tensor<T> FirstLogits(Tensor<T> prompt)
    {
        int length = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(prompt, new[] { 1, length, _o.LlmHiddenDim }));
        var all = logits ?? throw new InvalidOperationException("The decoder has no head.");
        return _engine.Reshape(_engine.TensorSlice(all, new[] { 0, length - 1, 0 }, new[] { 1, 1, _o.VocabSize }), new[] { 1, _o.VocabSize });
    }

    /// <summary>The LLM's loss: the cross-entropy of the transcript and the end token after the prompt (Hugging Face's
    /// causal LM loss, as the release computes it).</summary>
    public Tensor<T> LlmLoss(Tensor<T> prompt, IReadOnlyList<int> transcript)
    {
        var target = transcript.Concat(new[] { EndToken }).ToArray();
        var input = _engine.TensorConcatenate(new[] { prompt, TokenEmbeddings(target) }, 0);
        int length = input.Shape[0], promptLength = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, length, _o.LlmHiddenDim }));
        var labels = new int[1, length];
        for (int i = 0; i < promptLength; i++) labels[0, i] = -100;
        for (int i = 0; i < target.Length; i++) labels[0, promptLength + i] = target[i];
        return Llm.CausalLmLoss(logits ?? throw new InvalidOperationException("The decoder has no head."), labels);
    }

    /// <summary>The answer by greedy decoding after the prompt, without the end token.</summary>
    public int[] Greedy(Tensor<T> prompt, int maxTokens)
    {
        var cache = new LlamaCache<T>();
        var answer = new List<int>();
        var input = prompt;
        for (int step = 0; step < maxTokens; step++)
        {
            var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], _o.LlmHiddenDim }), null, null, cache, lastLogitsOnly: true);
            var row = logits ?? throw new InvalidOperationException("The decoder has no head.");
            int best = 0;
            for (int v = 1; v < _o.VocabSize; v++) if (NumOps.GreaterThan(row.GetFlat(v), row.GetFlat(best))) best = v;
            if (best == EndToken) break;
            answer.Add(best);
            input = TokenEmbeddings(new[] { best });
        }
        return answer.ToArray();
    }

    /// <summary>The CTC decoder's log-probabilities <c>[frames, CTC vocabulary]</c>; it reads the encoder output detached,
    /// so its loss never reaches the encoder.</summary>
    public Tensor<T> CtcLogProbabilities(Tensor<T> encoded)
    {
        var detached = new Tensor<T>(encoded.Shape.ToArray(), encoded.ToVector());
        return _engine.TensorLogSoftmax(CtcOutput.Forward(CtcDecoder.Forward(detached)), axis: -1);
    }

    /// <summary>The CTC loss of labels in the CTC vocabulary.</summary>
    public Tensor<T> CtcLoss(Tensor<T> encoded, IReadOnlyList<int> labels)
    {
        var logProbabilities = CtcLogProbabilities(encoded);
        int frames = logProbabilities.Shape[0];
        var ctc = _engine.TensorCTCLoss(_engine.Reshape(logProbabilities, new[] { frames, 1, _o.CtcVocabSize }),
            new Tensor<int>(labels.ToArray(), new[] { labels.Count }), new[] { frames }, new[] { labels.Count }, Blank);
        return _engine.Reshape(ctc, new[] { 1 });
    }

    /// <summary>The CTC decoder's greedy hypothesis: the best token per frame, repeats merged, blanks dropped (the first
    /// pass whose result retrieves hotwords).</summary>
    public int[] CtcGreedy(Tensor<T> encoded)
    {
        var logProbabilities = CtcLogProbabilities(encoded);
        int frames = logProbabilities.Shape[0], vocabulary = logProbabilities.Shape[1], previous = -1;
        var tokens = new List<int>();
        for (int f = 0; f < frames; f++)
        {
            int best = 0;
            for (int v = 1; v < vocabulary; v++) if (NumOps.GreaterThan(logProbabilities[f, v], logProbabilities[f, best])) best = v;
            if (best != previous && best != Blank) tokens.Add(best);
            previous = best;
        }
        return tokens.ToArray();
    }
}
