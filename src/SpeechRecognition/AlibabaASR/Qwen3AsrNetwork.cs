using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Whisper;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>The shape of a Qwen3-ASR model: its AuT audio encoder and its Qwen3 decoder.</summary>
internal sealed record Qwen3AsrConfiguration
{
    /// <summary>Log-mel bands (128).</summary>
    public int Mels { get; init; } = 128;
    /// <summary>The encoder's width (<c>d_model</c>).</summary>
    public int EncoderWidth { get; init; } = 1024;
    public int EncoderLayers { get; init; } = 24;
    public int EncoderHeads { get; init; } = 16;
    public int EncoderFeedForward { get; init; } = 4096;
    /// <summary>The convolutional front end's channels (<c>downsample_hidden_size</c>, 480).</summary>
    public int DownsampleChannels { get; init; } = 480;
    /// <summary>Frames per convolution chunk (<c>2 · n_window</c>, 100).</summary>
    public int ChunkFrames { get; init; } = 100;
    /// <summary>Chunks per attention window (<c>n_window_infer / (2 · n_window)</c>, 8).</summary>
    public int WindowChunks { get; init; } = 8;
    /// <summary>The decoder.</summary>
    public LlamaConfiguration Llm { get; init; } = new();
}

/// <summary>
/// Qwen3-ASR's networks (Shi et al. 2026, the Qwen3-ASR technical report, and the release's <c>modeling_qwen3_asr.py</c>):
/// the AuT audio encoder and the Qwen3 decoder that reads its output in place of the audio inside a ChatML prompt.
/// </summary>
/// <remarks>
/// The encoder splits the 128-band Whisper log-mel spectrogram into 100-frame chunks; three 3×3 stride-2 convolutions with
/// GELU (480 channels) reduce each chunk eightfold in time and frequency, a bias-free linear map flattens channels and
/// frequencies to the width, and sinusoidal positions restart in every chunk (13 tokens per second of audio, 12.5 Hz).
/// Pre-norm Transformer layers (biased projections, GELU feed-forward) attend within windows of eight chunks; a final
/// layer norm and a GELU projection pair map to the decoder's width.
/// </remarks>
internal sealed class Qwen3AsrNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    // Qwen3-Omni's added tokens after Qwen's own, in id order (151665–151704): Qwen3-ASR's tokenizer.
    private static readonly Lazy<QwenTokenizer> QwenText = new(() => new QwenTokenizer(
        new[]
        {
            "<tool_response>", "</tool_response>", "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
            "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<non_speech>", "<|audio_pad|>",
        }.Concat(Enumerable.Range(1, 27).Select(i => $"<blank{i}>")).Concat(new[] { "<asr_text>" })));
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public Qwen3AsrNetwork(IEngine engine, Qwen3AsrConfiguration configuration, List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Mels != 80 && c.Mels != 128) throw new ArgumentException("The Whisper log-mel front end has 80 or 128 bands.", nameof(configuration));
        if (c.EncoderWidth % c.EncoderHeads != 0 || c.EncoderWidth % 2 != 0)
            throw new ArgumentException("The encoder width must be even and divide by its heads.", nameof(configuration));
        if (c.ChunkFrames <= 0 || c.WindowChunks <= 0) throw new ArgumentException("Chunks and windows must be positive.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { EncoderLayers.Add(layer); return layer; }
        // PyTorch's default initialization: uniform in ±1/√fan-in for weights and biases.
        DenseLayer<T> Linear(int inputs, int outputs, bool bias = true)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bias ? bound * (2 * random.NextDouble() - 1) : 0.0).ToArray());
            return layer;
        }
        Conv1 = Add(new Conv2DLayer<T>(1, c.DownsampleChannels, 3, 3, 2, 2, 1, 1));
        Conv2 = Add(new Conv2DLayer<T>(c.DownsampleChannels, c.DownsampleChannels, 3, 3, 2, 2, 1, 1));
        Conv3 = Add(new Conv2DLayer<T>(c.DownsampleChannels, c.DownsampleChannels, 3, 3, 2, 2, 1, 1));
        ConvOut = Linear(c.DownsampleChannels * ReducedBands, c.EncoderWidth, bias: false);
        for (int i = 0; i < c.EncoderLayers; i++)
            Blocks.Add(new Block(Add(new LayerNormalizationLayer<T>(c.EncoderWidth)), Linear(c.EncoderWidth, c.EncoderWidth), Linear(c.EncoderWidth, c.EncoderWidth),
                Linear(c.EncoderWidth, c.EncoderWidth), Linear(c.EncoderWidth, c.EncoderWidth), Add(new LayerNormalizationLayer<T>(c.EncoderWidth)),
                Linear(c.EncoderWidth, c.EncoderFeedForward), Linear(c.EncoderFeedForward, c.EncoderWidth)));
        PostNorm = Add(new LayerNormalizationLayer<T>(c.EncoderWidth));
        Projection1 = Linear(c.EncoderWidth, c.EncoderWidth);
        Projection2 = Linear(c.EncoderWidth, c.Llm.Hidden);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, c.Llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(LlmLayers);
    }

    public Qwen3AsrConfiguration Configuration { get; }
    public Conv2DLayer<T> Conv1 { get; }
    public Conv2DLayer<T> Conv2 { get; }
    public Conv2DLayer<T> Conv3 { get; }
    public DenseLayer<T> ConvOut { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> PostNorm { get; }
    public DenseLayer<T> Projection1 { get; }
    public DenseLayer<T> Projection2 { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    // The bands after three stride-2 convolutions: ((((m + 1)/2 + 1)/2 + 1)/2.
    private int ReducedBands => (((Configuration.Mels + 1) / 2 + 1) / 2 + 1) / 2;

    /// <summary>Encoder tokens of <paramref name="frames"/> spectrogram frames in one chunk: three halvings, rounded up.</summary>
    public static int ChunkTokens(int frames) => (((frames - 1) / 2 + 1 - 1) / 2 + 1 - 1) / 2 + 1;

    /// <summary>Encoder tokens of a spectrogram of <paramref name="frames"/> frames (the release's
    /// <c>_get_feat_extract_output_lengths</c>).</summary>
    public int Tokens(int frames)
    {
        int full = Configuration.ChunkFrames, tail = frames % full;
        return frames / full * ChunkTokens(full) + (tail > 0 ? ChunkTokens(tail) : 0);
    }

    /// <summary>Whether the vocabulary covers Qwen3-ASR's tokenizer, so the prompt is ChatML text; a smaller vocabulary
    /// has no prompt text, and its last token closes the answer.</summary>
    public bool UsesQwenVocabulary => Configuration.Llm.VocabSize >= QwenText.Value.VocabularySize;

    public QwenTokenizer Tokenizer => QwenText.Value;

    /// <summary>The token that closes the answer: <c>&lt;|im_end|&gt;</c> (the tokenizer's end of sequence), or the last
    /// token without Qwen's vocabulary.</summary>
    public int EndToken => UsesQwenVocabulary ? Tokenizer.AddedTokenId("<|im_end|>") : Configuration.Llm.VocabSize - 1;

    /// <summary><c>&lt;asr_text&gt;</c>, which separates the language from the transcript, or -1 without Qwen's vocabulary.</summary>
    public int TextMarker => UsesQwenVocabulary ? Tokenizer.AddedTokenId("<asr_text>") : -1;

    /// <summary>The 16 kHz waveform's Whisper log-mel spectrogram <c>[mels, frames]</c>.</summary>
    public Tensor<T> Spectrogram(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        var span = audio.Data.Span;
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(span[i]);
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(samples, Configuration.Mels);
        int mels = mel.GetLength(0), frames = mel.GetLength(1);
        if (frames < 1) throw new ArgumentException("The recording is shorter than one spectrogram frame.", nameof(audio));
        var tensor = new Tensor<T>(new[] { mels, frames });
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < frames; f++) tensor[m, f] = NumOps.FromDouble(mel[m, f]);
        return tensor;
    }

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    /// <summary>The encoder's output <c>[tokens, decoder width]</c> of a spectrogram <c>[mels, frames]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> spectrogram)
    {
        var c = Configuration;
        int mels = spectrogram.Shape[0], frames = spectrogram.Shape[1], full = c.ChunkFrames;
        int chunks = (frames + full - 1) / full, width = Math.Min(frames, full);
        // Chunks [N, 1, mels, width], the last zero-padded, as pad_sequence lays them out.
        var pieces = new Tensor<T>[chunks];
        var lengths = new int[chunks];
        for (int i = 0; i < chunks; i++)
        {
            lengths[i] = Math.Min(full, frames - i * full);
            var piece = _engine.TensorSlice(spectrogram, new[] { 0, i * full }, new[] { mels, lengths[i] });
            if (lengths[i] < width) piece = _engine.TensorConcatenate(new[] { piece, new Tensor<T>(new[] { mels, width - lengths[i] }) }, 1);
            pieces[i] = _engine.Reshape(piece, new[] { 1, 1, mels, width });
        }
        var x = Gelu(Conv3.Forward(Gelu(Conv2.Forward(Gelu(Conv1.Forward(_engine.TensorConcatenate(pieces, 0)))))));
        int channels = x.Shape[1], bands = x.Shape[2], steps = x.Shape[3];
        // [N, C, F, T] -> [N, T, C·F], the bias-free projection, and positions 0 … T − 1 in every chunk.
        var flat = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 3, 1, 2 }), new[] { chunks * steps, channels * bands });
        var embedded = _engine.Reshape(ConvOut.Forward(flat), new[] { chunks, steps, c.EncoderWidth });
        embedded = _engine.TensorAdd(embedded, _engine.TensorBroadcastTo(
            _engine.Reshape(Positions(steps, c.EncoderWidth), new[] { 1, steps, c.EncoderWidth }), new[] { chunks, steps, c.EncoderWidth }));
        // The valid tokens of each chunk, concatenated.
        var valid = new List<Tensor<T>>();
        var tokenCounts = new int[chunks];
        for (int i = 0; i < chunks; i++)
        {
            tokenCounts[i] = ChunkTokens(lengths[i]);
            valid.Add(_engine.Reshape(_engine.TensorSlice(embedded, new[] { i, 0, 0 }, new[] { 1, tokenCounts[i], c.EncoderWidth }),
                new[] { tokenCounts[i], c.EncoderWidth }));
        }
        var h = valid.Count == 1 ? valid[0] : _engine.TensorConcatenate(valid.ToArray(), 0);
        // Attention windows: WindowChunks times the widest chunk's tokens (cu_seqlens), the remainder last.
        int window = tokenCounts.Max() * c.WindowChunks, total = h.Shape[0];
        foreach (var block in Blocks) h = Run(block, h, window, total);
        h = PostNorm.Forward(h);
        return Projection2.Forward(Gelu(Projection1.Forward(h)));
    }

    // SinusoidsPositionEmbedding: sines then cosines at timescales 10000^(−k/(channels/2 − 1)), positions from 0.
    private static Tensor<T> Positions(int length, int channels)
    {
        int half = channels / 2;
        double increment = Math.Log(10000.0) / (half - 1);
        var table = new Tensor<T>(new[] { length, channels });
        for (int t = 0; t < length; t++)
            for (int k = 0; k < half; k++)
            {
                double scaled = t * Math.Exp(-increment * k);
                table[t, k] = NumOps.FromDouble(Math.Sin(scaled));
                table[t, half + k] = NumOps.FromDouble(Math.Cos(scaled));
            }
        return table;
    }

    private Tensor<T> Run(Block block, Tensor<T> x, int window, int total)
    {
        var c = Configuration;
        int heads = c.EncoderHeads, dk = c.EncoderWidth / heads;
        var normed = block.AttentionNorm.Forward(x);
        var q = block.Query.Forward(normed);
        var k = block.Key.Forward(normed);
        var v = block.Value.Forward(normed);
        var contexts = new List<Tensor<T>>();
        for (int start = 0; start < total; start += window)
        {
            int length = Math.Min(window, total - start);
            Tensor<T> Heads(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(_engine.TensorSlice(t, new[] { start, 0 }, new[] { length, c.EncoderWidth }),
                new[] { length, heads, dk }), new[] { 1, 0, 2 });
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(q), _engine.TensorPermute(Heads(k), new[] { 0, 2, 1 })),
                NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
            var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(v));
            contexts.Add(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, c.EncoderWidth }));
        }
        var attended = contexts.Count == 1 ? contexts[0] : _engine.TensorConcatenate(contexts.ToArray(), 0);
        x = _engine.TensorAdd(x, block.Output.Forward(attended));
        var hidden = Gelu(block.FeedForward1.Forward(block.FeedForwardNorm.Forward(x)));
        return _engine.TensorAdd(x, block.FeedForward2.Forward(hidden));
    }

    /// <summary>
    /// The prompt embeddings <c>[L, decoder width]</c> (the release's chat template): <c>&lt;|im_start|&gt;system\n{context}
    /// &lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n&lt;|audio_start|&gt;</c>, the encoder's output in place of the audio pads,
    /// <c>&lt;|audio_end|&gt;&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>, then <c>language {language}&lt;asr_text&gt;</c>
    /// when the language is forced. Without Qwen's vocabulary the prompt is the audio alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> audioEmbeddings, string context, string? language)
    {
        if (!UsesQwenVocabulary) return audioEmbeddings;
        var head = Tokenizer.Encode($"<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>");
        var tail = Tokenizer.Encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n" + (language is null ? string.Empty : $"language {language}<asr_text>"));
        return _engine.TensorConcatenate(new[] { TokenEmbeddings(head), audioEmbeddings, TokenEmbeddings(tail) }, 0);
    }

    public Tensor<T> TokenEmbeddings(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        return _engine.Reshape(Llm.Embed(grid), new[] { ids.Count, Configuration.Llm.Hidden });
    }

    /// <summary>The logits <c>[1, vocabulary]</c> of the answer's first token.</summary>
    public Tensor<T> FirstLogits(Tensor<T> prompt)
    {
        int length = prompt.Shape[0], hidden = Configuration.Llm.Hidden, vocabulary = Configuration.Llm.VocabSize;
        var (_, logits) = Llm.Forward(_engine.Reshape(prompt, new[] { 1, length, hidden }));
        var all = logits ?? throw new InvalidOperationException("The decoder has no head.");
        return _engine.Reshape(_engine.TensorSlice(all, new[] { 0, length - 1, 0 }, new[] { 1, 1, vocabulary }), new[] { 1, vocabulary });
    }

    /// <summary>The fine-tuning loss: the cross-entropy of the answer and the end token after the prompt (the release's
    /// SFT collator masks the prompt with −100).</summary>
    public Tensor<T> Loss(Tensor<T> prompt, IReadOnlyList<int> answer)
    {
        var target = answer.Concat(new[] { EndToken }).ToArray();
        var input = _engine.TensorConcatenate(new[] { prompt, TokenEmbeddings(target) }, 0);
        int length = input.Shape[0], promptLength = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, length, Configuration.Llm.Hidden }));
        var labels = new int[1, length];
        for (int i = 0; i < promptLength; i++) labels[0, i] = -100;
        for (int i = 0; i < target.Length; i++) labels[0, promptLength + i] = target[i];
        return Llm.CausalLmLoss(logits ?? throw new InvalidOperationException("The decoder has no head."), labels);
    }

    /// <summary>The answer by greedy decoding after the prompt, without the end token.</summary>
    public int[] Greedy(Tensor<T> prompt, int maxTokens)
    {
        int hidden = Configuration.Llm.Hidden, vocabulary = Configuration.Llm.VocabSize;
        var cache = new LlamaCache<T>();
        var answer = new List<int>();
        var input = prompt;
        for (int step = 0; step < maxTokens; step++)
        {
            var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], hidden }), null, null, cache, lastLogitsOnly: true);
            var row = logits ?? throw new InvalidOperationException("The decoder has no head.");
            int best = 0;
            for (int v = 1; v < vocabulary; v++) if (NumOps.GreaterThan(row.GetFlat(v), row.GetFlat(best))) best = v;
            if (best == EndToken) break;
            answer.Add(best);
            input = TokenEmbeddings(new[] { best });
        }
        return answer.ToArray();
    }
}
