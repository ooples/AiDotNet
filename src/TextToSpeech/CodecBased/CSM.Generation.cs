using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class CSM<T>
{
    // ---------------------------------------------------------------- the codec

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <inheritdoc />
    /// <remarks>Mimi's codes of a 24 kHz recording, <c>[frames, 32]</c> (12.5 frames a second).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = Codec.Encode(Doubles(audio));
        var tokens = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int k = 0; k < codes.GetLength(1); k++) tokens[f, k] = Ops.FromDouble(codes[f, k]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>Mimi's audio of codes <c>[frames, 32]</c>, without the watermark synthesis adds.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = CodesOf(tokens, _options.MimiCodebookSize);
        return Wave(codes.GetLength(0) == 0 ? Array.Empty<double>() : Codec.Decode(codes));
    }

    private int[,] CodesOf(Tensor<T> tokens, int size)
    {
        int books = _options.AudioCodebooks;
        if (tokens.Rank != 2 || tokens.Shape[1] != books) throw new ArgumentException($"CSM's codes are [frames, {books}].", nameof(tokens));
        var codes = new int[tokens.Shape[0], books];
        for (int f = 0; f < tokens.Shape[0]; f++)
            for (int k = 0; k < books; k++)
            {
                codes[f, k] = (int)Math.Round(Ops.ToDouble(tokens[f, k]));
                if (codes[f, k] < 0 || codes[f, k] >= size) throw new ArgumentException($"Codes lie in [0, {size}).", nameof(tokens));
            }
        return codes;
    }

    /// <summary>The watermark model, read from <see cref="CSMOptions.WatermarkModelPath"/> on first use.</summary>
    internal SilentCipherNetwork<T> Watermarker
    {
        get
        {
            if (_watermarker is not null) return _watermarker;
            var path = _options.WatermarkModelPath;
            if (string.IsNullOrWhiteSpace(path))
                throw new InvalidOperationException("CSM watermarks its output as the release does: set CSMOptions.WatermarkModelPath to SilentCipher's "
                    + "44_1_khz/73999_iteration directory (sony/silentcipher), or set CSMOptions.Watermark to false.");
            var network = new SilentCipherNetwork<T>(Engine, new SilentCipherConfiguration(), new List<LayerBase<T>>());
            network.LoadCheckpointDirectory(path!);
            return _watermarker = network;
        }
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's Llama 3 token ids (synthesis builds the turn from their text).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenizer.Encode(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    private string TextOf(Tensor<T> ids) =>
        Tokenizer.Decode(ids.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))), skipSpecial: true);

    /// <inheritdoc />
    /// <remarks>Speaks <paramref name="text"/> as <see cref="TtsModelBase{T}.Voice"/>'s speaker (0 without a voice). A voice
    /// with a recording (<see cref="TtsVoice{T}.ReferenceAudio"/>, 24 kHz) and its transcript
    /// (<see cref="TtsVoice{T}.ReferenceTokens"/>) is one turn of context by that speaker; see
    /// <see cref="Generate"/> for a whole conversation.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var voice = Voice;
        var context = new List<CsmSegment<T>>();
        if (voice?.ReferenceAudio is not null || voice?.ReferenceTokens is not null)
        {
            if (voice.ReferenceAudio is null || voice.ReferenceTokens is null)
                throw new InvalidOperationException("A CSM voice is a turn of context: give both its recording (ReferenceAudio) and its transcript (ReferenceTokens).");
            context.Add(new CsmSegment<T> { Speaker = voice.SpeakerId, Text = TextOf(voice.ReferenceTokens), Audio = voice.ReferenceAudio });
        }
        return Generate(text, voice?.SpeakerId ?? 0, context);
    }

    /// <summary>
    /// Speaks <paramref name="text"/> as <paramref name="speaker"/>'s next turn after <paramref name="context"/>
    /// (<c>Generator.generate</c>): each earlier turn's text and audio frames, then this turn's text, then frames sampled
    /// until a frame of zeros or <see cref="CSMOptions.MaxAudioLengthMs"/>, decoded by Mimi and watermarked. Empty when the
    /// first frame sampled is the end.
    /// </summary>
    public Tensor<T> Generate(string text, int speaker, IReadOnlyList<CsmSegment<T>> context)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (!HasPaperLayers) throw NotNative();
        var frames = new List<CsmFrame>();
        foreach (var segment in context)
        {
            if (segment is null) throw new ArgumentException("A context turn is null.", nameof(context));
            frames.AddRange(TextFrames(segment.Text, segment.Speaker));
            frames.AddRange(AudioFrames(Codec.Encode(Doubles(segment.Audio))));
        }
        frames.AddRange(TextFrames(text, speaker));
        var sampled = GenerateFrames(frames);
        if (sampled.Count == 0) return Wave(Array.Empty<double>());
        var codes = new int[sampled.Count, _options.AudioCodebooks];
        for (int f = 0; f < sampled.Count; f++)
            for (int k = 0; k < _options.AudioCodebooks; k++) codes[f, k] = sampled[f][k];
        var audio = Codec.Decode(codes);
        if (_options.Watermark)
            audio = SilentCipherWatermark.Apply(Engine, Watermarker, audio, _options.SampleRate, _options.WatermarkKey, _options.WatermarkSdr);
        return Wave(audio);
    }

    /// <summary>The most frames generated: <see cref="CSMOptions.MaxAudioLengthMs"/> over a frame's duration (80 ms).</summary>
    internal int MaxGenerationFrames => (int)(_options.MaxAudioLengthMs / (1000.0 * HopSize / _options.SampleRate));

    /// <summary>
    /// The release's frame loop (<c>Model.generate_frame</c>): the backbone reads the prompt (then each new frame) with its
    /// cache; its last state gives codebook 0; the decoder, started afresh, reads that state and codebook 0's embedding and
    /// gives codebook 1, then reads each code's embedding to give the next. Stops at a frame of zeros, which is not kept.
    /// </summary>
    internal List<int[]> GenerateFrames(IReadOnlyList<CsmFrame> prompt)
    {
        var o = _options;
        int maxFrames = MaxGenerationFrames;
        if (prompt.Count >= o.MaxSequenceLength - maxFrames)
            throw new ArgumentException($"Inputs too long, must be below max_seq_len - max_generation_len: {o.MaxSequenceLength - maxFrames}.", nameof(prompt));
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var samples = new List<int[]>();
            var cache = new LlamaCache<T>();
            var hidden = Backbone.Forward(EmbedFrames(prompt), null, null, cache, lastLogitsOnly: false).Hidden;
            int h = o.HiddenDim, books = o.AudioCodebooks;
            for (int step = 0; step < maxFrames; step++)
            {
                var last = Engine.TensorSlice(hidden, new[] { 0, hidden.Shape[1] - 1, 0 }, new[] { 1, 1, h });
                var codes = new int[books];
                codes[0] = Sample(Codebook0Logits(Engine.Reshape(last, new[] { 1, h })), random);
                var decoderCache = new LlamaCache<T>();
                var input = Engine.TensorConcatenate(new[] { last, Engine.Reshape(AudioEmbeddings(0, new[] { codes[0] }), new[] { 1, 1, h }) }, 1);
                var state = DecoderHidden(input, decoderCache);
                for (int k = 1; k < books; k++)
                {
                    var row = Engine.Reshape(Engine.TensorSlice(state, new[] { 0, state.Shape[1] - 1, 0 }, new[] { 1, 1, o.DecoderHidden }), new[] { 1, o.DecoderHidden });
                    codes[k] = Sample(CodebookLogits(k, row), random);
                    if (k + 1 < books) state = DecoderHidden(Engine.Reshape(AudioEmbeddings(k, new[] { codes[k] }), new[] { 1, 1, h }), decoderCache);
                }
                if (codes.All(c => c == 0)) break;
                samples.Add(codes);
                hidden = Backbone.Forward(EmbedFrames(new[] { new CsmFrame(0, codes) }), null, null, cache, lastLogitsOnly: false).Hidden;
            }
            return samples;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>Sees each logit vector generation samples from (for comparing against the release).</summary>
    internal Action<double[]>? SampledLogits { get; set; }

    /// <summary>
    /// The release's <c>sample_topk</c>: logits over the temperature, everything below the k-th largest removed, then a
    /// draw from the softmax (the argmax of probabilities over exponential draws, as <c>_multinomial_sample_one_no_sync</c>).
    /// </summary>
    internal int Sample(Tensor<T> logits, Random random)
    {
        var o = _options;
        var values = logits.ToArray();
        SampledLogits?.Invoke(values.Select(v => Ops.ToDouble(v)).ToArray());
        var scores = new double[values.Length];
        for (int i = 0; i < scores.Length; i++) scores[i] = Ops.ToDouble(values[i]) / o.Temperature;
        if (o.ConstrainToCodebook)
            for (int i = o.MimiCodebookSize; i < scores.Length; i++) scores[i] = double.NegativeInfinity;
        double kth = scores.OrderByDescending(v => v).ElementAt(Math.Min(o.TopK, scores.Length) - 1);
        double max = scores.Max(), sum = 0;
        var probabilities = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++)
        {
            probabilities[i] = scores[i] < kth || double.IsNegativeInfinity(scores[i]) ? 0 : Math.Exp(scores[i] - max);
            sum += probabilities[i];
        }
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int i = 0; i < probabilities.Length; i++)
        {
            double q = -Math.Log(1.0 - random.NextDouble());
            double value = probabilities[i] / sum / q;
            if (value > bestValue)
            {
                bestValue = value;
                best = i;
            }
        }
        return best;
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's token ids (<see cref="PreprocessText"/>); the waveform
    /// <see cref="Synthesize"/> makes of their text.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        if (input.ToArray().Any(v => NumOps.ToDouble(v) < 0 || NumOps.ToDouble(v) >= _options.VocabSize))
            throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        return Synthesize(TextOf(input));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads the release's weights (<c>sesame/csm-1b</c>): its <c>ckpt.pt</c> or a <c>.safetensors</c> of the same state dict
    /// (<c>models.Model</c>'s names: torchtune's <c>backbone.*</c> and <c>decoder.*</c>, <c>text_embeddings</c>,
    /// <c>audio_embeddings</c>, <c>projection</c>, <c>codebook0_head</c>, <c>audio_head</c>), or the same model in
    /// transformers' <c>CsmForConditionalGeneration</c> names, or a directory holding one (<c>model.safetensors</c>, else
    /// <c>ckpt.pt</c>).
    /// </summary>
    public void LoadReferenceWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (Directory.Exists(path))
        {
            var file = new[] { "model.safetensors", "ckpt.pt" }.Select(f => Path.Combine(path, f)).FirstOrDefault(File.Exists)
                ?? throw new FileNotFoundException($"No model.safetensors or ckpt.pt in {path}.");
            path = file;
        }
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        var used = LoadTorchWeights(name => state.TryGetValue(name, out var v) ? v : null);
        // Causal masks and key-value caches are buffers of a model whose caches were set up; they are recomputed. A
        // transformers checkpoint also carries its Mimi (codec_model.*), in transformers' own names: the codec is loaded
        // from the release's Mimi by LoadCodecWeights.
        var unused = state.Keys.Where(k => !used.Contains(k) && !k.EndsWith("causal_mask", StringComparison.Ordinal) && !k.Contains(".kv_cache.")
            && !k.StartsWith("codec_model.", StringComparison.Ordinal)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    /// <summary>
    /// Loads a state dict with the release's names. torchtune's rotary embedding rotates adjacent pairs of each head's
    /// dimensions where the Llama decoder here rotates its two halves; the query and key rows are reordered to match, which
    /// leaves every attention score unchanged. Returns the names read.
    /// </summary>
    internal HashSet<string> LoadTorchWeights(Func<string, double[]?> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Need(string name, int count)
        {
            var values = read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
            used.Add(name);
            return values;
        }
        void LoadDecoder(LlamaNetwork<T> network, string prefix)
        {
            var c = network.Configuration;
            int hd = c.ResolvedHeadDim;
            network.LoadTorchWeights((name, shape) =>
            {
                int count = shape.Aggregate(1, (a, s) => a * s);
                if (name == "norm.weight") return Need($"{prefix}norm.scale", count);
                // layers.{i}.<module>
                int dot = name.IndexOf('.', "layers.".Length);
                string layer = name.Substring(0, dot), module = name.Substring(dot + 1);
                string tune = module switch
                {
                    "input_layernorm.weight" => "sa_norm.scale",
                    "self_attn.q_proj.weight" => "attn.q_proj.weight",
                    "self_attn.k_proj.weight" => "attn.k_proj.weight",
                    "self_attn.v_proj.weight" => "attn.v_proj.weight",
                    "self_attn.o_proj.weight" => "attn.output_proj.weight",
                    "post_attention_layernorm.weight" => "mlp_norm.scale",
                    "mlp.gate_proj.weight" => "mlp.w1.weight",
                    "mlp.up_proj.weight" => "mlp.w3.weight",
                    "mlp.down_proj.weight" => "mlp.w2.weight",
                    _ => throw new InvalidOperationException($"No release name for '{name}'."),
                };
                var values = Need($"{prefix}{layer}.{tune}", count);
                return module is "self_attn.q_proj.weight" or "self_attn.k_proj.weight" ? HalvesFromPairs(values, shape[0] / hd, hd, shape[1]) : values;
            }, prefix: "", headName: "");
        }
        var o = _options;
        // Hugging Face's CsmForConditionalGeneration names (transformers' port): Llama names, the rotary halves layout.
        bool port = read("backbone_model.norm.weight") is not null;
        if (port)
        {
            Backbone.LoadTorchWeights((name, shape) => Need(name, shape.Aggregate(1, (a, s) => a * s)), prefix: "backbone_model.", headName: "");
            AudioDecoder.LoadTorchWeights((name, shape) => Need(name, shape.Aggregate(1, (a, s) => a * s)), prefix: "depth_decoder.model.", headName: "");
        }
        else
        {
            LoadDecoder(Backbone, "backbone.");
            LoadDecoder(AudioDecoder, "decoder.");
        }
        string Name(string release, string transformers) => port ? transformers : release;
        TorchParameters.Linear(_textEmbeddings!, 1, o.VocabSize * o.HiddenDim, Need(Name("text_embeddings.weight", "embed_text_tokens.weight"), o.VocabSize * o.HiddenDim));
        int audioCount = o.AudioVocabSize * o.AudioCodebooks * o.HiddenDim;
        var audioTable = Need(Name("audio_embeddings.weight", "backbone_model.embed_tokens.embed_audio_tokens.weight"), audioCount);
        TorchParameters.Linear(_audioEmbeddings!, 1, audioCount, audioTable);
        if (port)
        {
            // The decoder's code table is tied to the backbone's (tie_codebooks_embeddings); a saved copy must agree.
            var tied = read("depth_decoder.model.embed_tokens.weight");
            if (tied is not null)
            {
                if (tied.Length != audioCount || !tied.SequenceEqual(audioTable))
                    throw new InvalidDataException("The checkpoint's decoder code table differs from the backbone's; CSM ties them.");
                used.Add("depth_decoder.model.embed_tokens.weight");
            }
        }
        TorchParameters.Linear(_projection!, o.HiddenDim, o.DecoderHidden,
            Need(Name("projection.weight", "depth_decoder.model.inputs_embeds_projector.weight"), o.DecoderHidden * o.HiddenDim));
        TorchParameters.Linear(_codebook0Head!, o.HiddenDim, o.AudioVocabSize, Need(Name("codebook0_head.weight", "lm_head.weight"), o.AudioVocabSize * o.HiddenDim));
        // audio_head [codebooks − 1, decoder hidden, vocabulary] is applied as h @ W; a linear layer takes Wᵀ.
        int per = o.DecoderHidden * o.AudioVocabSize;
        var heads = Need(Name("audio_head", "depth_decoder.codebooks_head.weight"), (o.AudioCodebooks - 1) * per);
        for (int k = 0; k < _audioHeads.Length; k++)
        {
            var transposed = new double[per];
            for (int i = 0; i < o.DecoderHidden; i++)
                for (int j = 0; j < o.AudioVocabSize; j++) transposed[j * o.DecoderHidden + i] = heads[k * per + i * o.AudioVocabSize + j];
            TorchParameters.Linear(_audioHeads[k], o.DecoderHidden, o.AudioVocabSize, transposed);
        }
        return used;
    }

    // Reorders each head's rows from torchtune's interleaved pairs (0, 1), (2, 3), … to the halves layout: the even dimensions
    // first, then the odd ones.
    private static double[] HalvesFromPairs(double[] weight, int heads, int headDim, int inputs)
    {
        var result = new double[weight.Length];
        int half = headDim / 2;
        for (int head = 0; head < heads; head++)
            for (int j = 0; j < headDim; j++)
            {
                int source = head * headDim + (j < half ? 2 * j : 2 * (j - half) + 1), target = head * headDim + j;
                Array.Copy(weight, source * inputs, result, target * inputs, inputs);
            }
        return result;
    }

    /// <summary>Loads Mimi: <c>kyutai/moshiko-pytorch-bf16</c>'s <c>tokenizer-e351c8d8-checkpoint125.safetensors</c>, as the
    /// release does.</summary>
    public void LoadCodecWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        var used = Codec.Load(name => state.TryGetValue(name, out var v) ? v : null);
        var unused = state.Keys.Where(k => !used.Contains(k)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The codec checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CSM-Native" : "CSM-ONNX",
            Description = "CSM: Sesame's conversational speech model, a Llama backbone and audio decoder over Mimi codes (Sesame, 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "CSM";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CSM<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
