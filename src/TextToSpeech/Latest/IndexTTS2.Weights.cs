namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// The files IndexTTS2's release reads (<c>infer_v2.py</c> and <c>ensure_models_available</c>).
/// </summary>
public sealed class IndexTts2Checkpoints
{
    /// <summary>The <c>IndexTeam/IndexTTS-2</c> directory: <c>gpt.pth</c>, <c>s2mel.pth</c>, <c>wav2vec2bert_stats.pt</c>,
    /// <c>feat1.pt</c>, <c>feat2.pt</c>, <c>bpe.model</c> and <c>qwen0.6bemo4-merge</c>.</summary>
    public required string Directory { get; init; }

    /// <summary>w2v-BERT 2.0's <c>model.safetensors</c> (<c>facebook/w2v-bert-2.0</c>).</summary>
    public required string SemanticModel { get; init; }

    /// <summary>MaskGCT's semantic codec, <c>semantic_codec/model.safetensors</c> (<c>amphion/MaskGCT</c>).</summary>
    public required string SemanticCodec { get; init; }

    /// <summary>CAM++'s <c>campplus_cn_common.bin</c> (<c>funasr/campplus</c>).</summary>
    public required string SpeakerEncoder { get; init; }

    /// <summary>BigVGAN-v2's <c>bigvgan_generator.pt</c> (<c>nvidia/bigvgan_v2_22khz_80band_256x</c>).</summary>
    public required string Vocoder { get; init; }

    /// <summary>Whether to load the text-to-emotion model too (Qwen3, about 0.6B parameters).</summary>
    public bool TextToEmotion { get; init; }
}

public partial class IndexTTS2<T>
{
    /// <summary>A checkpoint read once: each tensor becomes doubles when it is read and is then released, so loading a
    /// multi-gigabyte checkpoint never holds a double copy of all of it.</summary>
    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors;

        public Checkpoint(string path)
        {
            _tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            Keys = new HashSet<string>(_tensors.Keys, StringComparer.Ordinal);
        }

        public HashSet<string> Keys { get; }
        public HashSet<string> Used { get; } = new(StringComparer.Ordinal);

        public bool Contains(string name) => Keys.Contains(name);

        public double[]? Take(string name)
        {
            if (!_tensors.TryGetValue(name, out var tensor))
            {
                if (Used.Contains(name)) throw new InvalidOperationException($"'{name}' is read twice.");
                return null;
            }
            _tensors.Remove(name);
            Used.Add(name);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            return values;
        }

        public Func<string, int[], double[]> Reader(string prefix, string what) => (name, _) =>
            Take(prefix + name) ?? throw new InvalidDataException($"The {what} checkpoint has no tensor '{prefix + name}'.");

        public void RequireAllUsed(string what, Func<string, bool> unused)
        {
            var left = Keys.Where(k => !Used.Contains(k) && !unused(k)).ToList();
            if (left.Count > 0) throw new InvalidDataException($"The {what} checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
        }
    }

    /// <summary>
    /// Loads the release: the text-to-semantic model (<c>gpt.pth</c>; the text head, which inference never reads, is
    /// skipped), the semantic-to-mel module (<c>s2mel.pth</c>'s <c>cfm</c>, <c>length_regulator</c> and <c>gpt_layer</c>),
    /// w2v-BERT 2.0 with the feature statistics, the semantic codec, CAM++, BigVGAN-v2, the emotion table
    /// (<c>feat2.pt</c> with <c>feat1.pt</c>'s speaker vectors) and, if asked, the text-to-emotion model. Use
    /// <see cref="IndexTTS2Options.Release"/>'s configuration for these weights.
    /// </summary>
    public void LoadReferenceWeights(IndexTts2Checkpoints checkpoints)
    {
        ThrowIfDisposed();
        if (checkpoints is null) throw new ArgumentNullException(nameof(checkpoints));
        if (!HasPaperLayers) throw NotNative();
        var o = _options;
        string dir = checkpoints.Directory;
        if (o.BpeModelPath is null && File.Exists(Path.Combine(dir, "bpe.model"))) o.BpeModelPath = Path.Combine(dir, "bpe.model");

        SemanticModel.LoadTorchWeights(new Checkpoint(checkpoints.SemanticModel).Reader("", "w2v-BERT"));
        var stats = new Checkpoint(Path.Combine(dir, "wav2vec2bert_stats.pt"));
        SemanticCodec.LoadStatistics(stats.Take("mean")!, stats.Take("var")!);
        var codec = new Checkpoint(checkpoints.SemanticCodec);
        SemanticCodec.Load((name, _) => codec.Take(name));
        codec.RequireAllUsed("semantic codec", k => k.StartsWith("decoder.", StringComparison.Ordinal));

        var gpt = new Checkpoint(Path.Combine(dir, "gpt.pth"));
        string gptPrefix = gpt.Keys.Any(k => k.StartsWith("model.", StringComparison.Ordinal)) && !gpt.Contains("mel_head.weight") ? "model." : "";
        TextToSemantic.Load(gpt.Reader(gptPrefix, "gpt"));
        // The conformers' position tables are persistent buffers: the release runs on the checkpoint's float32 tables.
        if (gpt.Take(gptPrefix + "conditioning_encoder.embed.pos_enc.pe") is { } speakerPositions)
            TextToSemantic.Conditioning.LoadPositionTable(speakerPositions);
        if (gpt.Take(gptPrefix + "emo_conditioning_encoder.embed.pos_enc.pe") is { } emotionPositions)
            TextToSemantic.EmotionConditioning.LoadPositionTable(emotionPositions);
        gpt.RequireAllUsed("gpt", k => k.Contains("text_head.")
            || k.EndsWith("attn.bias", StringComparison.Ordinal) || k.EndsWith("attn.masked_bias", StringComparison.Ordinal)
            || (o.Duration == IndexTts2DurationConditioning.Paper && k.EndsWith("speed_emb.weight", StringComparison.Ordinal)));

        var s2mel = new Checkpoint(Path.Combine(dir, "s2mel.pth"));
        string Strip(string module, string name) =>
            s2mel.Contains($"net.{module}.module.{name}") ? $"net.{module}.module.{name}" : $"net.{module}.{name}";
        Func<string, int[], double[]> Read(string module) => (name, _) =>
            s2mel.Take(Strip(module, name)) ?? throw new InvalidDataException($"The s2mel checkpoint has no tensor '{Strip(module, name)}'.");
        var mel = SemanticToMel;
        mel.Regulator.Load(Engine, Read("length_regulator"), "");
        mel.Estimator.Load(Read("cfm"), name => s2mel.Contains(Strip("cfm", name)), "estimator.");
        var gptLayer = Read("gpt_layer");
        IndexTtsOps.LoadLinear(Engine, gptLayer, mel.Gpt1, "0", o.HiddenDim, 256);
        IndexTtsOps.LoadLinear(Engine, gptLayer, mel.Gpt2, "1", 256, 128);
        IndexTtsOps.LoadLinear(Engine, gptLayer, mel.Gpt3, "2", 128, o.SemanticHiddenSize);
        // The release builds but never runs the DiT's x_embedder, cond_embedder and content_mask_embedder, the regulator's
        // discrete-content embedding and mask token, the skip projections of blocks that send rather than receive, and
        // its buffers; s2mel.pth may also carry other modules' training state.
        s2mel.RequireAllUsed("s2mel", k => !k.StartsWith("net.cfm.", StringComparison.Ordinal) && !k.StartsWith("net.length_regulator.", StringComparison.Ordinal)
            && !k.StartsWith("net.gpt_layer.", StringComparison.Ordinal) || k.Contains("x_embedder") || k.Contains("cond_embedder")
            || k.Contains("content_mask_embedder") || k.EndsWith(".embedding.weight", StringComparison.Ordinal) || k.EndsWith("mask_token", StringComparison.Ordinal)
            || k.Contains("skip_in_linear") || k.EndsWith("input_pos", StringComparison.Ordinal) || k.EndsWith(".freqs", StringComparison.Ordinal)
            || k.Contains("freqs_cis") || k.Contains("causal_mask"));

        SpeakerEncoder.LoadTorchWeights(new Checkpoint(checkpoints.SpeakerEncoder).Reader("", "CAM++"));

        var vocoder = new Checkpoint(checkpoints.Vocoder);
        string vocoderPrefix = vocoder.Keys.Any(k => k.StartsWith("generator.", StringComparison.Ordinal)) ? "generator." : "";
        Vocoder.LoadTorch(name => vocoder.Take(vocoderPrefix + name));

        o.EmotionGroupSizes ??= new[] { 3, 17, 2, 8, 4, 5, 10, 24 };
        LoadEmotionTable(Rows(new Checkpoint(Path.Combine(dir, "feat2.pt")).Take("")!, o.HiddenDim),
            Rows(new Checkpoint(Path.Combine(dir, "feat1.pt")).Take("")!, o.StyleDim));

        if (checkpoints.TextToEmotion)
        {
            o.EmotionModelPath ??= Path.Combine(dir, "qwen0.6bemo4-merge");
            _ = TextToEmotion();
        }
    }

    internal static void LoadTextToEmotionState(IndexTtsTextToEmotion<T> t2e, string directory)
    {
        var state = new Checkpoint(Path.Combine(directory, "model.safetensors"));
        var random = new Random(0);
        t2e.Model.LoadTorchWeights((name, shape) =>
        {
            if (state.Take(name) is { } v) return v;
            // The release ships the adapter merged; a missing adapter starts as peft does (A uniform, B zero), which
            // leaves the merged model unchanged until it trains.
            if (name.EndsWith("lora_B.weight", StringComparison.Ordinal)) return new double[shape.Aggregate(1, (a, b) => a * b)];
            if (name.EndsWith("lora_A.weight", StringComparison.Ordinal)) return IndexTtsOps.Uniform(random, shape.Aggregate(1, (a, b) => a * b), shape[1]);
            throw new InvalidDataException($"The text-to-emotion checkpoint has no tensor '{name}'.");
        });
    }

    private static double[][] Rows(double[] values, int width) =>
        Enumerable.Range(0, values.Length / width).Select(r => values.Skip(r * width).Take(width).ToArray()).ToArray();

    /// <summary>Loads the text-to-emotion model's <c>model.safetensors</c> from <paramref name="directory"/> (a
    /// <c>Qwen3ForCausalLM</c> with tied embeddings).</summary>
    public void LoadTextToEmotion(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        _options.EmotionModelPath = directory;
        if (_t2e is null) _ = TextToEmotion();     // builds the model and loads the directory
        else LoadTextToEmotionState(_t2e, directory);
    }
}
