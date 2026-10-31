using AiDotNet.Audio.Speaker;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The files a <see cref="GPTSoVITS{T}"/> loads (the release's <c>GPT_SoVITS/pretrained_models</c>, Hugging Face
/// <c>lj1995/GPT-SoVITS</c>).</summary>
public sealed class GptSoVitsCheckpoints
{
    /// <summary>The GPT's checkpoint (an s1 <c>.ckpt</c>, e.g. <c>s1v3.ckpt</c>).</summary>
    public required string TextToSemantic { get; init; }

    /// <summary>The synthesizer's checkpoint (an s2 <c>.pth</c>, e.g. <c>v2Pro/s2Gv2ProPlus.pth</c>, <c>s2Gv3.pth</c>,
    /// <c>gsv-v4-pretrained/s2Gv4.pth</c>), or a LoRA fine-tune of a flow-matching one.</summary>
    public required string Synthesizer { get; init; }

    /// <summary>chinese-hubert-base's weights (<c>chinese-hubert-base/pytorch_model.bin</c>).</summary>
    public required string Hubert { get; init; }

    /// <summary>chinese-roberta-wwm-ext-large's weights (<c>chinese-roberta-wwm-ext-large/pytorch_model.bin</c>), which
    /// Chinese text needs.</summary>
    public string? Bert { get; init; }

    /// <summary>ERes2NetV2's weights (<c>sv/pretrained_eres2netv2w24s4ep4.ckpt</c>), which v2Pro and v2ProPlus need.</summary>
    public string? SpeakerVerification { get; init; }

    /// <summary>The vocoder (v3: <c>models--nvidia--bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt</c>; v4 and V5:
    /// <c>gsv-v4-pretrained/vocoder.pth</c> or <c>gsv-v5-pretrained/vocoder.pth</c>).</summary>
    public string? Vocoder { get; init; }

    /// <summary>For a LoRA <see cref="Synthesizer"/>: the pretrained synthesizer it adapts (<c>s2Gv3.pth</c>,
    /// <c>s2Gv4.pth</c>, …).</summary>
    public string? SynthesizerBase { get; init; }

    /// <summary>The discriminator's checkpoint for GAN fine-tuning (<c>s2D488k.pth</c>, <c>s2Dv2ProPlus.pth</c>, …).</summary>
    public string? Discriminator { get; init; }
}

public partial class GPTSoVITS<T>
{
    /// <summary>A checkpoint read once, its tensors released as they are read; nested dictionaries (the release's
    /// <c>weight</c>, the GPT's <c>model.</c>) are found by trying the prefixes.</summary>
    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors;
        private readonly string _prefix;

        public Checkpoint(string path, string probe, params string[] prefixes)
        {
            _tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            _prefix = prefixes.Concat(new[] { "" }).FirstOrDefault(p => _tensors.ContainsKey(p + probe))
                ?? throw new InvalidDataException($"'{path}' has no tensor '{probe}' under any of the release's prefixes.");
            Keys = new HashSet<string>(_tensors.Keys.Where(k => k.StartsWith(_prefix, StringComparison.Ordinal)).Select(k => k.Substring(_prefix.Length)),
                StringComparer.Ordinal);
        }

        public HashSet<string> Keys { get; }

        public double[]? Take(string name)
        {
            if (!_tensors.TryGetValue(_prefix + name, out var tensor)) return null;
            _tensors.Remove(_prefix + name);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            return values;
        }

        public int Length(string name) => _tensors[_prefix + name].Length;

        public double[] Require(string name) => Take(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{_prefix + name}'.");
    }

    /// <summary>
    /// Loads the release's weights for <see cref="GPTSoVITSOptions.Version"/>: the GPT, the synthesizer (merging a LoRA
    /// fine-tune into, or onto the adapters over, its base), chinese-hubert, chinese-roberta, ERes2NetV2 for v2Pro and the
    /// vocoder for v3 to V5.
    /// </summary>
    public void LoadReferenceWeights(GptSoVitsCheckpoints checkpoints)
    {
        ThrowIfDisposed();
        if (checkpoints is null) throw new ArgumentNullException(nameof(checkpoints));
        if (!HasPaperLayers) throw NotNative();
        var o = _options;

        var gpt = new Checkpoint(checkpoints.TextToSemantic, "bert_proj.weight", "weight.model.", "model.", "weight.");
        TextToSemantic.Load((name, _) => gpt.Require(name));

        if (IsCfm) LoadFlowSynthesizer(checkpoints);
        else
        {
            var s2 = new Checkpoint(checkpoints.Synthesizer, "ssl_proj.weight", "weight.");
            SoVits.Load(s2.Take);
            if (checkpoints.Discriminator is { } d)
            {
                var discriminator = new Checkpoint(d, "discriminators.0.convs.0.weight_v", "weight.", "model.");
                _discriminators!.LoadTorch(discriminator.Take, "discriminators.");
            }
        }

        var hubert = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(checkpoints.Hubert);
        Hubert.LoadTorchWeights(name => hubert.TryGetValue(name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null);

        if (checkpoints.Bert is { } bertPath)
        {
            var bert = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(bertPath);
            Bert.LoadTorchWeights((name, shape) =>
            {
                if (!bert.TryGetValue("bert." + name, out var tensor) && !bert.TryGetValue(name, out tensor))
                    throw new InvalidDataException($"The BERT checkpoint has no tensor '{name}'.");
                if (!tensor.Shape.ToArray().SequenceEqual(shape))
                    throw new InvalidDataException($"'{name}' is [{string.Join(", ", tensor.Shape.ToArray())}] in the checkpoint but [{string.Join(", ", shape)}] here.");
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
        }

        if (IsV2Pro)
        {
            string path = checkpoints.SpeakerVerification
                ?? throw new ArgumentException($"{VersionName} needs ERes2NetV2's weights (SpeakerVerification).", nameof(checkpoints));
            var sv = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            _speaker = new ERes2NetV2Encoder<T>(Engine, name => sv.TryGetValue(name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null,
                o.SpeakerChannels, blocks: o.SpeakerBlocks);
        }

        if (IsCfm)
        {
            string path = checkpoints.Vocoder ?? throw new ArgumentException($"{VersionName} needs its vocoder's weights (Vocoder).", nameof(checkpoints));
            var vocoder = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            string prefix = vocoder.Keys.Any(k => k.StartsWith("generator.", StringComparison.Ordinal)) ? "generator." : "";
            Func<string, double[]?> read = name => vocoder.TryGetValue(prefix + name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null;
            if (_bigVgan is not null) _bigVgan.LoadTorch(read);
            else _hifiGan!.LoadTorch(read);
        }
        _weightsGeneration++;
        _optimizers.Clear();
    }

    /// <summary>The flow-matching synthesizer: a pretrained checkpoint, or a LoRA one (its adapters under
    /// <c>cfm.base_model.model.</c>, the style encoder, bridge and wns1 in full) over <see cref="GptSoVitsCheckpoints.SynthesizerBase"/>,
    /// as <c>change_sovits_weights</c> loads it: onto the adapters when the model has them at the checkpoint's rank, else
    /// merged.</summary>
    private void LoadFlowSynthesizer(GptSoVitsCheckpoints checkpoints)
    {
        var cfm = FlowSynthesizer;
        var file = new Checkpoint(checkpoints.Synthesizer, "bridge.0.weight", "weight.");
        bool lora = file.Keys.Any(k => k.Contains(".lora_A."));
        if (!lora)
        {
            cfm.Load(file.Take);
            return;
        }
        string basePath = checkpoints.SynthesizerBase
            ?? throw new ArgumentException("A LoRA synthesizer checkpoint adapts a pretrained one; set SynthesizerBase.", nameof(checkpoints));
        var pretrained = new Checkpoint(basePath, "bridge.0.weight", "weight.");
        // The fine-tune's full tensors (style encoder, bridge, wns1) replace the base's; the DiT's base weights stay.
        cfm.Load(name => file.Take(name) ?? pretrained.Take(name));
        string loraPrefix = "cfm.base_model.model.estimator.";
        var probe = file.Keys.First(k => k.StartsWith(loraPrefix, StringComparison.Ordinal) && k.Contains(".lora_A."));
        var dit = cfm.Cfm.Estimator;
        int rank = file.Length(probe) / dit.Configuration.Dim;                                               // lora_A [rank, dim]
        Func<string, double[]?> read = file.Take;
        if (dit.HasLora)
        {
            if (_options.LoraRank != rank)
                throw new InvalidDataException($"The checkpoint's LoRA rank is {rank} but the model's adapters are rank {_options.LoraRank}; set LoraRank to match.");
            dit.LoadLora(read, loraPrefix, rank);
        }
        else
        {
            var adapters = new List<AiDotNet.NeuralNetworks.Layers.LayerBase<T>>();
            dit.AddLora(adapters, rank, rank, new Random(0));
            dit.LoadLora(read, loraPrefix, rank);
            dit.MergeLora();
        }
    }
}
