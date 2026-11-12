using AiDotNet.Audio.Speaker;
using AiDotNet.ModelLoading;

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
    /// <summary>Opens a checkpoint whose state sits under the first of <paramref name="prefixes"/> (or none) that holds
    /// <paramref name="probe"/> (the release's <c>weight</c>, the GPT's <c>model.</c>); returns the file and that prefix.</summary>
    private static (TorchCheckpointFile File, string Prefix) OpenCheckpoint(string path, string probe, params string[] prefixes)
    {
        var file = TorchCheckpointFile.Open(path);
        var prefix = prefixes.Concat(new[] { "" }).FirstOrDefault(p => file.Contains(p + probe));
        if (prefix is null)
        {
            file.Dispose();
            throw new InvalidDataException($"'{path}' has no tensor '{probe}' under any of the release's prefixes.");
        }
        return (file, prefix);
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

        var (gpt, gptPrefix) = OpenCheckpoint(checkpoints.TextToSemantic, "bert_proj.weight", "weight.model.", "model.", "weight.");
        using (gpt) TextToSemantic.Load(gpt.WithPrefix(gptPrefix));

        if (IsCfm) LoadFlowSynthesizer(checkpoints);
        else
        {
            var (s2, s2Prefix) = OpenCheckpoint(checkpoints.Synthesizer, "ssl_proj.weight", "weight.");
            using (s2) SoVits.Load(s2.WithPrefix(s2Prefix));
            if (checkpoints.Discriminator is { } d)
            {
                var (discriminator, prefix) = OpenCheckpoint(d, "discriminators.0.convs.0.weight_v", "weight.", "model.");
                using (discriminator)
                {
                    var source = discriminator.WithPrefix(prefix);
                    _discriminators!.LoadTorch(name => source.TryRead(name), "discriminators.");
                }
            }
        }

        using (var hubert = TorchCheckpointFile.Open(checkpoints.Hubert)) Hubert.LoadTorchWeights(hubert);

        if (checkpoints.Bert is { } bertPath)
        {
            using var bert = TorchCheckpointFile.Open(bertPath, strictAxes: true);
            Bert.LoadTorchWeights(bert.WithNames(name => bert.Contains("bert." + name) ? "bert." + name : name));
        }

        if (IsV2Pro)
        {
            string path = checkpoints.SpeakerVerification
                ?? throw new ArgumentException($"{VersionName} needs ERes2NetV2's weights (SpeakerVerification).", nameof(checkpoints));
            using var sv = TorchCheckpointFile.Open(path);
            _speaker = new ERes2NetV2Encoder<T>(Engine, sv, o.SpeakerChannels, blocks: o.SpeakerBlocks);
        }

        if (IsCfm)
        {
            string path = checkpoints.Vocoder ?? throw new ArgumentException($"{VersionName} needs its vocoder's weights (Vocoder).", nameof(checkpoints));
            using var vocoder = TorchCheckpointFile.Open(path);
            string prefix = vocoder.Names.Any(k => k.StartsWith("generator.", StringComparison.Ordinal)) ? "generator." : "";
            Func<string, double[]?> read = name => vocoder.TryRead(prefix + name);
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
        var (checkpoint, filePrefix) = OpenCheckpoint(checkpoints.Synthesizer, "bridge.0.weight", "weight.");
        using var opened = checkpoint;
        var file = checkpoint.WithPrefix(filePrefix);
        var keys = checkpoint.Names.Where(k => k.StartsWith(filePrefix, StringComparison.Ordinal)).Select(k => k.Substring(filePrefix.Length)).ToList();
        bool lora = keys.Any(k => k.Contains(".lora_A."));
        if (!lora)
        {
            cfm.Load(file);
            return;
        }
        string basePath = checkpoints.SynthesizerBase
            ?? throw new ArgumentException("A LoRA synthesizer checkpoint adapts a pretrained one; set SynthesizerBase.", nameof(checkpoints));
        var (pretrained, basePrefix) = OpenCheckpoint(basePath, "bridge.0.weight", "weight.");
        // The fine-tune's full tensors (style encoder, bridge, wns1) replace the base's; the DiT's base weights stay.
        using (pretrained) cfm.Load(file.Or(pretrained.WithPrefix(basePrefix)));
        string loraPrefix = "cfm.base_model.model.estimator.";
        var probe = keys.First(k => k.StartsWith(loraPrefix, StringComparison.Ordinal) && k.Contains(".lora_A."));
        var dit = cfm.Cfm.Estimator;
        long probeCount = (file.ShapeOf(probe) ?? Array.Empty<long>()).Aggregate(1L, (a, b) => a * b);
        int rank = (int)(probeCount / dit.Configuration.Dim);                                                // lora_A [rank, dim]
        var read = file;
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
