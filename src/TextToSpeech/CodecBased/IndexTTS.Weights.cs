using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class IndexTTS<T>
{
    /// <summary>The state under the prefix a checkpoint wraps it in (<c>model.</c> or <c>generator.</c>), found by
    /// <paramref name="probe"/>.</summary>
    private static TorchTensorSource Unwrapped(TorchCheckpointFile checkpoint, string probe)
    {
        foreach (var prefix in new[] { "", "model.", "generator.", "module." })
            if (checkpoint.Contains(prefix + probe)) return checkpoint.WithPrefix(prefix);
        throw new InvalidDataException($"The checkpoint has no tensor '{probe}' under any known prefix.");
    }

    /// <summary>
    /// Loads the release from its directory (<c>IndexTeam/Index-TTS</c>, or <c>IndexTeam/IndexTTS-1.5</c> with
    /// <see cref="IndexTTSOptions.Version1_5"/>'s configuration): the GPT (<c>gpt.pth</c>), BigVGAN with its ECAPA-TDNN
    /// (<c>bigvgan_generator.pth</c>), the DVAE's encoder and codebook (<c>dvae.pth</c>) and, when not set,
    /// <c>bpe.model</c> as the tokenizer.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        if (!HasPaperLayers) throw NotNative();
        if (_options.BpeModelPath is null && File.Exists(Path.Combine(directory, "bpe.model"))) _options.BpeModelPath = Path.Combine(directory, "bpe.model");

        using (var gpt = TorchCheckpointFile.Open(Path.Combine(directory, "gpt.pth")))
        {
            var state = Unwrapped(gpt, "mel_head.weight");
            TextToMel.Load(state);
            // The conformer's position table is a persistent buffer: the release runs on the checkpoint's (float32) copy.
            if (state.TryRead("conditioning_encoder.embed.pos_enc.pe") is { } positions) TextToMel.Conditioning.LoadPositionTable(positions);
            // GPT-2's causal-mask buffers and the release's inference copy are not weights.
            gpt.RequireAllUsed("gpt", k => k.EndsWith("attn.bias", StringComparison.Ordinal) || k.EndsWith("attn.masked_bias", StringComparison.Ordinal)
                || k.Contains("inference_model."));
        }

        using (var vocoder = TorchCheckpointFile.Open(Path.Combine(directory, "bigvgan_generator.pth")))
        {
            LoadVocoder(Unwrapped(vocoder, "conv_pre.bias"));
            // The anti-aliasing filters are fixed; BatchNorm's batch counters are not state the evaluation reads.
            vocoder.RequireAllUsed("BigVGAN", k => k.EndsWith(".filter", StringComparison.Ordinal) || k.EndsWith("num_batches_tracked", StringComparison.Ordinal));
        }

        using (var dvae = TorchCheckpointFile.Open(Path.Combine(directory, "dvae.pth")))
        {
            Dvae.Load(Unwrapped(dvae, "codebook.embed"), "");
            // The decoder only trained the DVAE; the codebook's running averages and the discretization loss's history are
            // training state.
            dvae.RequireAllUsed("DVAE", k => !k.Contains("encoder.") && !k.EndsWith("codebook.embed", StringComparison.Ordinal)
                || k.Contains("decoder."));
        }
    }

    /// <summary>Loads BigVGAN (its <c>speaker_encoder.*</c> into the ECAPA-TDNN).</summary>
    internal void LoadVocoder(TorchTensorSource source)
    {
        SpeakerEncoder.Load(source, "speaker_encoder.");
        Vocoder.LoadTorch(name => source.TryRead(name));
    }

    /// <summary>Loads the three state dictionaries, each read by its own names.</summary>
    internal void LoadState(Func<string, double[]?> gpt, Func<string, double[]?> vocoder, Func<string, double[]?> dvae)
    {
        TextToMel.Load(TorchTensorSource.From(gpt));
        LoadVocoder(TorchTensorSource.From(vocoder));
        Dvae.Load(TorchTensorSource.From(dvae), "");
    }
}
