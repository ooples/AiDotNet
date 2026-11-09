using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The files SpeechT5's release reads.</summary>
public sealed class SpeechT5Checkpoints
{
    /// <summary>The model: <c>microsoft/speecht5_tts</c>'s <c>pytorch_model.bin</c> (Hugging Face names), or an official
    /// fairseq checkpoint (its names, optionally under <c>model.</c>).</summary>
    public required string Model { get; init; }

    /// <summary>HiFi-GAN: <c>microsoft/speecht5_hifigan</c>'s <c>pytorch_model.bin</c>, with its spectrogram mean and
    /// scale; null keeps the vocoder as it is.</summary>
    public string? Vocoder { get; init; }

    /// <summary>speechbrain's x-vector extractor, <c>speechbrain/spkrec-xvect-voxceleb</c>'s <c>embedding_model.ckpt</c>;
    /// null keeps it as it is.</summary>
    public string? SpeakerEncoder { get; init; }
}

public partial class SpeechT5<T>
{
    /// <summary>
    /// Loads the release: the model (Hugging Face's <c>microsoft/speecht5_tts</c> or an official checkpoint; the
    /// pre-training modules too when the checkpoint holds them), HiFi-GAN (<c>microsoft/speecht5_hifigan</c>) and the
    /// x-vector extractor (<c>spkrec-xvect-voxceleb</c>). The default <see cref="SpeechT5Options"/> match these weights.
    /// </summary>
    public void LoadReferenceWeights(SpeechT5Checkpoints checkpoints)
    {
        ThrowIfDisposed();
        if (checkpoints is null) throw new ArgumentNullException(nameof(checkpoints));
        if (!HasPaperLayers) throw NotNative();
        using (var model = TorchCheckpointFile.Open(checkpoints.Model))
        {
            bool huggingFace = model.Contains("speecht5.encoder.prenet.embed_tokens.weight");
            string prefix = !huggingFace && model.Contains("model.encoder.layer_norm.weight") ? "model." : "";
            bool pretraining = model.Contains(prefix + "hubert_layer.final_proj.weight");
            LoadState(model.WithNames(name => huggingFace
                ? SpeechT5Core<T>.HuggingFaceName(name) ?? throw new InvalidDataException($"No Hugging Face name for '{name}'.")
                : prefix + name), pretraining && !huggingFace);
            // Not read: the batch norms' update counts, Hugging Face's stored positional tables (recomputed here) and, in an
            // official checkpoint, the CTC projection, the decoder's unused relative positions, the text decoder's own copy of
            // the shared embedding and the modules of other tasks.
            var ignored = new[] { "encoder.proj", "decoder.pos_emb", "text_decoder_prenet", "text_decoder_postnet", "speech_encoder_postnet",
                "speaker_decoder_postnet", "speech_encoder_prenet", "hubert_layer", "quantizer" }.Take(pretraining ? 6 : 9).Select(m => prefix + m).ToArray();
            model.RequireAllUsed("model", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal) || k.EndsWith(".pe", StringComparison.Ordinal)
                || (!huggingFace && (k.Contains(".norm_k.") || !k.StartsWith(prefix, StringComparison.Ordinal) || ignored.Any(m => k.StartsWith(m, StringComparison.Ordinal)))));
        }

        if (checkpoints.Vocoder is { } vocoderPath)
        {
            using var vocoder = TorchCheckpointFile.Open(vocoderPath);
            LoadVocoder(name => vocoder.TryRead(name));
            vocoder.RequireAllUsed("vocoder");
        }
        if (checkpoints.SpeakerEncoder is { } speakerPath)
        {
            using var speaker = TorchCheckpointFile.Open(speakerPath);
            SpeakerEncoder.Load((name, _) => speaker.Read(name, Array.Empty<int>()));
            speaker.RequireAllUsed("x-vector", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal));
            foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
        }
        _xvectors.Clear();
        _trainingSpeakers.Clear();
    }

    /// <summary>Loads the model from the official names (<see cref="SpeechT5Core{T}.Load(TorchTensorSource, bool)"/>); the post-net's batch norms
    /// then use their running statistics outside training.</summary>
    internal void LoadState(Func<string, int[], double[]> read, bool pretraining) => LoadState(TorchTensorSource.From(read), pretraining);

    /// <summary>Loads the model from the official names in <paramref name="source"/>.</summary>
    internal void LoadState(TorchTensorSource source, bool pretraining)
    {
        Core.Load(source, pretraining);
    }

    /// <summary>Loads Hugging Face's <c>SpeechT5HifiGan</c>: <c>conv_pre</c>, <c>upsampler.{i}</c>,
    /// <c>resblocks.{i}.convs1/convs2.{n}</c>, <c>conv_post</c> and the spectrogram <c>mean</c> and <c>scale</c>.</summary>
    internal void LoadVocoder(Func<string, double[]?> read)
    {
        int m = _options.MelChannels;
        Vocoder.LoadTorch(name => read(name.StartsWith("ups.", StringComparison.Ordinal) ? "upsampler." + name.Substring(4) : name));
        var mean = read("mean") ?? throw new InvalidDataException("The vocoder checkpoint has no 'mean'.");
        var scale = read("scale") ?? throw new InvalidDataException("The vocoder checkpoint has no 'scale'.");
        if (mean.Length != m || scale.Length != m) throw new InvalidDataException($"The vocoder's mean and scale have {m} values.");
        _vocoderMean = mean;
        _vocoderScale = scale;
    }
}
