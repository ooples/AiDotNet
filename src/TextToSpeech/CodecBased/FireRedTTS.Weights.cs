using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class FireRedTTS<T>
{
    /// <summary>The prefix a checkpoint's state sits under (the GPT's file wraps it in <c>model</c>).</summary>
    private static string Prefix(TorchTensorSource checkpoint, string probe)
    {
        foreach (var prefix in new[] { "", "model.", "state_dict.", "module." })
            if (checkpoint.Contains(prefix + probe)) return prefix;
        throw new InvalidDataException($"The checkpoint has no tensor '{probe}' under any known prefix.");
    }

    /// <summary>
    /// Loads the release from its directory (Hugging Face <c>fireredteam/FireRedTTS</c>): the GPT
    /// (<c>fireredtts_gpt.pt</c>), the flow and BigVGAN (<c>fireredtts_token2wav.pt</c>; the flow's frozen input embedding
    /// holds the speech tokenizer's codebooks) and the speaker encoder (<c>fireredtts_speaker.bin</c>). The speech
    /// tokenizer's encoder is not in this release (<see cref="LoadSpeechTokenizer"/>).
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        if (!HasPaperLayers) throw NotNative();

        using (var gpt = TorchCheckpointFile.Open(Path.Combine(directory, "fireredtts_gpt.pt")))
        {
            Gpt.Load(gpt.WithPrefix(Prefix(gpt, "mel_head.weight")), FireRedTtsGpt<T>.ReleaseTextTokenCount);
            // GPT-2's causal-mask buffers are not weights; the training step counter is not state.
            gpt.RequireAllUsed("GPT", k => k.EndsWith("attn.bias", StringComparison.Ordinal) || k.EndsWith("attn.masked_bias", StringComparison.Ordinal)
                || k == "step");
        }

        using (var token2wav = TorchCheckpointFile.Open(Path.Combine(directory, "fireredtts_token2wav.pt")))
        {
            LoadToken2Wav(token2wav);
            token2wav.RequireAllUsed("token2wav", k => k.EndsWith(".filter", StringComparison.Ordinal));
        }

        using (var speaker = TorchCheckpointFile.Open(Path.Combine(directory, "fireredtts_speaker.bin")))
        {
            SpeakerEncoder.Load(EcapaSource(speaker), "");
            speaker.RequireAllUsed("speaker", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal));
        }
    }

    /// <summary>Loads <c>fireredtts_token2wav.pt</c>'s state (<c>flow.*</c> with the codebooks, <c>generator.*</c>).</summary>
    internal void LoadToken2Wav(TorchTensorSource source)
    {
        for (int b = 0; b < 2; b++)
        {
            string name = $"flow.input_embedding.codebook.{b}.weight";
            if (!source.Contains(name)) throw new InvalidDataException("The token2wav checkpoint has no flow codebooks.");
            SpeechTokenizer.Quantizer.LoadCodebook(b, source, name);
        }
        Flow.Load(source, "flow.");
        Vocoder.LoadTorch(name => source.TryRead("generator." + name));
    }

    /// <summary>
    /// Loads the speech tokenizer: its encoder, quantizer and acoustic encoder from the FireRedTTS-1S release's
    /// <c>speech_tokenizer/codec.bin</c> (the same codebooks as FireRedTTS-1's flow), and HuBERT from a Transformers-format
    /// checkpoint of <c>TencentGameMate/chinese-hubert-large</c> (the 1S release ships fairseq's <c>hubert.pt</c>; its
    /// Hugging Face conversion holds the same weights). The decoders are not released.
    /// </summary>
    public void LoadSpeechTokenizer(string codecPath, string hubertPath)
    {
        ThrowIfDisposed();
        if (codecPath is null) throw new ArgumentNullException(nameof(codecPath));
        if (hubertPath is null) throw new ArgumentNullException(nameof(hubertPath));
        if (!HasPaperLayers) throw NotNative();
        if (!File.Exists(hubertPath)) throw new FileNotFoundException($"The checkpoint '{hubertPath}' does not exist.", hubertPath);
        using (var codec = TorchCheckpointFile.Open(codecPath))
        {
            SpeechTokenizer.Load(codec);
            SpeakerEncoder.Load(EcapaSource(codec.WithPrefix("speaker_encoder.")), "");
            codec.RequireAllUsed("speech tokenizer", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal) || k.EndsWith(".inited", StringComparison.Ordinal));
        }
        using var hubert = TorchCheckpointFile.Open(hubertPath);
        SpeechTokenizer.Hubert.LoadTorchWeights(hubert);
    }
}
