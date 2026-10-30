namespace AiDotNet.TextToSpeech.CodecBased;

public partial class FireRedTTS<T>
{
    /// <summary>A checkpoint read once: each tensor becomes doubles when it is read and is then released.</summary>
    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors;

        public Checkpoint(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint '{path}' does not exist.", path);
            _tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            Keys = new HashSet<string>(_tensors.Keys, StringComparer.Ordinal);
        }

        public HashSet<string> Keys { get; }
        private HashSet<string> Used { get; } = new(StringComparer.Ordinal);

        /// <summary>The prefix the state sits under (the GPT's file wraps it in <c>model</c>).</summary>
        public string Prefix(string probe)
        {
            foreach (var prefix in new[] { "", "model.", "state_dict.", "module." })
                if (Keys.Contains(prefix + probe)) return prefix;
            throw new InvalidDataException($"The checkpoint has no tensor '{probe}' under any known prefix.");
        }

        public double[]? Take(string name)
        {
            if (!_tensors.TryGetValue(name, out var tensor)) return null;
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

        var gpt = new Checkpoint(Path.Combine(directory, "fireredtts_gpt.pt"));
        string prefix = gpt.Prefix("mel_head.weight");
        Gpt.Load(gpt.Reader(prefix, "GPT"), FireRedTtsGpt<T>.ReleaseTextTokenCount);
        // GPT-2's causal-mask buffers are not weights; the training step counter is not state.
        gpt.RequireAllUsed("GPT", k => k.EndsWith("attn.bias", StringComparison.Ordinal) || k.EndsWith("attn.masked_bias", StringComparison.Ordinal)
            || k == "step");

        var token2wav = new Checkpoint(Path.Combine(directory, "fireredtts_token2wav.pt"));
        LoadToken2Wav(token2wav.Take);
        token2wav.RequireAllUsed("token2wav", k => k.EndsWith(".filter", StringComparison.Ordinal));

        var speaker = new Checkpoint(Path.Combine(directory, "fireredtts_speaker.bin"));
        SpeakerEncoder.Load(EcapaReader(speaker.Reader("", "speaker")), "");
        speaker.RequireAllUsed("speaker", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal));
    }

    /// <summary>Loads <c>fireredtts_token2wav.pt</c>'s state (<c>flow.*</c> with the codebooks, <c>generator.*</c>).</summary>
    internal void LoadToken2Wav(Func<string, double[]?> take)
    {
        for (int b = 0; b < 2; b++)
            SpeechTokenizer.Quantizer.LoadCodebook(b, take($"flow.input_embedding.codebook.{b}.weight")
                ?? throw new InvalidDataException("The token2wav checkpoint has no flow codebooks."));
        Flow.Load((name, _) => take(name) ?? throw new InvalidDataException($"The token2wav checkpoint has no tensor '{name}'."), "flow.");
        Vocoder.LoadTorch(name => take("generator." + name));
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
        var codec = new Checkpoint(codecPath);
        SpeechTokenizer.Load(codec.Reader("", "speech tokenizer"), k => codec.Keys.Contains(k));
        SpeakerEncoder.Load(EcapaReader(codec.Reader("speaker_encoder.", "speech tokenizer")), "");
        codec.RequireAllUsed("speech tokenizer", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal) || k.EndsWith(".inited", StringComparison.Ordinal));
        var hubert = new Checkpoint(hubertPath);
        SpeechTokenizer.Hubert.LoadTorchWeights(hubert.Take);
    }
}
