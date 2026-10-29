namespace AiDotNet.TextToSpeech.CodecBased;

public partial class IndexTTS<T>
{
    /// <summary>A checkpoint read once: each tensor becomes doubles when it is read and is then released.</summary>
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

        /// <summary>The prefix the state sits under: <c>model.</c> or <c>generator.</c> when the file wraps it.</summary>
        public string Prefix(string probe)
        {
            foreach (var prefix in new[] { "", "model.", "generator.", "module." })
                if (Keys.Contains(prefix + probe)) return prefix;
            throw new InvalidDataException($"The checkpoint has no tensor '{probe}' under any known prefix.");
        }

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

        var gpt = new Checkpoint(Path.Combine(directory, "gpt.pth"));
        string prefix = gpt.Prefix("mel_head.weight");
        TextToMel.Load(gpt.Reader(prefix, "gpt"));
        // The conformer's position table is a persistent buffer: the release runs on the checkpoint's (float32) copy.
        if (gpt.Take(prefix + "conditioning_encoder.embed.pos_enc.pe") is { } positions) TextToMel.Conditioning.LoadPositionTable(positions);
        // GPT-2's causal-mask buffers and the release's inference copy are not weights.
        gpt.RequireAllUsed("gpt", k => k.EndsWith("attn.bias", StringComparison.Ordinal) || k.EndsWith("attn.masked_bias", StringComparison.Ordinal)
            || k.Contains("inference_model."));

        var vocoder = new Checkpoint(Path.Combine(directory, "bigvgan_generator.pth"));
        LoadVocoder(vocoder.Reader(vocoder.Prefix("conv_pre.bias"), "BigVGAN"), name => vocoder.Take(vocoder.Prefix("conv_pre.bias") + name));
        // The anti-aliasing filters are fixed; BatchNorm's batch counters are not state the evaluation reads.
        vocoder.RequireAllUsed("BigVGAN", k => k.EndsWith(".filter", StringComparison.Ordinal) || k.EndsWith("num_batches_tracked", StringComparison.Ordinal));

        var dvae = new Checkpoint(Path.Combine(directory, "dvae.pth"));
        Dvae.Load(dvae.Reader(dvae.Prefix("codebook.embed"), "DVAE"), "");
        // The decoder only trained the DVAE; the codebook's running averages and the discretization loss's history are
        // training state.
        dvae.RequireAllUsed("DVAE", k => !k.Contains("encoder.") && !k.EndsWith("codebook.embed", StringComparison.Ordinal)
            || k.Contains("decoder."));
    }

    /// <summary>Loads BigVGAN (its <c>speaker_encoder.*</c> into the ECAPA-TDNN).</summary>
    internal void LoadVocoder(Func<string, int[], double[]> read, Func<string, double[]?> take)
    {
        SpeakerEncoder.Load(read, "speaker_encoder.");
        Vocoder.LoadTorch(take);
    }

    /// <summary>Loads the three state dictionaries, each read by its own names.</summary>
    internal void LoadState(Func<string, double[]?> gpt, Func<string, double[]?> vocoder, Func<string, double[]?> dvae)
    {
        Func<string, int[], double[]> Require(Func<string, double[]?> take, string what) =>
            (name, _) => take(name) ?? throw new InvalidDataException($"The {what} state has no tensor '{name}'.");
        TextToMel.Load(Require(gpt, "gpt"));
        LoadVocoder(Require(vocoder, "BigVGAN"), vocoder);
        Dvae.Load(Require(dvae, "DVAE"), "");
    }
}
