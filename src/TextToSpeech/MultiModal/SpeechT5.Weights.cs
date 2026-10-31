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
    /// <summary>A checkpoint's tensors as doubles, each read at most once.</summary>
    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors;

        public Checkpoint(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            _tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            Keys = new HashSet<string>(_tensors.Keys, StringComparer.Ordinal);
        }

        public HashSet<string> Keys { get; }
        public HashSet<string> Used { get; } = new(StringComparer.Ordinal);

        public double[]? Take(string name)
        {
            if (!_tensors.TryGetValue(name, out var tensor)) return null;
            _tensors.Remove(name);
            Used.Add(name);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            return values;
        }

        public void RequireAllUsed(string what, Func<string, bool> unused)
        {
            var left = Keys.Where(k => !Used.Contains(k) && !unused(k)).ToList();
            if (left.Count > 0) throw new InvalidDataException($"The {what} checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
        }
    }

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
        var model = new Checkpoint(checkpoints.Model);
        bool huggingFace = model.Keys.Contains("speecht5.encoder.prenet.embed_tokens.weight");
        string prefix = !huggingFace && model.Keys.Contains("model.encoder.layer_norm.weight") ? "model." : "";
        bool pretraining = model.Keys.Contains(prefix + "hubert_layer.final_proj.weight");
        LoadState((name, _) =>
        {
            string key = huggingFace ? SpeechT5Core<T>.HuggingFaceName(name) ?? throw new InvalidDataException($"No Hugging Face name for '{name}'.") : prefix + name;
            return model.Take(key) ?? throw new InvalidDataException($"The model checkpoint has no tensor '{key}'.");
        }, pretraining && !huggingFace);
        // Not read: the batch norms' update counts, Hugging Face's stored positional tables (recomputed here) and, in an
        // official checkpoint, the CTC projection, the decoder's unused relative positions, the text decoder's own copy of
        // the shared embedding and the modules of other tasks.
        var ignored = new[] { "encoder.proj", "decoder.pos_emb", "text_decoder_prenet", "text_decoder_postnet", "speech_encoder_postnet",
            "speaker_decoder_postnet", "speech_encoder_prenet", "hubert_layer", "quantizer" }.Take(pretraining ? 6 : 9).Select(m => prefix + m).ToArray();
        model.RequireAllUsed("model", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal) || k.EndsWith(".pe", StringComparison.Ordinal)
            || (!huggingFace && (k.Contains(".norm_k.") || !k.StartsWith(prefix, StringComparison.Ordinal) || ignored.Any(m => k.StartsWith(m, StringComparison.Ordinal)))));

        if (checkpoints.Vocoder is { } vocoderPath)
        {
            var vocoder = new Checkpoint(vocoderPath);
            LoadVocoder(name => vocoder.Take(name));
            vocoder.RequireAllUsed("vocoder", _ => false);
        }
        if (checkpoints.SpeakerEncoder is { } speakerPath)
        {
            var speaker = new Checkpoint(speakerPath);
            SpeakerEncoder.Load((name, _) => speaker.Take(name) ?? throw new InvalidDataException($"The x-vector checkpoint has no tensor '{name}'."));
            speaker.RequireAllUsed("x-vector", k => k.EndsWith("num_batches_tracked", StringComparison.Ordinal));
            foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
        }
        _xvectors.Clear();
        _trainingSpeakers.Clear();
    }

    /// <summary>Loads the model from the official names (<see cref="SpeechT5Core{T}.Load"/>); the post-net's batch norms
    /// then use their running statistics outside training.</summary>
    internal void LoadState(Func<string, int[], double[]> read, bool pretraining)
    {
        Core.Load(read, pretraining);
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
