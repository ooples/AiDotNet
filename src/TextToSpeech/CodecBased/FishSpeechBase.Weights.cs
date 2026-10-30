namespace AiDotNet.TextToSpeech.CodecBased;

public abstract partial class FishSpeechBase<T>
{
    /// <summary>The release's codec checkpoint file name in the model directory.</summary>
    public const string CodecCheckpointName = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth";

    /// <summary>A checkpoint read once, its names stripped of a wrapper's prefix: each tensor becomes doubles when it is
    /// read and is then released.</summary>
    private sealed class Checkpoint
    {
        private readonly Dictionary<string, Tensor<float>> _tensors;
        private readonly string _what;

        public Checkpoint(string path, string what, params string[] prefixes)
        {
            _what = what;
            var tensors = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            // The releases' loaders: a Lightning "state_dict", then a "model." (transformer) or "generator." (codec) prefix.
            foreach (var prefix in new[] { "state_dict." }.Concat(prefixes))
                if (tensors.Keys.Any(k => k.StartsWith(prefix, StringComparison.Ordinal)))
                    tensors = tensors.Where(p => p.Key.StartsWith(prefix, StringComparison.Ordinal))
                        .ToDictionary(p => p.Key.Substring(prefix.Length), p => p.Value, StringComparer.Ordinal);
            _tensors = tensors;
        }

        public double[]? Take(string name)
        {
            if (!_tensors.TryGetValue(name, out var tensor)) return null;
            _tensors.Remove(name);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            return values;
        }

        public double[] Read(string name, int[] shape)
        {
            var values = Take(name) ?? throw new InvalidDataException($"The {_what} checkpoint has no tensor '{name}'.");
            long expected = shape.Aggregate(1L, (a, b) => a * b);
            if (values.Length != expected)
                throw new InvalidDataException($"The {_what} checkpoint's '{name}' has {values.Length} values; [{string.Join(", ", shape)}] needs {expected}.");
            return values;
        }
    }

    /// <summary>
    /// Loads a release's directory (<c>fishaudio/fish-speech-1.5</c> or <c>-1.4</c>): the Dual-AR transformer
    /// (<c>model.pth</c>), Firefly-GAN (<see cref="CodecCheckpointName"/>) and the tokenizer (1.5:
    /// <c>tokenizer.tiktoken</c>; 1.4: <c>tokenizer.json</c>), unless <see cref="FishSpeechFamilyOptions.TokenizerPath"/>
    /// names another.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        if (!HasPaperLayers) throw NotNative();
        _options.TokenizerPath ??= Path.Combine(directory, _version == FishSpeechVersion.V15 ? "tokenizer.tiktoken" : "tokenizer.json");
        ResetTokenizer();
        _ = Tokenizer;
        LoadTransformer(Path.Combine(directory, "model.pth"));
        LoadCodec(Path.Combine(directory, CodecCheckpointName));
    }

    /// <summary>Loads the Dual-AR transformer from the release's <c>model.pth</c> (or any checkpoint with its names).</summary>
    public void LoadTransformer(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var checkpoint = new Checkpoint(path ?? throw new ArgumentNullException(nameof(path)), "transformer", "model.");
        Transformer.Load(checkpoint.Read);
        _preferenceReference = null;
    }

    /// <summary>Loads Firefly-GAN from the release's generator checkpoint.</summary>
    public void LoadCodec(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var checkpoint = new Checkpoint(path ?? throw new ArgumentNullException(nameof(path)), "codec", "generator.");
        Codec.Load(checkpoint.Take);
    }
}
