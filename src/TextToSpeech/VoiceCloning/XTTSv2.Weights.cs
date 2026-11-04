using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class XTTSv2<T>
{
    /// <summary>Loads named Coqui model/dVAE tensors and their original tokenizer and mel statistics.</summary>
    public void LoadAuthorWeights(string modelCheckpoint, string dvaeCheckpoint, string tokenizerPath, string melStatsPath)
    {
        RequireConvertedCheckpointTarget();
        var loader = new ComputerVision.Weights.WeightLoader();
        var model = UnwrapAuthorState(loader.LoadWeights(modelCheckpoint));
        var dvae = UnwrapAuthorState(loader.LoadWeights(dvaeCheckpoint));
        Func<string, int[], double[]> Reader(IReadOnlyDictionary<string, Tensor<float>> state) => (name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException($"XTTS checkpoint is missing '{name}'.");
            var values = tensor.ToVector().Select(v => (double)v).ToArray();
            if (!tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException($"XTTS tensor '{name}' has an incompatible shape.");
            return values;
        };
        _languageModel!.LoadTorchWeights(Reader(model));
        _dvae!.LoadTorchWeights(Reader(dvae));
        _decoder!.LoadTorchWeights(name => model.TryGetValue("hifigan_decoder.waveform_decoder." + name, out var value)
            ? value.ToVector().Select(v => (double)v).ToArray() : null);
        const string speakerPrefix = "hifigan_decoder.speaker_encoder.";
        var speaker = model.Where(kv => kv.Key.StartsWith(speakerPrefix, StringComparison.Ordinal)).ToDictionary(
            kv => kv.Key.Substring(speakerPrefix.Length), kv => new Tensor<T>(kv.Value.Shape.ToArray(),
                new Vector<T>(kv.Value.ToVector().Select(v => NumOps.FromDouble(v)))));
        _speakerEncoder!.LoadState(speaker);
        LoadTokenizer(tokenizerPath);
        LoadMelStats(melStatsPath);
    }

    // Original model.pth stores its state under "model" alongside non-neural trainer/config metadata.
    // WeightLoader preserves that root in names; extracted state dictionaries have no wrapper.
    internal static IReadOnlyDictionary<string, Tensor<float>> UnwrapAuthorState(
        IReadOnlyDictionary<string, Tensor<float>> state)
        => state.Keys.Any(name => name.StartsWith("model.", StringComparison.Ordinal))
            ? state.Where(entry => entry.Key.StartsWith("model.", StringComparison.Ordinal))
                .ToDictionary(entry => entry.Key.Substring(6), entry => entry.Value, StringComparer.Ordinal)
            : state;

    internal void LoadGptTorchWeights(Func<string, int[], double[]> read, bool includeConditioning = true)
    {
        RequireConvertedCheckpointTarget();
        _languageModel!.LoadTorchWeights(read, includeConditioning: includeConditioning);
    }

    private const string ConvertedParameterTensor = "xttsv2.parameters";

    /// <summary>Loads the 80 released mel normalization values from <c>mel_stats.pth</c> or safetensors.</summary>
    public void LoadMelStats(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Mel-statistics path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Mel statistics not found: {path}", path);
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        double[]? values = null;
        foreach (var entry in state)
        {
            var candidate = entry.Value.ToVector().Select(value => (double)value).ToArray();
            if (candidate.Length == _options.MelChannels)
            {
                values = candidate;
                break;
            }
        }
        SetMelStats(values ?? throw new InvalidDataException(
            $"The checkpoint contains no {_options.MelChannels}-value mel-statistics tensor."));
        _options.MelStatsPath = path;
    }

    internal void SetMelStats(double[] values)
    {
        Validation.Guard.NotNull(values);
        if (values.Length != _options.MelChannels || values.Any(value => double.IsNaN(value) || double.IsInfinity(value) || value == 0))
            throw new InvalidDataException($"XTTS-v2 mel statistics must contain {_options.MelChannels} finite, nonzero values.");
        _melStats = (double[])values.Clone();
    }

    /// <summary>Loads a strict conversion of the official <c>model.pth</c> and <c>dvae.pth</c> state dictionaries.</summary>
    /// <remarks>
    /// Coqui's checkpoint mixes Hugging Face GPT-2 tensors, weight-normalized convolutions, EMA quantizer buffers and
    /// H/ASP state. The AiDotNet converter preserves those tensors in this graph's serialization order under
    /// <c>xttsv2.parameters</c>. Exact length matching rejects partial checkpoints and configuration mismatches.
    /// <c>vocab.json</c> and <c>mel_stats.pth</c> remain unconverted and are loaded by their dedicated methods.
    /// </remarks>
    public void LoadConvertedWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Checkpoint path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        LoadConvertedWeights(name => state.TryGetValue(name, out var tensor)
            ? tensor.ToVector().Select(value => (double)value).ToArray()
            : null);
    }

    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        RequireConvertedCheckpointTarget();
        var values = read(ConvertedParameterTensor)
            ?? throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (values.Length != expected)
            throw new InvalidDataException(
                $"'{ConvertedParameterTensor}' has {values.Length} values; this XTTS-v2 configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    /// <summary>Loads a converted official graph plus its unmodified tokenizer and mel statistics.</summary>
    public void LoadReferenceWeights(string convertedCheckpointPath, string tokenizerPath, string melStatsPath)
    {
        LoadConvertedWeights(convertedCheckpointPath);
        LoadTokenizer(tokenizerPath);
        LoadMelStats(melStatsPath);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted XTTS-v2 weights require the native dVAE/GPT/H-ASP/HiFi-GAN graph.");
    }
}
