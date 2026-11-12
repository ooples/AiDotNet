using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class XTTSv2<T>
{
    /// <summary>Loads named Coqui model/dVAE tensors and their original tokenizer and mel statistics.</summary>
    public void LoadAuthorWeights(string modelCheckpoint, string dvaeCheckpoint, string tokenizerPath, string melStatsPath)
    {
        RequireConvertedCheckpointTarget();
        using var modelFile = AiDotNet.ModelLoading.TorchCheckpointFile.Open(modelCheckpoint, strictAxes: true);
        using var dvaeFile = AiDotNet.ModelLoading.TorchCheckpointFile.Open(dvaeCheckpoint, strictAxes: true);
        var (model, modelNames) = UnwrapAuthorState(modelFile);
        var (dvae, _) = UnwrapAuthorState(dvaeFile);
        _languageModel!.LoadTorchWeights(model);
        _dvae!.LoadTorchWeights(dvae);
        _decoder!.LoadTorchWeights(model.WithPrefix("hifigan_decoder.waveform_decoder."));
        const string speakerPrefix = "hifigan_decoder.speaker_encoder.";
        _speakerEncoder!.LoadState(model.WithPrefix(speakerPrefix),
            modelNames.Where(name => name.StartsWith(speakerPrefix, StringComparison.Ordinal)).Select(name => name.Substring(speakerPrefix.Length)));
        LoadTokenizer(tokenizerPath);
        LoadMelStats(melStatsPath);
    }

    // The checkpoint's tensors under its "model" root when it has one (see the dictionary form), and their names there.
    private static (AiDotNet.ModelLoading.TorchTensorSource Source, IReadOnlyList<string> Names) UnwrapAuthorState(
        AiDotNet.ModelLoading.TorchCheckpointFile file)
    {
        var names = file.Names;
        if (!names.Any(name => name.StartsWith("model.", StringComparison.Ordinal))) return (file, names.ToList());
        return (file.WithPrefix("model."), names.Where(name => name.StartsWith("model.", StringComparison.Ordinal))
            .Select(name => name.Substring(6)).ToList());
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
        // NumPy and ONNX files have no streaming reader; they load whole as before.
        using var file = Path.GetExtension(path).ToLowerInvariant() is ".npy" or ".onnx" ? null : AiDotNet.ModelLoading.TorchCheckpointFile.Open(path);
        LoadConvertedWeights(file ?? AiDotNet.ModelLoading.TorchTensorSource.From(new ComputerVision.Weights.WeightLoader().LoadWeights(path), $"the checkpoint {path}"));
    }

    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        LoadConvertedWeights(AiDotNet.ModelLoading.TorchTensorSource.From(read));
    }

    /// <summary>Copies <c>{ConvertedParameterTensor}</c> from <paramref name="source"/> straight into a parameter vector.</summary>
    internal void LoadConvertedWeights(AiDotNet.ModelLoading.TorchTensorSource source)
    {
        RequireConvertedCheckpointTarget();
        if (!source.Contains(ConvertedParameterTensor))
            throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (source.ShapeOf(ConvertedParameterTensor) is { } shape && shape.Aggregate(1L, (a, b) => a * b) is var count && count != expected)
            throw new InvalidDataException(
                $"'{ConvertedParameterTensor}' has {count} values; this XTTS-v2 configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
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
