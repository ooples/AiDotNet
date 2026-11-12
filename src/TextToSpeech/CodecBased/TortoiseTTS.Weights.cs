using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class TortoiseTTS<T>
{
    /// <summary>Loads all five original TorToiSe component checkpoints without a parameter-vector converter.</summary>
    public void LoadAuthorWeights(string autoregressivePath, string clvpPath, string diffusionPath,
        string speechTokenizerPath, string vocoderPath, string? textTokenizerPath = null,
        string? melNormalizationPath = null, string vocoderPrefix = "model_g.")
    {
        RequireConvertedCheckpointTarget();
        LoadAutoregressiveWeights(autoregressivePath);
        LoadClvpWeights(clvpPath);
        LoadDiffusionWeights(diffusionPath);
        LoadTokenizerWeights(speechTokenizerPath);
        LoadVocoderWeights(vocoderPath, vocoderPrefix);
        if (textTokenizerPath is not null) LoadTextTokenizer(textTokenizerPath);
        if (melNormalizationPath is not null) LoadMelNormalization(melNormalizationPath);
    }

    /// <summary>Loads the authors' original named autoregressive.pth state dictionary.</summary>
    public void LoadAutoregressiveWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, source => _autoregressive!.LoadTorchWeights(source));
    }

    /// <summary>Loads the authors' original named clvp2.pth state dictionary.</summary>
    public void LoadClvpWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, source => _clvp!.LoadTorchWeights(source));
    }

    /// <summary>Loads the authors' original named diffusion_decoder.pth state dictionary.</summary>
    public void LoadDiffusionWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, source => _diffusion!.LoadTorchWeights(source));
    }

    /// <summary>Loads the authors' original dVAE tensors, including EMA counts and codebook sums.</summary>
    public void LoadTokenizerWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, source => _tokenizer!.LoadTorchWeights(source));
    }

    /// <summary>Loads vocoder.pth's original model_g state, or an extracted UnivNet state with an empty prefix.</summary>
    public void LoadVocoderWeights(string path, string prefix = "model_g.")
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, source => _univNet!.LoadTorchWeights(source.WithPrefix(prefix)));
    }

    /// <summary>Loads a custom original mel_norms.pth tensor for the autoregressive front end.</summary>
    public void LoadMelNormalization(string path)
    {
        RequireConvertedCheckpointTarget();
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var tensor = state.Values.SingleOrDefault(value => value.Length == _options.AutoregressiveMelChannels)
            ?? throw new InvalidDataException("Expected one per-band mel normalization tensor.");
        var values = tensor.ToVector().Select(value => (double)value).ToArray();
        _melFrontend!.SetNormalization(values);
        _options.AutoregressiveMelNormalization = values;
    }

    /// <summary>Opens an original component checkpoint, each tensor's axes checked, and loads it with
    /// <paramref name="load"/>.</summary>
    private static void LoadAuthorState(string path, Action<AiDotNet.ModelLoading.TorchTensorSource> load)
    {
        using var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(path, strictAxes: true);
        load(file);
    }

    private const string ConvertedParameterTensor = "tortoise.parameters";

    /// <summary>Loads a legacy AiDotNet parameter-vector snapshot.</summary>
    /// <remarks>
    /// This restores optimizer parameters, not EMA codebook buffers or external tokenizer metadata. Use
    /// <see cref="LoadAuthorWeights"/> for the original five component checkpoints and their fitted state.
    /// </remarks>
    public void LoadConvertedWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Checkpoint path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        using var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(path);
        LoadConvertedWeights(file);
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
                $"'{ConvertedParameterTensor}' has {count} values; this TorToiSe configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted TorToiSe weights require the native VQ-VAE/GPT-2/CLVP/diffusion/UnivNet graph.");
    }
}
