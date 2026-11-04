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
        _autoregressive!.LoadTorchWeights(ReadAuthorState(path));
    }

    /// <summary>Loads the authors' original named clvp2.pth state dictionary.</summary>
    public void LoadClvpWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        _clvp!.LoadTorchWeights(ReadAuthorState(path));
    }

    /// <summary>Loads the authors' original named diffusion_decoder.pth state dictionary.</summary>
    public void LoadDiffusionWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        _diffusion!.LoadTorchWeights(ReadAuthorState(path));
    }

    /// <summary>Loads the authors' original dVAE tensors, including EMA counts and codebook sums.</summary>
    public void LoadTokenizerWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        _tokenizer!.LoadTorchWeights(ReadAuthorState(path));
    }

    /// <summary>Loads vocoder.pth's original model_g state, or an extracted UnivNet state with an empty prefix.</summary>
    public void LoadVocoderWeights(string path, string prefix = "model_g.")
    {
        RequireConvertedCheckpointTarget();
        var read = ReadAuthorState(path);
        _univNet!.LoadTorchWeights((name, shape) => read(prefix + name, shape));
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

    private static Func<string, int[], double[]> ReadAuthorState(string path)
    {
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return (name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException($"TorToiSe checkpoint is missing '{name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape) && !(shape.Length == 0 && tensor.Length == 1))
                throw new InvalidDataException($"TorToiSe tensor '{name}' has an incompatible shape.");
            return tensor.ToVector().Select(value => (double)value).ToArray();
        };
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
                $"'{ConvertedParameterTensor}' has {values.Length} values; this TorToiSe configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted TorToiSe weights require the native VQ-VAE/GPT-2/CLVP/diffusion/UnivNet graph.");
    }
}
