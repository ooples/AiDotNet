namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// Files used to assemble DiTTo-TTS. SpeechT5 and BigVGAN are public upstream checkpoints. The DiTTo authors did not
/// release their DiT, length predictor, or privately obtained CLaM-TTS Mel-VAE; those three paths therefore name
/// safetensors converted to the stable AiDotNet names described by each module's loader.
/// </summary>
public sealed class DiTToCheckpointPaths
{
    public required string SpeechT5 { get; init; }
    public required string MelVae { get; init; }
    public required string DiffusionTransformer { get; init; }
    public required string LengthPredictor { get; init; }
    public required string BigVgan { get; init; }
}

public partial class DiTToTTS<T>
{
    /// <summary>
    /// Loads the public SpeechT5/BigVGAN state and AiDotNet-converted DiTTo/CLaM tensors. This deliberately does not call
    /// the converted files "official": as of the paper publication there is no author-released DiTTo repository or model.
    /// </summary>
    public void LoadConvertedWeights(DiTToCheckpointPaths checkpoints)
    {
        Validation.Guard.NotNull(checkpoints); ThrowIfDisposed(); RequirePaperModel();
        var speech = ReadWeights(checkpoints.SpeechT5);
        TextEncoder!.LoadHuggingFace(Reader(speech, "SpeechT5"));
        MelVae!.Load(Reader(ReadWeights(checkpoints.MelVae), "CLaM-TTS Mel-VAE"), "");
        DiffusionTransformer!.Load(Reader(ReadWeights(checkpoints.DiffusionTransformer), "DiTTo diffusion Transformer"), "");
        LengthPredictor!.Load(Reader(ReadWeights(checkpoints.LengthPredictor), "DiTTo length predictor"), "");
        var bigvgan = ReadWeights(checkpoints.BigVgan);
        BigVgan!.LoadTorch(name => Optional(bigvgan, name) ?? Optional(bigvgan, "generator." + name));
    }

    private static Dictionary<string, double[]> ReadWeights(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("A checkpoint path is required.", nameof(path));
        IEnumerable<string> files;
        if (Directory.Exists(path))
        {
            files = Directory.GetFiles(path, "*.safetensors").OrderBy(p => p, StringComparer.Ordinal).ToArray();
            if (!files.Any()) throw new FileNotFoundException($"No safetensors files were found in {path}.", path);
        }
        else
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            files = new[] { path };
        }
        var weights = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (string file in files)
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
                weights[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        return weights;
    }

    private static Func<string, int[], double[]> Reader(IReadOnlyDictionary<string, double[]> weights, string component) =>
        (name, shape) =>
        {
            if (!weights.TryGetValue(name, out var values)) throw new InvalidDataException($"The {component} checkpoint has no tensor '{name}'.");
            int expected = shape.Aggregate(1, (a, b) => checked(a * b));
            if (values.Length != expected) throw new InvalidDataException($"Tensor '{name}' has {values.Length} values; expected {expected}.");
            return values;
        };

    private static double[]? Optional(IReadOnlyDictionary<string, double[]> weights, string name) =>
        weights.TryGetValue(name, out var values) ? values : null;
}
