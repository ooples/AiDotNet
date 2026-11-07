using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class SeedTTS<T>
{
    private const string ConvertedParameterTensor = "seedtts.parameters";

    /// <summary>Loads an explicitly converted checkpoint into an identical native Seed-TTS configuration.</summary>
    /// <remarks>
    /// ByteDance's official evaluation repository states that Seed-TTS source and weights are not released. This
    /// loader therefore does not imply official compatibility: the file must contain one tensor named
    /// <c>seedtts.parameters</c>, ordered exactly as <see cref="NeuralNetworks.NeuralNetworkBase{T}.GetParameters"/>
    /// for the same <see cref="SeedTTSOptions"/> configuration.
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
                $"'{ConvertedParameterTensor}' has {values.Length} values; this Seed-TTS configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted Seed-TTS weights require the native four-block graph, not caller-supplied layers or ONNX mode.");
    }
}
