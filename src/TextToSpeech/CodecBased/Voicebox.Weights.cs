using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Voicebox<T>
{
    private const string ConvertedParameterTensor = "voicebox.parameters";

    /// <summary>Loads a strictly converted paper-compatible Voicebox checkpoint.</summary>
    /// <remarks>
    /// Meta released neither Voicebox weights nor code. This loader is therefore for checkpoints independently trained
    /// against this architecture and exported in AiDotNet parameter order under <c>voicebox.parameters</c>; it must not be
    /// represented as an official Meta checkpoint. Exact length matching prevents accidental partial restoration into a
    /// compact test graph, a different phone table, or a different vocoder width.
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
                $"'{ConvertedParameterTensor}' has {values.Length} values; this Voicebox configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted Voicebox weights require the native duration/audio-flow/HiFi-GAN graph.");
    }
}
