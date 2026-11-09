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
                $"'{ConvertedParameterTensor}' has {count} values; this Voicebox configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted Voicebox weights require the native duration/audio-flow/HiFi-GAN graph.");
    }
}
