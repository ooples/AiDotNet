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
                $"'{ConvertedParameterTensor}' has {count} values; this Seed-TTS configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted Seed-TTS weights require the native four-block graph, not caller-supplied layers or ONNX mode.");
    }
}
