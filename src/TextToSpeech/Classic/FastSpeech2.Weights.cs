using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.Classic;

public partial class FastSpeech2<T>
{
    private const string ConvertedParameterTensor = "fastspeech2.parameters";

    /// <summary>Loads an AiDotNet-converted checkpoint for the paper-native FastSpeech 2 graph.</summary>
    /// <remarks>
    /// The authors released samples but no official model state. This conversion format is therefore deliberately
    /// identified as AiDotNet-specific: it contains one <c>fastspeech2.parameters</c> tensor in the order returned by
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}.GetParameters"/> for an identical
    /// <see cref="FastSpeech2Options"/> configuration. It is not the incompatible community implementation checkpoint.
    /// </remarks>
    public void LoadConvertedWeights(string path)
    {
        RequirePaperCheckpointTarget();
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
        RequirePaperCheckpointTarget();
        if (!source.Contains(ConvertedParameterTensor))
            throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (source.ShapeOf(ConvertedParameterTensor) is { } shape && shape.Aggregate(1L, (a, b) => a * b) is var count && count != expected)
            throw new InvalidDataException($"'{ConvertedParameterTensor}' has {count} values; this FastSpeech 2 configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequirePaperCheckpointTarget()
    {
        if (!_useNativeMode || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Converted FastSpeech 2 weights require the paper-native graph, not caller-supplied layers or ONNX mode.");
    }
}
