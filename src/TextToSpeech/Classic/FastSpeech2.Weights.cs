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
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        LoadConvertedWeights(name => state.TryGetValue(name, out var tensor)
            ? tensor.ToVector().Select(value => (double)value).ToArray()
            : null);
    }

    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        RequirePaperCheckpointTarget();
        var values = read(ConvertedParameterTensor)
            ?? throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (values.Length != expected)
            throw new InvalidDataException($"'{ConvertedParameterTensor}' has {values.Length} values; this FastSpeech 2 configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequirePaperCheckpointTarget()
    {
        if (!_useNativeMode || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Converted FastSpeech 2 weights require the paper-native graph, not caller-supplied layers or ONNX mode.");
    }
}
