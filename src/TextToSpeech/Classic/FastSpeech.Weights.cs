using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.Classic;

public partial class FastSpeech<T>
{
    private const string ConvertedParameterTensor = "fastspeech.parameters";

    /// <summary>Loads an AiDotNet-converted checkpoint for the paper-native FastSpeech graph.</summary>
    /// <remarks>
    /// The authors published demonstrations but no source checkpoint. The conversion file therefore contains one
    /// explicitly non-official tensor, <c>fastspeech.parameters</c>, in the order returned by
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}.GetParameters"/> for an identical
    /// <see cref="FastSpeechOptions"/> configuration. Any format supported by
    /// <see cref="ComputerVision.Weights.WeightLoader"/> may be used.
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
            throw new InvalidDataException($"'{ConvertedParameterTensor}' has {values.Length} values; this FastSpeech configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequirePaperCheckpointTarget()
    {
        if (!_useNativeMode || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Converted FastSpeech weights require the paper-native graph, not caller-supplied layers or ONNX mode.");
    }
}
