using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.Classic;

public partial class AlignTTS<T>
{
    private const string ConvertedParameterTensor = "aligntts.parameters";

    /// <summary>Loads an AiDotNet-converted checkpoint for the paper-native AlignTTS graph.</summary>
    /// <remarks>
    /// Zeng et al. did not release an official checkpoint or source state dictionary. This deliberately documented
    /// conversion format therefore contains one flat tensor, <c>aligntts.parameters</c>, in the stable order returned
    /// by <see cref="NeuralNetworks.NeuralNetworkBase{T}.GetParameters"/> for the same <see cref="AlignTTSOptions"/>
    /// configuration; it must not be presented as an official format. Any file type accepted by
    /// <see cref="ComputerVision.Weights.WeightLoader"/> may contain that tensor.
    /// </remarks>
    public void LoadConvertedWeights(string path)
    {
        ThrowIfDisposed();
        RequirePaperCheckpointTarget();
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Checkpoint path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);

        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        LoadConvertedWeights(name => state.TryGetValue(name, out var tensor)
            ? tensor.ToVector().Select(value => (double)value).ToArray()
            : null);
    }

    /// <summary>Loads the documented converted tensor through an in-memory reader.</summary>
    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        RequirePaperCheckpointTarget();
        var values = read(ConvertedParameterTensor)
            ?? throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (values.Length != expected)
            throw new InvalidDataException($"'{ConvertedParameterTensor}' has {values.Length} values; this AlignTTS configuration expects {expected}.");

        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequirePaperCheckpointTarget()
    {
        if (!_useNativeMode || _durationPredictor.Count == 0 || _mixDensityNetwork.Count == 0)
            throw new NotSupportedException("Converted AlignTTS weights require the paper-native graph, not caller-supplied layers or ONNX mode.");
    }
}
