using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.Classic;

public partial class AdaSpeech<T>
{
    private const string ConvertedParameterTensor = "adaspeech.parameters";

    /// <summary>
    /// Loads an AiDotNet-converted AdaSpeech checkpoint.
    /// </summary>
    /// <remarks>
    /// Chen et al. published the paper and audio samples but no official source checkpoint. Consequently this method
    /// does not claim compatibility with an official state dictionary: the conversion file contains one flat tensor,
    /// <c>adaspeech.parameters</c>, in the stable order returned by <see cref="NeuralNetworks.NeuralNetworkBase{T}.GetParameters"/>
    /// for the same <see cref="AdaSpeechOptions"/> configuration. SafeTensors, PyTorch, NumPy and other formats accepted
    /// by <see cref="ComputerVision.Weights.WeightLoader"/> may be used.
    /// </remarks>
    /// <param name="path">Path to the converted weight file.</param>
    public void LoadConvertedWeights(string path)
    {
        if (!_useNativeMode || _speakerEmbedding is null || _phonemePredictor is null)
            throw new NotSupportedException("Converted AdaSpeech weights require the paper-native model, not caller-supplied layers or ONNX mode.");
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
        if (!_useNativeMode || _speakerEmbedding is null || _phonemePredictor is null)
            throw new NotSupportedException("Converted AdaSpeech weights require the paper-native model, not caller-supplied layers or ONNX mode.");

        var values = read(ConvertedParameterTensor)
            ?? throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (values.Length != expected)
            throw new InvalidDataException($"'{ConvertedParameterTensor}' has {values.Length} values; this AdaSpeech configuration expects {expected}.");

        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }
}
