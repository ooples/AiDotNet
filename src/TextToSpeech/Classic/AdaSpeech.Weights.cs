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

        // NumPy and ONNX files have no streaming reader; they load whole as before.
        using var file = Path.GetExtension(path).ToLowerInvariant() is ".npy" or ".onnx" ? null : AiDotNet.ModelLoading.TorchCheckpointFile.Open(path);
        LoadConvertedWeights(file ?? AiDotNet.ModelLoading.TorchTensorSource.From(new ComputerVision.Weights.WeightLoader().LoadWeights(path), $"the checkpoint {path}"));
    }

    /// <summary>Loads the documented converted tensor through an in-memory reader.</summary>
    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        LoadConvertedWeights(AiDotNet.ModelLoading.TorchTensorSource.From(read));
    }

    /// <summary>Copies <c>{ConvertedParameterTensor}</c> from <paramref name="source"/> straight into a parameter vector.</summary>
    internal void LoadConvertedWeights(AiDotNet.ModelLoading.TorchTensorSource source)
    {
        if (!_useNativeMode || _speakerEmbedding is null || _phonemePredictor is null)
            throw new NotSupportedException("Converted AdaSpeech weights require the paper-native model, not caller-supplied layers or ONNX mode.");

        if (!source.Contains(ConvertedParameterTensor))
            throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (source.ShapeOf(ConvertedParameterTensor) is { } shape && shape.Aggregate(1L, (a, b) => a * b) is var count && count != expected)
            throw new InvalidDataException($"'{ConvertedParameterTensor}' has {count} values; this AdaSpeech configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }
}
