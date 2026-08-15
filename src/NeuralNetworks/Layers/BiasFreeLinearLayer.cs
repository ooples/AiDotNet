using AiDotNet.Attributes;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Trainable linear projection <c>y = xW</c> with no bias term.</summary>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8")]
public sealed class BiasFreeLinearLayer<T> : LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _outputSize;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _weights;

    public int InputSize => _inputSize;
    public int OutputSize => _outputSize;
    public override bool SupportsTraining => true;
    public override long ParameterCount => _weights.Length;

    public BiasFreeLinearLayer(int inputSize, int outputSize)
        : base([inputSize], [outputSize])
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));
        _inputSize = inputSize;
        _outputSize = outputSize;
        _weights = new Tensor<T>([inputSize, outputSize]);
        InitializeLayerWeights(_weights, inputSize, outputSize);
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank < 1 || input.Shape[^1] != _inputSize)
            throw new ArgumentException(
                $"Expected last dimension {_inputSize}, got [{string.Join(',', input.Shape)}].",
                nameof(input));
        return Engine.TensorMatMul(input, _weights);
    }

    public override Vector<T> GetParameters() => Vector<T>.FromMemory(_weights.Data);

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length)
            throw new ArgumentException($"Expected {_weights.Length} parameters, got {parameters.Length}.");
        parameters.AsSpan().CopyTo(_weights.Data.Span);
        Engine.InvalidatePersistentTensor(_weights);
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _weights.Length) return;
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] = NumOps.Subtract(_weights[i], NumOps.Multiply(learningRate, gradients[i]));
        Engine.InvalidatePersistentTensor(_weights);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputSize"] = _inputSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["OutputSize"] = _outputSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
