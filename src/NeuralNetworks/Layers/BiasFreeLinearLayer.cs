using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Trainable linear projection <c>y = xW</c> with no bias term.</summary>
/// <typeparam name="T">The numeric type used for inputs, weights, and outputs.</typeparam>
/// <remarks><para><b>For Beginners:</b> This layer changes the feature width using only a weight
/// matrix. Omitting the bias is useful when the surrounding architecture already supplies an
/// offset or when the paper explicitly requires a pure projection.</para></remarks>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8")]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class BiasFreeLinearLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputSize <= 0 || inputRank < 1) return null;
        var features = new OutputAxisContract(
            TensorAxis.Features, AxisRelation.Fixed(_outputSize));
        OutputAxisContract Pass(TensorAxis axis) =>
            new(axis, AxisRelation.Same(axis));
        return inputRank switch
        {
            1 => [features],
            2 => [Pass(TensorAxis.Batch), features],
            3 => [Pass(TensorAxis.Batch), Pass(TensorAxis.Time), features],
            4 =>
            [
                Pass(TensorAxis.Batch), Pass(TensorAxis.Channels),
                Pass(TensorAxis.Height), features
            ],
            _ => null
        };
    }

    private readonly int _inputSize;
    private readonly int _outputSize;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weights;

    public int InputSize => _inputSize;
    public int OutputSize => _outputSize;
    public override bool SupportsTraining => true;
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

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank < 1 || input.Shape[^1] != _inputSize)
            throw new ArgumentException(
                $"Expected last dimension {_inputSize}, got [{string.Join(",", input.Shape)}].",
                nameof(input));
        return Engine.TensorMatMul(input, _weights);
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
