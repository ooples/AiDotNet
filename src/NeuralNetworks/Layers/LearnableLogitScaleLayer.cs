using AiDotNet.Attributes;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Applies exp(clamp(logScale, 0, log(100))) to similarity logits.</summary>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4", TestConstructorArgs = "0.07")]
[ElementWiseShape(Note = "A learned scalar rescales values without changing any axis.")]
[AutoParameters]
public partial class LearnableLogitScaleLayer<T> : LayerBase<T>
{
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logScale;

    public override bool SupportsTraining => true;
    public LearnableLogitScaleLayer(double temperature = 0.07)
        : base([-1], [-1])
    {
        if (temperature <= 0) throw new ArgumentOutOfRangeException(nameof(temperature));
        _logScale = new Tensor<T>([1]);
        _logScale[0] = NumOps.FromDouble(Math.Log(1.0 / temperature));
        RegisterTrainableParameter(_logScale, PersistentTensorRole.Weights);
    }

    public T Scale => NumOps.Exp(NumOps.FromDouble(Math.Max(
        0.0, Math.Min(NumOps.ToDouble(_logScale[0]), 4.605170185988092))));

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var clamped = Engine.TensorClamp(
            _logScale, NumOps.Zero, NumOps.FromDouble(4.605170185988092));
        var scale = Engine.TensorExp(clamped);
        return Engine.TensorMultiply(input, scale);
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradient = GetParameterGradients();
        if (gradient.Length == 1)
            _logScale[0] = NumOps.Subtract(_logScale[0], NumOps.Multiply(learningRate, gradient[0]));
        Engine.InvalidatePersistentTensor(_logScale);
    }

    public override void ResetState()
    {
    }
}
