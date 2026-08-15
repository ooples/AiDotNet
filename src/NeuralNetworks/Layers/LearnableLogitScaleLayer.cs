using AiDotNet.Attributes;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Applies exp(clamp(logScale, 0, log(100))) to similarity logits.</summary>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4", TestConstructorArgs = "0.07")]
public class LearnableLogitScaleLayer<T> : LayerBase<T>
{
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _logScale;

    public override bool SupportsTraining => true;
    public override long ParameterCount => 1;

    public LearnableLogitScaleLayer(double temperature = 0.07)
        : base([-1], [-1])
    {
        if (temperature <= 0) throw new ArgumentOutOfRangeException(nameof(temperature));
        _logScale = new Tensor<T>([1]);
        _logScale[0] = NumOps.FromDouble(Math.Log(1.0 / temperature));
        RegisterTrainableParameter(_logScale, PersistentTensorRole.Weights);
    }

    public T Scale => NumOps.Exp(NumOps.FromDouble(Math.Clamp(
        NumOps.ToDouble(_logScale[0]), 0.0, 4.605170185988092)));

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var clamped = Engine.TensorClamp(
            _logScale, NumOps.Zero, NumOps.FromDouble(4.605170185988092));
        var scale = Engine.TensorExp(clamped);
        return Engine.TensorMultiply(input, scale);
    }

    public override Vector<T> GetParameters() => Vector<T>.FromMemory(_logScale.Data);

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 1) throw new ArgumentException("Expected one log-scale parameter.");
        _logScale[0] = parameters[0];
        Engine.InvalidatePersistentTensor(_logScale);
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
