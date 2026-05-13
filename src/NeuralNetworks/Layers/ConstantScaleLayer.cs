using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Multiplies its input by a fixed (non-trainable) scalar. Useful for
/// paper-canonical embedding rescaling — Vaswani 2017 §3.4 (preserved by
/// T5 / LLaMA / Gemma / Qwen2 / ChatGLM3) multiplies token embeddings by
/// √d_model before feeding them into the transformer stack.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4", TestConstructorArgs = "1.0")]
public partial class ConstantScaleLayer<T> : LayerBase<T>
{
    private readonly T _scale;

    public override bool SupportsTraining => false;

    public ConstantScaleLayer(double scale)
        : base(new[] { -1 }, new[] { -1 })
    {
        _scale = NumOps.FromDouble(scale);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input) =>
        Engine.TensorMultiplyScalar(input, _scale);

    /// <inheritdoc/>
    public override long ParameterCount => 0;

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
            throw new ArgumentException(
                $"ConstantScaleLayer has no trainable parameters; got {parameters.Length}.");
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void ClearGradients() { base.ClearGradients(); }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { /* no-op */ }

    /// <inheritdoc/>
    public override void ResetState() { /* no-op */ }
}
