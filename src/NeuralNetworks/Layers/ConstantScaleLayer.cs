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
// Multiplies every value by a constant; shape is never touched, at any rank.
[ElementWiseShape(Note = "Scales by a fixed constant. Shape untouched at any rank.")]
[AutoParameters]
public partial class ConstantScaleLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly T _scale;

    /// <summary>
    /// Singleton scale held as a Tensor so Forward can use the same
    /// tape-tracked <see cref="IEngine.TensorBroadcastMultiply{T}"/> path
    /// that RMSNorm uses for its γ gain. <c>Engine.TensorMultiplyScalar</c>
    /// records as a unary-scalar op that does NOT propagate gradient back
    /// to its tensor input on the autodiff tape — using it in Forward
    /// breaks gradient flow upstream (caught by
    /// <c>T5Conditioner_Training_ChangesParameters</c>: the EmbeddingLayer
    /// stops receiving gradients when ConstantScaleLayer sits between it
    /// and the transformer blocks). <see cref="IEngine.TensorBroadcastMultiply{T}"/>
    /// IS tape-tracked (used by every Norm/Embedding/MHA forward), so the
    /// composition <c>broadcastMultiply(input, scaleTensor)</c> preserves
    /// gradient flow back to <c>input</c> while still producing the same
    /// scalar-scaled output.
    /// </summary>
    // A constant scale by definition -- saved so a reload reproduces it, never updated by the optimizer.
    [Buffer]
    private readonly Tensor<T> _scaleTensor;

    public override bool SupportsTraining => false;

    public ConstantScaleLayer(double scale)
        : base(new[] { -1 }, new[] { -1 })
    {
        _scale = NumOps.FromDouble(scale);
        _scaleTensor = new Tensor<T>(new[] { 1 });
        _scaleTensor[0] = _scale;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) =>
        Engine.TensorBroadcastMultiply(input, _scaleTensor);

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void ClearGradients() { base.ClearGradients(); }

    /// <inheritdoc/>
    public override void ResetState() { /* no-op */ }
}
