using System;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Applies logit soft-capping — <c>y = cap · tanh(x / cap)</c> — elementwise, bounding values to
/// <c>(-cap, cap)</c>. Used by Gemma-2 on the final logits (and, in the reference, on attention scores).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The layer has no parameters; it is a fixed nonlinearity placed after the LM head so imported Gemma-2
/// models reproduce the reference's bounded output distribution.
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "30.0")]
public partial class LogitSoftcapLayer<T> : LayerBase<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly double _cap;

    public override bool SupportsTraining => false;

    /// <summary>The soft-cap magnitude; outputs lie in (-cap, cap).</summary>
    public double Cap => _cap;

    /// <summary>Creates a logit soft-cap layer.</summary>
    /// <param name="cap">The soft-cap magnitude (Gemma-2 final logits: 30; attention scores: 50).</param>
    public LogitSoftcapLayer(double cap)
        : base(new[] { -1 }, new[] { -1 })
    {
        if (!(cap > 0.0)) throw new ArgumentOutOfRangeException(nameof(cap), "cap must be positive.");
        _cap = cap;
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var output = new Tensor<T>(input._shape);
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double x = Convert.ToDouble(src[i]);
            dst[i] = Ops.FromDouble(_cap * Math.Tanh(x / _cap));
        }
        return output;
    }

    /// <inheritdoc/>
    public override long ParameterCount => 0;

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
            throw new ArgumentException("LogitSoftcapLayer has no parameters.");
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void ClearGradients() { }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { }

    /// <inheritdoc/>
    public override void ResetState() { }
}
