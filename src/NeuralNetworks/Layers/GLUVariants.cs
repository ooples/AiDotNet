using AiDotNet.ActivationFunctions;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

public enum GLUGateType { Sigmoid, Swish, GELU, ReLU, Bilinear }

// Every type below is a GatedLinearUnitLayer that only fixes the gate activation - none of them
// touches a shape. The layouts therefore restate the base's exactly: rank 2 only, because the base's
// ForwardTraced runs `input.MatrixMultiply(...)` against a [outputDimension, inputDimension] weight,
// which is a matrix product rather than a batched one.
//
// The OutputAxesFor bodies delegate rather than restate. The emitted width is the base's private
// `_outputDimension`, so a subclass CANNOT write `Fixed(...)` for it; and declaring the layouts here
// without a body would let ShapeContractGenerator fill the gap with its default `Same(role)` for
// every axis - which is wrong, because the feature axis is resized to _outputDimension, not carried
// through. Delegating both suppresses that generation and keeps one source of truth.

[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class SwiGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>, IShapeContract
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public SwiGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new SwishActivation<T>()) { }

    /// <inheritdoc />
    public new IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => base.OutputAxesFor(inputRank);
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class GeGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>, IShapeContract
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public GeGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new GELUActivation<T>()) { }

    /// <inheritdoc />
    public new IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => base.OutputAxesFor(inputRank);
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class ReGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>, IShapeContract
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public ReGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new ReLUActivation<T>()) { }

    /// <inheritdoc />
    public new IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => base.OutputAxesFor(inputRank);
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class BilinearGLUFeedForwardLayer<T> : GatedLinearUnitLayer<T>, IShapeContract
{
    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    public BilinearGLUFeedForwardLayer(int outputSize)
        : base(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()) { }

    /// <inheritdoc />
    public new IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => base.OutputAxesFor(inputRank);
}
