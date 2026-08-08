using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Passes its input through unchanged in the forward direction while blocking gradient flow
/// backwards — the layer form of a stop-gradient / detach barrier.
/// </summary>
/// <remarks>
/// <para>Useful wherever an architecture deliberately trains two components as independent
/// functional units inside a single sequential stack. The canonical example is ABINet's
/// <b>Autonomous</b> principle (Fang et al., CVPR 2021, arXiv:2103.06495), which blocks gradient
/// flow between the vision model and the language model so the language model is forced to learn
/// explicit language modelling rather than becoming an extension of the visual features. The
/// paper measures roughly 0.9% worse accuracy when that gradient is allowed to flow.</para>
/// <para>Forward is the identity, so inference is completely unaffected; only the backward pass
/// changes. Implemented with <c>Engine.StopGradient</c> so the tape records the barrier itself
/// rather than the layer silently detaching outside the graph.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = false, ChangesShape = false, ExpectedInputRank = -1, Cost = ComputeCost.Low, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
// Passes values straight through and blocks the gradient; shape is never touched, at any rank.
[ElementWiseShape(Note = "Identity forward, gradient barrier backward. Shape untouched at any rank.")]
[AutoParameters]
public partial class StopGradientLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>Initializes a new stop-gradient barrier.</summary>
    public StopGradientLayer()
        : base(new[] { -1 }, new[] { -1 })
    {
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Engine.StopGradient(input);

    /// <inheritdoc/>
    /// <remarks>The barrier passes its input through untouched, so its output shape is its input shape.</remarks>
    protected override bool IsShapePreserving => true;

    /// <inheritdoc/>
    public override void ResetState()
    {
    }
}
