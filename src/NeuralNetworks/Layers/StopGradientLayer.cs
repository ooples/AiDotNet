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
public class StopGradientLayer<T> : LayerBase<T>
{
    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    public override long ParameterCount => 0;

    /// <summary>Initializes a new stop-gradient barrier.</summary>
    public StopGradientLayer()
        : base(new[] { -1 }, new[] { -1 })
    {
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input) => Engine.StopGradient(input);

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => new(0);

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
            throw new ArgumentException("StopGradientLayer has no parameters.", nameof(parameters));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
    }
}
