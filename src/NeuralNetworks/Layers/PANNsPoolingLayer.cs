using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements CNN14's channel-preserving frequency/time aggregation for NCHW features.
/// </summary>
/// <remarks>
/// The released PANNs CNN14 first averages the mel-frequency axis, then adds the
/// maximum and mean across time. Engine reductions are used so gradients remain on
/// the training tape. This layer is internal because it is an architectural detail
/// of the PANNs factory rather than a general channels-last global-pooling variant.
/// </remarks>
internal sealed class PANNsPoolingLayer<T> : LayerBase<T>
{
    public PANNsPoolingLayer()
        : base(new[] { -1, -1, -1 }, new[] { -1 })
    {
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    public override long ParameterCount => 0;

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(
                $"PANNsPoolingLayer expects NCHW rank-4 input, got rank {input.Rank}.",
                nameof(input));
        }

        ResolveShapes(
            new[] { input.Shape[1], input.Shape[2], input.Shape[3] },
            new[] { input.Shape[1] });
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        // [B,C,T,F] -> mean_F -> [B,C,T] -> (max_T + mean_T) -> [B,C].
        var frequencyMean = Engine.ReduceMean(input, new[] { 3 }, keepDims: false);
        var temporalMean = Engine.ReduceMean(frequencyMean, new[] { 2 }, keepDims: false);
        var temporalMax = Engine.ReduceMax(frequencyMean, new[] { 2 }, keepDims: false, out _);
        return Engine.TensorAdd(temporalMax, temporalMean);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
            throw new ArgumentException($"PANNsPoolingLayer has no parameters; got {parameters.Length}.", nameof(parameters));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { }

    /// <inheritdoc/>
    public override void ResetState() { }
}
