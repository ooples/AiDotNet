using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// A standard bidirectional Transformer denoiser block for the native Seed implementation.
/// The technical report does not disclose ByteDance's internal block configuration.
/// </summary>
internal sealed class SeedTTSDiffusionBlock<T>
{
    private readonly IEngine _engine;

    internal SeedTTSDiffusionBlock(IEngine engine, List<LayerBase<T>> layers, int width, int heads)
    {
        _engine = engine;
        Attention = new MultiHeadAttentionLayer<T>(heads, width / heads, activationFunction: new IdentityActivation<T>());
        Norm1 = new LayerNormalizationLayer<T>(width);
        FeedForwardIn = new DenseLayer<T>(width * 4, new IdentityActivation<T>() as IActivationFunction<T>);
        FeedForwardOut = new DenseLayer<T>(width, new IdentityActivation<T>() as IActivationFunction<T>);
        Norm2 = new LayerNormalizationLayer<T>(width);
        layers.Add(Attention); layers.Add(Norm1); layers.Add(FeedForwardIn); layers.Add(FeedForwardOut); layers.Add(Norm2);
    }

    internal MultiHeadAttentionLayer<T> Attention { get; }
    internal LayerNormalizationLayer<T> Norm1 { get; }
    internal DenseLayer<T> FeedForwardIn { get; }
    internal DenseLayer<T> FeedForwardOut { get; }
    internal LayerNormalizationLayer<T> Norm2 { get; }

    internal Tensor<T> Forward(Tensor<T> input)
    {
        var hidden = Norm1.Forward(_engine.TensorAdd(input, Attention.Forward(input)));
        var flattened = _engine.Reshape(hidden, new[] { hidden.Shape[0] * hidden.Shape[1], hidden.Shape[2] });
        var expanded = ActivationHelper.ExactGelu(FeedForwardIn.Forward(flattened), _engine);
        var residual = _engine.Reshape(FeedForwardOut.Forward(expanded), hidden.Shape.ToArray());
        return Norm2.Forward(_engine.TensorAdd(hidden, residual));
    }
}
