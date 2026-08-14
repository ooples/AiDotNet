using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Parameter-owning Bark boundary around an EnCodec-compatible audio codec.
/// </summary>
/// <remarks>
/// A codec is a model in its own right, but Bark owns it as its fourth stage. Registering the
/// nested codec layers here makes the ordinary layer graph the single source of truth for
/// parameters, gradients, serialization, cloning, and device placement.
/// </remarks>
[AutoParameters]
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, SupportsBackpropagation = false, ChangesShape = true, Cost = ComputeCost.High)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
internal partial class BarkCodecLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly IAudioCodec<T> _codec;

    internal BarkCodecLayer(IAudioCodec<T> codec)
        : base([LayerShape.Dynamic, LayerShape.Dynamic], [LayerShape.Dynamic])
    {
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        if (codec is NeuralNetworkBase<T> network)
        {
            foreach (var layer in network.Layers)
            {
                if (layer is LayerBase<T> trainable)
                    RegisterSubLayer(trainable);
            }
        }
    }

    internal IAudioCodec<T> Codec => _codec;

    public override bool SupportsTraining => false;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? [new OutputAxisContract(
                TensorAxis.Time,
                AxisRelation.Unknown("The injected codec determines samples per frame."))]
            : null;

    internal int[,] Encode(Tensor<T> audio) => _codec.Encode(audio);

    internal Task<int[,]> EncodeAsync(Tensor<T> audio, CancellationToken cancellationToken)
        => _codec.EncodeAsync(audio, cancellationToken);

    internal Tensor<T> Decode(int[,] tokens) => _codec.Decode(tokens);

    internal Task<Tensor<T>> DecodeAsync(int[,] tokens, CancellationToken cancellationToken)
        => _codec.DecodeAsync(tokens, cancellationToken);

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Shape.Length != 2)
            throw new ArgumentException("Bark codec input must have shape [codebook, frame].", nameof(input));

        int codebooks = input.Shape[0];
        int frames = input.Shape[1];
        var tokens = new int[codebooks, frames];
        for (int codebook = 0; codebook < codebooks; codebook++)
        {
            for (int frame = 0; frame < frames; frame++)
                tokens[codebook, frame] = Convert.ToInt32(NumOps.ToDouble(input[codebook, frame]));
        }
        return _codec.Decode(tokens);
    }

    public override void ResetState()
    {
        if (_codec is NeuralNetworkBase<T> network)
            network.ResetState();
    }
}
