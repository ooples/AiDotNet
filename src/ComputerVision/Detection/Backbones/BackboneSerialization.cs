using System.IO;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Binary serialization helpers shared by detection backbones (ResNet,
/// CSPDarknet, EfficientNet, SwinTransformer). Replaces the per-wrapper
/// <c>WriteParameters</c> / <c>ReadParameters</c> methods that lived on
/// the legacy <c>Conv2D</c> / <c>Dense</c> / <c>MultiHeadSelfAttention</c>
/// shims under the deleted <c>ConvUtils</c> file.
/// </summary>
/// <remarks>
/// Each helper writes <c>length</c> followed by the layer's flat parameter
/// vector (as <c>double</c> for cross-precision portability), then the same for
/// the layer's non-trainable state (a BatchNorm's running statistics). The reader
/// mirrors that and pushes the vector back through <c>SetParameters</c>;
/// for lazy layers that haven't seen their first <c>Forward</c> yet, the
/// pre-Forward Deserialize → SetParameters replay path on the layer
/// itself buffers the vector and applies it in <c>OnFirstForward</c>.
/// </remarks>
internal static class BackboneSerialization
{
    public static void WriteLayerParameters<T>(BinaryWriter writer, LayerBase<T> layer)
    {
        var p = layer.GetParameters();
        writer.Write(p.Length);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < p.Length; i++)
            writer.Write(ops.ToDouble(p[i]));
        var extras = layer is ILayerSerializationExtras<T> e ? e.GetExtraParameters() : new Vector<T>(0);
        writer.Write(extras.Length);
        for (int i = 0; i < extras.Length; i++)
            writer.Write(ops.ToDouble(extras[i]));
    }

    public static void ReadLayerParameters<T>(BinaryReader reader, LayerBase<T> layer)
    {
        int len = reader.ReadInt32();
        if (len < 0)
            throw new InvalidDataException(
                $"Negative parameter length ({len}) on the wire — corrupt parameter stream.");
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new T[len];
        for (int i = 0; i < len; i++)
            values[i] = ops.FromDouble(reader.ReadDouble());
        if (len > 0) layer.SetParameters(new Vector<T>(values));
        int extraLength = reader.ReadInt32();
        if (extraLength < 0)
            throw new InvalidDataException(
                $"Negative state length ({extraLength}) on the wire — corrupt parameter stream.");
        var extras = new T[extraLength];
        for (int i = 0; i < extraLength; i++)
            extras[i] = ops.FromDouble(reader.ReadDouble());
        if (extraLength == 0) return;
        if (layer is not ILayerSerializationExtras<T> e)
            throw new InvalidDataException($"{layer.GetType().Name} carries no state, but the stream has {extraLength} values.");
        e.SetExtraParameters(new Vector<T>(extras));
    }
}
