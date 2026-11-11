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
/// vector (as <c>double</c> for cross-precision portability). The reader
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
        foreach (double value in CvTensorOps<T>.ToDoubleArray(p.ToArray()))
            writer.Write(value);
    }

    public static void ReadLayerParameters<T>(BinaryReader reader, LayerBase<T> layer)
    {
        int len = reader.ReadInt32();
        if (len < 0)
            throw new InvalidDataException(
                $"Negative parameter length ({len}) on the wire — corrupt parameter stream.");
        if (len == 0) return;
        var wire = new double[len];
        for (int i = 0; i < len; i++)
            wire[i] = reader.ReadDouble();
        layer.SetParameters(new Vector<T>(CvTensorOps<T>.FromDoubleArray(wire, new[] { len }).ToArray()));
    }

    /// <summary>Writes every value of <paramref name="tensor"/> as a <see cref="double"/>, in order.</summary>
    public static void WriteValues<T>(BinaryWriter writer, Tensor<T> tensor)
        => CvTensorOps<T>.WriteDoubles(writer, tensor);

    /// <summary>Reads <c>tensor.Length</c> doubles and writes them through into <paramref name="tensor"/>.</summary>
    public static void ReadValuesInto<T>(BinaryReader reader, Tensor<T> tensor)
        => CvTensorOps<T>.ReadDoublesInto(reader, tensor);
}
