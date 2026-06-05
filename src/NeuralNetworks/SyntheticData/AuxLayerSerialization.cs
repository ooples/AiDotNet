using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Serialization helpers for "auxiliary" sub-networks that synthetic-data generators keep
/// <i>outside</i> the base <see cref="NeuralNetworkBase{T}.Layers"/> collection (output heads,
/// decoders, generator batch-norm, timestep projections, ...).
/// </summary>
/// <remarks>
/// <para>
/// The base <see cref="NeuralNetworkBase{T}"/> serialization only persists the layers in the
/// <c>Layers</c> list. Generators that train additional sub-networks held in private fields must
/// therefore persist those weights themselves, or a saved/cloned model would silently fall back
/// to freshly-initialized auxiliary weights and generate garbage.
/// </para>
/// <para>
/// This helper writes each auxiliary layer's input/output shape, its trainable parameters, and —
/// for layers that carry non-trainable buffers (e.g. <see cref="BatchNormalizationLayer{T}"/>
/// running mean/variance via <see cref="ILayerSerializationExtras{T}"/>) — those extras as well,
/// so the round-trip is bit-for-bit faithful.
/// </para>
/// <para><b>For Beginners:</b> some of these generators have small helper networks that live in
/// private variables instead of the main layer list. When you save and reload the model, the main
/// layer list is restored automatically, but these helpers are not — unless we save them too. This
/// class does exactly that: it writes the helper network's shape and learned numbers to the file
/// and reads them back when loading.</para>
/// </remarks>
internal static class AuxLayerSerialization
{
    /// <summary>
    /// Writes an optional auxiliary layer (shape + trainable parameters + serialization extras).
    /// A leading boolean records whether the layer was present so <see cref="Read{T}"/> can mirror it.
    /// </summary>
    public static void Write<T>(BinaryWriter writer, ILayer<T>? layer)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));

        if (layer is null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);

        var numOps = MathHelper.GetNumericOperations<T>();
        WriteIntArray(writer, layer.GetInputShape());
        WriteIntArray(writer, layer.GetOutputShape());

        var parameters = layer.GetParameters();
        writer.Write(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            writer.Write(numOps.ToDouble(parameters[i]));
        }

        if (layer is ILayerSerializationExtras<T> extras)
        {
            var extraParameters = extras.GetExtraParameters();
            writer.Write(extraParameters.Length);
            for (int i = 0; i < extraParameters.Length; i++)
            {
                writer.Write(numOps.ToDouble(extraParameters[i]));
            }
        }
        else
        {
            writer.Write(0);
        }
    }

    /// <summary>
    /// Reads an optional auxiliary layer previously written by <see cref="Write{T}"/>.
    /// </summary>
    /// <param name="reader">The reader positioned at the serialized layer.</param>
    /// <param name="factory">
    /// Builds a fresh layer instance sized for the deserialized input/output shapes. The helper then
    /// resolves the layer's shape and restores its parameters and extras.
    /// </param>
    /// <returns>The restored layer, or <see langword="null"/> if none was written.</returns>
    public static ILayer<T>? Read<T>(BinaryReader reader, Func<int[], int[], ILayer<T>> factory)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        if (factory is null) throw new ArgumentNullException(nameof(factory));

        if (!reader.ReadBoolean())
        {
            return null;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int[] inputShape = ReadIntArray(reader);
        int[] outputShape = ReadIntArray(reader);

        var layer = factory(inputShape, outputShape);
        if (inputShape.Length > 0 && inputShape[0] > 0 && layer is LayerBase<T> resolvable)
        {
            resolvable.ResolveFromShape(inputShape);
        }

        int parameterCount = reader.ReadInt32();
        var parameters = new Vector<T>(parameterCount);
        for (int i = 0; i < parameterCount; i++)
        {
            parameters[i] = numOps.FromDouble(reader.ReadDouble());
        }
        if (parameterCount > 0)
        {
            layer.SetParameters(parameters);
        }

        int extraCount = reader.ReadInt32();
        var extraParameters = new Vector<T>(extraCount);
        for (int i = 0; i < extraCount; i++)
        {
            extraParameters[i] = numOps.FromDouble(reader.ReadDouble());
        }
        if (extraCount > 0 && layer is ILayerSerializationExtras<T> extras)
        {
            extras.SetExtraParameters(extraParameters);
        }

        return layer;
    }

    private static void WriteIntArray(BinaryWriter writer, int[]? values)
    {
        if (values is null)
        {
            writer.Write(0);
            return;
        }

        writer.Write(values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }

    private static int[] ReadIntArray(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var values = new int[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = reader.ReadInt32();
        }

        return values;
    }
}
