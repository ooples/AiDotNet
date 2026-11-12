using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
/// <para>
/// Three flavours are provided:
/// <list type="bullet">
/// <item><see cref="Write{T}"/> / <see cref="Read{T}"/> — a single optional layer whose size is
/// embedded so it can be rebuilt from scratch.</item>
/// <item><see cref="WriteParameters{T}"/> / <see cref="ReadParametersInto{T}"/> — a list of layers
/// whose <i>structure</i> is rebuilt deterministically by the caller (only the learned parameters
/// and extras round-trip).</item>
/// <item><see cref="WriteLayerList{T}"/> / <see cref="ReadLayerList{T, TLayer}"/> — a homogeneous
/// list rebuilt entirely from the serialized data (shape + state) via a simple factory.</item>
/// </list>
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

        WriteIntArray(writer, layer.GetInputShape());
        WriteIntArray(writer, layer.GetOutputShape());
        WriteLayerState(writer, layer);
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

        int[] inputShape = ReadIntArray(reader);
        int[] outputShape = ReadIntArray(reader);

        var layer = factory(inputShape, outputShape);
        if (inputShape.Length > 0 && inputShape[0] > 0 && layer is LayerBase<T> resolvable)
        {
            resolvable.ResolveFromPublishedShape(inputShape);
        }

        ReadLayerState(reader, layer);
        return layer;
    }

    /// <summary>
    /// Writes the trainable parameters (and serialization extras) of every layer in
    /// <paramref name="layers"/>. The layer <i>structure</i> is assumed to be rebuilt
    /// deterministically by the caller before <see cref="ReadParametersInto{T}"/> is invoked.
    /// </summary>
    public static void WriteParameters<T>(BinaryWriter writer, IReadOnlyList<ILayer<T>> layers)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        if (layers is null) throw new ArgumentNullException(nameof(layers));

        writer.Write(layers.Count);
        for (int i = 0; i < layers.Count; i++)
        {
            WriteLayerState(writer, layers[i]);
        }
    }

    /// <summary>
    /// Restores parameters (and serialization extras) written by <see cref="WriteParameters{T}"/>
    /// into the matching, already-rebuilt layers in <paramref name="layers"/>.
    /// </summary>
    public static void ReadParametersInto<T>(BinaryReader reader, IReadOnlyList<ILayer<T>> layers)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        if (layers is null) throw new ArgumentNullException(nameof(layers));

        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            ILayer<T>? target = i < layers.Count ? layers[i] : null;
            ReadLayerState(reader, target);
        }
    }

    /// <summary>
    /// Writes a homogeneous list of layers (count + each layer's shape and state) so it can be
    /// rebuilt entirely by <see cref="ReadLayerList{T, TLayer}"/> without any external structure.
    /// </summary>
    public static void WriteLayerList<T>(BinaryWriter writer, IReadOnlyList<ILayer<T>> layers)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        if (layers is null) throw new ArgumentNullException(nameof(layers));

        writer.Write(layers.Count);
        for (int i = 0; i < layers.Count; i++)
        {
            Write(writer, layers[i]);
        }
    }

    /// <summary>
    /// Rebuilds a homogeneous list of layers previously written by <see cref="WriteLayerList{T}"/>,
    /// replacing the contents of <paramref name="target"/>.
    /// </summary>
    /// <typeparam name="TLayer">The concrete layer type held in the list.</typeparam>
    /// <param name="reader">The reader positioned at the serialized list.</param>
    /// <param name="target">The list to clear and repopulate.</param>
    /// <param name="factory">Builds a fresh layer instance sized for the deserialized shapes.</param>
    public static void ReadLayerList<T, TLayer>(BinaryReader reader, List<TLayer> target, Func<int[], int[], TLayer> factory)
        where TLayer : class, ILayer<T>
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (factory is null) throw new ArgumentNullException(nameof(factory));

        target.Clear();
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            var layer = Read<T>(reader, (inShape, outShape) => factory(inShape, outShape)) as TLayer;
            if (layer is not null)
            {
                target.Add(layer);
            }
        }
    }

    private static void WriteLayerState<T>(BinaryWriter writer, ILayer<T> layer)
    {
        WriteElements(writer, layer.GetParameters());

        if (layer is ILayerSerializationExtras<T> extras)
        {
            WriteElements(writer, extras.GetExtraParameters());
        }
        else
        {
            writer.Write(0);
        }
    }

    private static void ReadLayerState<T>(BinaryReader reader, ILayer<T>? layer)
    {
        var parameters = ReadElements<T>(reader);
        if (parameters.Length > 0 && layer is not null)
        {
            layer.SetParameters(parameters);
        }

        var extraParameters = ReadElements<T>(reader);
        if (extraParameters.Length > 0 && layer is ILayerSerializationExtras<T> extras)
        {
            extras.SetExtraParameters(extraParameters);
        }
    }

    /// <summary>
    /// Writes <paramref name="values"/> as its element count, the element size in bytes, and the
    /// raw element bytes of <typeparamref name="T"/>. The bytes are taken straight from the
    /// vector's span (<c>MemoryMarshal</c> byte view), so values round-trip bit-for-bit in
    /// <typeparamref name="T"/> without a per-element conversion.
    /// </summary>
    private static void WriteElements<T>(BinaryWriter writer, Vector<T> values)
    {
        int elementSize = GetElementSize<T>();
        writer.Write(values.Length);
        if (values.Length == 0)
        {
            return;
        }

        writer.Write(elementSize);
        ReadOnlySpan<T> span = values.AsSpan();
        int byteCount = checked(span.Length * elementSize);
        ref byte first = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(span));
#if NETCOREAPP2_1_OR_GREATER || NETSTANDARD2_1_OR_GREATER
        writer.Write(MemoryMarshal.CreateReadOnlySpan(ref first, byteCount));
#else
        var buffer = new byte[byteCount];
        Unsafe.CopyBlockUnaligned(ref buffer[0], ref first, (uint)byteCount);
        writer.Write(buffer);
#endif
    }

    /// <summary>
    /// Reads a vector written by <see cref="WriteElements{T}"/>, copying the raw bytes straight
    /// into the new vector's storage.
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// The stream was written for an element type of a different size, or is truncated.
    /// </exception>
    private static Vector<T> ReadElements<T>(BinaryReader reader)
    {
        int elementSize = GetElementSize<T>();
        int count = reader.ReadInt32();
        if (count < 0)
        {
            throw new InvalidDataException($"Negative element count {count} in auxiliary layer state.");
        }

        var values = new Vector<T>(count);
        if (count == 0)
        {
            return values;
        }

        int storedElementSize = reader.ReadInt32();
        if (storedElementSize != elementSize)
        {
            throw new InvalidDataException(
                $"Auxiliary layer state was written with {storedElementSize}-byte elements, but "
                + $"{typeof(T).Name} uses {elementSize}-byte elements.");
        }

        int byteCount = checked(count * elementSize);
        byte[] bytes = reader.ReadBytes(byteCount);
        if (bytes.Length != byteCount)
        {
            throw new InvalidDataException(
                $"Auxiliary layer state is truncated: expected {byteCount} bytes, read {bytes.Length}.");
        }

        Span<T> target = values.AsWritableSpan();
        Unsafe.CopyBlockUnaligned(
            ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(target)),
            ref bytes[0],
            (uint)byteCount);
        return values;
    }

    private static int GetElementSize<T>()
    {
        if (!typeof(T).IsValueType)
        {
            throw new NotSupportedException(
                $"Auxiliary layer serialization needs a value-type element, but got {typeof(T).Name}.");
        }

        return Unsafe.SizeOf<T>();
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
