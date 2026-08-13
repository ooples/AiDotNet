using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models;

/// <summary>
/// Attaches declared state to a payload whose format this code does not know.
/// </summary>
/// <remarks>
/// <para>
/// There are twenty-six parallel model base hierarchies, each with its own <c>byte[] Serialize()</c>
/// and its own format -- some JSON, some binary, some nested metadata. They are siblings over the
/// same interfaces rather than a hierarchy, so there is no one place to put a state block and no
/// single format to put it in.
/// </para>
/// <para>
/// So the state is appended as a SUFFIX and located from the END: the trailer is the magic and the
/// block length, which means the existing payload in front of it is untouched and does not need to
/// be understood. A payload written before this existed simply has no trailer, so it reads back
/// exactly as it always did -- old checkpoints keep working instead of failing on a format they
/// could not have known about.
/// </para>
/// </remarks>
public static class ModelStateEnvelope
{
    private const int Magic = unchecked((int)0xA1D057A7);
    private const int TrailerLength = sizeof(int) * 2;

    /// <summary>Appends the declared state to a payload, or returns it unchanged when none exists.</summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="state">The model's declared state.</param>
    /// <param name="payload">Whatever the base already produced.</param>
    /// <returns>The payload, with a state trailer when there is state to carry.</returns>
    public static byte[] Append<T>(ModelStateRegistry<T> state, byte[] payload)
    {
        if (state is null || state.Count == 0) return payload;

        using var buffer = new MemoryStream();
        using (var writer = new BinaryWriter(buffer, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            state.WriteAll(writer);
            writer.Flush();
        }

        var block = buffer.ToArray();
        var result = new byte[payload.Length + block.Length + TrailerLength];

        Buffer.BlockCopy(payload, 0, result, 0, payload.Length);
        Buffer.BlockCopy(block, 0, result, payload.Length, block.Length);
        Buffer.BlockCopy(BitConverter.GetBytes(block.Length), 0, result, payload.Length + block.Length, sizeof(int));
        Buffer.BlockCopy(BitConverter.GetBytes(Magic), 0, result, payload.Length + block.Length + sizeof(int), sizeof(int));

        return result;
    }

    /// <summary>Applies and strips a state trailer, returning the payload the base should read.</summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="state">The model's declared state.</param>
    /// <param name="payload">The stored bytes.</param>
    /// <returns>The payload without its trailer, or the original when there is none.</returns>
    public static byte[] Extract<T>(ModelStateRegistry<T> state, byte[] payload)
    {
        if (payload is null || payload.Length < TrailerLength) return payload ?? Array.Empty<byte>();

        int magic = BitConverter.ToInt32(payload, payload.Length - sizeof(int));
        if (magic != Magic) return payload;

        int blockLength = BitConverter.ToInt32(payload, payload.Length - TrailerLength);
        int innerLength = payload.Length - TrailerLength - blockLength;
        if (blockLength < 0 || innerLength < 0) return payload;

        if (state is not null && state.Count > 0)
        {
            using var buffer = new MemoryStream(payload, innerLength, blockLength);
            using var reader = new BinaryReader(buffer, System.Text.Encoding.UTF8, leaveOpen: true);
            state.ReadAll(reader);
        }

        var inner = new byte[innerLength];
        Buffer.BlockCopy(payload, 0, inner, 0, innerLength);
        return inner;
    }
}

/// <summary>
/// The declared home for model state that is not a flat parameter vector.
/// </summary>
/// <typeparam name="T">The model's numeric type.</typeparam>
/// <remarks>
/// <para>
/// <see cref="ModelBase{T, TInput, TOutput}"/> persists whatever the model declared through its
/// parameter components, which covers every model whose learned state IS its parameter vector. It is
/// not everything. A k-nearest-neighbours model's state is the training set; a random forest's is a
/// list of trees; a GAM's is its fitted knot vectors; a kernel ridge model's is its centres and dual
/// coefficients. None of that fits a flat vector, none of it had anywhere to be declared, and so
/// every one of those models hand-wrote a Serialize/Deserialize pair -- which is two places to forget
/// the same field.
/// </para>
/// <para>
/// This is the model-side analogue of <c>LayerBase.RegisterBuffer</c>. A model DECLARES a piece of
/// state once, by name, with a getter and a setter; the base writes and reads it. The author never
/// touches a <see cref="BinaryWriter"/>, and because both halves are driven by the one registration
/// they cannot drift apart -- the failure mode a hand-written pair cannot detect.
/// </para>
/// <para>
/// Keyed by NAME, and a name in the payload with no matching registration is skipped rather than
/// fatal, so adding state does not invalidate existing checkpoints. Values are written in
/// registration order, but read by name, so re-ordering registrations is also safe.
/// </para>
/// </remarks>
public sealed class ModelStateRegistry<T>
{
    private readonly List<Entry> _entries = new();
    private readonly HashSet<string> _names = new(StringComparer.Ordinal);

    private sealed class Entry
    {
        public string Name = string.Empty;
        public Action<BinaryWriter> Write = _ => { };
        public Action<BinaryReader> Read = _ => { };
    }

    /// <summary>Gets the number of declared state entries.</summary>
    public int Count => _entries.Count;

    private void Add(string name, Action<BinaryWriter> write, Action<BinaryReader> read)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("State name must not be empty.", nameof(name));

        // A duplicate name would make the payload ambiguous on the way back in, and the value that
        // won would depend on registration order -- exactly the kind of order dependence that makes
        // a restore differ from a save for reasons nobody can see.
        if (!_names.Add(name))
            throw new ArgumentException($"State '{name}' is already declared on this model.", nameof(name));

        _entries.Add(new Entry { Name = name, Write = write, Read = read });
    }

    /// <summary>Declares a vector, such as a fitted knot vector or a set of dual coefficients.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Vector<T>?> get, Action<Vector<T>?> set)
        => Add(name,
            w => WriteVector(w, get()),
            r => set(ReadVector(r)));

    /// <summary>Declares a matrix, such as the retained training set of an instance-based model.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Matrix<T>?> get, Action<Matrix<T>?> set)
        => Add(name,
            w => WriteMatrix(w, get()),
            r => set(ReadMatrix(r)));

    /// <summary>Declares a list of vectors, such as per-feature knots or per-output coefficients.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    /// <remarks>
    /// A per-feature collection is one piece of state, not N of them: restoring some entries and not
    /// others gives a model that is fitted for part of its input and defaulted for the rest, which
    /// predicts without complaining.
    /// </remarks>
    public void Declare(string name, Func<List<Vector<T>>?> get, Action<List<Vector<T>>?> set)
        => Add(name,
            w =>
            {
                var list = get();
                if (list is null) { w.Write(-1); return; }
                w.Write(list.Count);
                foreach (var v in list) WriteVector(w, v);
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var list = new List<Vector<T>>(count);
                for (int i = 0; i < count; i++) list.Add(ReadVector(r) ?? new Vector<T>(0));
                set(list);
            });

    /// <summary>Declares a tensor.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Tensor<T>?> get, Action<Tensor<T>?> set)
        => Add(name,
            w => WriteTensor(w, get()),
            r => set(ReadTensor(r)));

    /// <summary>Declares an integer array, such as node indices or a feature mapping.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<int[]?> get, Action<int[]?> set)
        => Add(name,
            w => WriteInts(w, get()),
            r => set(ReadInts(r)));

    /// <summary>Declares a double array.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<double[]?> get, Action<double[]?> set)
        => Add(name,
            w => WriteDoubles(w, get()),
            r => set(ReadDoubles(r)));

    // Scalars. A hyperparameter that PREDICTION reads is state, however small: k-nearest-neighbours
    // restored its training set correctly and still predicted differently, because K came back as the
    // constructor default and the model was voting over the wrong number of neighbours. A field does
    // not have to be big to change the answer.

    /// <summary>Declares an integer, such as a neighbour count or a tree depth.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareInt32(string name, Func<int> get, Action<int> set)
        => Add(name, w => w.Write(get()), r => set(r.ReadInt32()));

    /// <summary>Declares a double, such as a temperature or a learned threshold.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareDouble(string name, Func<double> get, Action<double> set)
        => Add(name, w => w.Write(get()), r => set(r.ReadDouble()));

    /// <summary>Declares a boolean, such as a fitted flag or a mode switch.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareBoolean(string name, Func<bool> get, Action<bool> set)
        => Add(name, w => w.Write(get()), r => set(r.ReadBoolean()));

    /// <summary>Declares a numeric value held as the model's own numeric type.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareScalar(string name, Func<T> get, Action<T> set)
        => Add(name,
            w => w.Write(Convert.ToDouble(get())),
            r => set(Ops.FromDouble(r.ReadDouble())));

    /// <summary>Declares a string, such as a fitted category name or a chosen kernel.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareString(string name, Func<string?> get, Action<string?> set)
        => Add(name,
            w => { var v = get(); w.Write(v is not null); if (v is not null) w.Write(v); },
            r => set(r.ReadBoolean() ? r.ReadString() : null));

    /// <summary>
    /// Declares a child model whose own state travels with this one -- an ensemble member, a base
    /// learner, a per-output head.
    /// </summary>
    /// <typeparam name="TInput">The child's input type.</typeparam>
    /// <typeparam name="TOutput">The child's output type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current children, in a stable order.</param>
    /// <remarks>
    /// The children are NOT reconstructed here. A restore writes each child's saved bytes into the
    /// child the parent already built, so the parent's own construction decides how many children
    /// there are and of what type -- which is configuration, and configuration is replayed by the
    /// recorded constructor rather than carried in a state payload.
    /// </remarks>
    public void DeclareChildren<TInput, TOutput>(
        string name,
        Func<IReadOnlyList<IFullModel<T, TInput, TOutput>>?> get)
        => Add(name,
            w =>
            {
                var children = get();
                if (children is null) { w.Write(-1); return; }

                w.Write(children.Count);
                foreach (var child in children)
                {
                    var bytes = child?.Serialize() ?? Array.Empty<byte>();
                    w.Write(bytes.Length);
                    w.Write(bytes);
                }
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) return;

                var children = get();
                for (int i = 0; i < count; i++)
                {
                    int length = r.ReadInt32();
                    var bytes = r.ReadBytes(length);

                    // A parent rebuilt with a different child count is a configuration mismatch, and
                    // loading the first N anyway would leave the tail silently untrained.
                    if (children is not null && i < children.Count && length > 0)
                    {
                        children[i]?.Deserialize(bytes);
                    }
                }
            });

    /// <summary>Writes every declared entry, name-tagged and length-prefixed.</summary>
    /// <param name="writer">The writer receiving the state block.</param>
    /// <remarks>
    /// Each entry's payload is length-prefixed so an unknown name can be SKIPPED on the way in.
    /// Without that, a checkpoint containing state this build no longer declares would desynchronise
    /// the stream and corrupt everything after it.
    /// </remarks>
    public void WriteAll(BinaryWriter writer)
    {
        writer.Write(_entries.Count);

        foreach (var entry in _entries)
        {
            writer.Write(entry.Name);

            using var buffer = new MemoryStream();
            using (var inner = new BinaryWriter(buffer, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                entry.Write(inner);
                inner.Flush();
            }

            var bytes = buffer.ToArray();
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }
    }

    /// <summary>Restores every entry the payload and this model have in common.</summary>
    /// <param name="reader">The reader positioned at the state block.</param>
    public void ReadAll(BinaryReader reader)
    {
        int count = reader.ReadInt32();

        var byName = new Dictionary<string, Entry>(StringComparer.Ordinal);
        foreach (var entry in _entries) byName[entry.Name] = entry;

        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            int length = reader.ReadInt32();
            var bytes = reader.ReadBytes(length);

            if (!byName.TryGetValue(name, out var entry)) continue;

            using var buffer = new MemoryStream(bytes);
            using var inner = new BinaryReader(buffer, System.Text.Encoding.UTF8, leaveOpen: true);
            entry.Read(inner);
        }
    }

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static void WriteVector(BinaryWriter w, Vector<T>? v)
    {
        if (v is null) { w.Write(-1); return; }
        w.Write(v.Length);
        for (int i = 0; i < v.Length; i++) w.Write(Convert.ToDouble(v[i]));
    }

    private static Vector<T>? ReadVector(BinaryReader r)
    {
        int length = r.ReadInt32();
        if (length < 0) return null;
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++) v[i] = Ops.FromDouble(r.ReadDouble());
        return v;
    }

    private static void WriteMatrix(BinaryWriter w, Matrix<T>? m)
    {
        if (m is null) { w.Write(-1); return; }
        w.Write(m.Rows);
        w.Write(m.Columns);
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                w.Write(Convert.ToDouble(m[i, j]));
    }

    private static Matrix<T>? ReadMatrix(BinaryReader r)
    {
        int rows = r.ReadInt32();
        if (rows < 0) return null;
        int columns = r.ReadInt32();
        var m = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                m[i, j] = Ops.FromDouble(r.ReadDouble());
        return m;
    }

    private static void WriteTensor(BinaryWriter w, Tensor<T>? t)
    {
        if (t is null) { w.Write(-1); return; }
        var shape = t.Shape;
        w.Write(shape.Length);
        for (int i = 0; i < shape.Length; i++) w.Write(shape[i]);
        w.Write(t.Length);
        for (int i = 0; i < t.Length; i++) w.Write(Convert.ToDouble(t[i]));
    }

    private static Tensor<T>? ReadTensor(BinaryReader r)
    {
        int rank = r.ReadInt32();
        if (rank < 0) return null;
        var shape = new int[rank];
        for (int i = 0; i < rank; i++) shape[i] = r.ReadInt32();
        int length = r.ReadInt32();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < length && i < t.Length; i++) t[i] = Ops.FromDouble(r.ReadDouble());
        return t;
    }

    private static void WriteInts(BinaryWriter w, int[]? a)
    {
        if (a is null) { w.Write(-1); return; }
        w.Write(a.Length);
        foreach (var value in a) w.Write(value);
    }

    private static int[]? ReadInts(BinaryReader r)
    {
        int length = r.ReadInt32();
        if (length < 0) return null;
        var a = new int[length];
        for (int i = 0; i < length; i++) a[i] = r.ReadInt32();
        return a;
    }

    private static void WriteDoubles(BinaryWriter w, double[]? a)
    {
        if (a is null) { w.Write(-1); return; }
        w.Write(a.Length);
        foreach (var value in a) w.Write(value);
    }

    private static double[]? ReadDoubles(BinaryReader r)
    {
        int length = r.ReadInt32();
        if (length < 0) return null;
        var a = new double[length];
        for (int i = 0; i < length; i++) a[i] = r.ReadDouble();
        return a;
    }
}
