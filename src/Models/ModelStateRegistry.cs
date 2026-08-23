using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
        => Extract(state, payload, restoreAfterParameters: null);

    /// <summary>
    /// Applies structural state and strips the envelope, deferring exact native-precision
    /// parameter shadows until the ordinary flat parameter vector has been restored.
    /// </summary>
    public static byte[] ExtractBeforeParameters<T>(ModelStateRegistry<T> state, byte[] payload)
        => Extract(state, payload, restoreAfterParameters: false);

    /// <summary>
    /// Applies only exact native-precision parameter shadows from an envelope. The returned inner
    /// payload is provided for symmetry and can be ignored by callers that already parsed it.
    /// </summary>
    public static byte[] ExtractAfterParameters<T>(ModelStateRegistry<T> state, byte[] payload)
        => Extract(state, payload, restoreAfterParameters: true);

    private static byte[] Extract<T>(
        ModelStateRegistry<T> state,
        byte[] payload,
        bool? restoreAfterParameters)
    {
        if (payload is null) throw new ArgumentNullException(nameof(payload));
        if (payload.Length < TrailerLength) return payload;

        int magic = BitConverter.ToInt32(payload, payload.Length - sizeof(int));
        if (magic != Magic) return payload;

        int blockLength = BitConverter.ToInt32(payload, payload.Length - TrailerLength);
        int innerLength = payload.Length - TrailerLength - blockLength;
        if (blockLength < 0 || innerLength < 0) return payload;

        if (state is not null && state.Count > 0)
        {
            using var buffer = new MemoryStream(payload, innerLength, blockLength);
            using var reader = new BinaryReader(buffer, System.Text.Encoding.UTF8, leaveOpen: true);
            if (!restoreAfterParameters.HasValue) state.ReadAll(reader);
            else if (restoreAfterParameters.Value) state.ReadAfterParameters(reader);
            else state.ReadBeforeParameters(reader);
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
    // A child-list payload used to begin directly with its count. A negative marker keeps old
    // payloads readable while allowing new payloads to record each child's concrete runtime type.
    private const int TypedChildListMarker = unchecked((int)0xA1D0C11D);

    private static readonly JsonSerializerSettings ObjectStateSettings = new()
    {
        ConstructorHandling = ConstructorHandling.AllowNonPublicDefaultConstructor,
        Formatting = Formatting.None,
        TypeNameHandling = TypeNameHandling.None,
        Converters = { new ModelSerializerJsonConverter() }
    };

    private readonly List<Entry> _entries = new();
    private readonly HashSet<string> _names = new(StringComparer.Ordinal);

    private sealed class Entry
    {
        public string Name = string.Empty;
        public Action<BinaryWriter> Write = _ => { };
        public Action<BinaryReader> Read = _ => { };
        public bool RestoreAfterParameters;
    }

    /// <summary>Gets the number of declared state entries.</summary>
    public int Count => _entries.Count;

    private void Add(
        string name,
        Action<BinaryWriter> write,
        Action<BinaryReader> read,
        bool restoreAfterParameters = false)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("State name must not be empty.", nameof(name));

        // A duplicate name would make the payload ambiguous on the way back in, and the value that
        // won would depend on registration order -- exactly the kind of order dependence that makes
        // a restore differ from a save for reasons nobody can see.
        if (!_names.Add(name))
            throw new ArgumentException($"State '{name}' is already declared on this model.", nameof(name));

        _entries.Add(new Entry
        {
            Name = name,
            Write = write,
            Read = read,
            RestoreAfterParameters = restoreAfterParameters
        });
    }

    /// <summary>Declares a vector, such as a fitted knot vector or a set of dual coefficients.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Vector<T>?> get, Action<Vector<T>?> set)
        => Add(name,
            w => WriteVector(w, get()),
            r => set(ReadVector(r)));

    /// <summary>Declares a byte vector, such as quantized optimizer moments.</summary>
    public void DeclareByteVector(string name, Func<Vector<byte>?> get, Action<Vector<byte>?> set)
        => Add(name,
            w =>
            {
                var vector = get();
                if (vector is null) { w.Write(-1); return; }
                w.Write(vector.Length);
                for (int i = 0; i < vector.Length; i++) w.Write(vector[i]);
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) { set(null); return; }
                var vector = new Vector<byte>(length);
                for (int i = 0; i < length; i++) vector[i] = r.ReadByte();
                set(vector);
            });

    /// <summary>Declares a double vector held by a model whose primary numeric type may differ.</summary>
    public void DeclareDoubleVector(string name, Func<Vector<double>?> get, Action<Vector<double>?> set)
        => Add(name,
            w =>
            {
                var vector = get();
                if (vector is null) { w.Write(-1); return; }
                w.Write(vector.Length);
                for (int i = 0; i < vector.Length; i++) w.Write(vector[i]);
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) { set(null); return; }
                var vector = new Vector<double>(length);
                for (int i = 0; i < length; i++) vector[i] = r.ReadDouble();
                set(vector);
            });

    /// <summary>Declares a matrix, such as the retained training set of an instance-based model.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Matrix<T>?> get, Action<Matrix<T>?> set)
        => Add(name,
            w => WriteMatrix(w, get()),
            r => set(ReadMatrix(r)));

    /// <summary>Declares an assignable fitted object, array, list, or dictionary.</summary>
    /// <typeparam name="TState">The compile-time state type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs the restored value.</param>
    /// <remarks>
    /// This is the general object-state path used by generated declarations for learned structures
    /// that are not numeric tensors: nested tree nodes, ensemble records, jagged arrays and similar
    /// model-owned data. Nested models still travel through their own serializer, so a POCO record
    /// that contains a model does not reduce that child to its public properties.
    /// </remarks>
    public void DeclareObject<TState>(string name, Func<TState?> get, Action<TState?> set)
        where TState : class
        => Add(name,
            w => WriteObjectState(w, get()),
            r => set(ReadObjectState<TState>(r, name)));

    /// <summary>Declares a readonly list or dictionary and restores its contents in place.</summary>
    /// <typeparam name="TState">The concrete collection type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the collection instance created by the model constructor.</param>
    /// <remarks>
    /// A readonly collection field means the reference is configuration, not that its fitted
    /// contents are immutable. The generator cannot assign that field, so the registry clears and
    /// refills the existing collection. Failing loudly when the constructor left it null prevents a
    /// successful-looking restore that silently drops the payload.
    /// </remarks>
    public void DeclareObjectInPlace<TState>(string name, Func<TState?> get)
        where TState : class
        => Add(name,
            w => WriteObjectState(w, get()),
            r =>
            {
                var restored = ReadObjectState<TState>(r, name);
                var current = get();
                if (current is null)
                {
                    throw new InvalidOperationException(
                        $"State '{name}' is held in a readonly collection, but its constructor left "
                        + "the collection null, so the restored contents have nowhere to go.");
                }

                CopyCollectionState(name, current, restored);
            });

    /// <summary>
    /// Declares a deterministic repair that runs after the ordinary state entries in the payload.
    /// </summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="restore">Rebuilds state derived from the entries restored before it.</param>
    /// <remarks>
    /// The entry intentionally writes no payload. Its presence in the name-framed state block makes
    /// the callback version-safe, while registration order ensures options and fitted dimensions are
    /// available before generated code reconstructs helpers that are derived from them.
    /// </remarks>
    public void DeclareAfterRestore(string name, Action restore)
    {
        if (restore is null) throw new ArgumentNullException(nameof(restore));
        Add(name, _ => { }, _ => restore());
    }

    /// <summary>Declares a readonly list of vectors and restores its contents in place.</summary>
    public void DeclareInPlace(string name, Func<List<Vector<T>>?> get)
        => Declare(name, get, restored => RestoreCollectionInPlace(name, get, restored));

    /// <summary>Declares a readonly list of matrices and restores its contents in place.</summary>
    public void DeclareInPlace(string name, Func<List<Matrix<T>>?> get)
        => Declare(name, get, restored => RestoreCollectionInPlace(name, get, restored));

    /// <summary>Declares a readonly list of tensors and restores its contents in place.</summary>
    public void DeclareInPlace(string name, Func<List<Tensor<T>>?> get)
        => Declare(name, get, restored => RestoreCollectionInPlace(name, get, restored));

    /// <summary>Declares a readonly string-keyed vector table and restores it in place.</summary>
    public void DeclareInPlace(string name, Func<Dictionary<string, Vector<T>>?> get)
        => Declare(name, get, restored => RestoreCollectionInPlace(name, get, restored));

    /// <summary>Declares a readonly integer-keyed vector table and restores it in place.</summary>
    public void DeclareInPlace(string name, Func<Dictionary<int, Vector<T>>?> get)
        => Declare(name, get, restored => RestoreCollectionInPlace(name, get, restored));

    private static void RestoreCollectionInPlace<TState>(
        string name,
        Func<TState?> get,
        TState? restored)
        where TState : class
    {
        var current = get();
        if (current is null)
        {
            throw new InvalidOperationException(
                $"State '{name}' is held in a readonly collection, but its constructor left "
                + "the collection null, so the restored contents have nowhere to go.");
        }

        CopyCollectionState(name, current, restored);
    }

    private static void WriteObjectState<TState>(BinaryWriter writer, TState? value)
        where TState : class
    {
        if (value is null) { writer.Write(false); return; }
        writer.Write(true);
        writer.Write(JsonConvert.SerializeObject(value, ObjectStateSettings));
    }

    private static TState? ReadObjectState<TState>(BinaryReader reader, string name)
        where TState : class
    {
        if (!reader.ReadBoolean()) return null;
        string json = reader.ReadString();
        try
        {
            return JsonConvert.DeserializeObject<TState>(json, ObjectStateSettings)
                ?? throw new InvalidOperationException(
                    $"State '{name}' deserialized to null for '{typeof(TState).FullName}'.");
        }
        catch (JsonException exception)
        {
            throw new InvalidOperationException(
                $"State '{name}' is not valid serialized object state for "
                + $"'{typeof(TState).FullName}'.",
                exception);
        }
    }

    private static void CopyCollectionState<TState>(string name, TState current, TState? restored)
        where TState : class
    {
        if (current is IDictionary currentDictionary)
        {
            currentDictionary.Clear();
            if (restored is IDictionary restoredDictionary)
            {
                foreach (DictionaryEntry pair in restoredDictionary)
                    currentDictionary.Add(pair.Key, pair.Value);
            }
            return;
        }

        if (current is IList currentList)
        {
            currentList.Clear();
            if (restored is IList restoredList)
            {
                foreach (var item in restoredList) currentList.Add(item);
            }
            return;
        }

        throw new InvalidOperationException(
            $"State '{name}' requested in-place restoration for '{typeof(TState).FullName}', "
            + "which is neither a list nor a dictionary.");
    }

    /// <summary>
    /// Preserves nested models inside generated object state through their canonical serializer.
    /// </summary>
    private sealed class ModelSerializerJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
            => typeof(IModelSerializer).IsAssignableFrom(objectType);

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value is null) { writer.WriteNull(); return; }
            if (value is not IModelSerializer model)
                throw new JsonSerializationException(
                    $"'{value.GetType().FullName}' does not implement IModelSerializer.");

            writer.WriteStartObject();
            writer.WritePropertyName("modelType");
            writer.WriteValue(value.GetType().AssemblyQualifiedName);
            writer.WritePropertyName("payload");
            writer.WriteValue(Convert.ToBase64String(model.Serialize()));
            writer.WriteEndObject();
        }

        public override object? ReadJson(
            JsonReader reader,
            Type objectType,
            object? existingValue,
            JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null) return null;

            var data = JObject.Load(reader);
            string? typeName = data["modelType"]?.Value<string>();
            string? payloadText = data["payload"]?.Value<string>();
            Type? concrete = string.IsNullOrWhiteSpace(typeName)
                ? objectType
                : Type.GetType(typeName!, throwOnError: false);

            if (concrete is null || !objectType.IsAssignableFrom(concrete)
                || !typeof(IModelSerializer).IsAssignableFrom(concrete))
            {
                throw new JsonSerializationException(
                    $"Nested model type '{typeName}' cannot be restored as '{objectType.FullName}'.");
            }

            var model = CreateSerializable(concrete, "generated object state");
            try
            {
                model.Deserialize(Convert.FromBase64String(payloadText ?? string.Empty));
            }
            catch (FormatException exception)
            {
                throw new JsonSerializationException("Nested model payload is not valid base64.", exception);
            }
            return model;
        }
    }

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

    /// <summary>Declares a list of tensors, such as a temporal memory bank.</summary>
    public void Declare(string name, Func<List<Tensor<T>>?> get, Action<List<Tensor<T>>?> set)
        => Add(name,
            w =>
            {
                var list = get();
                if (list is null) { w.Write(-1); return; }
                w.Write(list.Count);
                foreach (var tensor in list) WriteTensor(w, tensor);
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var list = new List<Tensor<T>>(count);
                for (int i = 0; i < count; i++)
                    list.Add(ReadTensor(r) ?? new Tensor<T>([0]));
                set(list);
            });

    /// <summary>Declares an integer array, such as node indices or a feature mapping.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<int[]?> get, Action<int[]?> set)
        => Add(name,
            w => WriteInts(w, get()),
            r => set(ReadInts(r)));

    /// <summary>
    /// Describes ONE node of a recursive structure, so the registry can walk the whole of it.
    /// </summary>
    /// <typeparam name="TNode">The node type.</typeparam>
    /// <remarks>
    /// <para>
    /// A decision tree, a Hoeffding tree, an M5 model tree -- their learned model is a node graph,
    /// not a vector, and it was the one shape nothing here could express. Every tree model therefore
    /// kept a hand-written Serialize that walked the graph itself, and every ENSEMBLE of trees
    /// inherited that: a forest cannot round-trip until its members can.
    /// </para>
    /// <para>
    /// The model describes a single node -- its own fields, and which of its members are children --
    /// and the registry does the recursion, the null markers and the ordering. Nothing in the model
    /// touches a reader or a writer, and the description is small enough to read at a glance.
    /// </para>
    /// </remarks>
    public sealed class NodeShape<TNode> where TNode : class
    {
        internal Func<TNode>? Factory;
        internal readonly List<(Action<TNode, BinaryWriter> Write, Action<TNode, BinaryReader> Read)> Fields = new();
        internal readonly List<(Func<TNode, TNode?> Get, Action<TNode, TNode?> Set)> Children = new();

        /// <summary>Declares how to make an empty node.</summary>
        /// <param name="factory">Creates a node with no fields set.</param>
        /// <returns>This shape, for chaining.</returns>
        public NodeShape<TNode> Create(Func<TNode> factory)
        {
            Factory = factory;
            return this;
        }

        /// <summary>Declares an integer field on the node.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Writes it.</param>
        /// <returns>This shape, for chaining.</returns>
        public NodeShape<TNode> Int32(Func<TNode, int> get, Action<TNode, int> set)
        {
            Fields.Add(((n, w) => w.Write(get(n)), (n, r) => set(n, r.ReadInt32())));
            return this;
        }

        /// <summary>Declares a 64-bit integer field on the node.</summary>
        public NodeShape<TNode> Int64(Func<TNode, long> get, Action<TNode, long> set)
        {
            Fields.Add(((n, w) => w.Write(get(n)), (n, r) => set(n, r.ReadInt64())));
            return this;
        }

        /// <summary>Declares a <see cref="double"/> field on the node.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Installs a restored value.</param>
        /// <remarks>
        /// Distinct from <see cref="Scalar"/>, which carries the model's own <typeparamref name="T"/>.
        /// A tree's split threshold is frequently declared as a plain <c>double</c> regardless of the
        /// model's numeric type, and routing that through Scalar would convert it to T and back,
        /// changing the value on any T narrower than double.
        /// </remarks>
        public NodeShape<TNode> Double(Func<TNode, double> get, Action<TNode, double> set)
        {
            Fields.Add(((n, w) => w.Write(get(n)), (n, r) => set(n, r.ReadDouble())));
            return this;
        }

        /// <summary>Declares a <see cref="double"/> array field on the node, such as a leaf's curve.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Installs a restored value.</param>
        /// <remarks>
        /// Null and empty are distinguished by a -1 length, matching the registry's own array
        /// declarations: a leaf that never accumulated a curve is not the same as one whose curve
        /// is empty.
        /// </remarks>
        public NodeShape<TNode> DoubleArray(Func<TNode, double[]?> get, Action<TNode, double[]?> set)
        {
            Fields.Add((
                (n, w) =>
                {
                    var a = get(n);
                    if (a is null) { w.Write(-1); return; }
                    w.Write(a.Length);
                    foreach (var value in a) w.Write(value);
                },
                (n, r) =>
                {
                    int length = r.ReadInt32();
                    if (length < 0) { set(n, null); return; }
                    var a = new double[length];
                    for (int i = 0; i < length; i++) a[i] = r.ReadDouble();
                    set(n, a);
                }));
            return this;
        }

        /// <summary>Declares a boolean field on the node.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Writes it.</param>
        /// <returns>This shape, for chaining.</returns>
        public NodeShape<TNode> Boolean(Func<TNode, bool> get, Action<TNode, bool> set)
        {
            Fields.Add(((n, w) => w.Write(get(n)), (n, r) => set(n, r.ReadBoolean())));
            return this;
        }

        /// <summary>Declares a field held as the model's numeric type.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Writes it.</param>
        /// <returns>This shape, for chaining.</returns>
        public NodeShape<TNode> Scalar(Func<TNode, T> get, Action<TNode, T> set)
        {
            Fields.Add((
                (n, w) => w.Write(Convert.ToDouble(get(n))),
                (n, r) => set(n, Ops.FromDouble(r.ReadDouble()))));
            return this;
        }

        /// <summary>Declares a vector field on the node, such as class probabilities.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Writes it.</param>
        /// <returns>This shape, for chaining.</returns>
        public NodeShape<TNode> Vector(Func<TNode, Vector<T>?> get, Action<TNode, Vector<T>?> set)
        {
            Fields.Add((
                (n, w) => WriteVector(w, get(n)),
                (n, r) => set(n, ReadVector(r))));
            return this;
        }

        /// <summary>Declares one of the node's children.</summary>
        /// <param name="get">Reads it.</param>
        /// <param name="set">Attaches it.</param>
        /// <returns>This shape, for chaining.</returns>
        public NodeShape<TNode> Child(Func<TNode, TNode?> get, Action<TNode, TNode?> set)
        {
            Children.Add((get, set));
            return this;
        }
    }

    /// <summary>Declares a recursive node graph, such as a decision tree.</summary>
    /// <typeparam name="TNode">The node type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="getRoot">Reads the root.</param>
    /// <param name="setRoot">Installs a restored root.</param>
    /// <param name="describe">Describes one node.</param>
    public void DeclareGraph<TNode>(
        string name,
        Func<TNode?> getRoot,
        Action<TNode?> setRoot,
        Action<NodeShape<TNode>> describe)
        where TNode : class
    {
        var shape = new NodeShape<TNode>();
        describe(shape);

        if (shape.Factory is null)
            throw new ArgumentException($"State '{name}' must declare Create so its nodes can be rebuilt.", nameof(describe));

        Add(name,
            w => WriteNode(w, getRoot(), shape),
            r => setRoot(ReadNode(r, shape)));
    }

    private static void WriteNode<TNode>(BinaryWriter w, TNode? node, NodeShape<TNode> shape)
        where TNode : class
    {
        // A presence flag per node, so an absent child costs one byte and a null root is representable.
        if (node is null) { w.Write(false); return; }

        w.Write(true);
        foreach (var field in shape.Fields) field.Write(node, w);
        foreach (var child in shape.Children) WriteNode(w, child.Get(node), shape);
    }

    private static TNode? ReadNode<TNode>(BinaryReader r, NodeShape<TNode> shape)
        where TNode : class
    {
        if (!r.ReadBoolean()) return null;

        var node = shape.Factory!();
        foreach (var field in shape.Fields) field.Read(node, r);
        foreach (var child in shape.Children) child.Set(node, ReadNode(r, shape));
        return node;
    }

    /// <summary>Declares a LIST of node graphs — a forest, where each entry is its own root.</summary>
    /// <typeparam name="TNode">The node type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current roots.</param>
    /// <param name="set">Installs the restored roots.</param>
    /// <param name="describe">Describes one node, exactly as <see cref="DeclareGraph"/> does.</param>
    /// <remarks>
    /// <para>
    /// The gap this fills: <see cref="DeclareGraph"/> carries ONE root, and
    /// <see cref="DeclareChildList{TChild}"/> carries many children but demands they implement
    /// <c>IModelSerializer</c>. An ensemble's trees are neither — many roots, and the node is a
    /// plain private type with no serialization surface of its own. Without this overload every
    /// forest model had to hand-write the walk, which is what ADN0060 reports and what ADN0062
    /// reports from the other direction ("no ModelStateRegistry declaration, so nothing would
    /// persist it").
    /// </para>
    /// <para>
    /// Unlike <see cref="DeclareChildList{TChild}"/> the roots are REPLACED rather than restored in
    /// place: how many trees a forest has is fitted, not configuration, so the count comes from the
    /// payload rather than from whatever the constructor happened to build.
    /// </para>
    /// <para>
    /// A null root is dropped rather than preserved positionally. <see cref="WriteNode"/> writes a
    /// presence flag per node, so the stream stays aligned either way; a forest with a null tree in
    /// it has no meaning, and keeping the slot would require a nullable element type that every
    /// caller would then have to defend against.
    /// </para>
    /// </remarks>
    public void DeclareGraphList<TNode>(
        string name,
        Func<List<TNode>?> get,
        Action<List<TNode>?> set,
        Action<NodeShape<TNode>> describe)
        where TNode : class
    {
        var shape = new NodeShape<TNode>();
        describe(shape);

        if (shape.Factory is null)
            throw new ArgumentException($"State '{name}' must declare Create so its nodes can be rebuilt.", nameof(describe));

        Add(name,
            w =>
            {
                var roots = get();
                // -1 distinguishes "no list" from "an empty list", matching DeclareChildList.
                if (roots is null) { w.Write(-1); return; }
                w.Write(roots.Count);
                foreach (var root in roots) WriteNode(w, root, shape);
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }

                var roots = new List<TNode>(count);
                for (int i = 0; i < count; i++)
                {
                    var node = ReadNode(r, shape);
                    if (node is not null) roots.Add(node);
                }
                set(roots);
            });
    }

    /// <summary>Declares an array held as the model's own numeric type.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareArray(string name, Func<T[]?> get, Action<T[]?> set)
        => Add(name,
            w =>
            {
                var a = get();
                if (a is null) { w.Write(-1); return; }
                w.Write(a.Length);
                foreach (var value in a) w.Write(Convert.ToDouble(value));
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) { set(null); return; }
                var a = new T[length];
                for (int i = 0; i < length; i++) a[i] = Ops.FromDouble(r.ReadDouble());
                set(a);
            });

    /// <summary>Declares a JAGGED array held as the model's own numeric type.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    /// <remarks>
    /// Jagged rather than rectangular deliberately: the shape this exists for is per-feature bin
    /// thresholds, where each feature has its own bin count. A Matrix would have to pad to the
    /// widest row and then carry the real widths separately, which is two facts that can disagree.
    /// Each row keeps its own length, and a null row is representable so the outer and inner
    /// nullability both round-trip.
    /// </remarks>
    public void DeclareJaggedArray(string name, Func<T[][]?> get, Action<T[][]?> set)
        => Add(name,
            w =>
            {
                var rows = get();
                if (rows is null) { w.Write(-1); return; }
                w.Write(rows.Length);
                foreach (var row in rows)
                {
                    if (row is null) { w.Write(-1); continue; }
                    w.Write(row.Length);
                    foreach (var value in row) w.Write(Convert.ToDouble(value));
                }
            },
            r =>
            {
                int outer = r.ReadInt32();
                if (outer < 0) { set(null); return; }
                var rows = new T[outer][];
                for (int i = 0; i < outer; i++)
                {
                    int inner = r.ReadInt32();
                    if (inner < 0) continue;
                    var row = new T[inner];
                    for (int j = 0; j < inner; j++) row[j] = Ops.FromDouble(r.ReadDouble());
                    rows[i] = row;
                }
                set(rows);
            });

    /// <summary>Declares a list of matrices, such as per-class or per-category probability tables.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<List<Matrix<T>>?> get, Action<List<Matrix<T>>?> set)
        => Add(name,
            w =>
            {
                var list = get();
                if (list is null) { w.Write(-1); return; }
                w.Write(list.Count);
                foreach (var m in list) WriteMatrix(w, m);
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var list = new List<Matrix<T>>(count);
                for (int i = 0; i < count; i++) list.Add(ReadMatrix(r) ?? new Matrix<T>(0, 0));
                set(list);
            });

    /// <summary>Declares a single nested model, such as a DQN agent's target network.</summary>
    /// <typeparam name="TChild">The child's type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the child.</param>
    /// <remarks>
    /// The child is restored IN PLACE, not replaced: the parent builds it in its constructor, so what
    /// travels is its state and not its identity. A target network that came back as a fresh instance
    /// would leave the agent bootstrapping from an untrained copy of itself.
    /// </remarks>
    public void DeclareChild<TChild>(string name, Func<TChild?> get)
        where TChild : class, IModelSerializer
        => Add(name,
            w =>
            {
                var child = get();
                if (child is null) { w.Write(-1); return; }
                var bytes = child.Serialize();
                w.Write(bytes.Length);
                w.Write(bytes);
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) return;
                var bytes = r.ReadBytes(length);
                if (length > 0) get()?.Deserialize(bytes);
            });

    /// <summary>Declares an assignable child that may not exist until its fitted state is restored.</summary>
    /// <typeparam name="TChild">The child's concrete or abstract serializer type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current child, if materialized.</param>
    /// <param name="set">Installs a child constructed from the serialized state.</param>
    /// <remarks>
    /// Some nested state is created by <c>Fit</c>, not by the parent's constructor. Restoring only
    /// in place silently discarded that child's bytes on a fresh destination. This overload retains
    /// the ordinary in-place path when an instance exists and otherwise constructs the declared
    /// child through the same parameterless-or-all-optional convention used for child lists.
    /// </remarks>
    public void DeclareChild<TChild>(string name, Func<TChild?> get, Action<TChild?> set)
        where TChild : class, IModelSerializer
        => Add(name,
            w =>
            {
                var child = get();
                if (child is null) { w.Write(-1); return; }
                var bytes = child.Serialize();
                w.Write(bytes.Length);
                w.Write(bytes);
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) { set(null); return; }

                var bytes = r.ReadBytes(length);
                var child = get();
                if (child is null)
                {
                    child = CreateChild<TChild>(name, string.Empty);
                    set(child);
                }

                if (length > 0) child.Deserialize(bytes);
            });

    /// <summary>
    /// Declares an assignable fitted child whose parent already owns the canonical construction
    /// factory for its abstract or interface-typed slot.
    /// </summary>
    /// <typeparam name="TChild">The child's serializer contract.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current fitted child.</param>
    /// <param name="set">Installs a child created during restore.</param>
    /// <param name="create">Builds the configured concrete child when the slot is empty.</param>
    /// <remarks>
    /// A parent factory is stronger construction evidence than reflection: it preserves the exact
    /// configured implementation even when <typeparamref name="TChild"/> is an interface and the
    /// concrete child requires constructor arguments. Stacking classifiers are the canonical shape.
    /// </remarks>
    public void DeclareChild<TChild>(
        string name,
        Func<TChild?> get,
        Action<TChild?> set,
        Func<TChild> create)
        where TChild : class, IModelSerializer
    {
        if (create is null) throw new ArgumentNullException(nameof(create));
        Add(name,
            w =>
            {
                var child = get();
                if (child is null) { w.Write(-1); return; }
                var bytes = child.Serialize();
                w.Write(bytes.Length);
                w.Write(bytes);
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) { set(null); return; }

                var bytes = r.ReadBytes(length);
                var child = get();
                if (child is null)
                {
                    child = create() ?? throw new InvalidOperationException(
                        $"State '{name}' used its configured child factory, but the factory returned null.");
                    set(child);
                }

                if (length > 0) child.Deserialize(bytes);
            });
    }

    /// <summary>Declares a nested parameter source, such as a duelling agent's target network.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the source.</param>
    /// <remarks>
    /// A parameter source carries its state as a vector rather than as a serialized payload, so that
    /// is what travels. Restored in place: the parent constructed it, and a target network that came
    /// back as a fresh instance would leave the agent bootstrapping from an untrained copy of itself.
    /// </remarks>
    public void DeclareParameterSource(string name, Func<IParameterSource<T>?> get)
        => Add(name,
            w => WriteVector(w, get()?.GetParameters()),
            r =>
            {
                var values = ReadVector(r);
                if (values is not null) get()?.SetParameters(values);
            });

    /// <summary>Declares a list of layers the model owns directly, such as a conv stack.</summary>
    /// <typeparam name="TLayer">The layer type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the layers, in a stable order.</param>
    /// <remarks>
    /// <para>
    /// A model that keeps layers in a plain <c>List</c> had no way to declare them: every existing
    /// overload takes a vector, a matrix, a tensor or an <c>IModelSerializer</c>, and a layer is none
    /// of those. So the generator skipped the member silently and the layers' weights travelled
    /// nowhere. DeepANT lost both convolution layers this way -- they came back as the
    /// placeholder-shaped shells its deserialization constructor builds, 96 kernel values collapsed
    /// to 1, and the model's prediction changed sign across a round-trip.
    /// </para>
    /// <para>
    /// Networks do not hit this, because their layers belong to the network base; it is models on
    /// other bases, holding layers directly, that had no declaration to make.
    /// </para>
    /// <para>
    /// RESTORED IN PLACE, like <see cref="DeclareChildList{TChild}"/> and for the same reason: the
    /// parent's constructor builds these layers at their configured widths, so how many there are and
    /// how wide they are is configuration that the constructor already replays. Only the learned
    /// values need to travel, and each layer's own <c>Serialize</c>/<c>Deserialize</c> pair already
    /// carries its parameter layout, resolved shape and buffers.
    /// </para>
    /// <para>
    /// A count mismatch is therefore a real disagreement about configuration rather than something to
    /// paper over: the extra payloads are consumed so the reader stays aligned for whatever follows,
    /// and the surplus layers keep their constructed values.
    /// </para>
    /// </remarks>
    public void DeclareLayerList<TLayer>(string name, Func<List<TLayer>?> get)
        where TLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
        => Add(name,
            w =>
            {
                var layers = get();
                if (layers is null) { w.Write(-1); return; }
                w.Write(layers.Count);
                foreach (var layer in layers)
                {
                    using var ms = new MemoryStream();
                    using (var lw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
                    {
                        layer?.Serialize(lw);
                    }

                    var bytes = ms.ToArray();
                    w.Write(bytes.Length);
                    w.Write(bytes);
                }
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) return;
                var layers = get();
                for (int i = 0; i < count; i++)
                {
                    int length = r.ReadInt32();
                    var bytes = r.ReadBytes(length);

                    // Length-framed per layer so an unreadable or surplus payload costs only that
                    // layer, never the alignment of the entries that follow it.
                    if (layers is null || i >= layers.Count || length == 0) { continue; }

                    using var ms = new MemoryStream(bytes);
                    using var lr = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                    layers[i]?.Deserialize(lr);
                }
            });

    /// <summary>Declares a list of nested models, such as an agent's per-actor target networks.</summary>
    /// <typeparam name="TChild">The child type.</typeparam>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the children, in a stable order.</param>
    /// <remarks>
    /// Restored in place and by position, because the parent builds these in its constructor: how
    /// many there are is configuration, and configuration is replayed by the recorded constructor.
    /// </remarks>
    public void DeclareChildList<TChild>(string name, Func<List<TChild>?> get)
        where TChild : class, IModelSerializer
        => Add(name,
            w =>
            {
                var children = get();
                w.Write(TypedChildListMarker);
                if (children is null) { w.Write(-1); return; }
                w.Write(children.Count);
                foreach (var child in children)
                {
                    // The declared element may be an interface or abstract base. A fresh clone has
                    // no fitted children to inspect, so the concrete type must travel with its bytes.
                    w.Write(child?.GetType().AssemblyQualifiedName ?? string.Empty);
                    var bytes = child?.Serialize() ?? Array.Empty<byte>();
                    w.Write(bytes.Length);
                    w.Write(bytes);
                }
            },
            r =>
            {
                int header = r.ReadInt32();
                bool carriesTypes = header == TypedChildListMarker;
                int count = carriesTypes ? r.ReadInt32() : header;
                if (count < 0) return;
                var children = get();
                for (int i = 0; i < count; i++)
                {
                    string typeName = carriesTypes ? r.ReadString() : string.Empty;
                    int length = r.ReadInt32();
                    var bytes = r.ReadBytes(length);
                    if (length == 0 || children is null) continue;

                    // GROW THE LIST. Restoring in place is right when the parent rebuilt its children
                    // first, and silently wrong the moment it did not: a CLONE is constructed empty, so
                    // `i < children.Count` was false for every child and the whole payload was read and
                    // dropped. RandomForest, ExtremelyRandomizedTrees and AdaBoostR2 all round-tripped
                    // through Serialize and Deserialize perfectly and still cloned into a forest with no
                    // trees, which is exactly the silent loss this work exists to remove -- the bytes
                    // were there, and nothing was listening.
                    while (children.Count <= i)
                    {
                        children.Add(CreateChild<TChild>(name, typeName));
                    }

                    children[i]?.Deserialize(bytes);
                }
            });

    /// <summary>Builds an empty child for a restored list to fill.</summary>
    /// <param name="name">The state name, for the error message when it cannot be built.</param>
    /// <param name="typeName">The concrete runtime type saved beside the child payload.</param>
    /// <returns>A new child.</returns>
    /// <remarks>
    /// LOUD when it cannot. Returning null here, or skipping the child, would put back the silent drop
    /// this exists to fix -- and it would look like a model that restored fine and predicts wrongly,
    /// which is the hardest kind of defect to find. A child that cannot be built without arguments
    /// needs its parent to build the list before restoring, and the message says so.
    /// </remarks>
    private static TChild CreateChild<TChild>(string name, string typeName)
        where TChild : class, IModelSerializer
    {
        Type childType = typeof(TChild);
        if (!string.IsNullOrEmpty(typeName))
        {
            Type? recordedType = Type.GetType(typeName, throwOnError: false);
            if (recordedType is null || !childType.IsAssignableFrom(recordedType))
            {
                throw new InvalidOperationException(
                    $"State '{name}' recorded child type '{typeName}', which cannot be restored as "
                    + $"{childType.Name}.");
            }

            childType = recordedType;
        }

        try
        {
            if (Activator.CreateInstance(childType, nonPublic: true) is TChild child) return child;
        }
        catch (MissingMethodException)
        {
            // Falls through to the optional-argument attempt below.
        }

        // A CONSTRUCTOR WHOSE PARAMETERS ARE ALL OPTIONAL IS CALLABLE WITH NO ARGUMENTS, but
        // Activator's parameterless lookup cannot see it. DecisionTreeRegression declares exactly
        // one constructor, `(DecisionTreeOptions? options = null, IRegularization? regularization =
        // null)`, so `new DecisionTreeRegression<T>()` compiles while reflection reported the type as
        // having no constructor at all -- and RandomForestRegression, whose payload holds trees a
        // freshly constructed clone has none of, failed its round-trip on that alone.
        //
        // CloneEngine already binds this shape with OptionalParamBinding, which turns a Type.Missing
        // slot into the declared default. Doing the same here fixes every child type of this shape,
        // rather than asking each one to declare a second, empty constructor.
        var withOptionalArguments = childType
            .GetConstructors(System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.NonPublic
                | System.Reflection.BindingFlags.Instance)
            .FirstOrDefault(c => c.GetParameters().Length > 0
                && c.GetParameters().All(p => p.IsOptional));

        if (withOptionalArguments is not null)
        {
            var arguments = new object?[withOptionalArguments.GetParameters().Length];
            for (int i = 0; i < arguments.Length; i++) arguments[i] = Type.Missing;

            if (withOptionalArguments.Invoke(
                    System.Reflection.BindingFlags.OptionalParamBinding,
                    binder: null,
                    arguments,
                    culture: null) is TChild built)
            {
                return built;
            }
        }

        throw new InvalidOperationException(
            $"State '{name}' carries a list of {typeof(TChild).Name}, and the model being restored has "
            + "fewer of them than the payload holds. Restoring cannot create one because "
            + $"{childType.Name} has no constructor callable without arguments. Either give it one, "
            + "or have the model build its list before Deserialize runs.");
    }

    /// <summary>Declares a decision tree, carried whole.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the root.</param>
    /// <param name="set">Installs a restored root.</param>
    /// <remarks>
    /// Every decision-tree model kept its structure in a <c>Root</c> field no overload could carry, so
    /// each walked the tree by hand -- and a by-hand walk carries what its author remembered. The
    /// shared one wrote FeatureIndex, SplitValue, Prediction and IsLeaf, and dropped <c>Threshold</c>
    /// and <c>LinearModel</c>. M5ModelTree fits a LINEAR MODEL at every leaf, so its restored tree had
    /// the right shape, the right splits and constant leaves: it predicted 3 where the original said 2.
    /// A leaf model travels as its concrete type name plus its own payload, because the field is
    /// declared as a base type and the restore has to rebuild whatever was actually fitted.
    /// </remarks>
    public void DeclareTree(
        string name,
        Func<DecisionTreeNode<T>?> get,
        Action<DecisionTreeNode<T>?> set)
        => Add(name,
            w => WriteNode(w, get()),
            r => set(ReadNode(r, name)));

    private static void WriteNode(BinaryWriter w, DecisionTreeNode<T>? node)
    {
        if (node is null) { w.Write(false); return; }
        w.Write(true);

        w.Write(node.FeatureIndex);
        w.Write(Ops.ToDouble(node.SplitValue));
        w.Write(Ops.ToDouble(node.Threshold));
        w.Write(Ops.ToDouble(node.Prediction));
        w.Write(node.IsLeaf);

        if (node.LinearModel is null)
        {
            w.Write(false);
        }
        else
        {
            w.Write(true);
            var concrete = node.LinearModel.GetType();
            w.Write(concrete.AssemblyQualifiedName ?? concrete.FullName ?? concrete.Name);
            var payload = node.LinearModel.Serialize();
            w.Write(payload.Length);
            w.Write(payload);
        }

        WriteNode(w, node.Left);
        WriteNode(w, node.Right);
    }

    private static DecisionTreeNode<T>? ReadNode(BinaryReader r, string name)
    {
        if (!r.ReadBoolean()) return null;

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = r.ReadInt32(),
            SplitValue = Ops.FromDouble(r.ReadDouble()),
            Threshold = Ops.FromDouble(r.ReadDouble()),
            Prediction = Ops.FromDouble(r.ReadDouble()),
            IsLeaf = r.ReadBoolean(),
        };

        if (r.ReadBoolean())
        {
            string typeName = r.ReadString();
            int length = r.ReadInt32();
            var payload = r.ReadBytes(length);

            var concrete = Type.GetType(typeName, throwOnError: false);
            if (concrete is null)
            {
                throw new InvalidOperationException(
                    $"State '{name}' holds a leaf model of type '{typeName}', which this runtime cannot "
                    + "load, so the tree cannot be restored as it was fitted.");
            }

            var leaf = CreateSerializable(concrete, name);
            leaf.Deserialize(payload);
            node.LinearModel = leaf as RegressionBase<T>;
        }

        node.Left = ReadNode(r, name);
        node.Right = ReadNode(r, name);
        return node;
    }

    /// <summary>Builds an empty instance of a type named in a payload.</summary>
    /// <param name="type">The concrete type to build.</param>
    /// <param name="name">The state name, for the error message.</param>
    /// <returns>The new instance.</returns>
    /// <remarks>
    /// Accepts an all-optional constructor for the same reason <see cref="CreateChild{TChild}"/> does:
    /// reflection's parameterless lookup cannot see one, and most models declare exactly that shape.
    /// </remarks>
    private static IModelSerializer CreateSerializable(Type type, string name)
    {
        try
        {
            if (Activator.CreateInstance(type, nonPublic: true) is IModelSerializer built) return built;
        }
        catch (MissingMethodException)
        {
        }

        var withOptionalArguments = type
            .GetConstructors(System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.NonPublic
                | System.Reflection.BindingFlags.Instance)
            .FirstOrDefault(c => c.GetParameters().Length > 0
                && c.GetParameters().All(p => p.IsOptional));

        if (withOptionalArguments is not null)
        {
            var arguments = new object?[withOptionalArguments.GetParameters().Length];
            for (int i = 0; i < arguments.Length; i++) arguments[i] = Type.Missing;

            if (withOptionalArguments.Invoke(
                    System.Reflection.BindingFlags.OptionalParamBinding,
                    binder: null,
                    arguments,
                    culture: null) is IModelSerializer built)
            {
                return built;
            }
        }

        throw new InvalidOperationException(
            $"State '{name}' holds a leaf model of type '{type.Name}', which has no constructor callable "
            + "without arguments, so restoring cannot build one to read it back into.");
    }

    /// <summary>Declares the options object a model predicts with.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the options instance.</param>
    /// <remarks>
    /// <para>
    /// Restored IN PLACE, like a child model, so a readonly <c>_options</c> field works: what travels
    /// is the settings, not the identity of the object holding them.
    /// </para>
    /// <para>
    /// Needed because configuration is not merely descriptive -- it decides what the model predicts.
    /// KNearestNeighborsRegression answers with <c>_options.K</c> neighbours, so a payload that
    /// carried its training data but not its K restored a model that ran, and answered differently.
    /// Cloning already carries configuration through the clone plan; this is the serialize half of
    /// the same contract.
    /// </para>
    /// <para>
    /// SCALARS ONLY, DELIBERATELY. Settings that are objects -- a regularization strategy, a kernel,
    /// a delegate -- are reproduced by the constructor that built them, and writing them here would
    /// mean a second, weaker copy of what the clone plan already does properly. Restricting the scope
    /// is not the same as dropping state silently: the boundary is a property's type, it is the same
    /// on both sides of the round-trip, and it is stated here rather than discovered from a wrong
    /// prediction.
    /// </para>
    /// </remarks>
    public void DeclareOptions(string name, Func<ModelOptions?> get)
        => Add(name,
            w =>
            {
                var options = get();
                if (options is null) { w.Write(-1); return; }

                var properties = ScalarOptionProperties(options.GetType());
                w.Write(properties.Count);
                foreach (var property in properties)
                {
                    WriteScalarOption(w, property.GetValue(options));
                }
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) return;

                var options = get();
                // The reader must consume its bytes whether or not there is anywhere to put them,
                // or every later declaration reads from the wrong offset.
                var properties = options is null
                    ? new List<System.Reflection.PropertyInfo>()
                    : ScalarOptionProperties(options.GetType());

                for (int i = 0; i < count; i++)
                {
                    var target = i < properties.Count ? properties[i] : null;
                    var value = ReadScalarOption(r, target?.PropertyType);
                    if (options is null || target is null) continue;
                    target.SetValue(options, value);
                }
            });

    /// <summary>The settable scalar settings of an options type, in a stable order.</summary>
    /// <param name="type">The options type.</param>
    /// <returns>The properties carried by <see cref="DeclareOptions"/>.</returns>
    /// <remarks>
    /// Ordered by name so the reader walks what the writer wrote. Reflection order is not specified
    /// and can differ between runtimes, which would silently pair one setting's bytes with another's.
    /// </remarks>
    private static List<System.Reflection.PropertyInfo> ScalarOptionProperties(Type type)
        => type.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance)
            .Where(p => p.CanRead && p.CanWrite && p.GetIndexParameters().Length == 0)
            .Where(p => IsCarriedScalar(p.PropertyType))
            .OrderBy(p => p.Name, StringComparer.Ordinal)
            .ToList();

    private static bool IsCarriedScalar(Type type)
    {
        var bare = Nullable.GetUnderlyingType(type) ?? type;
        return bare.IsEnum
            || bare == typeof(int) || bare == typeof(long) || bare == typeof(double)
            || bare == typeof(float) || bare == typeof(bool) || bare == typeof(string);
    }

    // Each value is TAGGED with its own type. The reader must be able to consume a value it has
    // nowhere to put -- an options object that is null, or a setting that no longer exists -- and
    // without a tag it would have to guess a width and desynchronise every later declaration.
    private const byte OptionNull = 0;
    private const byte OptionBool = 1;
    private const byte OptionInt = 2;
    private const byte OptionLong = 3;
    private const byte OptionDouble = 4;
    private const byte OptionFloat = 5;
    private const byte OptionString = 6;
    private const byte OptionEnum = 7;

    private static void WriteScalarOption(BinaryWriter w, object? value)
    {
        switch (value)
        {
            case null: w.Write(OptionNull); break;
            case bool v: w.Write(OptionBool); w.Write(v); break;
            case int v: w.Write(OptionInt); w.Write(v); break;
            case long v: w.Write(OptionLong); w.Write(v); break;
            case double v: w.Write(OptionDouble); w.Write(v); break;
            case float v: w.Write(OptionFloat); w.Write(v); break;
            case string v: w.Write(OptionString); w.Write(v); break;
            // An enum travels as its underlying integer, so renaming a member does not move values.
            default:
                w.Write(OptionEnum);
                w.Write(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture));
                break;
        }
    }

    private static object? ReadScalarOption(BinaryReader r, Type? target)
    {
        byte tag = r.ReadByte();
        object? value = tag switch
        {
            OptionNull => null,
            OptionBool => r.ReadBoolean(),
            OptionInt => r.ReadInt32(),
            OptionLong => r.ReadInt64(),
            OptionDouble => r.ReadDouble(),
            OptionFloat => r.ReadSingle(),
            OptionString => r.ReadString(),
            OptionEnum => r.ReadInt64(),
            _ => throw new InvalidOperationException(
                $"An options payload carries an unknown value tag {tag}."),
        };

        if (value is null || target is null) return null;

        var bare = Nullable.GetUnderlyingType(target) ?? target;
        if (bare.IsEnum)
        {
            // Enum.ToObject requires the value to have the enum's actual underlying CLR type on
            // newer runtimes. The payload deliberately normalizes every enum to Int64, so convert
            // it back before constructing the enum (byte-backed options exposed this regression).
            var underlying = Enum.GetUnderlyingType(bare);
            var converted = Convert.ChangeType(
                value, underlying, System.Globalization.CultureInfo.InvariantCulture);
            return Enum.ToObject(bare, converted!);
        }

        return Convert.ChangeType(value, bare, System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>Declares an integer vector, such as a set of selected feature indices.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Vector<int>?> get, Action<Vector<int>?> set)
        => Add(name,
            w =>
            {
                var v = get();
                if (v is null) { w.Write(-1); return; }
                w.Write(v.Length);
                for (int i = 0; i < v.Length; i++) w.Write(v[i]);
            },
            r =>
            {
                int length = r.ReadInt32();
                if (length < 0) { set(null); return; }
                var v = new Vector<int>(length);
                for (int i = 0; i < length; i++) v[i] = r.ReadInt32();
                set(v);
            });

    /// <summary>Declares a keyed set of vectors, such as per-layer optimiser moments.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    /// <remarks>
    /// Written key-first so the restore rebuilds the same mapping. Dictionary order is not stable,
    /// so the pairs are sorted by key -- otherwise the same model could produce two different
    /// payloads and neither would be wrong.
    /// </remarks>
    /// <summary>Declares a STRING-keyed table of vectors, such as per-edge or per-operation weights.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    /// <remarks>
    /// The int-keyed overload below cannot serve this: a supernet keys its weights by an edge name
    /// like "node2_op3", and mapping those onto ints would need a side table that is itself state.
    /// Keys are written in sorted order for the same reason as the int-keyed one - a Dictionary has
    /// no inherent order, and an unstable order makes two payloads for identical state differ.
    /// </remarks>
    public void Declare(string name, Func<Dictionary<string, Vector<T>>?> get, Action<Dictionary<string, Vector<T>>?> set)
        => Add(name,
            w =>
            {
                var map = get();
                if (map is null) { w.Write(-1); return; }
                w.Write(map.Count);
                foreach (var pair in map.OrderBy(p => p.Key, StringComparer.Ordinal))
                {
                    w.Write(pair.Key);
                    WriteVector(w, pair.Value);
                }
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var map = new Dictionary<string, Vector<T>>(count, StringComparer.Ordinal);
                for (int i = 0; i < count; i++)
                {
                    string key = r.ReadString();
                    map[key] = ReadVector(r) ?? new Vector<T>(0);
                }
                set(map);
            });

    public void Declare(string name, Func<Dictionary<int, Vector<T>>?> get, Action<Dictionary<int, Vector<T>>?> set)
        => Add(name,
            w =>
            {
                var map = get();
                if (map is null) { w.Write(-1); return; }
                w.Write(map.Count);
                foreach (var pair in map.OrderBy(p => p.Key))
                {
                    w.Write(pair.Key);
                    WriteVector(w, pair.Value);
                }
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var map = new Dictionary<int, Vector<T>>(count);
                for (int i = 0; i < count; i++)
                {
                    int key = r.ReadInt32();
                    map[key] = ReadVector(r) ?? new Vector<T>(0);
                }
                set(map);
            });

    /// <summary>Declares an array of vectors, such as per-feature sorted values or per-point distances.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Vector<T>[]?> get, Action<Vector<T>[]?> set)
        => Add(name,
            w =>
            {
                var a = get();
                if (a is null) { w.Write(-1); return; }
                w.Write(a.Length);
                foreach (var v in a) WriteVector(w, v);
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var a = new Vector<T>[count];
                for (int i = 0; i < count; i++) a[i] = ReadVector(r) ?? new Vector<T>(0);
                set(a);
            });

    /// <summary>Declares an array of matrices, such as one probability table per category.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<Matrix<T>[]?> get, Action<Matrix<T>[]?> set)
        => Add(name,
            w =>
            {
                var a = get();
                if (a is null) { w.Write(-1); return; }
                w.Write(a.Length);
                foreach (var m in a) WriteMatrix(w, m);
            },
            r =>
            {
                int count = r.ReadInt32();
                if (count < 0) { set(null); return; }
                var a = new Matrix<T>[count];
                for (int i = 0; i < count; i++) a[i] = ReadMatrix(r) ?? new Matrix<T>(0, 0);
                set(a);
            });

    /// <summary>Declares a double array.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void Declare(string name, Func<double[]?> get, Action<double[]?> set)
        => Add(name,
            w => WriteDoubles(w, get()),
            r => set(ReadDoubles(r)));

    /// <summary>
    /// Declares a double array that shadows a flat <typeparamref name="T"/> parameter slot.
    /// Its native double payload is restored after the ordinary parameter vector so a float model
    /// does not lose the working storage's extra precision during a checkpoint round trip.
    /// </summary>
    public void DeclareExact(string name, Func<double[]?> get, Action<double[]?> set)
        => Add(name,
            w => WriteDoubles(w, get()),
            r => set(ReadDoubles(r)),
            restoreAfterParameters: true);

    /// <summary>Declares a readonly double array and restores its contents without replacing it.</summary>
    public void DeclareInPlace(string name, Func<double[]?> get)
        => Add(name,
            w => WriteDoubles(w, get()),
            r => CopyDoublesInPlace(name, get(), ReadDoubles(r)));

    /// <summary>
    /// Declares a readonly double parameter array whose exact payload wins after vector restore.
    /// </summary>
    public void DeclareExactInPlace(string name, Func<double[]?> get)
        => Add(name,
            w => WriteDoubles(w, get()),
            r => CopyDoublesInPlace(name, get(), ReadDoubles(r)),
            restoreAfterParameters: true);

    /// <summary>Declares a readonly jagged double array and restores its contents in place.</summary>
    public void DeclareInPlace(string name, Func<double[][]?> get)
        => Add(name,
            w => WriteJaggedDoubles(w, get()),
            r => CopyJaggedDoublesInPlace(name, get(), ReadJaggedDoubles(r)));

    /// <summary>
    /// Declares a readonly jagged double parameter array whose exact payload wins after vector restore.
    /// </summary>
    public void DeclareExactInPlace(string name, Func<double[][]?> get)
        => Add(name,
            w => WriteJaggedDoubles(w, get()),
            r => CopyJaggedDoublesInPlace(name, get(), ReadJaggedDoubles(r)),
            restoreAfterParameters: true);

    /// <summary>
    /// Declares a replaceable jagged double parameter array whose exact payload wins after vector restore.
    /// </summary>
    public void DeclareExact(string name, Func<double[][]?> get, Action<double[][]?> set)
        => Add(name,
            w => WriteJaggedDoubles(w, get()),
            r => set(ReadJaggedDoubles(r)),
            restoreAfterParameters: true);

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

    /// <summary>Declares a 64-bit integer, such as an online model's sample count.</summary>
    public void DeclareInt64(string name, Func<long> get, Action<long> set)
        => Add(name, w => w.Write(get()), r => set(r.ReadInt64()));

    /// <summary>Declares a double, such as a temperature or a learned threshold.</summary>
    /// <param name="name">A stable name, unique within the model.</param>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Installs a restored value.</param>
    public void DeclareDouble(string name, Func<double> get, Action<double> set)
        => Add(name, w => w.Write(get()), r => set(r.ReadDouble()));

    /// <summary>
    /// Declares a double scalar that shadows a flat <typeparamref name="T"/> parameter slot.
    /// </summary>
    public void DeclareExactDouble(string name, Func<double> get, Action<double> set)
        => Add(name, w => w.Write(get()), r => set(r.ReadDouble()), restoreAfterParameters: true);

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
    public void DeclareChildren<TChild, TInput, TOutput>(
        string name,
        Func<IList<TChild>?> get)
        where TChild : class, IFullModel<T, TInput, TOutput>
        => Add(name,
            w =>
            {
                var children = get();
                if (children is null) { w.Write(-1); return; }

                w.Write(children.Count);
                foreach (var child in children)
                {
                    // The concrete TYPE travels with the bytes. An ensemble does not build its
                    // members in its constructor -- they are created during training -- so a freshly
                    // constructed clone has an EMPTY list and there is nothing to restore into. The
                    // type is what lets the members be rebuilt rather than silently dropped, which
                    // would leave an ensemble that predicts from no members at all.
                    w.Write(child?.GetType().AssemblyQualifiedName ?? string.Empty);
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
                if (children is null) return;

                for (int i = 0; i < count; i++)
                {
                    string typeName = r.ReadString();
                    int length = r.ReadInt32();
                    var bytes = r.ReadBytes(length);
                    if (length == 0) continue;

                    if (i < children.Count)
                    {
                        children[i]?.Deserialize(bytes);
                        continue;
                    }

                    if (children.IsReadOnly) continue;

                    if (CreateChild<TInput, TOutput>(typeName) is not TChild child) continue;

                    child.Deserialize(bytes);
                    children.Add(child);
                }
            });

    /// <summary>Rebuilds a child from the type name saved beside its bytes.</summary>
    /// <typeparam name="TInput">The child's input type.</typeparam>
    /// <typeparam name="TOutput">The child's output type.</typeparam>
    /// <param name="typeName">The assembly-qualified name recorded at save time.</param>
    /// <returns>A new child, or <see langword="null"/> when it cannot be constructed.</returns>
    /// <remarks>
    /// Invokes a constructor whose parameters are ALL optional, binding each to its declared default,
    /// which is what most models in this library offer. <c>Activator.CreateInstance(Type)</c> is not
    /// enough on its own: it requires a genuinely parameterless constructor and declines the
    /// all-optional ones that are the common shape here.
    /// </remarks>
    private static IFullModel<T, TInput, TOutput>? CreateChild<TInput, TOutput>(string typeName)
    {
        if (string.IsNullOrEmpty(typeName)) return null;

        var type = Type.GetType(typeName, throwOnError: false);
        if (type is null) return null;

        foreach (var constructor in type.GetConstructors())
        {
            var parameters = constructor.GetParameters();
            if (Array.Exists(parameters, p => !p.IsOptional)) continue;

            var arguments = new object?[parameters.Length];
            for (int i = 0; i < arguments.Length; i++) arguments[i] = Type.Missing;

            return constructor.Invoke(
                System.Reflection.BindingFlags.OptionalParamBinding,
                binder: null,
                arguments,
                culture: null) as IFullModel<T, TInput, TOutput>;
        }

        return null;
    }

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
    public void ReadAll(BinaryReader reader) => ReadAll(reader, restoreAfterParameters: null);

    /// <summary>
    /// Restores structural/non-parameter state while leaving exact native-precision parameter
    /// shadows for the post-parameter phase.
    /// </summary>
    public void ReadBeforeParameters(BinaryReader reader)
        => ReadAll(reader, restoreAfterParameters: false);

    /// <summary>
    /// Restores only native-precision parameter shadows. This must run after the flat
    /// <typeparamref name="T"/> vector has been distributed to its parameter sources.
    /// </summary>
    public void ReadAfterParameters(BinaryReader reader)
        => ReadAll(reader, restoreAfterParameters: true);

    private void ReadAll(BinaryReader reader, bool? restoreAfterParameters)
    {
        int count = reader.ReadInt32();

        var byName = new Dictionary<string, Entry>(StringComparer.Ordinal);
        foreach (var entry in _entries) byName[entry.Name] = entry;

        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            int length = reader.ReadInt32();
            var bytes = reader.ReadBytes(length);

            if (!byName.TryGetValue(name, out var entry)
                || (restoreAfterParameters.HasValue
                    && entry.RestoreAfterParameters != restoreAfterParameters.Value))
            {
                continue;
            }

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

    private static void CopyDoublesInPlace(string name, double[]? destination, double[]? source)
    {
        if (source is null)
        {
            if (destination is null) return;
            throw new InvalidDataException(
                $"State '{name}' was null in the checkpoint but is construction-owned in this model.");
        }
        if (destination is null || destination.Length != source.Length)
        {
            throw new InvalidDataException(
                $"State '{name}' requires a {source.Length}-value construction-owned array, but the "
                + $"destination has {destination?.Length.ToString() ?? "no"} values.");
        }
        Array.Copy(source, destination, source.Length);
    }

    private static void WriteJaggedDoubles(BinaryWriter w, double[][]? values)
    {
        if (values is null) { w.Write(-1); return; }
        w.Write(values.Length);
        foreach (var row in values) WriteDoubles(w, row);
    }

    private static double[][]? ReadJaggedDoubles(BinaryReader r)
    {
        int count = r.ReadInt32();
        if (count < 0) return null;
        var values = new double[count][];
        for (int i = 0; i < count; i++) values[i] = ReadDoubles(r) ?? Array.Empty<double>();
        return values;
    }

    private static void CopyJaggedDoublesInPlace(
        string name,
        double[][]? destination,
        double[][]? source)
    {
        if (source is null)
        {
            if (destination is null) return;
            throw new InvalidDataException(
                $"State '{name}' was null in the checkpoint but is construction-owned in this model.");
        }
        if (destination is null || destination.Length != source.Length)
        {
            throw new InvalidDataException(
                $"State '{name}' requires {source.Length} construction-owned rows, but the destination "
                + $"has {destination?.Length.ToString() ?? "no"} rows.");
        }
        for (int i = 0; i < source.Length; i++)
            CopyDoublesInPlace($"{name}[{i}]", destination[i], source[i]);
    }
}
