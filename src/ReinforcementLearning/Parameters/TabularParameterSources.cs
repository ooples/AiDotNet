using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Parameters;

/// <summary>
/// Exposes a tabular agent's Q-table as a parameter source, so it can be registered like a network
/// instead of every agent hand-writing the same flatten-and-refill loops.
/// </summary>
/// <remarks>
/// <para>
/// Seventeen agents stored their values as
/// <c>Dictionary&lt;string, Dictionary&lt;int, T&gt;&gt;</c> and each wrote its own ParameterCount,
/// GetParameters and SetParameters over it. They were the same three loops seventeen times, which
/// is three chances per agent for the count and the vector to disagree.
/// </para>
/// <para>
/// The table is held BY REFERENCE, deliberately. A tabular agent discovers states as it explores,
/// so its parameter count genuinely grows during training; a snapshot taken at registration would
/// describe a model that no longer exists. Every member reads the live dictionary.
/// </para>
/// <para>
/// Row order is the dictionary's enumeration order, which is what the hand-written code used, so
/// serialization order is unchanged by this conversion. That order is stable for a dictionary that
/// is only ever added to, which is the tabular usage -- but it is NOT guaranteed across a removal,
/// so an agent that starts pruning states needs an explicit ordering here.
/// </para>
/// <para><b>For Beginners:</b> this presents a lookup table of action values as one flat list of
/// numbers, so saving and loading it works the same way it does for a neural network.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the table's values.</typeparam>
public sealed class QTableParameterSource<T> : IParameterSource<T>, AiDotNet.Models.Parameters.IParameterTopologySource
{
    private readonly Dictionary<string, Dictionary<int, T>> _table;
    private readonly int _actionSize;
    private readonly INumericOperations<T> _ops;

    /// <summary>
    /// Wraps <paramref name="table"/>, reading <paramref name="actionSize"/> values per state.
    /// </summary>
    public QTableParameterSource(Dictionary<string, Dictionary<int, T>> table, int actionSize)
    {
        _table = table ?? throw new ArgumentNullException(nameof(table));
        _actionSize = actionSize;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    /// <remarks>Clamped to at least one row, matching what the hand-written counts did: an agent
    /// that has seen no states still reports a non-empty surface, so callers that size a buffer
    /// from the count do not produce a zero-length vector.</remarks>
    public long ParameterCount => (long)Math.Max(_table.Count, 1) * _actionSize;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        if (_table.Count == 0)
        {
            // An empty table still reports one zero row so the parameter count is never zero.
            return Vector<T>.CreateDefault(checked((int)ParameterCount), _ops.Zero);
        }

        var zero = _ops.Zero;
        int actionSize = _actionSize;
        return Vector<T>.FromEnumerable(_table.Values.SelectMany(
            row => Enumerable.Range(0, actionSize).Select(action => row.TryGetValue(action, out var v) ? v : zero)));
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        // Pure data movement back into the dictionary rows: stream the flat vector in order and
        // stop when it runs out, exactly like the indexed refill.
        using var values = ((IEnumerable<T>)parameters).GetEnumerator();
        foreach (var row in _table.Values)
        {
            for (int action = 0; action < _actionSize; action++)
            {
                if (!values.MoveNext())
                {
                    return;
                }

                row[action] = values.Current;
            }
        }
    }

    /// <inheritdoc />
    public void WriteParameterTopology(BinaryWriter writer)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        writer.Write(_table.Count);
        foreach (var state in _table)
        {
            writer.Write(state.Key);
            writer.Write(state.Value.Count);
            foreach (var action in state.Value.Keys) writer.Write(action);
        }
    }

    /// <inheritdoc />
    public void ReadParameterTopology(BinaryReader reader)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        int stateCount = reader.ReadInt32();
        if (stateCount < 0) throw new InvalidDataException($"Q-table topology has negative state count {stateCount}.");
        _table.Clear();
        for (int stateIndex = 0; stateIndex < stateCount; stateIndex++)
        {
            string state = reader.ReadString();
            int actionCount = reader.ReadInt32();
            if (actionCount < 0)
                throw new InvalidDataException($"Q-table topology has negative action count {actionCount}.");
            var actions = new Dictionary<int, T>();
            for (int actionIndex = 0; actionIndex < actionCount; actionIndex++)
                actions.Add(reader.ReadInt32(), _ops.Zero);
            _table.Add(state, actions);
        }
    }
}

/// <summary>
/// Exposes a state-value table (<c>Dictionary&lt;string, T&gt;</c>) as a parameter source, for the
/// agents that learn V(s) rather than Q(s, a).
/// </summary>
/// <remarks>Held by reference and enumerated in dictionary order, for the same reasons as
/// <see cref="QTableParameterSource{T}"/>.</remarks>
/// <typeparam name="T">The numeric type of the table's values.</typeparam>
public sealed class ValueTableParameterSource<T> : IParameterSource<T>, AiDotNet.Models.Parameters.IParameterTopologySource
{
    private readonly Dictionary<string, T> _table;

    /// <summary>Wraps <paramref name="table"/>.</summary>
    public ValueTableParameterSource(Dictionary<string, T> table)
    {
        _table = table ?? throw new ArgumentNullException(nameof(table));
    }

    /// <inheritdoc />
    public long ParameterCount => _table.Count;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        return Vector<T>.FromEnumerable(_table.Values);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        using var values = ((IEnumerable<T>)parameters).GetEnumerator();
        foreach (var key in _table.Keys.ToList())
        {
            if (!values.MoveNext()) break;
            _table[key] = values.Current;
        }
    }

    /// <inheritdoc />
    public void WriteParameterTopology(BinaryWriter writer)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        writer.Write(_table.Count);
        foreach (var key in _table.Keys) writer.Write(key);
    }

    /// <inheritdoc />
    public void ReadParameterTopology(BinaryReader reader)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        int count = reader.ReadInt32();
        if (count < 0) throw new InvalidDataException($"Value-table topology has negative count {count}.");
        _table.Clear();
        // Topology only: the values are zero until the parameters themselves are read.
        var zero = MathHelper.GetNumericOperations<T>().Zero;
        for (int i = 0; i < count; i++) _table.Add(reader.ReadString(), zero);
    }
}

/// <summary>
/// Exposes a linear agent's weight matrix as a parameter source, row-major.
/// </summary>
/// <remarks>
/// The four linear agents (LSPI, LSTD, LinearQLearning, LinearSARSA) keep their weights in a
/// <see cref="Matrix{T}"/>, which the parameter generator cannot discover -- it only recognises
/// tensors -- so each hand-wrote its own flattening. Registering this instead puts them on the same
/// footing as every other component.
/// </remarks>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
public sealed class MatrixParameterSource<T> : IParameterSource<T>
{
    private readonly Func<Matrix<T>> _get;

    /// <summary>Wraps whatever matrix <paramref name="accessor"/> currently returns, so an agent
    /// that REPLACES its matrix rather than mutating it stays correctly described.</summary>
    public MatrixParameterSource(Func<Matrix<T>> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get { var m = _get(); return (long)m.Rows * m.Columns; }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        // Row-major flatten: one bulk copy of the matrix storage.
        return Vector<T>.FromSpan(_get().AsSpan());
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var m = _get();
        // Row-major refill of as many entries as the vector holds, as one bulk copy.
        int count = Math.Min(parameters.Length, m.Rows * m.Columns);
        parameters.AsSpan().Slice(0, count).CopyTo(m.AsWritableSpan());
    }
}

/// <summary>
/// Exposes a plain <see cref="Vector{T}"/> of learned values -- a bandit's action preferences, for
/// instance -- as a parameter source.
/// </summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
public sealed class VectorParameterSource<T> : IParameterSource<T>
{
    private readonly Func<Vector<T>> _get;

    /// <summary>Wraps whatever vector <paramref name="accessor"/> currently returns.</summary>
    public VectorParameterSource(Func<Vector<T>> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount => _get().Length;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        return Vector<T>.FromSpan(_get().AsSpan());
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var v = _get();
        int count = Math.Min(v.Length, parameters.Length);
        parameters.AsSpan().Slice(0, count).CopyTo(v.AsWritableSpan());
    }
}

/// <summary>
/// Exposes a Q-table as the entries it ACTUALLY holds, rather than padding each state out to the
/// full action count.
/// </summary>
/// <remarks>
/// <para>
/// This is a genuinely different contract from <see cref="QTableParameterSource{T}"/> and the two
/// must not be swapped. Six agents pad every state to ActionSize and clamp the table to one row;
/// another six walk only the action entries that exist. For a sparsely-visited table those give
/// different lengths, so using the wrong one would resize the model.
/// </para>
/// <para>
/// <paramref name="padEmptyToOne"/> reproduces the one variation among these agents: some append a
/// single zero when the table is empty so the surface is never zero-length.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the table's values.</typeparam>
public sealed class QTableEntriesParameterSource<T> : IParameterSource<T>, AiDotNet.Models.Parameters.IParameterTopologySource
{
    private readonly Dictionary<string, Dictionary<int, T>> _table;
    private readonly bool _padEmptyToOne;
    private readonly bool _sortStateKeys;
    private readonly bool _ascendingActions;
    private readonly INumericOperations<T> _ops;

    /// <summary>Wraps <paramref name="table"/>, counting only the entries present.</summary>
    /// <param name="table">The Q-table, held by reference.</param>
    /// <param name="padEmptyToOne">Report one parameter when the table is empty.</param>
    /// <param name="sortStateKeys">Walk states in sorted key order rather than dictionary order.
    /// DynaQPlusAgent and PrioritizedSweepingAgent sort, so their serialization order is stable
    /// across runs regardless of the order states were first visited; the others do not, and
    /// changing that for them would renumber every value in an existing checkpoint.</param>
    /// <param name="ascendingActions">Walk each state's actions by ascending index rather than
    /// dictionary order, which is what a <c>for (a = 0; a &lt; ActionSize; a++)</c> loop did.</param>
    public QTableEntriesParameterSource(
        Dictionary<string, Dictionary<int, T>> table,
        bool padEmptyToOne = false,
        bool sortStateKeys = false,
        bool ascendingActions = false)
    {
        _table = table ?? throw new ArgumentNullException(nameof(table));
        _padEmptyToOne = padEmptyToOne;
        _sortStateKeys = sortStateKeys;
        _ascendingActions = ascendingActions;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    private IEnumerable<Dictionary<int, T>> Rows()
    {
        if (!_sortStateKeys)
        {
            foreach (var row in _table.Values) yield return row;
            yield break;
        }
        foreach (var key in _table.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            yield return _table[key];
        }
    }

    private IEnumerable<int> Actions(Dictionary<int, T> row) =>
        _ascendingActions ? row.Keys.OrderBy(a => a) : row.Keys.ToList();

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long n = 0;
            foreach (var row in _table.Values) n += row.Count;
            return n == 0 && _padEmptyToOne ? 1 : n;
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        int count = checked((int)ParameterCount);
        var flat = Vector<T>.FromEnumerable(Rows().SelectMany(row => Actions(row).Select(action => row[action])));
        if (flat.Length == count)
        {
            return flat;
        }

        // Only the padded empty table differs from the entry count: a single zero.
        return Vector<T>.CreateDefault(count, _ops.Zero);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        using var values = ((IEnumerable<T>)parameters).GetEnumerator();
        foreach (var row in Rows())
        {
            foreach (var action in Actions(row).ToList())
            {
                if (!values.MoveNext()) return;
                row[action] = values.Current;
            }
        }
    }

    /// <inheritdoc />
    public void WriteParameterTopology(BinaryWriter writer)
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        writer.Write(_table.Count);
        foreach (var state in _table)
        {
            writer.Write(state.Key);
            writer.Write(state.Value.Count);
            foreach (var action in state.Value.Keys) writer.Write(action);
        }
    }

    /// <inheritdoc />
    public void ReadParameterTopology(BinaryReader reader)
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        int stateCount = reader.ReadInt32();
        if (stateCount < 0) throw new InvalidDataException($"Q-table topology has negative state count {stateCount}.");
        _table.Clear();
        for (int stateIndex = 0; stateIndex < stateCount; stateIndex++)
        {
            string state = reader.ReadString();
            int actionCount = reader.ReadInt32();
            if (actionCount < 0)
                throw new InvalidDataException($"Q-table topology has negative action count {actionCount}.");
            var actions = new Dictionary<int, T>();
            for (int actionIndex = 0; actionIndex < actionCount; actionIndex++)
                actions.Add(reader.ReadInt32(), _ops.Zero);
            _table.Add(state, actions);
        }
    }
}

/// <summary>
/// A pair of per-arm integer counters exposed as one INTERLEAVED parameter vector:
/// <c>[first[0], second[0], first[1], second[1], ...]</c>.
/// </summary>
/// <remarks>
/// Thompson sampling keeps a Beta posterior per arm as a success and a failure count, and its
/// hand-written surface interleaved them rather than concatenating the two arrays. That ordering is
/// its serialization order, so it is preserved exactly here -- concatenating instead would still
/// produce a vector of the right LENGTH while silently renumbering every existing checkpoint.
///
/// The counters are integers. Values round-trip through <typeparamref name="T"/> and are truncated
/// back to int on restore, matching the original code; a fractional value written into this surface
/// is floored, not rejected.
/// </remarks>
public sealed class InterleavedCountParameterSource<T> : IParameterSource<T>
{
    private readonly Vector<int> _first;
    private readonly Vector<int> _second;
    private readonly int _pairs;

    /// <summary>Creates a source over <paramref name="pairs"/> (first, second) counter pairs.</summary>
    public InterleavedCountParameterSource(Vector<int> first, Vector<int> second, int pairs)
    {
        _first = first ?? throw new ArgumentNullException(nameof(first));
        _second = second ?? throw new ArgumentNullException(nameof(second));
        _pairs = pairs;
    }

    /// <inheritdoc />
    public long ParameterCount => (long)_pairs * 2;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        if (_pairs == 0)
        {
            return new Vector<T>(0);
        }

        // Stack the two count vectors as the columns of a [pairs, 2] tensor: its row-major layout
        // is the interleaved first/second order. Cast converts every count to T in one bulk pass.
        var engine = AiDotNetEngine.Current;
        var interleaved = engine.TensorStack(
            new[] { _first.GetSubVector(0, _pairs).AsTensor(), _second.GetSubVector(0, _pairs).AsTensor() }, 1);
        return interleaved.Cast<T>().ToVector();
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int pairs = Math.Min(_pairs, parameters.Length / 2);
        if (pairs == 0)
        {
            return;
        }

        // Truncate toward zero (the (int) cast), de-interleave with stride-2 gathers, and convert
        // the now-integral values to int in bulk.
        var engine = AiDotNetEngine.Current;
        var truncated = engine.Truncate(parameters.GetSubVector(0, pairs * 2));
        var first = engine.StridedGather(truncated, 0, 2, pairs).AsTensor().Cast<int>();
        var second = engine.StridedGather(truncated, 1, 2, pairs).AsTensor().Cast<int>();
        first.AsSpan().CopyTo(_first.AsWritableSpan());
        second.AsSpan().CopyTo(_second.AsWritableSpan());
    }
}
