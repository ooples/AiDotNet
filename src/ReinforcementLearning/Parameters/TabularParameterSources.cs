using AiDotNet.Interfaces;
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
public sealed class QTableParameterSource<T> : IParameterSource<T>
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
        var result = new Vector<T>(checked((int)ParameterCount));
        int idx = 0;
        foreach (var row in _table.Values)
        {
            for (int action = 0; action < _actionSize; action++)
            {
                result[idx++] = row.TryGetValue(action, out var v) ? v : _ops.Zero;
            }
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int idx = 0;
        foreach (var row in _table.Values)
        {
            for (int action = 0; action < _actionSize && idx < parameters.Length; action++)
            {
                row[action] = parameters[idx++];
            }
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
public sealed class ValueTableParameterSource<T> : IParameterSource<T>
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
        var result = new Vector<T>(_table.Count);
        int idx = 0;
        foreach (var value in _table.Values) result[idx++] = value;
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int idx = 0;
        foreach (var key in _table.Keys.ToList())
        {
            if (idx >= parameters.Length) break;
            _table[key] = parameters[idx++];
        }
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
        var m = _get();
        var result = new Vector<T>(m.Rows * m.Columns);
        int idx = 0;
        for (int r = 0; r < m.Rows; r++)
        {
            for (int c = 0; c < m.Columns; c++) result[idx++] = m[r, c];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var m = _get();
        int idx = 0;
        for (int r = 0; r < m.Rows; r++)
        {
            for (int c = 0; c < m.Columns && idx < parameters.Length; c++) m[r, c] = parameters[idx++];
        }
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
        var v = _get();
        var result = new Vector<T>(v.Length);
        for (int i = 0; i < v.Length; i++) result[i] = v[i];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var v = _get();
        for (int i = 0; i < v.Length && i < parameters.Length; i++) v[i] = parameters[i];
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
public sealed class QTableEntriesParameterSource<T> : IParameterSource<T>
{
    private readonly Dictionary<string, Dictionary<int, T>> _table;
    private readonly bool _padEmptyToOne;
    private readonly INumericOperations<T> _ops;

    /// <summary>Wraps <paramref name="table"/>, counting only the entries present.</summary>
    public QTableEntriesParameterSource(Dictionary<string, Dictionary<int, T>> table, bool padEmptyToOne = false)
    {
        _table = table ?? throw new ArgumentNullException(nameof(table));
        _padEmptyToOne = padEmptyToOne;
        _ops = MathHelper.GetNumericOperations<T>();
    }

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
        var result = new Vector<T>(checked((int)ParameterCount));
        int idx = 0;
        foreach (var row in _table.Values)
        {
            foreach (var value in row.Values)
            {
                if (idx >= result.Length) break;
                result[idx++] = value;
            }
        }
        while (idx < result.Length) result[idx++] = _ops.Zero;
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int idx = 0;
        foreach (var row in _table.Values)
        {
            foreach (var action in row.Keys.ToList())
            {
                if (idx >= parameters.Length) return;
                row[action] = parameters[idx++];
            }
        }
    }
}
