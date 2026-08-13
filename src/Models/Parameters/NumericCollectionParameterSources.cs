using System.Globalization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Parameters;

internal sealed class NumericCollectionParameterSource<T, TValue> :
    IParameterSource<T>, IParameterLayoutSource
{
    private readonly Func<IEnumerable<KeyValuePair<string, TValue>>?> _get;
    private readonly Func<TValue, long> _count;
    private readonly Action<TValue, Vector<T>, int> _read;
    private readonly Action<TValue, Vector<T>, int> _write;

    public NumericCollectionParameterSource(
        Func<IEnumerable<KeyValuePair<string, TValue>>?> get,
        Func<TValue, long> count,
        Action<TValue, Vector<T>, int> read,
        Action<TValue, Vector<T>, int> write)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _count = count ?? throw new ArgumentNullException(nameof(count));
        _read = read ?? throw new ArgumentNullException(nameof(read));
        _write = write ?? throw new ArgumentNullException(nameof(write));
    }

    private List<KeyValuePair<string, TValue>> Values()
    {
        var result = new List<KeyValuePair<string, TValue>>();
        var values = _get();
        if (values is null) return result;
        foreach (var value in values) result.Add(value);
        result.Sort((left, right) => StringComparer.Ordinal.Compare(left.Key, right.Key));
        for (int i = 1; i < result.Count; i++)
        {
            if (string.Equals(result[i - 1].Key, result[i].Key, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Collection parameter key '{result[i].Key}' is not unique after canonicalization.");
        }
        return result;
    }

    public long ParameterCount
    {
        get
        {
            long total = 0;
            var values = Values();
            for (int i = 0; i < values.Count; i++) total = checked(total + _count(values[i].Value));
            return total;
        }
    }

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var values = Values();
        var slots = new List<ParameterSlotDescriptor>(Math.Max(values.Count, 1));
        for (int i = 0; i < values.Count; i++)
        {
            long count = _count(values[i].Value);
            slots.Add(new ParameterSlotDescriptor(
                values[i].Key, ParameterSlotRole.Trainable,
                count == 0 ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized,
                count));
        }
        if (slots.Count == 0)
        {
            slots.Add(new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable, ParameterReadiness.ParameterFree, 0));
        }
        return slots;
    }

    public Vector<T> GetParameters()
    {
        var values = Values();
        long total = 0;
        for (int i = 0; i < values.Count; i++) total = checked(total + _count(values[i].Value));
        var result = new Vector<T>(checked((int)total));
        int offset = 0;
        for (int i = 0; i < values.Count; i++)
        {
            _read(values[i].Value, result, offset);
            offset = checked(offset + (int)_count(values[i].Value));
        }
        return result;
    }

    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var values = Values();
        long expected = 0;
        for (int i = 0; i < values.Count; i++) expected = checked(expected + _count(values[i].Value));
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

        int offset = 0;
        for (int i = 0; i < values.Count; i++)
        {
            _write(values[i].Value, parameters, offset);
            offset = checked(offset + (int)_count(values[i].Value));
        }
    }
}

internal static class ParameterCollectionKeys
{
    public static string Index(int index) => $"index={index:D8}";

    public static string Canonical<TKey>(TKey key)
    {
        string text = key is IFormattable formattable
            ? formattable.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty
            : key?.ToString() ?? "<null>";
        return "key=" + Uri.EscapeDataString(text);
    }
}

/// <summary>Deterministic collection ordering shared by generated model and network surfaces.</summary>
public static class ParameterCollectionOrdering
{
    /// <summary>Returns an empty sequence for an absent collection without snapshotting its values.</summary>
    public static IEnumerable<TValue> Present<TValue>(IEnumerable<TValue>? values)
        => values ?? Array.Empty<TValue>();

    /// <summary>Returns only materialized members from a nullable reference collection.</summary>
    public static IEnumerable<TValue> PresentNonNull<TValue>(IEnumerable<TValue?>? values)
        where TValue : class
    {
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value is not null) yield return value;
        }
    }

    /// <summary>Returns dictionary values ordered by the canonical invariant representation of their keys.</summary>
    public static IEnumerable<TValue> OrderedValues<TKey, TValue>(
        IEnumerable<KeyValuePair<TKey, TValue>>? values)
    {
        if (values is null) yield break;
        var ordered = new List<KeyValuePair<string, TValue>>();
        foreach (var value in values)
            ordered.Add(new KeyValuePair<string, TValue>(
                ParameterCollectionKeys.Canonical(value.Key), value.Value));
        ordered.Sort((left, right) => StringComparer.Ordinal.Compare(left.Key, right.Key));
        for (int i = 1; i < ordered.Count; i++)
        {
            if (string.Equals(ordered[i - 1].Key, ordered[i].Key, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Collection parameter key '{ordered[i].Key}' is not unique after canonicalization.");
        }
        for (int i = 0; i < ordered.Count; i++) yield return ordered[i].Value;
    }
}

/// <summary>An index-stable tensor collection exposed as one parameter source.</summary>
public sealed class TensorCollectionParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly NumericCollectionParameterSource<T, Tensor<T>> _inner;

    public TensorCollectionParameterSource(Func<IEnumerable<Tensor<T>?>?> get)
    {
        _inner = new NumericCollectionParameterSource<T, Tensor<T>>(
            () => Enumerate(get), tensor => tensor.Length,
            (tensor, destination, offset) =>
            {
                for (int i = 0; i < tensor.Length; i++) destination[offset + i] = tensor[i];
            },
            (tensor, source, offset) =>
            {
                for (int i = 0; i < tensor.Length; i++) tensor[i] = source[offset + i];
            });
    }

    private static IEnumerable<KeyValuePair<string, Tensor<T>>> Enumerate(Func<IEnumerable<Tensor<T>?>?> get)
    {
        int index = 0;
        var values = get();
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value is not null)
                yield return new KeyValuePair<string, Tensor<T>>(ParameterCollectionKeys.Index(index), value);
            index++;
        }
    }

    public long ParameterCount => _inner.ParameterCount;
    public Vector<T> GetParameters() => _inner.GetParameters();
    public void SetParameters(Vector<T> parameters) => _inner.SetParameters(parameters);
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() => _inner.GetParameterLayout();
}

/// <summary>An index-stable matrix collection exposed as one parameter source.</summary>
public sealed class MatrixCollectionParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly NumericCollectionParameterSource<T, Matrix<T>> _inner;

    public MatrixCollectionParameterSource(Func<IEnumerable<Matrix<T>?>?> get)
    {
        _inner = new NumericCollectionParameterSource<T, Matrix<T>>(
            () => Enumerate(get), matrix => (long)matrix.Rows * matrix.Columns,
            (matrix, destination, offset) =>
            {
                int at = offset;
                for (int r = 0; r < matrix.Rows; r++)
                    for (int c = 0; c < matrix.Columns; c++) destination[at++] = matrix[r, c];
            },
            (matrix, source, offset) =>
            {
                int at = offset;
                for (int r = 0; r < matrix.Rows; r++)
                    for (int c = 0; c < matrix.Columns; c++) matrix[r, c] = source[at++];
            });
    }

    private static IEnumerable<KeyValuePair<string, Matrix<T>>> Enumerate(Func<IEnumerable<Matrix<T>?>?> get)
    {
        int index = 0;
        var values = get();
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value is not null)
                yield return new KeyValuePair<string, Matrix<T>>(ParameterCollectionKeys.Index(index), value);
            index++;
        }
    }

    public long ParameterCount => _inner.ParameterCount;
    public Vector<T> GetParameters() => _inner.GetParameters();
    public void SetParameters(Vector<T> parameters) => _inner.SetParameters(parameters);
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() => _inner.GetParameterLayout();
}

/// <summary>An index-stable vector collection exposed as one parameter source.</summary>
public sealed class VectorCollectionParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly NumericCollectionParameterSource<T, Vector<T>> _inner;

    public VectorCollectionParameterSource(Func<IEnumerable<Vector<T>?>?> get)
    {
        _inner = new NumericCollectionParameterSource<T, Vector<T>>(
            () => Enumerate(get), vector => vector.Length,
            (vector, destination, offset) =>
            {
                for (int i = 0; i < vector.Length; i++) destination[offset + i] = vector[i];
            },
            (vector, source, offset) =>
            {
                for (int i = 0; i < vector.Length; i++) vector[i] = source[offset + i];
            });
    }

    private static IEnumerable<KeyValuePair<string, Vector<T>>> Enumerate(Func<IEnumerable<Vector<T>?>?> get)
    {
        int index = 0;
        var values = get();
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value is not null)
                yield return new KeyValuePair<string, Vector<T>>(ParameterCollectionKeys.Index(index), value);
            index++;
        }
    }

    public long ParameterCount => _inner.ParameterCount;
    public Vector<T> GetParameters() => _inner.GetParameters();
    public void SetParameters(Vector<T> parameters) => _inner.SetParameters(parameters);
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() => _inner.GetParameterLayout();
}

/// <summary>A key-stable tensor dictionary exposed as one parameter source.</summary>
public sealed class KeyedTensorCollectionParameterSource<T, TKey> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly NumericCollectionParameterSource<T, Tensor<T>> _inner;

    public KeyedTensorCollectionParameterSource(Func<IEnumerable<KeyValuePair<TKey, Tensor<T>>>?> get)
    {
        _inner = new NumericCollectionParameterSource<T, Tensor<T>>(
            () => Enumerate(get), tensor => tensor.Length,
            (tensor, destination, offset) =>
            {
                for (int i = 0; i < tensor.Length; i++) destination[offset + i] = tensor[i];
            },
            (tensor, source, offset) =>
            {
                for (int i = 0; i < tensor.Length; i++) tensor[i] = source[offset + i];
            });
    }

    private static IEnumerable<KeyValuePair<string, Tensor<T>>> Enumerate(
        Func<IEnumerable<KeyValuePair<TKey, Tensor<T>>>?> get)
    {
        var values = get();
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value.Value is not null)
                yield return new KeyValuePair<string, Tensor<T>>(
                    ParameterCollectionKeys.Canonical(value.Key), value.Value);
        }
    }

    public long ParameterCount => _inner.ParameterCount;
    public Vector<T> GetParameters() => _inner.GetParameters();
    public void SetParameters(Vector<T> parameters) => _inner.SetParameters(parameters);
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() => _inner.GetParameterLayout();
}

/// <summary>A key-stable matrix dictionary exposed as one parameter source.</summary>
public sealed class KeyedMatrixCollectionParameterSource<T, TKey> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly NumericCollectionParameterSource<T, Matrix<T>> _inner;

    public KeyedMatrixCollectionParameterSource(Func<IEnumerable<KeyValuePair<TKey, Matrix<T>>>?> get)
    {
        _inner = new NumericCollectionParameterSource<T, Matrix<T>>(
            () => Enumerate(get), matrix => (long)matrix.Rows * matrix.Columns,
            (matrix, destination, offset) =>
            {
                int at = offset;
                for (int r = 0; r < matrix.Rows; r++)
                    for (int c = 0; c < matrix.Columns; c++) destination[at++] = matrix[r, c];
            },
            (matrix, source, offset) =>
            {
                int at = offset;
                for (int r = 0; r < matrix.Rows; r++)
                    for (int c = 0; c < matrix.Columns; c++) matrix[r, c] = source[at++];
            });
    }

    private static IEnumerable<KeyValuePair<string, Matrix<T>>> Enumerate(
        Func<IEnumerable<KeyValuePair<TKey, Matrix<T>>>?> get)
    {
        var values = get();
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value.Value is not null)
                yield return new KeyValuePair<string, Matrix<T>>(
                    ParameterCollectionKeys.Canonical(value.Key), value.Value);
        }
    }

    public long ParameterCount => _inner.ParameterCount;
    public Vector<T> GetParameters() => _inner.GetParameters();
    public void SetParameters(Vector<T> parameters) => _inner.SetParameters(parameters);
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() => _inner.GetParameterLayout();
}

/// <summary>A key-stable vector dictionary exposed as one parameter source.</summary>
public sealed class KeyedVectorCollectionParameterSource<T, TKey> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly NumericCollectionParameterSource<T, Vector<T>> _inner;

    public KeyedVectorCollectionParameterSource(Func<IEnumerable<KeyValuePair<TKey, Vector<T>>>?> get)
    {
        _inner = new NumericCollectionParameterSource<T, Vector<T>>(
            () => Enumerate(get), vector => vector.Length,
            (vector, destination, offset) =>
            {
                for (int i = 0; i < vector.Length; i++) destination[offset + i] = vector[i];
            },
            (vector, source, offset) =>
            {
                for (int i = 0; i < vector.Length; i++) vector[i] = source[offset + i];
            });
    }

    private static IEnumerable<KeyValuePair<string, Vector<T>>> Enumerate(
        Func<IEnumerable<KeyValuePair<TKey, Vector<T>>>?> get)
    {
        var values = get();
        if (values is null) yield break;
        foreach (var value in values)
        {
            if (value.Value is not null)
                yield return new KeyValuePair<string, Vector<T>>(
                    ParameterCollectionKeys.Canonical(value.Key), value.Value);
        }
    }

    public long ParameterCount => _inner.ParameterCount;
    public Vector<T> GetParameters() => _inner.GetParameters();
    public void SetParameters(Vector<T> parameters) => _inner.SetParameters(parameters);
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() => _inner.GetParameterLayout();
}

/// <summary>A scalar dictionary exposed in canonical key order.</summary>
/// <remarks>
/// The source keeps the keys in the manifest, not merely the current enumeration order. This is
/// the parameter adapter for tabular models whose learned values live directly in dictionaries;
/// adding or re-inserting an unrelated key therefore cannot reorder an existing checkpoint.
/// </remarks>
public sealed class KeyedScalarCollectionParameterSource<T, TKey> :
    IParameterSource<T>, IParameterLayoutSource
    where TKey : notnull
{
    private readonly Func<IDictionary<TKey, T>?> _get;

    /// <summary>Creates a write-through source over the dictionary returned by <paramref name="get"/>.</summary>
    public KeyedScalarCollectionParameterSource(Func<IDictionary<TKey, T>?> get)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
    }

    private List<(string StableId, TKey Key)> Entries()
    {
        var result = new List<(string StableId, TKey Key)>();
        var values = _get();
        if (values is null) return result;

        foreach (var value in values)
            result.Add((ParameterCollectionKeys.Canonical(value.Key), value.Key));

        result.Sort((left, right) => StringComparer.Ordinal.Compare(left.StableId, right.StableId));
        for (int i = 1; i < result.Count; i++)
        {
            if (string.Equals(result[i - 1].StableId, result[i].StableId, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Collection parameter key '{result[i].StableId}' is not unique after canonicalization.");
        }
        return result;
    }

    /// <inheritdoc />
    public long ParameterCount => Entries().Count;

    /// <inheritdoc />
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var entries = Entries();
        if (entries.Count == 0)
        {
            return new[]
            {
                new ParameterSlotDescriptor(
                    "$", ParameterSlotRole.Trainable, ParameterReadiness.ParameterFree, 0)
            };
        }

        var slots = new List<ParameterSlotDescriptor>(entries.Count);
        for (int i = 0; i < entries.Count; i++)
        {
            slots.Add(new ParameterSlotDescriptor(
                entries[i].StableId, ParameterSlotRole.Trainable,
                ParameterReadiness.Materialized, 1));
        }
        return slots;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var values = _get();
        var entries = Entries();
        var result = new Vector<T>(entries.Count);
        if (values is null) return result;
        for (int i = 0; i < entries.Count; i++) result[i] = values[entries[i].Key];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var values = _get();
        var entries = Entries();
        if (parameters.Length != entries.Count)
            throw new ArgumentException(
                $"Expected {entries.Count} keyed scalar parameters, got {parameters.Length}.",
                nameof(parameters));
        if (values is null) return;
        for (int i = 0; i < entries.Count; i++) values[entries[i].Key] = parameters[i];
    }
}

/// <summary>A two-level scalar dictionary exposed in canonical outer- and inner-key order.</summary>
/// <remarks>
/// Tabular reinforcement-learning agents commonly store a state dictionary whose values are
/// action dictionaries. Both key levels are part of the durable identity, so sparse or ragged
/// tables round-trip without shifting later values onto a different state or action.
/// </remarks>
public sealed class NestedKeyedScalarCollectionParameterSource<T, TOuterKey, TInnerKey> :
    IParameterSource<T>, IParameterLayoutSource
    where TOuterKey : notnull
    where TInnerKey : notnull
{
    private readonly Func<IDictionary<TOuterKey, Dictionary<TInnerKey, T>>?> _get;

    /// <summary>Creates a write-through source over the nested dictionary returned by <paramref name="get"/>.</summary>
    public NestedKeyedScalarCollectionParameterSource(
        Func<IDictionary<TOuterKey, Dictionary<TInnerKey, T>>?> get)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
    }

    private List<(string StableId, TOuterKey OuterKey, TInnerKey InnerKey)> Entries()
    {
        var result = new List<(string StableId, TOuterKey OuterKey, TInnerKey InnerKey)>();
        var values = _get();
        if (values is null) return result;

        foreach (var outer in values)
        {
            string outerId = ParameterCollectionKeys.Canonical(outer.Key);
            if (outer.Value is null) continue;
            foreach (var inner in outer.Value)
            {
                result.Add((outerId + "/" + ParameterCollectionKeys.Canonical(inner.Key),
                    outer.Key, inner.Key));
            }
        }

        result.Sort((left, right) => StringComparer.Ordinal.Compare(left.StableId, right.StableId));
        for (int i = 1; i < result.Count; i++)
        {
            if (string.Equals(result[i - 1].StableId, result[i].StableId, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Collection parameter key '{result[i].StableId}' is not unique after canonicalization.");
        }
        return result;
    }

    /// <inheritdoc />
    public long ParameterCount => Entries().Count;

    /// <inheritdoc />
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var entries = Entries();
        if (entries.Count == 0)
        {
            return new[]
            {
                new ParameterSlotDescriptor(
                    "$", ParameterSlotRole.Trainable, ParameterReadiness.ParameterFree, 0)
            };
        }

        var slots = new List<ParameterSlotDescriptor>(entries.Count);
        for (int i = 0; i < entries.Count; i++)
        {
            slots.Add(new ParameterSlotDescriptor(
                entries[i].StableId, ParameterSlotRole.Trainable,
                ParameterReadiness.Materialized, 1));
        }
        return slots;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var values = _get();
        var entries = Entries();
        var result = new Vector<T>(entries.Count);
        if (values is null) return result;
        for (int i = 0; i < entries.Count; i++)
            result[i] = values[entries[i].OuterKey][entries[i].InnerKey];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var values = _get();
        var entries = Entries();
        if (parameters.Length != entries.Count)
            throw new ArgumentException(
                $"Expected {entries.Count} nested keyed scalar parameters, got {parameters.Length}.",
                nameof(parameters));
        if (values is null) return;
        for (int i = 0; i < entries.Count; i++)
            values[entries[i].OuterKey][entries[i].InnerKey] = parameters[i];
    }
}
