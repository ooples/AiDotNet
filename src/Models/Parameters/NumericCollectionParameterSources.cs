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
