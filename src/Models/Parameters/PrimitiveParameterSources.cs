using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Parameters;

/// <summary>Exposes an explicitly-marked <see cref="double"/> field as one write-through parameter.</summary>
public sealed class DoubleScalarParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly Func<double> _get;
    private readonly Action<double> _set;
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public DoubleScalarParameterSource(Func<double> get, Action<double> set)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _set = set ?? throw new ArgumentNullException(nameof(set));
    }

    public long ParameterCount => 1;

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() =>
        new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 1)
        };

    public Vector<T> GetParameters() => new(new[] { _numOps.FromDouble(_get()) });

    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != 1)
            throw new ArgumentException($"Expected 1 parameter, got {parameters.Length}.", nameof(parameters));
        _set(_numOps.ToDouble(parameters[0]));
    }
}

/// <summary>Exposes an explicitly-marked <see cref="double"/> array as a write-through parameter source.</summary>
public sealed class DoubleArrayParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly Func<double[]?> _get;
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public DoubleArrayParameterSource(Func<double[]?> get)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
    }

    public long ParameterCount => _get()?.LongLength ?? 0;

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var value = _get();
        return new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable,
                value is null ? ParameterReadiness.ShapeDeferred
                    : value.Length == 0 ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized,
                value?.LongLength)
        };
    }

    public Vector<T> GetParameters()
    {
        var value = _get();
        if (value is null) return new Vector<T>(0);
        var result = new Vector<T>(value.Length);
        for (int i = 0; i < value.Length; i++) result[i] = _numOps.FromDouble(value[i]);
        return result;
    }

    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var value = _get();
        if (value is null)
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        if (parameters.Length != value.Length)
            throw new ArgumentException(
                $"Expected {value.Length} parameters, got {parameters.Length}.", nameof(parameters));
        for (int i = 0; i < value.Length; i++) value[i] = _numOps.ToDouble(parameters[i]);
    }
}

/// <summary>Exposes an explicitly-marked jagged <see cref="double"/> array in stable row-major order.</summary>
public sealed class DoubleJaggedParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly Func<double[][]?> _get;
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public DoubleJaggedParameterSource(Func<double[][]?> get)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
    }

    public long ParameterCount
    {
        get
        {
            var value = _get();
            if (value is null) return 0;
            long count = 0;
            for (int i = 0; i < value.Length; i++)
                count = checked(count + (value[i]?.LongLength ?? 0));
            return count;
        }
    }

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var value = _get();
        if (value is null)
        {
            return new[]
            {
                new ParameterSlotDescriptor(
                    "$", ParameterSlotRole.Trainable, ParameterReadiness.ShapeDeferred, null)
            };
        }

        if (value.Length == 0)
        {
            return new[]
            {
                new ParameterSlotDescriptor(
                    "$", ParameterSlotRole.Trainable,
                    ParameterReadiness.ShapeResolvedUnmaterialized, 0)
            };
        }

        var slots = new ParameterSlotDescriptor[value.Length];
        for (int i = 0; i < value.Length; i++)
        {
            var row = value[i];
            slots[i] = new ParameterSlotDescriptor(
                ParameterCollectionKeys.Index(i), ParameterSlotRole.Trainable,
                row is null ? ParameterReadiness.ShapeDeferred
                    : row.Length == 0 ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized,
                row?.LongLength);
        }
        return slots;
    }

    public Vector<T> GetParameters()
    {
        var value = _get();
        if (value is null) return new Vector<T>(0);
        var result = new Vector<T>(checked((int)ParameterCount));
        int at = 0;
        for (int i = 0; i < value.Length; i++)
        {
            var row = value[i];
            if (row is null)
                throw new ParameterLayoutNotReadyException(
                    "read", new ParameterLayoutSnapshot(GetParameterLayout()));
            for (int j = 0; j < row.Length; j++) result[at++] = _numOps.FromDouble(row[j]);
        }
        return result;
    }

    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var value = _get();
        if (value is null)
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));

        int at = 0;
        for (int i = 0; i < value.Length; i++)
        {
            var row = value[i];
            if (row is null)
                throw new ParameterLayoutNotReadyException(
                    "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
            for (int j = 0; j < row.Length; j++) row[j] = _numOps.ToDouble(parameters[at++]);
        }
    }
}
