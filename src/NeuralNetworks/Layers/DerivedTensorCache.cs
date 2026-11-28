using System.Threading;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Non-generic lifecycle contract for layer-owned derived parameter caches so <see cref="LayerBase{T}"/> can
/// invalidate and release persistent device buffers uniformly during weight updates and disposal.
/// </summary>
internal interface IDerivedParameterCache
{
    void Invalidate();
    void Release();
}

/// <summary>
/// Process-wide and thread-local runtime switch controlling whether inference derived-parameter caches are active.
/// Configured end-to-end via <c>AiModelBuilder.ConfigureExecutionAcceleration</c> and <c>AiModelResult.Predict</c>.
/// </summary>
internal static class DerivedTensorCacheSettings
{
    private static int s_defaultEnabled = 1;

    [ThreadStatic]
    private static int t_threadOverride;

    internal static bool DefaultEnabled
    {
        get => Volatile.Read(ref s_defaultEnabled) != 0;
        set => Interlocked.Exchange(ref s_defaultEnabled, value ? 1 : 0);
    }

    internal static void SetThreadOverride(bool? enabled)
        => t_threadOverride = enabled.HasValue ? (enabled.Value ? 1 : -1) : 0;

    internal static bool IsEnabled
        => t_threadOverride != 0 ? t_threadOverride > 0 : DefaultEnabled;
}

/// <summary>
/// Thread-safe inference cache holding one tensor derived from up to four source tensors (such as a stacked gate
/// weight, a weight-normalized convolution kernel, or a transposed projection matrix) and rebuilding it only when a
/// source reference, storage identity, tensor <see cref="TensorBase{T}.Version"/>, shared storage
/// <see cref="TensorBase{T}.GpuCacheVersion"/>, training generation, variant key, or execution <see cref="IEngine"/>
/// changes.
/// </summary>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal sealed class DerivedTensorCache<T> : IDerivedParameterCache
{
    private const int MaxSources = 4;

    /// <summary>The number of training steps or weight-surface mutations noted so far, process-wide.</summary>
    internal static long TrainingGeneration => DerivedTensorCacheGeneration.Current;

    /// <summary>Records that a training step or parameter mutation occurred so cached derivations verify or rebuild.</summary>
    internal static void NoteTrainingStep() => DerivedTensorCacheGeneration.Advance();

    /// <summary>
    /// Returns <c>true</c> when the current execution context is a non-differentiable, non-tracing pass
    /// eligible to serve cached derived parameter tensors.
    /// </summary>
    internal static bool CanUse(IEngine engine, bool requireEvalMode = false, bool isTrainingMode = false)
        => (!requireEvalMode || !isTrainingMode)
           && DerivedTensorCacheSettings.IsEnabled
           && (AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed
               || (AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is null
                   && !AiDotNet.Tensors.Engines.Autodiff.GradientRecording.IsRecording<T>()
                   && !AiDotNet.Tensors.Engines.Autodiff.GradientRecording.IsTrainingTrace))
           && !AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive
           && !AiDotNet.NeuralNetworks.GraphCaptureCompatibility.IsActive
           && ReferenceEquals(engine, engine.Unwrap());

    private readonly object _gate = new();
    private readonly Tensor<T>?[] _sources = new Tensor<T>?[MaxSources];
    private readonly object?[] _storageIdentities = new object?[MaxSources];
    private readonly int[] _storageMutationVersions = new int[MaxSources];
    private readonly int[] _versions = new int[MaxSources];
    private readonly int[] _gpuCacheVersions = new int[MaxSources];
    private readonly bool _disposeReplacedValues;
    private readonly bool _registerPersistent;
    private readonly PersistentTensorRole _persistentRole;

    private long _generation;
    private long _variantKey;
    private int _sourceCount;
    private IEngine? _engine;
    private Tensor<T>? _value;
    private bool _registeredPersistent;

    internal DerivedTensorCache(
        bool disposeReplacedValues = false,
        bool registerPersistent = false,
        PersistentTensorRole persistentRole = PersistentTensorRole.Weights)
    {
        _disposeReplacedValues = disposeReplacedValues;
        _registerPersistent = registerPersistent;
        _persistentRole = persistentRole;
    }

    private bool OwnsCachedValue(Tensor<T>? value)
    {
        if (value is null) return false;
        for (int i = 0; i < _sourceCount; i++)
        {
            if (ReferenceEquals(value, _sources[i])) return false;
        }
        return true;
    }

    private static bool IsSource(Tensor<T> candidate, Tensor<T>[] sources, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (ReferenceEquals(candidate, sources[i])) return true;
        }
        return false;
    }

    private static bool IsSparseTensor(Tensor<T> tensor)
        => tensor.GetType().Name.StartsWith("SparseTensor", StringComparison.Ordinal);

    private static void Retire(Tensor<T> value, IEngine? engine, bool unregisterPersistent, bool disposeValue)
    {
        if (engine is not null)
        {
            if (unregisterPersistent && !IsSparseTensor(value))
            {
                engine.UnregisterPersistentTensor(value);
            }
            engine.InvalidatePersistentTensor(value);
        }

        if (disposeValue)
        {
            value.Dispose();
        }
    }

    /// <summary>
    /// Gets the cached derivation of the supplied source tensors, or computes and caches it under
    /// <see cref="TensorScope.Suspend"/> and <see cref="AiDotNet.Tensors.Engines.Autodiff.NoGradScope{T}"/>.
    /// </summary>
    internal Tensor<T> GetOrBuild(
        IEngine engine,
        Func<IEngine, Tensor<T>[], Tensor<T>> build,
        Tensor<T> source0,
        Tensor<T>? source1 = null,
        Tensor<T>? source2 = null,
        Tensor<T>? source3 = null,
        long variantKey = 0)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (build is null) throw new ArgumentNullException(nameof(build));
        if (source0 is null) throw new ArgumentNullException(nameof(source0));

        int count = source1 is null ? 1 : source2 is null ? 2 : source3 is null ? 3 : 4;
        lock (_gate)
        {
            long generation = TrainingGeneration;
            if (_value is { } cached
                && _generation == generation
                && _variantKey == variantKey
                && ReferenceEquals(_engine, engine)
                && SourcesMatch(count, source0, source1, source2, source3))
            {
                return cached;
            }

            var sources = new Tensor<T>[count];
            sources[0] = source0;
            if (count > 1 && source1 is not null) sources[1] = source1;
            if (count > 2 && source2 is not null) sources[2] = source2;
            if (count > 3 && source3 is not null) sources[3] = source3;

            Tensor<T> value;
            bool registeredNew = false;
            using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
            using (TensorScope.Suspend())
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                value = build(engine, sources)
                    ?? throw new InvalidOperationException("DerivedTensorCache build callback returned null.");
                if (_registerPersistent && !IsSource(value, sources, count) && !IsSparseTensor(value))
                {
                    engine.RegisterPersistentTensor(value, _persistentRole);
                    registeredNew = true;
                }
            }

            var replaced = _value;
            var replacedEngine = _engine;
            bool replacedRegistered = _registeredPersistent;
            bool ownsReplaced = !ReferenceEquals(replaced, value) && OwnsCachedValue(replaced);

            Array.Clear(_sources, 0, MaxSources);
            Array.Clear(_storageIdentities, 0, MaxSources);
            for (int i = 0; i < count; i++)
            {
                _sources[i] = sources[i];
                _storageIdentities[i] = sources[i].StorageIdentity;
                _storageMutationVersions[i] = sources[i].StorageMutationVersion;
                _versions[i] = sources[i].Version;
                _gpuCacheVersions[i] = sources[i].GpuCacheVersion;
            }

            _sourceCount = count;
            _engine = engine;
            _generation = generation;
            _variantKey = variantKey;
            _value = value;
            _registeredPersistent = registeredNew;

            if (ownsReplaced && replaced is not null)
            {
                Retire(replaced, replacedEngine, replacedRegistered, _disposeReplacedValues);
            }

            return value;
        }
    }

    /// <summary>Drops the cached value so the next <see cref="GetOrBuild"/> rebuilds it.</summary>
    public void Invalidate() => ClearCore(disposeOwned: _disposeReplacedValues);

    /// <summary>Releases persistent engine registrations and drops cached references during layer disposal.</summary>
    public void Release() => ClearCore(disposeOwned: _disposeReplacedValues);

    private void ClearCore(bool disposeOwned)
    {
        lock (_gate)
        {
            var replaced = _value;
            var engine = _engine;
            bool wasRegistered = _registeredPersistent;
            bool ownsReplaced = OwnsCachedValue(replaced);

            _value = null;
            _engine = null;
            _registeredPersistent = false;
            _sourceCount = 0;
            Array.Clear(_sources, 0, MaxSources);
            Array.Clear(_storageIdentities, 0, MaxSources);

            if (ownsReplaced && replaced is not null)
            {
                Retire(replaced, engine, wasRegistered, disposeOwned);
            }
        }
    }

    private bool SourcesMatch(int count, Tensor<T> source0, Tensor<T>? source1, Tensor<T>? source2, Tensor<T>? source3)
    {
        if (_sourceCount != count || !Matches(0, source0)) return false;
        if (count > 1 && !Matches(1, source1)) return false;
        if (count > 2 && !Matches(2, source2)) return false;
        return count < 4 || Matches(3, source3);
    }

    private bool Matches(int index, Tensor<T>? source)
        => source is not null
           && ReferenceEquals((object?)_sources[index], (object)source)
           && ReferenceEquals(_storageIdentities[index], source.StorageIdentity)
           && _storageMutationVersions[index] == source.StorageMutationVersion
           && _versions[index] == source.Version
           && _gpuCacheVersions[index] == source.GpuCacheVersion;
}

/// <summary>
/// Thread-safe inference cache holding one or two tensors derived together from up to four source tensors (such as
/// Compact WY reflector factors <c>(UHatCol, TUHat)</c> or a materialized orthogonal operator <c>(Q, null)</c>).
/// </summary>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal sealed class DerivedTensorPairCache<T> : IDerivedParameterCache
{
    private const int MaxSources = 4;

    private readonly object _gate = new();
    private readonly Tensor<T>?[] _sources = new Tensor<T>?[MaxSources];
    private readonly object?[] _storageIdentities = new object?[MaxSources];
    private readonly int[] _storageMutationVersions = new int[MaxSources];
    private readonly int[] _versions = new int[MaxSources];
    private readonly int[] _gpuCacheVersions = new int[MaxSources];
    private readonly bool _disposeReplacedValues;
    private readonly bool _registerPersistent;
    private readonly PersistentTensorRole _persistentRole;

    private long _generation;
    private long _variantKey;
    private int _sourceCount;
    private IEngine? _engine;
    private Tensor<T>? _first;
    private Tensor<T>? _second;
    private bool _hasValue;
    private bool _firstRegistered;
    private bool _secondRegistered;

    internal DerivedTensorPairCache(
        bool disposeReplacedValues = false,
        bool registerPersistent = true,
        PersistentTensorRole persistentRole = PersistentTensorRole.Weights)
    {
        _disposeReplacedValues = disposeReplacedValues;
        _registerPersistent = registerPersistent;
        _persistentRole = persistentRole;
    }

    private bool OwnsCachedValue(Tensor<T>? value)
    {
        if (value is null) return false;
        for (int i = 0; i < _sourceCount; i++)
        {
            if (ReferenceEquals(value, _sources[i])) return false;
        }
        return true;
    }

    private static bool IsSource(Tensor<T> candidate, Tensor<T>[] sources, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (ReferenceEquals(candidate, sources[i])) return true;
        }
        return false;
    }

    private static bool IsSparseTensor(Tensor<T> tensor)
        => tensor.GetType().Name.StartsWith("SparseTensor", StringComparison.Ordinal);

    private static void Retire(Tensor<T> value, IEngine? engine, bool unregisterPersistent, bool disposeValue)
    {
        if (engine is not null)
        {
            if (unregisterPersistent && !IsSparseTensor(value))
            {
                engine.UnregisterPersistentTensor(value);
            }
            engine.InvalidatePersistentTensor(value);
        }

        if (disposeValue)
        {
            value.Dispose();
        }
    }

    internal (Tensor<T> First, Tensor<T>? Second) GetOrBuild(
        IEngine engine,
        Func<IEngine, Tensor<T>[], (Tensor<T> First, Tensor<T>? Second)> build,
        Tensor<T> source0,
        Tensor<T>? source1 = null,
        Tensor<T>? source2 = null,
        Tensor<T>? source3 = null,
        long variantKey = 0)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (build is null) throw new ArgumentNullException(nameof(build));
        if (source0 is null) throw new ArgumentNullException(nameof(source0));

        int count = source1 is null ? 1 : source2 is null ? 2 : source3 is null ? 3 : 4;
        lock (_gate)
        {
            long generation = DerivedTensorCacheGeneration.Current;
            if (_hasValue
                && _first is { } cachedFirst
                && _generation == generation
                && _variantKey == variantKey
                && ReferenceEquals(_engine, engine)
                && SourcesMatch(count, source0, source1, source2, source3))
            {
                return (cachedFirst, _second);
            }

            var sources = new Tensor<T>[count];
            sources[0] = source0;
            if (count > 1 && source1 is not null) sources[1] = source1;
            if (count > 2 && source2 is not null) sources[2] = source2;
            if (count > 3 && source3 is not null) sources[3] = source3;

            (Tensor<T> First, Tensor<T>? Second) pair;
            bool regFirst = false;
            bool regSecond = false;
            using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
            using (TensorScope.Suspend())
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                pair = build(engine, sources);
                if (pair.First is null)
                    throw new InvalidOperationException("DerivedTensorPairCache build callback returned a null First tensor.");
                if (_registerPersistent && !IsSource(pair.First, sources, count) && !IsSparseTensor(pair.First))
                {
                    engine.RegisterPersistentTensor(pair.First, _persistentRole);
                    regFirst = true;
                }
                if (_registerPersistent && pair.Second is not null && !ReferenceEquals(pair.Second, pair.First)
                    && !IsSource(pair.Second, sources, count) && !IsSparseTensor(pair.Second))
                {
                    engine.RegisterPersistentTensor(pair.Second, _persistentRole);
                    regSecond = true;
                }
            }

            var oldFirst = _first;
            var oldSecond = _second;
            var oldEngine = _engine;
            bool oldFirstReg = _firstRegistered;
            bool oldSecondReg = _secondRegistered;
            bool retireFirst = !ReferenceEquals(oldFirst, pair.First) && !ReferenceEquals(oldFirst, pair.Second) && OwnsCachedValue(oldFirst);
            bool retireSecond = !ReferenceEquals(oldSecond, pair.First) && !ReferenceEquals(oldSecond, pair.Second)
                && !ReferenceEquals(oldSecond, oldFirst) && OwnsCachedValue(oldSecond);

            Array.Clear(_sources, 0, MaxSources);
            Array.Clear(_storageIdentities, 0, MaxSources);
            for (int i = 0; i < count; i++)
            {
                _sources[i] = sources[i];
                _storageIdentities[i] = sources[i].StorageIdentity;
                _storageMutationVersions[i] = sources[i].StorageMutationVersion;
                _versions[i] = sources[i].Version;
                _gpuCacheVersions[i] = sources[i].GpuCacheVersion;
            }

            _sourceCount = count;
            _engine = engine;
            _generation = generation;
            _variantKey = variantKey;
            _first = pair.First;
            _second = pair.Second;
            _hasValue = true;
            _firstRegistered = regFirst;
            _secondRegistered = regSecond;

            if (retireFirst && oldFirst is not null)
            {
                Retire(oldFirst, oldEngine, oldFirstReg, _disposeReplacedValues);
            }
            if (retireSecond && oldSecond is not null)
            {
                Retire(oldSecond, oldEngine, oldSecondReg, _disposeReplacedValues);
            }

            return pair;
        }
    }

    public void Invalidate() => ClearCore(disposeOwned: _disposeReplacedValues);

    public void Release() => ClearCore(disposeOwned: _disposeReplacedValues);

    private void ClearCore(bool disposeOwned)
    {
        lock (_gate)
        {
            var oldFirst = _first;
            var oldSecond = _second;
            var oldEngine = _engine;
            bool oldFirstReg = _firstRegistered;
            bool oldSecondReg = _secondRegistered;
            bool retireFirst = OwnsCachedValue(oldFirst);
            bool retireSecond = !ReferenceEquals(oldSecond, oldFirst) && OwnsCachedValue(oldSecond);

            _first = null;
            _second = null;
            _hasValue = false;
            _engine = null;
            _firstRegistered = false;
            _secondRegistered = false;
            _sourceCount = 0;
            Array.Clear(_sources, 0, MaxSources);
            Array.Clear(_storageIdentities, 0, MaxSources);

            if (retireFirst && oldFirst is not null)
            {
                Retire(oldFirst, oldEngine, oldFirstReg, disposeOwned);
            }
            if (retireSecond && oldSecond is not null)
            {
                Retire(oldSecond, oldEngine, oldSecondReg, disposeOwned);
            }
        }
    }

    private bool SourcesMatch(int count, Tensor<T> source0, Tensor<T>? source1, Tensor<T>? source2, Tensor<T>? source3)
    {
        if (_sourceCount != count || !Matches(0, source0)) return false;
        if (count > 1 && !Matches(1, source1)) return false;
        if (count > 2 && !Matches(2, source2)) return false;
        return count < 4 || Matches(3, source3);
    }

    private bool Matches(int index, Tensor<T>? source)
        => source is not null
           && ReferenceEquals((object?)_sources[index], (object)source)
           && ReferenceEquals(_storageIdentities[index], source.StorageIdentity)
           && _storageMutationVersions[index] == source.StorageMutationVersion
           && _versions[index] == source.Version
           && _gpuCacheVersions[index] == source.GpuCacheVersion;
}

/// <summary>
/// The process-wide training generation every <see cref="DerivedTensorCache{T}"/> checks, advanced by every training step
/// (eager, taped, traced, or a compiled replay that updates weights on the device without advancing their versions)
/// and parameter-surface mutation.
/// </summary>
internal static class DerivedTensorCacheGeneration
{
    private static long s_generation;

    internal static long Current => Interlocked.Read(ref s_generation);

    internal static void Advance() => Interlocked.Increment(ref s_generation);
}
