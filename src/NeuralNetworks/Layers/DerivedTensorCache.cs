using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Holds one tensor derived from up to four source tensors (a stacked gate weight, a folded bias) and rebuilds it only
/// when it could differ: a source was replaced, a source's <see cref="Tensor{T}.Version"/> moved (every in-place write
/// advances it: an optimizer step, a <c>SetParameters</c>, a deserialization), or the engine changed.
/// </summary>
/// <remarks>
/// <para>
/// For inference-only derivations. The value is built outside any tensor arena, so an arena reset between predicts
/// cannot reclaim it, and under a no-grad scope, so it holds no tape history. A differentiable forward must compute its
/// derivation inline instead, connected to the tape.
/// </para>
/// <para>
/// A writer that changes a source without advancing its version (a device-side optimizer update of a GPU-resident
/// copy) must call <see cref="Invalidate"/>. A clone is a new layer with a new cache, so nothing is shared between
/// copies.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal sealed class DerivedTensorCache<T>
{
    private const int MaxSources = 4;

    private readonly Tensor<T>?[] _sources = new Tensor<T>?[MaxSources];
    private readonly int[] _versions = new int[MaxSources];
    private int _sourceCount;
    private IEngine? _engine;
    private Tensor<T>? _value;

    /// <summary>
    /// The cached derivation of the given sources, built with <paramref name="build"/> (passed the engine and the
    /// non-null sources in order) when there is none yet or it is stale. Sources after the first null are not used.
    /// </summary>
    internal Tensor<T> GetOrBuild(IEngine engine, Func<IEngine, Tensor<T>[], Tensor<T>> build,
        Tensor<T> source0, Tensor<T>? source1 = null, Tensor<T>? source2 = null, Tensor<T>? source3 = null)
    {
        int count = source1 is null ? 1 : source2 is null ? 2 : source3 is null ? 3 : 4;
        if (_value is { } cached && ReferenceEquals(_engine, engine) && _sourceCount == count
            && Matches(0, source0) && (count < 2 || Matches(1, source1)) && (count < 3 || Matches(2, source2))
            && (count < 4 || Matches(3, source3)))
            return cached;

        var sources = new Tensor<T>[count];
        sources[0] = source0;
        if (count > 1 && source1 is not null) sources[1] = source1;
        if (count > 2 && source2 is not null) sources[2] = source2;
        if (count > 3 && source3 is not null) sources[3] = source3;

        Tensor<T> value;
        using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            value = build(engine, sources);

        Array.Clear(_sources, 0, MaxSources);
        for (int i = 0; i < count; i++)
        {
            _sources[i] = sources[i];
            _versions[i] = sources[i].Version;
        }

        _sourceCount = count;
        _engine = engine;
        _value = value;
        return value;
    }

    /// <summary>Drops the cached value, so the next <see cref="GetOrBuild"/> rebuilds it.</summary>
    internal void Invalidate()
    {
        _value = null;
        _engine = null;
        _sourceCount = 0;
        Array.Clear(_sources, 0, MaxSources);
    }

    private bool Matches(int index, Tensor<T>? source)
        => source is not null && ReferenceEquals(_sources[index], source) && _versions[index] == source.Version;
}
