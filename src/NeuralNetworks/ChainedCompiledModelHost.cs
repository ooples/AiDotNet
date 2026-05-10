using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Composite compilation helper that wraps a sequence of <see cref="CompiledModelHost{T}"/>
/// stages and replays them as a single chained forward pass — each stage compiled
/// and cached independently, but the overall <see cref="Predict"/> presents a
/// single eager-forward surface.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <remarks>
/// <para>
/// Composite models — SDXL (text-encoder × 2 + UNet + VAE), multi-stage VLMs
/// (vision-encoder + Q-Former + decoder), encoder-decoder transformers — have
/// stages whose shapes evolve independently across calls. A single
/// <see cref="CompiledModelHost{T}"/> traces and caches against the input shape
/// of the first stage, but downstream stages may see fresh shapes per call
/// (text length, image batch, denoising-step count) that don't fit the cached
/// plan. Wrapping each stage in its own host lets the cache key by per-stage
/// input shape, so the early stages stay hot across calls and only the
/// shape-varying tail recompiles.
/// </para>
/// <para>
/// Each stage carries its own <c>structureVersion</c> token — when a stage's
/// layer graph mutates (lazy-init, weight swap, LoRA hot-swap) the caller bumps
/// only that stage's version. The other stages' compiled plans are unaffected.
/// </para>
/// <code>
/// // SDXL composite hosting:
/// public class SDXLModel&lt;T&gt; : NeuralNetworkBase&lt;T&gt; {
///     private readonly ChainedCompiledModelHost&lt;T&gt; _gen = new(stageCount: 4);
///     // stage 0 = text encoder 1, 1 = text encoder 2, 2 = UNet, 3 = VAE decode
///
///     public Tensor&lt;T&gt; Generate(Tensor&lt;T&gt; tokenIds) =&gt; _gen.Predict(
///         input: tokenIds,
///         versions: new[] { _txtEnc1.StructureVersion, _txtEnc2.StructureVersion,
///                          _unet.StructureVersion, _vae.StructureVersion },
///         stages: new Func&lt;Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;[] {
///             x =&gt; _txtEnc1.Forward(x), x =&gt; _txtEnc2.Forward(x),
///             x =&gt; _unet.Forward(x),    x =&gt; _vae.Decode(x)
///         });
/// }
/// </code>
/// <para>
/// <b>Disposal cascade:</b> <see cref="Dispose"/> propagates to every stage host
/// — callers don't track per-stage cleanup. <see cref="InvalidateAll"/> wipes
/// every stage cache (typical use: weight swap that touches every stage).
/// </para>
/// </remarks>
internal sealed class ChainedCompiledModelHost<T> : IDisposable
{
    /// <summary>Per-stage compiled hosts, one per pipeline stage. Nulled in
    /// <see cref="Dispose"/> after element disposal so the array reference is
    /// released for GC.</summary>
    private CompiledModelHost<T>[]? _stageHosts;

    private bool _disposed;

    /// <summary>
    /// Create a chained host for a fixed-stage-count pipeline. Each stage gets
    /// its own <see cref="CompiledModelHost{T}"/>; passing
    /// <paramref name="modelIdentity"/> stamps the stage hosts with
    /// <c>"{modelIdentity}.stage{i}"</c> identities for disk plan caching.
    /// </summary>
    /// <param name="stageCount">Number of forward stages (must be ≥ 1).</param>
    /// <param name="modelIdentity">Optional stable model name (e.g. the
    /// concrete <c>this.GetType().FullName</c>) used as a prefix for each
    /// stage's disk plan cache key. <c>null</c> disables disk caching.</param>
    public ChainedCompiledModelHost(int stageCount, string? modelIdentity = null)
    {
        if (stageCount < 1)
            throw new System.ArgumentOutOfRangeException(nameof(stageCount), "Need at least 1 stage.");
        _stageHosts = new CompiledModelHost<T>[stageCount];
        for (int i = 0; i < stageCount; i++)
        {
            string? id = modelIdentity is null ? null : $"{modelIdentity}.stage{i}";
            _stageHosts[i] = new CompiledModelHost<T>(modelIdentity: id);
        }
    }

    /// <summary>Number of stages the chain was built for. Throws
    /// <see cref="ObjectDisposedException"/> if the host has been disposed.</summary>
    public int StageCount => (_stageHosts ?? throw new System.ObjectDisposedException(nameof(ChainedCompiledModelHost<T>))).Length;

    /// <summary>
    /// Run the chained forward: stage 0 sees <paramref name="input"/>, each
    /// subsequent stage sees the prior stage's output. Each stage is compiled
    /// against its own structure version and input shape via its own
    /// <see cref="CompiledModelHost{T}"/>.
    /// </summary>
    /// <param name="input">Input to the first stage.</param>
    /// <param name="versions">Per-stage structure versions
    /// (length must equal <see cref="StageCount"/>). Bumping a stage's
    /// version drops only that stage's plan cache; the rest stay hot.</param>
    /// <param name="stages">Per-stage eager-forward lambdas
    /// (length must equal <see cref="StageCount"/>). Each lambda takes the
    /// stage's input and returns its output, which becomes the next stage's
    /// input. Invoked at most once per (stage, shape, version) tuple — replay
    /// happens via the compiled plan on subsequent calls.</param>
    public Tensor<T> Predict(Tensor<T> input, int[] versions, System.Func<Tensor<T>, Tensor<T>>[] stages)
    {
        if (input is null) throw new System.ArgumentNullException(nameof(input));
        if (versions is null) throw new System.ArgumentNullException(nameof(versions));
        if (stages is null) throw new System.ArgumentNullException(nameof(stages));
        var hosts = _stageHosts ?? throw new System.ObjectDisposedException(nameof(ChainedCompiledModelHost<T>));
        if (versions.Length != hosts.Length)
            throw new System.ArgumentException(
                $"versions length ({versions.Length}) must equal stageCount ({hosts.Length}).",
                nameof(versions));
        if (stages.Length != hosts.Length)
            throw new System.ArgumentException(
                $"stages length ({stages.Length}) must equal stageCount ({hosts.Length}).",
                nameof(stages));

        var current = input;
        for (int i = 0; i < hosts.Length; i++)
        {
            // Capture per-stage closure variables so the eager-forward lambda
            // sees the right tensor when its host actually invokes it. Without
            // these locals the closure would capture the loop variable and
            // every stage's lambda would call stages[stageCount-1] with the
            // final tensor.
            var stageInput = current;
            var stageFn = stages[i];
            current = hosts[i].Predict(stageInput, versions[i], () => stageFn(stageInput));
        }
        return current;
    }

    /// <summary>
    /// Invalidate every stage's plan cache. Call when a non-version-tracked
    /// global change has happened (e.g. <c>Engine</c> swap, batch-size mode
    /// change) and the per-stage version counters won't catch it.
    /// </summary>
    public void InvalidateAll()
    {
        if (_disposed || _stageHosts is null) return;
        for (int i = 0; i < _stageHosts.Length; i++)
            _stageHosts[i].Invalidate();
    }

    /// <summary>
    /// Invalidate a single stage's plan cache. Use when one stage's weights
    /// have been mutated in a way the caller's structure-version counter
    /// doesn't capture (e.g. an external LoRA hot-swap).
    /// </summary>
    public void InvalidateStage(int stageIndex)
    {
        if (_disposed || _stageHosts is null) return;
        if ((uint)stageIndex >= (uint)_stageHosts.Length)
            throw new System.ArgumentOutOfRangeException(nameof(stageIndex));
        _stageHosts[stageIndex].Invalidate();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        var hosts = _stageHosts;
        if (hosts is not null)
        {
            for (int i = 0; i < hosts.Length; i++)
                hosts[i].Dispose();
            // Release the array reference for GC. Subsequent accesses on the
            // public surface (Predict, StageCount) throw ObjectDisposedException;
            // InvalidateAll / InvalidateStage no-op via the same null check.
            _stageHosts = null;
        }
    }
}
