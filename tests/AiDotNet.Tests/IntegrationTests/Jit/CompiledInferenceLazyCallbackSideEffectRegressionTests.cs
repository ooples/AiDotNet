using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Jit;

/// <summary>
/// Regression pin for the side-effect contract that closed issues #1352 and
/// #1353 (failed JIT trace inside LayerNorm and the associated trace-time
/// mean/variance state-mutation channel). Both issues were closed as
/// not-reproducible on AiDotNet 0.204 + AiDotNet.Tensors 0.81.3 — the
/// textbook repros pass cleanly because the #1331-family shape-tracking
/// fixes neutralized the upstream "Destination is too short" trigger AND
/// because <see cref="CpuEngine"/>'s LayerNorm/RmsNorm/BatchNorm/GroupNorm/
/// InstanceNorm/Dropout lazy callbacks now pre-realize their nodes at trace
/// time via <c>eagerResult.AsSpan().CopyTo(lazyResult.AsWritableSpan())</c>:
/// <c>AsWritableSpan()</c> on a tensor with a non-null <c>LazySource</c>
/// auto-materializes the node, setting <c>IsRealized=true</c> and running
/// the callback exactly once. By the time
/// <see cref="CompiledModelCache{T}.GetOrCompileInference(Tensor{T}, System.Func{Tensor{T}})"/>'s
/// <c>using</c>-scope <c>Dispose</c> hits the safety-net <c>Realize()</c>
/// on a failed trace, every recorded node short-circuits via
/// <c>if (IsRealized) return;</c>.
///
/// <para>
/// The contract this test pins: a <c>forward</c> closure that throws after
/// recording lazy LayerNorm nodes must NOT cause the scope's auto-realize
/// to re-execute those callbacks during dispose. The signal is a spy
/// engine's eager-LayerNorm counter — if the count after dispose exceeds
/// the count at the throw point, a future change has regressed the
/// pre-realization optimization and reopened the #1352/#1353 mutation
/// channel. Without that channel closed, every consumer that wraps
/// <c>GetOrCompileInference</c> in try/catch (the documented JIT fallback
/// pattern) leaks state corruption back into their model when the trace
/// fails.
/// </para>
/// </summary>
[Collection("NonParallelIntegration")]
public class CompiledInferenceLazyCallbackSideEffectRegressionTests
{
    private readonly ITestOutputHelper _output;

    public CompiledInferenceLazyCallbackSideEffectRegressionTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// When the <c>forward</c> closure passed to
    /// <see cref="CompiledModelCache{T}.GetOrCompileInference(Tensor{T}, System.Func{Tensor{T}})"/>
    /// throws after recording two chained lazy LayerNorm nodes, the lazy-
    /// graph scope's auto-realize on disposal must not re-execute any of
    /// the recorded callbacks. Two chained nodes are recorded so neither
    /// can be removed by the graph compiler's
    /// <c>DeadCodeEliminationPass</c> (node A has a consumer, node B is a
    /// leaf — both survive DCE). A spy engine subclasses
    /// <see cref="CpuEngine"/> and counts each invocation of the eager
    /// LayerNorm kernel; the count snapshotted at the throw point must
    /// equal the count after <see cref="CompiledModelCache{T}"/> has
    /// disposed its internal scope and propagated the original exception.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public async Task GetOrCompileInference_ForwardThrowsAfterTwoLayerNorms_DoesNotReplayLazyCallbacksOnDispose()
    {
        await Task.Yield();

        var spy = new LayerNormSpyEngine();
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = spy;
        try
        {
            const int B = 2;
            const int F = 8;
            var input = MakeInput(B, F);
            var gamma = MakeGamma(F);
            var beta = MakeBeta(F);

            int eagerCountAtThrow = -1;
            bool twoLayerNormsRecorded = false;

            using var cache = new CompiledModelCache<float>();

            var thrown = Assert.ThrowsAny<System.Exception>(() =>
                cache.GetOrCompileInference(input, () =>
                {
                    // Two chained LayerNorms: neither survives DCE
                    // independently (A has consumer B; B is a leaf), so
                    // both stay in the realized node list and would BOTH
                    // re-fire if scope.Dispose's safety-net Realize ran
                    // on the partial graph.
                    var ln1 = AiDotNetEngine.Current.LayerNorm(
                        input, gamma, beta, 1e-5,
                        out _, out _);
                    _ = AiDotNetEngine.Current.LayerNorm(
                        ln1, gamma, beta, 1e-5,
                        out _, out _);

                    eagerCountAtThrow = spy.EagerInvocationCount;
                    twoLayerNormsRecorded = true;

                    // Force partial-trace failure. The scope's auto-
                    // realize fires from the using-block's implicit
                    // finally — see LazyTensorScope.Dispose.
                    throw new System.InvalidOperationException(
                        "AIDN-1352-1353 forced-trace-failure sentinel");
                }));

            Assert.True(
                twoLayerNormsRecorded,
                "Test precondition failed: lazy LayerNorm ops never ran, " +
                "so the scope's auto-realize channel can't be exercised.");

            // The original exception must propagate unmasked. The
            // partial-trace path's safety-net Realize can only mask this
            // by throwing its own exception during dispose; with the
            // pre-realization optimization in place it short-circuits
            // every node and exits cleanly.
            Assert.IsType<System.InvalidOperationException>(thrown);
            Assert.Equal(
                "AIDN-1352-1353 forced-trace-failure sentinel",
                thrown.Message);

            // The decisive regression signal: did scope.Dispose's
            // Realize() re-execute any lazy LayerNorm callback?
            int eagerCountAfterDispose = spy.EagerInvocationCount;
            _output.WriteLine(
                $"eager LayerNorm calls: at throw={eagerCountAtThrow}, " +
                $"post-Dispose={eagerCountAfterDispose}, " +
                $"delta={eagerCountAfterDispose - eagerCountAtThrow}");
            Assert.Equal(eagerCountAtThrow, eagerCountAfterDispose);
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }

    /// <summary>
    /// <see cref="CpuEngine"/> subclass that counts every invocation of
    /// the LayerNorm entry point. Each user-visible LayerNorm under
    /// GraphMode currently produces three spy hits (the outer dispatch,
    /// the GraphMode-branch recursive eager call after scope is nulled,
    /// and the trace-time auto-materialization triggered by
    /// <c>AsWritableSpan</c>), so two chained LayerNorms produce six
    /// hits at the throw point. The exact factor is implementation-
    /// dependent and not what the test asserts on — the assertion is
    /// on the DELTA across scope.Dispose, which must remain zero.
    /// </summary>
    private sealed class LayerNormSpyEngine : CpuEngine
    {
        private int _eagerCount;
        public int EagerInvocationCount => Volatile.Read(ref _eagerCount);

        public override Tensor<T> LayerNorm<T>(
            Tensor<T> input,
            Tensor<T> gamma,
            Tensor<T> beta,
            double epsilon,
            out Tensor<T> mean,
            out Tensor<T> variance)
        {
            Interlocked.Increment(ref _eagerCount);
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    private static Tensor<float> MakeInput(int batch, int features)
    {
        var t = new Tensor<float>(new[] { batch, features });
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < features; f++)
                t[b, f] = (b * 13 + f * 7 + 1) * 0.1f;
        return t;
    }

    private static Tensor<float> MakeGamma(int features)
    {
        var t = new Tensor<float>(new[] { features });
        for (int f = 0; f < features; f++) t[f] = 1.0f;
        return t;
    }

    private static Tensor<float> MakeBeta(int features)
    {
        var t = new Tensor<float>(new[] { features });
        for (int f = 0; f < features; f++) t[f] = 0.0f;
        return t;
    }
}
