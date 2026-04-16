using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for <see cref="CompiledModelHost{T}"/> — covers invalidation on
/// structure-version bump, eager-path fallback when compilation is disabled or
/// when the host is disposed, thread-safe behavior under concurrent access,
/// and narrow-catch of non-fatal exceptions.
/// </summary>
public class CompiledModelHostTests
{
    private static Tensor<float> MakeInput(int[] shape)
    {
        int len = 1;
        foreach (var d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.1f;
        return new Tensor<float>(data, shape);
    }

    /// <summary>
    /// When <see cref="TensorCodecOptions.EnableCompilation"/> is false, Predict must
    /// invoke the eager lambda and return its output without ever allocating a compile
    /// cache — the eager path is the safety valve users depend on when compilation
    /// is disabled.
    /// </summary>
    [Fact]
    public void Predict_EagerFallback_WhenCompilationDisabled()
    {
        using var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        var expected = MakeInput(new[] { 2, 3 });
        int eagerCalls = 0;

        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            var result1 = host.Predict(input, structureVersion: 0, eagerForward: () => { eagerCalls++; return expected; });
            var result2 = host.Predict(input, structureVersion: 0, eagerForward: () => { eagerCalls++; return expected; });

            Assert.Same(expected, result1);
            Assert.Same(expected, result2);
            Assert.Equal(2, eagerCalls); // Each call runs the eager lambda — no compile.
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// After Dispose, Predict must short-circuit to the eager path — the compile
    /// cache is torn down and any attempt to use it would throw. Verifies the
    /// contract documented on <see cref="CompiledModelHost{T}.Dispose"/>.
    /// </summary>
    [Fact]
    public void Predict_EagerFallback_AfterDispose()
    {
        var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        var expected = MakeInput(new[] { 2, 3 });

        host.Dispose();

        // Compilation enabled but host disposed — must still run eager without throwing.
        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            int eagerCalls = 0;
            var result = host.Predict(input, structureVersion: 0, eagerForward: () => { eagerCalls++; return expected; });
            Assert.Same(expected, result);
            Assert.Equal(1, eagerCalls);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// Dispose must be idempotent — calling it twice must not throw. Services that
    /// wire Dispose into both explicit shutdown and finalizer paths depend on this.
    /// </summary>
    [Fact]
    public void Dispose_IsIdempotent()
    {
        var host = new CompiledModelHost<float>();
        host.Dispose();
        host.Dispose(); // Must not throw.
    }

    /// <summary>
    /// When the caller bumps <c>structureVersion</c> between Predict calls, the host
    /// must drop any cached plans compiled at the old version. We can observe this
    /// indirectly: the eager lambda is re-invoked on the version bump (for re-tracing)
    /// even though the input shape is unchanged.
    /// </summary>
    [Fact]
    public void Predict_InvalidatesCache_OnStructureVersionBump()
    {
        using var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        var output = MakeInput(new[] { 2, 3 });
        int eagerCalls = 0;

        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            host.Predict(input, structureVersion: 0, eagerForward: () => { eagerCalls++; return output; });
            int callsAfterV0 = eagerCalls;

            // Bump the version — must force cache invalidation and re-trace.
            host.Predict(input, structureVersion: 1, eagerForward: () => { eagerCalls++; return output; });
            int callsAfterV1 = eagerCalls;

            // Version bump must cause at least one additional eager-lambda call
            // (the re-trace). Strict monotonic bump from the cache rebuild.
            Assert.True(callsAfterV1 > callsAfterV0,
                $"Version bump should force re-trace but eagerCalls stayed at {callsAfterV0}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// Explicit Invalidate() must drop cached plans so the NEXT Predict triggers a
    /// re-trace (observed via the eager lambda being invoked again).
    /// </summary>
    [Fact]
    public void Invalidate_ForcesRecompileOnNextPredict()
    {
        using var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        var output = MakeInput(new[] { 2, 3 });
        int eagerCalls = 0;

        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            host.Predict(input, structureVersion: 0, eagerForward: () => { eagerCalls++; return output; });
            int callsBefore = eagerCalls;

            host.Invalidate();

            host.Predict(input, structureVersion: 0, eagerForward: () => { eagerCalls++; return output; });
            int callsAfter = eagerCalls;

            Assert.True(callsAfter > callsBefore,
                $"Invalidate() should force re-trace but eagerCalls stayed at {callsBefore}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// When the eager lambda throws a recoverable exception during tracing, the host
    /// must propagate the exception (there's no cached plan to fall back to, and
    /// swallowing would hide user-visible bugs in the forward pass). This exercises
    /// the catch-exception-filter path: recoverable exceptions bubble up so callers
    /// see the root cause.
    /// </summary>
    [Fact]
    public void Predict_PropagatesEagerException_OnFirstTrace()
    {
        using var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });

        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            Assert.ThrowsAny<Exception>(() =>
                host.Predict(input, structureVersion: 0, eagerForward: () =>
                    throw new InvalidOperationException("forward-pass bug")));
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// Concurrent Predict + Invalidate must not crash, corrupt the cache, or
    /// produce wrong outputs. The host synchronizes lifecycle mutations under
    /// <c>_sync</c>; this test stress-tests that guarantee under contention.
    /// </summary>
    [Fact]
    public async Task ConcurrentPredictAndInvalidate_DoesNotCrash()
    {
        using var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        var output = MakeInput(new[] { 2, 3 });

        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            int errors = 0;

            var predictTask = Task.Run(() =>
            {
                int version = 0;
                while (!cts.IsCancellationRequested)
                {
                    try
                    {
                        var result = host.Predict(
                            input,
                            structureVersion: Interlocked.Increment(ref version) / 100,
                            eagerForward: () => output);
                        Assert.NotNull(result);
                    }
                    catch (ObjectDisposedException)
                    {
                        // Host disposed by another thread — acceptable during
                        // concurrent dispose scenarios in future tests.
                    }
                    catch
                    {
                        Interlocked.Increment(ref errors);
                    }
                }
            });

            var invalidateTask = Task.Run(() =>
            {
                while (!cts.IsCancellationRequested)
                {
                    try { host.Invalidate(); }
                    catch { Interlocked.Increment(ref errors); }
                }
            });

            await Task.WhenAll(predictTask, invalidateTask);

            Assert.Equal(0, errors);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// Concurrent Predict + Dispose must terminate cleanly. Predict calls occurring
    /// after Dispose must fall back to eager (not throw), per the documented contract.
    /// </summary>
    [Fact]
    public async Task ConcurrentPredictAndDispose_FallsBackToEagerAfterDispose()
    {
        var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        var output = MakeInput(new[] { 2, 3 });

        var previousOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));
            int errors = 0;
            int successfulPredicts = 0;

            var predictTask = Task.Run(() =>
            {
                while (!cts.IsCancellationRequested)
                {
                    try
                    {
                        host.Predict(input, structureVersion: 0, eagerForward: () => output);
                        Interlocked.Increment(ref successfulPredicts);
                    }
                    catch
                    {
                        Interlocked.Increment(ref errors);
                    }
                }
            });

            // Let some Predicts run first, then dispose mid-flight.
            await Task.Delay(100);
            host.Dispose();

            await predictTask;

            // After dispose, predict must still work via eager fallback (no errors).
            Assert.Equal(0, errors);
            Assert.True(successfulPredicts > 0);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    /// <summary>
    /// Null eagerForward must throw ArgumentNullException immediately — no silent
    /// deferral to the compile path where the error would be obscured.
    /// </summary>
    [Fact]
    public void Predict_ThrowsOnNullEagerForward()
    {
        using var host = new CompiledModelHost<float>();
        var input = MakeInput(new[] { 2, 3 });
        Assert.Throws<ArgumentNullException>(() =>
            host.Predict(input, structureVersion: 0, eagerForward: null!));
    }
}
