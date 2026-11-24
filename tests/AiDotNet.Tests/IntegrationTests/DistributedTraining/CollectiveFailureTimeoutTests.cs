using System;
using System.Diagnostics;
using System.Globalization;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// A rank that hangs, dies or never starts surfaces on the other ranks as an exception within the backend's timeout,
/// never as a hang (PyTorch's process-group timeout). Each test bounds its own wait well past the timeout, so a backend
/// that blocks forever fails the test instead of hanging the run.
/// </summary>
[Collection("SequentialMultiProcessGpuDistributed")]
public sealed class CollectiveFailureTimeoutTests
{
    private const int TimeoutMs = 2000;
    // A timeout must fire within its bound, not hang; the slack absorbs a loaded CI machine's scheduling and socket teardown.
    private const int SchedulingSlackMs = 20_000;
    private static readonly TimeSpan Guard = TimeSpan.FromSeconds(30);

    private static (string? Addr, string? Port) UseLoopbackRendezvous()
    {
        var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        int port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        var previous = (Environment.GetEnvironmentVariable("AIDOTNET_MASTER_ADDR"), Environment.GetEnvironmentVariable("AIDOTNET_MASTER_PORT"));
        Environment.SetEnvironmentVariable("AIDOTNET_MASTER_ADDR", "127.0.0.1");
        Environment.SetEnvironmentVariable("AIDOTNET_MASTER_PORT", port.ToString(CultureInfo.InvariantCulture));
        return previous;
    }

    private static void Restore((string? Addr, string? Port) previous)
    {
        Environment.SetEnvironmentVariable("AIDOTNET_MASTER_ADDR", previous.Addr);
        Environment.SetEnvironmentVariable("AIDOTNET_MASTER_PORT", previous.Port);
    }

    // Runs the surviving rank's collective and returns what it threw, failing if it is still blocked after the guard.
    private static Exception Surfaced(Action collective, out TimeSpan took)
    {
        var watch = Stopwatch.StartNew();
        var run = Task.Run(collective);
        bool finished = Task.WhenAny(run, Task.Delay(Guard)).GetAwaiter().GetResult() == run;
        Assert.True(finished, $"the collective was still blocked after {Guard.TotalSeconds} s: a hang, not an error");
        took = watch.Elapsed;
        Assert.True(run.IsFaulted, "the collective completed although its peer never took part");
        return run.Exception!.GetBaseException();
    }

    [Fact(Timeout = 120000)]
    public async Task Gloo_APeerThatHangs_FailsTheCollectiveWithinTheTimeout()
    {
        var previous = UseLoopbackRendezvous();
        var g0 = new GlooCommunicationBackend<float>(0, 2) { TimeoutMilliseconds = TimeoutMs };
        var g1 = new GlooCommunicationBackend<float>(1, 2) { TimeoutMilliseconds = TimeoutMs };
        try
        {
            await Task.WhenAll(Task.Run(() => g0.Initialize()), Task.Run(() => g1.Initialize()));
            // Rank 1 stays connected but never joins the all-reduce.
            var error = Surfaced(() => g0.AllReduce(new Vector<float>(new float[1024]), ReductionOperation.Sum), out var took);
            Assert.True(error is TimeoutException or System.IO.IOException, $"unexpected {error.GetType().Name}: {error.Message}");
            Assert.True(took < TimeSpan.FromMilliseconds(TimeoutMs + SchedulingSlackMs), $"took {took.TotalMilliseconds:F0} ms for a {TimeoutMs} ms timeout");
        }
        finally
        {
            try { g0.Shutdown(); } catch (Exception) { /* already broken */ }
            try { g1.Shutdown(); } catch (Exception) { /* already broken */ }
            Restore(previous);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Gloo_APeerThatDies_FailsTheCollective()
    {
        var previous = UseLoopbackRendezvous();
        var g0 = new GlooCommunicationBackend<float>(0, 2) { TimeoutMilliseconds = TimeoutMs };
        var g1 = new GlooCommunicationBackend<float>(1, 2) { TimeoutMilliseconds = TimeoutMs };
        try
        {
            await Task.WhenAll(Task.Run(() => g0.Initialize()), Task.Run(() => g1.Initialize()));
            g1.Shutdown();   // its sockets close, as a killed process's do
            var error = Surfaced(() => g0.AllReduce(new Vector<float>(new float[1024]), ReductionOperation.Sum), out _);
            Assert.True(error is TimeoutException or System.IO.IOException or System.Net.Sockets.SocketException or InvalidOperationException,
                $"unexpected {error.GetType().Name}: {error.Message}");
        }
        finally
        {
            try { g0.Shutdown(); } catch (Exception) { /* already broken */ }
            Restore(previous);
        }
    }

    [Fact]
    public void Gloo_ARankThatNeverStarts_FailsRendezvous()
    {
        var previous = UseLoopbackRendezvous();
        var g0 = new GlooCommunicationBackend<float>(0, 2) { TimeoutMilliseconds = TimeoutMs };
        try
        {
            var error = Surfaced(() => g0.Initialize(), out var took);
            Assert.IsType<TimeoutException>(error);
            Assert.True(took < TimeSpan.FromMilliseconds(TimeoutMs + SchedulingSlackMs), $"took {took.TotalMilliseconds:F0} ms");
        }
        finally
        {
            try { g0.Shutdown(); } catch (Exception) { /* never initialized */ }
            Restore(previous);
        }
    }

    [Fact]
    public void SharedMemory_APeerThatHangs_FailsTheCollectiveWithinTheTimeout()
    {
        string session = "aidn_timeout_" + Guid.NewGuid().ToString("N");
        var s0 = new SharedMemoryCommunicationBackend<float>(0, 2, session, 1L << 20, TimeoutMs);
        var s1 = new SharedMemoryCommunicationBackend<float>(1, 2, session, 1L << 20, TimeoutMs);
        try
        {
            Task.WaitAll(Task.Run(() => s0.Initialize()), Task.Run(() => s1.Initialize()));
            var error = Surfaced(() => s0.AllReduce(new Vector<float>(new float[1024]), ReductionOperation.Sum), out var took);
            Assert.True(error is TimeoutException or System.IO.IOException, $"unexpected {error.GetType().Name}: {error.Message}");
            Assert.True(took < TimeSpan.FromMilliseconds(TimeoutMs + SchedulingSlackMs), $"took {took.TotalMilliseconds:F0} ms");
        }
        finally
        {
            try { s0.Shutdown(); } catch (Exception) { /* already broken */ }
            try { s1.Shutdown(); } catch (Exception) { /* already broken */ }
        }
    }

    [Fact]
    public void SharedMemory_APeerWhoseHandleCannotBeQueried_IsStillSeenToHaveExited()
    {
        using var peer = Process.Start(new ProcessStartInfo("cmd.exe", "/c exit 0") { UseShellExecute = false, CreateNoWindow = true })!;
        peer.WaitForExit();
        static bool Restricted(Process p) => throw new System.ComponentModel.Win32Exception(5, "Access is denied");
        // A restricted handle used to read as "alive", leaving only the collective timeout to notice the dead rank.
        Assert.True(SharedMemoryCommunicationBackend<float>.HasPeerExited(peer, Restricted));
        using var self = Process.GetCurrentProcess();
        Assert.False(SharedMemoryCommunicationBackend<float>.HasPeerExited(self, Restricted));
    }
}

/// <summary>
/// The multi-process and rendezvous tests run alone: they set the process-wide rendezvous variables
/// (AIDOTNET_MASTER_ADDR, AIDOTNET_MASTER_PORT) and bind loopback ports, which a test running beside them would read.
/// </summary>
[CollectionDefinition("SequentialMultiProcessGpuDistributed", DisableParallelization = true)]
public sealed class SequentialMultiProcessGpuDistributedCollection { }
