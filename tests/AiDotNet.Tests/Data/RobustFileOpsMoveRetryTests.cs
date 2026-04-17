using System.Reflection;
using AiDotNet.Data;
using Xunit;

namespace AiDotNetTests.UnitTests.Data;

/// <summary>
/// Tests for <see cref="RobustFileOps"/>.MoveWithRetryAsync / MoveWithRetry.
///
/// Regression: on Windows the freshly-written temp file can be briefly
/// locked by Windows Defender or the search indexer while a download or
/// atomic save calls File.Move; the original implementation had no retry,
/// so a single transient IOException would abort the operation and
/// typically wipe the temp file in a <c>finally</c> block. RobustFileOps
/// retries on IOException / UnauthorizedAccessException with linear
/// backoff before propagating. Used by DatasetDownloader, WeightDownloader,
/// HuggingFaceModelLoader, OnnxModelDownloader, M4DatasetLoader, and the
/// BTreeIndex atomic flush path.
/// </summary>
public class RobustFileOpsMoveRetryTests
{
    private static readonly MethodInfo MoveWithRetryAsyncMethod =
        typeof(RobustFileOps).GetMethod(
            "MoveWithRetryAsync",
            BindingFlags.Static | BindingFlags.NonPublic)
        ?? throw new InvalidOperationException("MoveWithRetryAsync not found; helper may have been reverted.");

    private static readonly MethodInfo MoveWithRetrySyncMethod =
        typeof(RobustFileOps).GetMethod(
            "MoveWithRetry",
            BindingFlags.Static | BindingFlags.NonPublic)
        ?? throw new InvalidOperationException("MoveWithRetry (sync) not found; helper may have been reverted.");

    private static Task InvokeMoveAsync(string src, string dst, int maxAttempts = 5, int delayMs = 1)
        => (Task)MoveWithRetryAsyncMethod.Invoke(
            null,
            new object[] { src, dst, CancellationToken.None, maxAttempts, delayMs })!;

    private static void InvokeMoveSync(string src, string dst, int maxAttempts = 5, int delayMs = 1)
        => MoveWithRetrySyncMethod.Invoke(
            null,
            new object[] { src, dst, maxAttempts, delayMs });

    [Fact]
    public async Task Move_Succeeds_WhenNoContention()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_move_{Guid.NewGuid()}.bin");
        string dst = Path.Combine(Path.GetTempPath(), $"aidotnet_move_dst_{Guid.NewGuid()}.bin");
        File.WriteAllBytes(src, [1, 2, 3, 4]);
        try
        {
            await InvokeMoveAsync(src, dst);
            Assert.True(File.Exists(dst));
            Assert.False(File.Exists(src));
            Assert.Equal(new byte[] { 1, 2, 3, 4 }, File.ReadAllBytes(dst));
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
        }
    }

    /// <summary>
    /// Simulates the Windows antivirus / indexer lock pattern: open the source
    /// file with exclusive sharing for a short window, then release it.
    /// The retry loop must tolerate the transient IOException and succeed
    /// once the lock is released.
    /// </summary>
    [Fact]
    public async Task Move_SucceedsAfter_TransientSharingViolation()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_move_{Guid.NewGuid()}.bin");
        string dst = Path.Combine(Path.GetTempPath(), $"aidotnet_move_dst_{Guid.NewGuid()}.bin");
        File.WriteAllBytes(src, [9, 9, 9]);

        // Hold an exclusive handle for ~250 ms then release, letting the
        // retry loop (linear 1ms * attempt backoff, max 5 attempts) get
        // through. The retry delay is scaled up to 100 ms per attempt here
        // so the test is not racy; the production default is 200 ms.
        var lockRelease = new TaskCompletionSource<bool>();
        _ = Task.Run(async () =>
        {
            try
            {
                using var holder = new FileStream(src, FileMode.Open, FileAccess.Read, FileShare.None);
                await Task.Delay(TimeSpan.FromMilliseconds(250));
            }
            finally
            {
                lockRelease.TrySetResult(true);
            }
        });

        try
        {
            await InvokeMoveAsync(src, dst, maxAttempts: 8, delayMs: 100);
            Assert.True(File.Exists(dst), "Destination should exist after retry-backed move");
            Assert.Equal(new byte[] { 9, 9, 9 }, File.ReadAllBytes(dst));
            await lockRelease.Task;  // drain background task for clean teardown
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
        }
    }

    /// <summary>
    /// Sync variant: used by BTreeIndex.Flush and any other sync save path
    /// that cannot await. Mirrors Move_Succeeds_WhenNoContention for the
    /// synchronous entry point.
    /// </summary>
    [Fact]
    public void MoveSync_Succeeds_WhenNoContention()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_movesync_{Guid.NewGuid()}.bin");
        string dst = Path.Combine(Path.GetTempPath(), $"aidotnet_movesync_dst_{Guid.NewGuid()}.bin");
        File.WriteAllBytes(src, [7, 7, 7]);
        try
        {
            InvokeMoveSync(src, dst);
            Assert.True(File.Exists(dst));
            Assert.False(File.Exists(src));
            Assert.Equal(new byte[] { 7, 7, 7 }, File.ReadAllBytes(dst));
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
        }
    }

    /// <summary>
    /// If every retry attempt fails, the final attempt must propagate the
    /// underlying exception rather than silently succeed or hang.
    /// </summary>
    [Fact]
    public async Task Move_Propagates_WhenLockNeverReleases()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_move_{Guid.NewGuid()}.bin");
        string dst = Path.Combine(Path.GetTempPath(), $"aidotnet_move_dst_{Guid.NewGuid()}.bin");
        File.WriteAllBytes(src, [1]);

        // Hold an exclusive read handle for the duration of the call — all
        // attempts will fail with IOException and the final one should
        // surface.
        using var holder = new FileStream(src, FileMode.Open, FileAccess.Read, FileShare.None);
        try
        {
            await Assert.ThrowsAsync<IOException>(
                () => InvokeMoveAsync(src, dst, maxAttempts: 3, delayMs: 1));
            Assert.False(File.Exists(dst), "Destination must not exist when the move fails");
        }
        finally
        {
            holder.Dispose();
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
        }
    }
}
