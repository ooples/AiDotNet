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
///
/// Calls are direct (not reflection-based) — ``AiDotNet`` marks this
/// assembly as <see cref="System.Runtime.CompilerServices.InternalsVisibleToAttribute"/>
/// so ``RobustFileOps``'s internal methods are compile-time visible.
/// That catches rename / signature regressions at build time instead of
/// runtime.
/// </summary>
public class RobustFileOpsMoveRetryTests
{

    [Fact]
    public async Task Move_Succeeds_WhenNoContention()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_move_{Guid.NewGuid()}.bin");
        string dst = Path.Combine(Path.GetTempPath(), $"aidotnet_move_dst_{Guid.NewGuid()}.bin");
        File.WriteAllBytes(src, [1, 2, 3, 4]);
        try
        {
            await RobustFileOps.MoveWithRetryAsync(src, dst, CancellationToken.None, maxAttempts: 5, initialDelayMs: 1);
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

        // Acquire an exclusive handle on the source, synchronize with the
        // main test so the retry path is actually exercised (otherwise
        // Task.Run scheduling could let the main thread call File.Move
        // before the lock exists and the test passes trivially without
        // ever triggering retry), hold the handle for ~250 ms, then
        // release so the retry loop eventually succeeds. Retry delay is
        // scaled up to 100 ms per attempt here to keep the race window
        // deterministic; production default is 200 ms.
        var lockAcquired = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var lockRelease = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        _ = Task.Run(async () =>
        {
            try
            {
                using var holder = new FileStream(src, FileMode.Open, FileAccess.Read, FileShare.None);
                lockAcquired.TrySetResult(true);
                await Task.Delay(TimeSpan.FromMilliseconds(250));
            }
            catch (Exception ex)
            {
                // If the FileStream open itself fails, release both
                // synchronization primitives so the main test fails fast
                // with a clear error rather than hanging on the wait.
                lockAcquired.TrySetException(ex);
            }
            finally
            {
                lockRelease.TrySetResult(true);
            }
        });

        // Wait for the background task to actually acquire the exclusive
        // handle before we start the move. Without this await the test
        // is a race between Task.Run scheduling and the main thread, and
        // the retry path would not be exercised on the "fast" side of
        // the race — a silently-passing test.
        await lockAcquired.Task;

        try
        {
            await RobustFileOps.MoveWithRetryAsync(src, dst, CancellationToken.None, maxAttempts: 8, initialDelayMs: 100);
            Assert.True(File.Exists(dst), "Destination should exist after retry-backed move");
            Assert.False(File.Exists(src), "Source temp should have been consumed by the move");
            Assert.Equal(new byte[] { 9, 9, 9 }, File.ReadAllBytes(dst));
        }
        finally
        {
            // Drain the background task before teardown regardless of the
            // assertion outcome. If we didn't await here and an assertion
            // above threw, the background task could still own the
            // FileStream when File.Delete(src) runs, masking the real
            // failure with a confusing "file in use" IOException.
            await lockRelease.Task;
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
            RobustFileOps.MoveWithRetry(src, dst, maxAttempts: 5, initialDelayMs: 1);
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
    /// Atomic replace sibling. Used by BTreeIndex.Flush to replace an
    /// existing index file; must tolerate the same AV / indexer lock that
    /// <see cref="RobustFileOps.MoveWithRetry"/> does and preserve the
    /// optional backup file.
    /// </summary>
    [Fact]
    public void ReplaceSync_Succeeds_WhenNoContention()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_replacesync_{Guid.NewGuid()}.bin");
        string dst = Path.Combine(Path.GetTempPath(), $"aidotnet_replacesync_dst_{Guid.NewGuid()}.bin");
        string bak = dst + ".bak";
        File.WriteAllBytes(src, [5, 5, 5]);
        File.WriteAllBytes(dst, [1, 2, 3]);  // destination must exist for File.Replace
        try
        {
            RobustFileOps.ReplaceWithRetry(src, dst, bak, maxAttempts: 5, initialDelayMs: 1);
            Assert.True(File.Exists(dst), "Destination should exist after replace");
            Assert.False(File.Exists(src), "Source temp should have been consumed");
            Assert.Equal(new byte[] { 5, 5, 5 }, File.ReadAllBytes(dst));
            Assert.True(File.Exists(bak), "Backup should contain the previous destination contents");
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(bak));
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
            if (File.Exists(bak)) File.Delete(bak);
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
                () => RobustFileOps.MoveWithRetryAsync(src, dst, CancellationToken.None, maxAttempts: 3, initialDelayMs: 1));
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
