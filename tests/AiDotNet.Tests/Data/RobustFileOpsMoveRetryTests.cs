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
    /// Cross-platform retry-trigger: the destination's parent directory
    /// doesn't exist when the move starts, so File.Move throws
    /// DirectoryNotFoundException (a subclass of IOException). A
    /// background task creates the parent ~250 ms later, after which a
    /// retry attempt succeeds. This exercises the same retry path as
    /// the Windows AV / indexer scenario without depending on Windows-
    /// specific FileShare.None semantics — on Linux, opening a
    /// FileStream with FileShare.None does not actually block File.Move,
    /// so the original lock-based simulation passed trivially without
    /// ever exercising retry on the Linux CI runner.
    /// </summary>
    [Fact]
    public async Task Move_SucceedsAfter_TransientMissingParentDirectory()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_move_{Guid.NewGuid()}.bin");
        string parentDir = Path.Combine(Path.GetTempPath(), $"aidotnet_move_parent_{Guid.NewGuid()}");
        string dst = Path.Combine(parentDir, "dst.bin");
        File.WriteAllBytes(src, [9, 9, 9]);

        // Background task creates the parent directory after ~250 ms,
        // letting the retry loop succeed once the directory exists.
        // Retry schedule: 100, 200, 300, 400 ms (initialDelayMs * attempt),
        // so by attempt 3 or 4 the directory will be in place.
        var dirReady = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        _ = Task.Run(async () =>
        {
            try
            {
                await Task.Delay(TimeSpan.FromMilliseconds(250));
                Directory.CreateDirectory(parentDir);
            }
            finally
            {
                dirReady.TrySetResult(true);
            }
        });

        try
        {
            await RobustFileOps.MoveWithRetryAsync(src, dst, CancellationToken.None, maxAttempts: 8, initialDelayMs: 100);
            Assert.True(File.Exists(dst), "Destination should exist after retry-backed move");
            Assert.False(File.Exists(src), "Source temp should have been consumed by the move");
            Assert.Equal(new byte[] { 9, 9, 9 }, File.ReadAllBytes(dst));
        }
        finally
        {
            // Drain the background task before teardown so cleanup is
            // deterministic regardless of which assertion threw.
            await dirReady.Task;
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
            if (Directory.Exists(parentDir)) Directory.Delete(parentDir, recursive: true);
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
    ///
    /// Cross-platform retry-trigger: destination's parent directory never
    /// exists, so every File.Move attempt throws DirectoryNotFoundException
    /// (an IOException subclass). Assert.ThrowsAsync&lt;DirectoryNotFoundException&gt;
    /// verifies the specific cross-platform trigger used by this test.
    /// </summary>
    [Fact]
    public async Task Move_Propagates_WhenParentDirectoryNeverCreated()
    {
        string src = Path.Combine(Path.GetTempPath(), $"aidotnet_move_{Guid.NewGuid()}.bin");
        string parentDir = Path.Combine(Path.GetTempPath(), $"aidotnet_move_parent_{Guid.NewGuid()}");
        string dst = Path.Combine(parentDir, "dst.bin");
        File.WriteAllBytes(src, [1]);

        // Parent directory is never created — every retry attempt fails,
        // and the final attempt must propagate the IOException to the
        // caller (rather than swallow it).
        try
        {
            await Assert.ThrowsAsync<DirectoryNotFoundException>(
                () => RobustFileOps.MoveWithRetryAsync(src, dst, CancellationToken.None, maxAttempts: 3, initialDelayMs: 1));
            Assert.False(File.Exists(dst), "Destination must not exist when the move fails");
            Assert.True(File.Exists(src), "Source must remain in place when the move fails");
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
            if (Directory.Exists(parentDir)) Directory.Delete(parentDir, recursive: true);
        }
    }
}
