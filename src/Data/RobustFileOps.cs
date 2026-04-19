namespace AiDotNet.Data;

/// <summary>
/// Filesystem helpers that tolerate transient locks — particularly the
/// "sharing violation" window that Windows Defender or the search indexer
/// can hold on a freshly written file for a few hundred milliseconds after
/// the writer's handle is released.
/// </summary>
/// <remarks>
/// <para>
/// Every download path in this library writes to a temp file and then
/// renames to the final location. On Windows, an antivirus scanner or the
/// search indexer can hold a secondary handle on the temp file briefly
/// after the writer closes it — long enough that the immediate
/// <c>File.Move</c> fails with <see cref="IOException"/>. Without retry
/// logic, a single transient lock aborts the entire download; the caller
/// then typically wipes the temp file in a <c>finally</c> block and must
/// re-download from scratch, which will race again the next time.
/// </para>
/// <para>
/// <see cref="MoveWithRetryAsync"/> retries on <see cref="IOException"/>
/// and <see cref="UnauthorizedAccessException"/> with linear backoff. The
/// final attempt still propagates the real exception so callers see a
/// clear failure rather than a silent no-op.
/// </para>
/// </remarks>
internal static class RobustFileOps
{
    /// <summary>
    /// Moves <paramref name="sourcePath"/> to <paramref name="destinationPath"/>,
    /// tolerating transient Windows-style sharing violations.
    /// </summary>
    /// <param name="sourcePath">The source file path (typically a temp file).</param>
    /// <param name="destinationPath">The destination path (final location).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <param name="maxAttempts">Maximum total attempts (default 5).</param>
    /// <param name="initialDelayMs">Initial delay between retries; each subsequent
    /// retry waits <c>attempt * initialDelayMs</c> milliseconds (default 200 ms,
    /// so default schedule is 200 / 400 / 600 / 800 ms).</param>
    /// <exception cref="IOException">Final attempt failed with an IO error.</exception>
    /// <exception cref="UnauthorizedAccessException">Final attempt failed with an access error.</exception>
    /// <remarks>
    /// <para>
    /// The retry tolerates <see cref="IOException"/> (e.g. sharing violation on
    /// Windows) and <see cref="UnauthorizedAccessException"/> (e.g. ACL
    /// contention mid-move). Other exceptions are surfaced immediately.
    /// </para>
    /// </remarks>
    internal static async Task MoveWithRetryAsync(
        string sourcePath,
        string destinationPath,
        CancellationToken cancellationToken,
        int maxAttempts = 5,
        int initialDelayMs = 200)
    {
        ValidateRetryArguments(maxAttempts, initialDelayMs);

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                File.Move(sourcePath, destinationPath);
                return;
            }
            // On the final attempt the `when` clause is false, so the
            // catch is skipped and the original exception propagates to
            // the caller — exactly the intended "clear failure" behavior.
            catch (IOException) when (attempt < maxAttempts)
            {
                await Task.Delay(TimeSpan.FromMilliseconds(initialDelayMs * attempt), cancellationToken);
            }
            catch (UnauthorizedAccessException) when (attempt < maxAttempts)
            {
                await Task.Delay(TimeSpan.FromMilliseconds(initialDelayMs * attempt), cancellationToken);
            }
        }
    }

    /// <summary>
    /// Synchronous sibling of <see cref="MoveWithRetryAsync"/>. For call sites
    /// that can't go async (e.g. atomic index flushes from a sync save path).
    /// Uses <see cref="Thread.Sleep"/> between attempts; prefer the async
    /// variant when an awaitable context is available.
    /// </summary>
    internal static void MoveWithRetry(
        string sourcePath,
        string destinationPath,
        int maxAttempts = 5,
        int initialDelayMs = 200)
    {
        ValidateRetryArguments(maxAttempts, initialDelayMs);

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                File.Move(sourcePath, destinationPath);
                return;
            }
            // Final attempt falls through `when (attempt < maxAttempts)`,
            // so the underlying exception naturally propagates.
            catch (IOException) when (attempt < maxAttempts)
            {
                Thread.Sleep(initialDelayMs * attempt);
            }
            catch (UnauthorizedAccessException) when (attempt < maxAttempts)
            {
                Thread.Sleep(initialDelayMs * attempt);
            }
        }
    }

    /// <summary>
    /// Atomically replaces <paramref name="destinationPath"/> with
    /// <paramref name="sourcePath"/>, optionally keeping a backup at
    /// <paramref name="destinationBackupPath"/>. Retries on transient
    /// <see cref="IOException"/> / <see cref="UnauthorizedAccessException"/>
    /// with linear backoff, mirroring <see cref="MoveWithRetry"/>.
    /// Synchronous; used by atomic index / checkpoint flush paths that are
    /// not async.
    /// </summary>
    /// <remarks>
    /// On Windows <see cref="File.Replace(string, string, string)"/> is the
    /// in-place atomic rename, but it can still fail with a sharing
    /// violation if Windows Defender or the search indexer is mid-scan of
    /// either file.
    /// </remarks>
    internal static void ReplaceWithRetry(
        string sourcePath,
        string destinationPath,
        string? destinationBackupPath,
        int maxAttempts = 5,
        int initialDelayMs = 200)
    {
        ValidateRetryArguments(maxAttempts, initialDelayMs);

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                File.Replace(sourcePath, destinationPath, destinationBackupPath);
                return;
            }
            // Final attempt: the `when` gate is false so the exception
            // propagates directly to the caller.
            catch (IOException) when (attempt < maxAttempts)
            {
                Thread.Sleep(initialDelayMs * attempt);
            }
            catch (UnauthorizedAccessException) when (attempt < maxAttempts)
            {
                Thread.Sleep(initialDelayMs * attempt);
            }
        }
    }

    /// <summary>
    /// Validates retry parameters. Rejects silent-success configurations —
    /// <c>maxAttempts &lt; 1</c> would exit the retry loop without ever
    /// touching the filesystem and without throwing, which is a confusing
    /// failure mode for any caller that forwards user-supplied config.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxAttempts"/>
    /// is less than 1, or <paramref name="initialDelayMs"/> is negative.</exception>
    private static void ValidateRetryArguments(int maxAttempts, int initialDelayMs)
    {
        if (maxAttempts < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxAttempts),
                maxAttempts,
                "maxAttempts must be at least 1.");
        }

        if (initialDelayMs < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(initialDelayMs),
                initialDelayMs,
                "initialDelayMs must be non-negative.");
        }
    }
}
