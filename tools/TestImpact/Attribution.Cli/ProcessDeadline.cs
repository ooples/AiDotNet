using System.Diagnostics;

// One deadline covers redirected I/O as well as process exit. A child may stall
// either before EOF or after closing stdout; both must release the caller.
internal static class ProcessDeadline
{
    internal static async Task<T> Run<T>(Process process, TimeSpan timeout, Func<CancellationToken, Task<T>> read)
    {
        using var deadline = new CancellationTokenSource(timeout);
        try
        {
            T result = await read(deadline.Token);
            await process.WaitForExitAsync(deadline.Token);
            return result;
        }
        catch
        {
            try { process.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) when (process.HasExited) { }
            await process.WaitForExitAsync();
            throw;
        }
    }
}
