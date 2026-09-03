namespace AiDotNet.Evolve.Cli;

/// <summary>The process entry point: supplies the real console and the real interrupt handling, nothing else.</summary>
internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            // A search can hold hours of work, so the first interrupt asks it to stop and write its checkpoint rather
            // than killing the process. A second interrupt is left to the operating system.
            eventArgs.Cancel = !cancellation.IsCancellationRequested;
            cancellation.Cancel();
            Console.Error.WriteLine("Stopping after the current batch; press again to abandon the run.");
        };

        return await EvolveCommandLine
            .ExecuteAsync(args, Console.Out, Console.Error, cancellation.Token)
            .ConfigureAwait(false);
    }
}
