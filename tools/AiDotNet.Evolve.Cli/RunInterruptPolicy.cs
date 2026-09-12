using AiDotNet.Evolution;

namespace AiDotNet.Evolve.Cli;

/// <summary>Separates a checkpoint-safe stop request from cancellation and process termination.</summary>
internal sealed class RunInterruptPolicy(EvolutionRunControl? control, CancellationTokenSource cancellation, TextWriter error)
{
    private int _interrupts;

    /// <returns>Whether the console should suppress operating-system termination.</returns>
    internal bool Interrupt()
    {
        int interrupt = Interlocked.Increment(ref _interrupts);
        if (control is not null && interrupt == 1)
        {
            control.RequestStop();
            error.WriteLine("Graceful stop requested; draining the current batch. Checkpoint durability is not yet confirmed. Press again to cancel.");
            return true;
        }
        if (interrupt == (control is null ? 1 : 2))
        {
            cancellation.Cancel();
            error.WriteLine("Cancellation requested; in-flight work may be rolled back. Press again to terminate the process.");
            return true;
        }
        return false;
    }
}
