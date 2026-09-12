using AiDotNet.Evolution;

namespace AiDotNet.Evolve.Cli;

/// <summary>The process entry point: supplies the real console and the real interrupt handling, nothing else.</summary>
internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        using var cancellation = new CancellationTokenSource();
        var control = args.Length > 0 && string.Equals(args[0], "run", StringComparison.OrdinalIgnoreCase)
            ? new EvolutionRunControl() : null;
        var interrupts = new RunInterruptPolicy(control, cancellation, Console.Error);
        ConsoleCancelEventHandler handler = (_, eventArgs) => eventArgs.Cancel = interrupts.Interrupt();
        Console.CancelKeyPress += handler;
        try
        {
            return await EvolveCommandLine.ExecuteAsync(args, Console.Out, Console.Error, cancellation.Token, control)
                .ConfigureAwait(false);
        }
        finally { Console.CancelKeyPress -= handler; }
    }
}
