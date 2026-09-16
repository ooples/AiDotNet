namespace AiDotNet.TestImpact.Xunit;

internal enum RuntimeInitializationStatus { Missing, Recorded, Conflicting }
internal sealed record RuntimeInitializationBinding(RuntimeInitializationStatus Status, RuntimeEnvironmentBinding? Inputs);

// Opt-in instrumentation only. Record BEFORE the initializer normalizes env
// variables: reading only its final environment loses the original inputs.
public static class RuntimeContractInitialization
{
    private static readonly RuntimeInitializationLedger Cpu = new();

    public static void RecordCpuStartup() => Cpu.Record(RuntimeContractEnvironment.Capture());

    internal static RuntimeInitializationBinding CpuStartup => Cpu.Snapshot;
}

internal sealed class RuntimeInitializationLedger
{
    private readonly object gate = new();
    private RuntimeInitializationBinding binding = new(RuntimeInitializationStatus.Missing, null);

    internal void Record(RuntimeEnvironmentBinding inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        lock (gate)
        {
            if (binding.Status == RuntimeInitializationStatus.Missing)
                binding = new(RuntimeInitializationStatus.Recorded, inputs);
            else if (binding.Inputs != inputs)
                binding = binding with { Status = RuntimeInitializationStatus.Conflicting };
        }
    }

    internal RuntimeInitializationBinding Snapshot { get { lock (gate) return binding; } }
}
