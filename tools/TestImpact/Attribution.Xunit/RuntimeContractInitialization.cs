namespace AiDotNet.TestImpact.Xunit;

// Opt-in instrumentation only. Record BEFORE the initializer normalizes env
// variables: reading only its final environment loses the original inputs.
public static class RuntimeContractInitialization
{
    private static readonly RuntimeInitializationLedger Cpu = new();

    public static void RecordCpuStartup() => Cpu.Record(RuntimeContractEnvironment.Capture());
    public static void RecordCpuResetInput(RuntimeCpuResetInput input) => Cpu.RecordReset(input);
    public static void RecordCpuCompletion(bool cpuActive, int maxDegreeOfParallelism) =>
        Cpu.Complete(new(cpuActive ? RuntimeCpuMode.Cpu : RuntimeCpuMode.Other, maxDegreeOfParallelism));

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

    internal void RecordReset(RuntimeCpuResetInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        lock (gate)
        {
            if (binding.Status != RuntimeInitializationStatus.Recorded || binding.ResetInput is not null || binding.Completion is not null ||
                !Enum.IsDefined(input.Mode) || !Enum.IsDefined(input.Logging))
                binding = binding with { Status = RuntimeInitializationStatus.Conflicting };
            else binding = binding with { ResetInput = input };
        }
    }

    internal void Complete(RuntimeCpuCompletion completion)
    {
        ArgumentNullException.ThrowIfNull(completion);
        lock (gate)
        {
            if (binding.Status != RuntimeInitializationStatus.Recorded ||
                binding.Completion is not null && binding.Completion != completion)
                binding = binding with { Status = RuntimeInitializationStatus.Conflicting };
            else binding = binding with { Completion = completion };
        }
    }
}
