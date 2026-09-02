namespace AiDotNet.Enums;

/// <summary>Selects where a candidate program is executed when a program-evolution run scores it.</summary>
/// <remarks>
/// <para>
/// Generated program text is untrusted input, so the choice of execution boundary is a security decision rather
/// than a performance one. This enumeration names the three boundaries AiDotNet supports and deliberately makes the
/// safe one the default: <see cref="OutOfProcessWorker"/> launches an isolated child process with a wall-clock
/// timeout, byte caps on captured output, a scrubbed environment, and an operating-system memory cap, and it never
/// loads candidate code into the hosting application.
/// </para>
/// <para>
/// <see cref="InProcessUnsafe"/> exists only so that a caller who genuinely wants it must say so twice: selecting it
/// is rejected by <see cref="AiDotNet.Configuration.ProgramSandboxOptions.Validate"/> unless
/// <see cref="AiDotNet.Configuration.ProgramSandboxOptions.AllowUnsafeInProcessExecution"/> is also set to
/// <c>true</c>. No engine shipped in this library implements that mode; it is a marker for hosts that supply their
/// own in-process runner and have accepted the consequences.
/// </para>
/// <para><b>For Beginners:</b> When evolution writes a program, something has to run it to find out whether it
/// works — and that program may do anything at all, because a language model wrote it. This setting says where it
/// runs. The default starts a separate, tightly limited process, which is the option you want: if the generated
/// code hangs, allocates endlessly, or crashes, only that child process is affected. The serving option hands the
/// job to a container-backed service instead. The last option is a warning label, not a feature.</para>
/// </remarks>
public enum ProgramSandboxMode
{
    /// <summary>Runs each candidate in an isolated child process with time, memory, output, and environment limits.</summary>
    /// <remarks>
    /// This is the default and the mode implemented by
    /// <see cref="AiDotNet.ProgramSynthesis.Execution.ProcessProgramExecutionEngine"/>. It needs no container
    /// runtime and no network access, only an interpreter or compiler already installed on the machine.
    /// </remarks>
    OutOfProcessWorker = 0,

    /// <summary>Delegates execution to an AiDotNet.Serving instance, which sandboxes the program in a container.</summary>
    /// <remarks>
    /// This is the strongest boundary because the program runs with no network, a read-only mount, and container
    /// CPU and memory limits, but it requires a reachable serving deployment. Pair it with
    /// <see cref="AiDotNet.ProgramSynthesis.Serving.ServingProgramExecutionEngine"/>.
    /// </remarks>
    Serving = 1,

    /// <summary>Runs candidate code inside the hosting process, with no isolation whatsoever.</summary>
    /// <remarks>
    /// Selecting this mode throws from <see cref="AiDotNet.Configuration.ProgramSandboxOptions.Validate"/> unless
    /// <see cref="AiDotNet.Configuration.ProgramSandboxOptions.AllowUnsafeInProcessExecution"/> is <c>true</c>.
    /// A program that runs in the host process can read every secret the host holds, so treat this as equivalent to
    /// executing an unreviewed script downloaded from the internet.
    /// </remarks>
    InProcessUnsafe = 2
}
