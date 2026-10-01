using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tests.TestInfrastructure;

/// <summary>
/// Engine resets for tests that share one process-wide <see cref="AiDotNetEngine.Current"/>.
/// </summary>
internal static class TestEngines
{
    /// <summary>
    /// Puts the process on a CPU engine, keeping the current one when it already is a plain <see cref="CpuEngine"/>.
    /// </summary>
    /// <remarks>
    /// <see cref="AiDotNetEngine.ResetToCpu"/> installs a NEW <see cref="CpuEngine"/> on every call. Classes run in
    /// parallel and every model reads the engine on each operation, so a reset in one class swapped the engine under a
    /// model training in another, and that step's update was lost (FusedOptimizerParityTests' LAMB parity and
    /// FTTransformerClassifierTests' training tests failed only beside other classes, and passed alone). On a CPU host
    /// the reset had nothing to undo.
    /// </remarks>
    public static void EnsureCpu()
    {
        if (AiDotNetEngine.Current.GetType() != typeof(CpuEngine))
            AiDotNetEngine.ResetToCpu();
    }
}
