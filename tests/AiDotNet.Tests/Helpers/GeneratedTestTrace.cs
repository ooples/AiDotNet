namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Emits an opt-in construction marker for source-generated tests on memory-sensitive CI shards.
/// </summary>
public static class GeneratedTestTrace
{
    /// <summary>Records the generated test class before its model/component factory can allocate.</summary>
    public static void Record(Type testType)
    {
        string? tracePath = Environment.GetEnvironmentVariable("AIDOTNET_TEST_TRACE_FILE");
        if (string.IsNullOrWhiteSpace(tracePath))
            return;

        try
        {
            string? traceDirectory = Path.GetDirectoryName(tracePath);
            if (!string.IsNullOrEmpty(traceDirectory))
                Directory.CreateDirectory(traceDirectory);

            File.AppendAllText(
                tracePath,
                $"{DateTime.UtcNow:O} [test-start] {testType.FullName}{Environment.NewLine}");
        }
        catch (IOException)
        {
            // Diagnostics must never change a test outcome.
        }
        catch (UnauthorizedAccessException)
        {
            // Diagnostics must never change a test outcome.
        }
    }
}
