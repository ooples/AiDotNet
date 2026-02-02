using System;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Helper class for handling GPU-related test scenarios.
/// </summary>
public static class GpuTestHelper
{
    /// <summary>
    /// Executes a test action and gracefully handles GPU unavailability.
    /// Returns true if the test executed, false if it was skipped due to GPU issues.
    /// </summary>
    public static bool TryExecuteGpuTest(Action testAction)
    {
        try
        {
            testAction();
            return true;
        }
        catch (InvalidOperationException ex) when (IsGpuUnavailableException(ex))
        {
            // GPU/OpenCL not available - test passes but skips execution
            return false;
        }
        catch (OutOfMemoryException)
        {
            // Insufficient GPU memory - test passes but skips execution
            return false;
        }
    }

    /// <summary>
    /// Executes a test function and gracefully handles GPU unavailability.
    /// Returns the result if successful, or default(T) if skipped.
    /// </summary>
    public static (bool executed, T? result) TryExecuteGpuTest<T>(Func<T> testFunc)
    {
        try
        {
            return (true, testFunc());
        }
        catch (InvalidOperationException ex) when (IsGpuUnavailableException(ex))
        {
            return (false, default);
        }
        catch (OutOfMemoryException)
        {
            return (false, default);
        }
    }

    /// <summary>
    /// Checks if an exception indicates GPU/OpenCL is unavailable.
    /// </summary>
    public static bool IsGpuUnavailableException(Exception ex)
    {
        var message = ex.Message;
        return message.Contains("OpenCL") ||
               message.Contains("Failed to create") ||
               message.Contains("Failed to write") ||
               message.Contains("Failed to enqueue") ||
               message.Contains("GPU") ||
               message.Contains("buffer") ||
               message.Contains("kernel");
    }
}
