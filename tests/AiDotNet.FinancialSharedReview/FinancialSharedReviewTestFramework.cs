using Xunit.Abstractions;
using Xunit.Sdk;

[assembly: Xunit.TestFramework(
    "AiDotNet.Tests.FinancialSharedReviewTestFramework", "AiDotNetTests")]

namespace AiDotNet.Tests;

/// <summary>
/// Initializes the existing CPU fixture before unchanged shared-boundary tests, including net471.
/// </summary>
public sealed class FinancialSharedReviewTestFramework : XunitTestFramework
{
    /// <summary>Initializes the existing process-wide test policy before discovery or execution.</summary>
    /// <param name="messageSink">The xUnit diagnostic message sink.</param>
    public FinancialSharedReviewTestFramework(IMessageSink messageSink) : base(messageSink)
    {
        TestModuleInitializer.EnsureInitialized();
    }
}
