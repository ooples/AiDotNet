using System;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

[Collection(AiDotNet.Tests.TestInfrastructure.DiagnosticsEnvironmentCollection.Name)]
public class InferenceDiagnosticsTests
{
    [Fact]
    public void InferenceDiagnostics_Disabled_DoesNotRecord()
    {
        var original = Environment.GetEnvironmentVariable("AIDOTNET_DIAGNOSTICS");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", null);
            InferenceDiagnostics.Clear();

            InferenceDiagnostics.RecordDecision("Test", "Feature", enabled: true, reason: "Reason");

            Assert.Empty(InferenceDiagnostics.Snapshot());
        }
        finally
        {
            InferenceDiagnostics.Clear();
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", original);
        }
    }

    [Fact]
    public void InferenceDiagnostics_Enabled_Records()
    {
        var original = Environment.GetEnvironmentVariable("AIDOTNET_DIAGNOSTICS");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", "1");
            InferenceDiagnostics.Clear();

            InferenceDiagnostics.RecordDecision("Test", "Feature", enabled: true, reason: "Reason");

            var entries = InferenceDiagnostics.Snapshot();
            Assert.Contains(entries, e => e.Area == "Test" && e.Feature == "Feature" && e.Enabled);
        }
        finally
        {
            InferenceDiagnostics.Clear();
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", original);
        }
    }
}
