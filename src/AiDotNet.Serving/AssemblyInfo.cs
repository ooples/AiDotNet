using System.Runtime.CompilerServices;

// Expose internal members of AiDotNet.Serving to the test assembly.
// Currently used by OpenTelemetryRegistration.DisposeTelemetry, which is
// an internal test-teardown hook that production code should not call.
[assembly: InternalsVisibleTo("AiDotNet.Serving.Tests")]
