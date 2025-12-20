using System;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.TestInfrastructure;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DiagnosticsEnvironmentCollection : ICollectionFixture<DiagnosticsEnvironmentCollection.Fixture>
{
    public const string Name = "DiagnosticsEnv";

    public sealed class Fixture : IDisposable
    {
        private readonly string? _original;

        public Fixture()
        {
            _original = Environment.GetEnvironmentVariable("AIDOTNET_DIAGNOSTICS");
        }

        public void Dispose()
        {
            InferenceDiagnostics.Clear();
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", _original);
        }
    }
}

