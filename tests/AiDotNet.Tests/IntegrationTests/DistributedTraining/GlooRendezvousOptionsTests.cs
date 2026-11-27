using System;
using AiDotNet.DistributedTraining;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// The Gloo rendezvous is a typed option (MasterAddress/MasterPort); AIDOTNET_MASTER_ADDR/PORT are only a deprecated
/// fallback that warns. The typed options let process groups in one process rendezvous independently: tests in
/// parallel collections that shared the process-wide variables hung when one changed the port under another.
/// </summary>
[Collection("SequentialMultiProcessGpuDistributed")]
public sealed class GlooRendezvousOptionsTests
{
    [Fact]
    public void TypedRendezvousWins_AndTheDeprecatedVariablesAreAFallbackThatWarns()
    {
        var previous = (Environment.GetEnvironmentVariable("AIDOTNET_MASTER_ADDR"), Environment.GetEnvironmentVariable("AIDOTNET_MASTER_PORT"));
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_MASTER_ADDR", "10.9.8.7");
            Environment.SetEnvironmentVariable("AIDOTNET_MASTER_PORT", "1234");

            var typed = new GlooCommunicationBackend<float>(0, 2) { MasterAddress = "127.0.0.1", MasterPort = 29611 };
            Assert.Equal(("127.0.0.1", "29611"), typed.ResolvedRendezvous);

            var legacy = new GlooCommunicationBackend<float>(0, 2);
            Assert.Equal(("10.9.8.7", "1234"), legacy.ResolvedRendezvous);
            Assert.True(GlooCommunicationBackend<float>.EnvironmentRendezvousWarned);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_MASTER_ADDR", previous.Item1);
            Environment.SetEnvironmentVariable("AIDOTNET_MASTER_PORT", previous.Item2);
        }
    }
}
