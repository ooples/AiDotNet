using System;
using System.Linq;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// One network failing to dispose must not leave the agent's other networks alive.
///
/// <para>Both agent bases disposed their networks in a plain loop, so the first network whose Dispose threw
/// ended the loop and every network after it was never released. Both now release every network and then
/// report all failures together as an <see cref="AggregateException"/>, the library's convention for
/// multi-resource disposal (see <c>DataPipeline</c>).</para>
/// </summary>
public partial class AgentDisposalFailureTests
{
    [Fact]
    public void Deep_agent_releases_every_network_even_when_one_throws()
    {
        var failing = new ThrowingNetwork();
        var healthy = new CountingNetwork();
        var agent = new TwoNetworkDeepAgent(failing, healthy);

        var thrown = Assert.Throws<AggregateException>(() => agent.Dispose());

        Assert.Equal(1, healthy.DisposeCalls);
        Assert.Contains(thrown.InnerExceptions, e => e is InvalidOperationException && e.Message == ThrowingNetwork.Message);
    }

    [Fact]
    public void Trading_agent_releases_every_network_even_when_one_throws()
    {
        var failing = new ThrowingNetwork();
        var healthy = new CountingNetwork();
        var agent = new TwoNetworkTradingAgent(failing, healthy);

        var thrown = Assert.Throws<AggregateException>(() => agent.Dispose());

        Assert.Equal(1, healthy.DisposeCalls);
        Assert.Contains(thrown.InnerExceptions, e => e is InvalidOperationException && e.Message == ThrowingNetwork.Message);
    }

    [Fact]
    public void Deep_agent_releases_a_network_registered_twice_exactly_once()
    {
        var shared = new CountingNetwork();
        var agent = new TwoNetworkDeepAgent(shared, shared);

        agent.Dispose();
        agent.Dispose();

        Assert.Equal(1, shared.DisposeCalls);
    }

    private static NeuralNetworkArchitecture<double> Arch() => new(inputFeatures: 2, outputSize: 2);

    private sealed partial class ThrowingNetwork : NeuralNetwork<double>
    {
        public const string Message = "simulated dispose failure";

        public ThrowingNetwork() : base(Arch())
        {
        }

        protected override void Dispose(bool disposing) => throw new InvalidOperationException(Message);
    }

    private sealed partial class CountingNetwork : NeuralNetwork<double>
    {
        public CountingNetwork() : base(Arch())
        {
        }

        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }

    // The failing network is registered FIRST: a loop that stops at the first failure never reaches the second.
    private sealed partial class TwoNetworkDeepAgent : DeepReinforcementLearningAgentBase<double>
    {
        public TwoNetworkDeepAgent(NeuralNetwork<double> first, NeuralNetwork<double> second)
            : base(new ReinforcementLearningOptions<double>())
        {
            Networks.Add(first);
            Networks.Add(second);
        }

        public override int FeatureCount => 2;

        public override Vector<double> SelectAction(Vector<double> state, bool training = true) => new Vector<double>(2);

        public override void StoreExperience(
            Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)
        {
        }

        public override double Train() => 0.0;

        public override ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double> { FeatureCount = 2 };

        public override byte[] Serialize() => Array.Empty<byte>();

        public override void Deserialize(byte[] data)
        {
        }
    }

    private sealed partial class TwoNetworkTradingAgent : TradingAgentBase<double>
    {
        public TwoNetworkTradingAgent(NeuralNetwork<double> first, NeuralNetwork<double> second)
            : base(new TradingAgentOptions<double> { StateSize = 2, ActionSize = 2 })
        {
            Networks.Add(first);
            Networks.Add(second);
        }

        public override int FeatureCount => 2;

        public override Vector<double> SelectAction(Vector<double> state, bool training = true) => new Vector<double>(2);

        public override void StoreExperience(
            Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)
        {
        }

        public override double Train() => 0.0;

        public override ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double> { FeatureCount = 2 };

        public override byte[] Serialize() => Array.Empty<byte>();

        public override void Deserialize(byte[] data)
        {
        }
    }
}
