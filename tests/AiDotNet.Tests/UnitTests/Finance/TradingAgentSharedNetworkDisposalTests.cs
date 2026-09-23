using System;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// A network registered more than once with a trading agent -- two fields holding one instance -- is torn
/// down exactly once when the agent is disposed, and a later direct dispose of that network does not tear it
/// down again.
/// </summary>
public partial class TradingAgentSharedNetworkDisposalTests
{
    [Fact]
    public void A_network_held_by_two_fields_is_torn_down_exactly_once()
    {
        var shared = new CountingNetwork();
        var agent = new TwoFieldsOneNetworkAgent(shared);

        agent.Dispose();
        agent.Dispose();

        Assert.Equal(1, shared.DisposeCalls);
    }

    [Fact]
    public void Disposing_the_network_again_after_its_agent_is_a_no_op()
    {
        var shared = new CountingNetwork();
        var agent = new TwoFieldsOneNetworkAgent(shared);

        agent.Dispose();
        shared.Dispose();

        Assert.Equal(1, shared.DisposeCalls);
    }

    private sealed partial class CountingNetwork : NeuralNetwork<double>
    {
        public CountingNetwork()
            : base(new NeuralNetworkArchitecture<double>(inputFeatures: 2, outputSize: 2))
        {
        }

        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }

    /// <summary>An agent whose online and "target" fields alias one network, as a buggy agent might.</summary>
    private sealed partial class TwoFieldsOneNetworkAgent : TradingAgentBase<double>
    {
        private readonly CountingNetwork _online;
        private readonly CountingNetwork _target;

        public TwoFieldsOneNetworkAgent(CountingNetwork network)
            : base(new TradingAgentOptions<double> { StateSize = 2, ActionSize = 2 })
        {
            _online = network;
            _target = network;
            Networks.Add(_online);
            Networks.Add(_target);
        }

        public override int FeatureCount => 2;

        public override Vector<double> SelectAction(Vector<double> state, bool training = true)
            => new Vector<double>(2);

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
