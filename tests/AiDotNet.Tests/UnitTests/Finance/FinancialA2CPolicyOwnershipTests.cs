using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

[Trait("category", "unit")]
public sealed class FinancialA2CPolicyOwnershipTests
{
    public enum ExternalWrite { TensorElement, TensorCopy, NetworkChunks, NetworkParameters, LayerParameters }

    public FinancialA2CPolicyOwnershipTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(ExternalWrite.TensorElement)]
    [InlineData(ExternalWrite.TensorCopy)]
    [InlineData(ExternalWrite.NetworkChunks)]
    [InlineData(ExternalWrite.NetworkParameters)]
    [InlineData(ExternalWrite.LayerParameters)]
    public void Ordinary_live_actor_writes_discard_pending_old_policy_data(ExternalWrite write)
    {
        using var agent = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        var state = State(4, 1);
        agent.StoreExperience(state, OneHot(3, 0), 2.0, state, true);
        ChangeActor(agent, write);
        var changed = agent.GetParameters().ToArray();
        Assert.Equal(0.0, agent.Train());
        Assert.Equal(changed, agent.GetParameters().ToArray());
        agent.StoreExperience(state, OneHot(3, 1), 2.0, state, true);
        agent.Train();
        Assert.False(changed.SequenceEqual(agent.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(ExternalWrite.TensorElement)]
    [InlineData(ExternalWrite.TensorCopy)]
    [InlineData(ExternalWrite.NetworkChunks)]
    [InlineData(ExternalWrite.NetworkParameters)]
    [InlineData(ExternalWrite.LayerParameters)]
    public void Returned_actions_cannot_be_stored_after_an_observable_policy_change(ExternalWrite write)
    {
        using var agent = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        var state = State(4, 1);
        var action = agent.SelectAction(state, training: true);
        ChangeActor(agent, write);
        Assert.Throws<InvalidOperationException>(() => agent.StoreExperience(state, action, 2.0, state, true));
        Assert.Equal(0.0, agent.Train());
    }

    [Fact]
    public void Returned_action_cannot_cross_an_explicit_agent_restore_even_when_values_are_identical()
    {
        using var agent = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        var state = State(4, 1);
        var action = agent.SelectAction(state, training: true);
        agent.SetParameters(agent.GetParameters());
        Assert.Throws<InvalidOperationException>(() => agent.StoreExperience(state, action, 2.0, state, true));
    }

    [Fact]
    public void Known_greedy_evaluation_actions_are_not_accepted_as_sampled_behavior()
    {
        using var agent = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        var state = State(4, 1);
        var action = agent.SelectAction(state, training: false);
        Assert.Throws<InvalidOperationException>(() => agent.StoreExperience(state, action, 2.0, state, true));
    }

    [Fact]
    public void Sampled_action_can_be_stored_before_the_policy_changes()
    {
        using var agent = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        var state = State(4, 1);
        var action = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, action, 2.0, state, true);
        var before = agent.GetParameters().ToArray();
        agent.Train();
        Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
    }

    [Fact]
    public void Failed_restore_at_the_topology_boundary_discards_pending_behavior()
    {
        using var agent = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        using var source = FinancialA2CRolloutContractTests.CreateVanillaAgent();
        var changed = source.GetParameters();
        changed[0] = 0.5;
        source.SetParameters(changed);
        var payload = source.Serialize();
        using var payloadStream = new MemoryStream(payload, writable: false);
        using var reader = new BinaryReader(payloadStream);
        reader.ReadInt32(); // Agent envelope version.
        reader.ReadString(); // Runtime type identity.
        int parameterCount = reader.ReadInt32();
        for (int i = 0; i < parameterCount; i++) reader.ReadDouble();
        var truncated = payload.Take(checked((int)payloadStream.Position)).ToArray();
        var state = State(4, 1);
        agent.StoreExperience(state, OneHot(3, 0), 2.0, state, true);
        Assert.Throws<EndOfStreamException>(() => agent.Deserialize(truncated));
        var afterFailure = agent.GetParameters().ToArray();
        Assert.Equal(0.0, agent.Train());
        Assert.Equal(afterFailure, agent.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Legacy_detached_layout_keeps_collection_but_honors_explicit_policy_boundaries(bool replace)
    {
        var options = Options(A2C, 4, 3, seed: 14);
        options.BatchSize = 2;
        options.WarmupSteps = 0;
        options.EntropyCoefficient = 0;
        using var layer = new LegacyLayer();
        var actorArchitecture = Arch(4, 3);
        actorArchitecture.Layers.Add(layer);
        using var agent = new FinancialA2CAgent<double>(actorArchitecture, Arch(4, 1), options);
        var state = State(4, 1);
        agent.SelectAction(state, training: true);
        agent.SetParameters(new Vector<double>((int)agent.ParameterCount));
        var actor = Policy(agent);
        var firstSnapshot = Assert.Single(actor.GetParameterStateChunks());
        var secondSnapshot = Assert.Single(actor.GetParameterStateChunks());
        Assert.False(firstSnapshot.IsWritableInPlace);
        Assert.NotSame(firstSnapshot.Tensor, secondSnapshot.Tensor);
        agent.StoreExperience(state, OneHot(3, 0), 2.0, state, true);
        if (replace) agent.SetParameters(agent.GetParameters());
        agent.StoreExperience(state, OneHot(3, 1), 2.0, state, true);
        var before = agent.GetParameters().ToArray();
        var loss = agent.Train();
        if (replace)
        {
            Assert.Equal(0.0, loss);
            Assert.Equal(before, agent.GetParameters().ToArray());
        }
        else
        {
            Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
        }
    }

    private static NeuralNetworkBase<double> Policy(FinancialA2CAgent<double> agent) =>
        Networks(agent, A2C, 4, 3).Single(n => n.Role == FinancialNetworkRole.Policy).Network;

    private static void ChangeActor(FinancialA2CAgent<double> agent, ExternalWrite write)
    {
        var actor = Policy(agent);
        var before = actor.GetParameters().ToArray();
        var tensor = actor.GetParameterChunks().First();
        switch (write)
        {
            case ExternalWrite.TensorElement:
                tensor.SetFlat(0, tensor.GetFlat(0) + 0.25);
                break;
            case ExternalWrite.TensorCopy:
                var values = tensor.ToArray();
                values[0] += 0.25;
                tensor.CopyFromArray(values);
                break;
            case ExternalWrite.NetworkChunks:
                var chunks = actor.GetParameterChunks().Select(t => t.Clone()).ToArray();
                chunks[0].SetFlat(0, chunks[0].GetFlat(0) + 0.25);
                actor.SetParameterChunks(chunks);
                break;
            case ExternalWrite.NetworkParameters:
                var parameters = actor.GetParameters();
                parameters[0] += 0.25;
                actor.SetParameters(parameters);
                break;
            case ExternalWrite.LayerParameters:
                var layer = actor.Layers[0];
                var layerParameters = layer.GetParameters();
                layerParameters[0] += 0.25;
                layer.SetParameters(layerParameters);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(write));
        }
        Assert.False(before.SequenceEqual(actor.GetParameters().ToArray()));
    }

    // A supported legacy ILayer deliberately exposes only detached flat state chunks.
    // Actual computation/training delegates to Dense; no fake policy or replay behavior.
    internal sealed class LegacyLayer : ILayer<double>, ITrainableLayer<double>, IDisposable
    {
        private readonly DenseLayer<double> _inner = new(3, (IActivationFunction<double>)new IdentityActivation<double>());
        public LegacyLayer() => _inner.Forward(new Tensor<double>(new[] { 1, 4 }));
        public int[] GetInputShape() => _inner.GetInputShape();
        public int[] GetOutputShape() => _inner.GetOutputShape();
        public LayerShape GetOutputLayerShape() => _inner.GetOutputLayerShape();
        public bool IsShapeResolved => _inner.IsShapeResolved;
        public Tensor<double>? GetWeights() => _inner.GetWeights();
        public Tensor<double>? GetBiases() => _inner.GetBiases();
        public Tensor<double> Forward(Tensor<double> input) => _inner.Forward(input);
        public Tensor<double> ForwardWithPrecisionCheck(Tensor<double> input) => _inner.ForwardWithPrecisionCheck(input);
        public Tensor<double> ForwardGpu(params Tensor<double>[] inputs) => _inner.ForwardGpu(inputs);
        public string LayerName => _inner.LayerName;
        public bool CanExecuteOnGpu => _inner.CanExecuteOnGpu;
        public bool SupportsTraining => _inner.SupportsTraining;
        public void SetTrainingMode(bool training) => _inner.SetTrainingMode(training);
        public IReadOnlyList<ILayer<double>> GetSubLayers() => Array.Empty<ILayer<double>>();
        public long ParameterCount => _inner.ParameterCount;
        public int ParameterReads { get; private set; }
        public Vector<double> GetParameters() { ParameterReads++; return _inner.GetParameters(); }
        public void SetParameters(Vector<double> parameters) => _inner.SetParameters(parameters);
        public void UpdateParameters(Vector<double> parameters) => _inner.UpdateParameters(parameters);
        public void UpdateParameters(double rate) => _inner.UpdateParameters(rate);
        public Vector<double> GetParameterGradients() => _inner.GetParameterGradients();
        public void ClearGradients() => _inner.ClearGradients();
        public void ResetState() => _inner.ResetState();
        public IReadOnlyList<Tensor<double>> GetTrainableParameters() => _inner.GetTrainableParameters();
        public void SetTrainableParameters(IReadOnlyList<Tensor<double>> parameters) => _inner.SetTrainableParameters(parameters);
        public void ZeroGrad() => _inner.ZeroGrad();
        public bool SupportsGpuTraining => _inner.SupportsGpuTraining;
        public void UpdateParametersGpu(IGpuOptimizerConfig config) => _inner.UpdateParametersGpu(config);
        public void UploadWeightsToGpu() => _inner.UploadWeightsToGpu();
        public void DownloadWeightsFromGpu() => _inner.DownloadWeightsFromGpu();
        public void ZeroGradientsGpu() => _inner.ZeroGradientsGpu();
        public void Serialize(BinaryWriter writer) => _inner.Serialize(writer);
        public void Deserialize(BinaryReader reader) => _inner.Deserialize(reader);
        public IEnumerable<ActivationFunction> GetActivationTypes() => _inner.GetActivationTypes();
        public Dictionary<string, string> GetDiagnostics() => _inner.GetDiagnostics();
        public IEnumerable<string> GetParameterNames() => _inner.GetParameterNames();
        public bool TryGetParameter(string name, out Tensor<double>? tensor) => _inner.TryGetParameter(name, out tensor);
        public bool SetParameter(string name, Tensor<double> value) => _inner.SetParameter(name, value);
        public int[]? GetParameterShape(string name) => _inner.GetParameterShape(name);
        public int NamedParameterCount => _inner.NamedParameterCount;
        public WeightLoadValidation ValidateWeights(IEnumerable<string> names, Func<string, string?>? mapping = null) => _inner.ValidateWeights(names, mapping);
        public WeightLoadResult LoadWeights(Dictionary<string, Tensor<double>> weights, Func<string, string?>? mapping = null, bool strict = false) => _inner.LoadWeights(weights, mapping, strict);
        public void Dispose() => _inner.Dispose();
    }
}
