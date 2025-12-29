using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class BaseClassesIntegrationTests
{
    [Fact]
    public void PolicyBase_ValidateStateAndAction_ThrowsForInvalidInput()
    {
        var policy = new TestPolicy();

        Assert.Throws<ArgumentNullException>(() => policy.InvokeValidateState(null!));
        Assert.Throws<ArgumentException>(() => policy.InvokeValidateState(new Vector<double>(0)));
        Assert.Throws<ArgumentException>(() => policy.InvokeValidateActionSize(expected: 2, actual: 1));
    }

    [Fact]
    public void PolicyBase_Dispose_MarksDisposed()
    {
        var policy = new TestPolicy();

        Assert.False(policy.IsDisposed);
        policy.Dispose();
        Assert.True(policy.IsDisposed);
    }

    [Fact]
    public void ExplorationStrategyBase_ClampAndValidate_Work()
    {
        var strategy = new TestExplorationStrategy();
        var action = new Vector<double>(3);
        action[0] = -2.0;
        action[1] = 0.5;
        action[2] = 2.0;

        var clamped = strategy.Clamp(action, min: -1.0, max: 1.0);

        Assert.Equal(-1.0, clamped[0]);
        Assert.Equal(0.5, clamped[1]);
        Assert.Equal(1.0, clamped[2]);

        Assert.Throws<ArgumentException>(() => strategy.ValidateSize(expected: 2, actual: 1));
    }

    [Fact]
    public void ExplorationStrategyBase_BoxMullerSample_IsFinite()
    {
        var strategy = new TestExplorationStrategy();

        var sample = strategy.SampleNormal(new Random(7));

        Assert.False(double.IsNaN(sample));
        Assert.False(double.IsInfinity(sample));
    }

    [Fact]
    public void DeepReinforcementLearningAgentBase_ParameterCount_SumsNetworks()
    {
        var agent = new TestDeepAgent(CreateOptions(), jitPolicy: null);

        Assert.Equal(agent.NetworkParameterCount, agent.ParameterCount);
    }

    [Fact]
    public void DeepReinforcementLearningAgentBase_ExportComputationGraph_UsesJitPolicy()
    {
        var jitPolicy = new StubJitPolicy { SupportsJitCompilation = true };
        var agent = new TestDeepAgent(CreateOptions(), jitPolicy);

        Assert.True(agent.SupportsJitCompilation);

        var inputs = new List<ComputationNode<double>>();
        var output = agent.ExportComputationGraph(inputs);

        Assert.Single(inputs);
        Assert.Same(inputs[0], output);
    }

    [Fact]
    public void DeepReinforcementLearningAgentBase_ExportComputationGraph_ThrowsWhenUnsupported()
    {
        var jitPolicy = new StubJitPolicy { SupportsJitCompilation = false };
        var agent = new TestDeepAgent(CreateOptions(), jitPolicy);

        Assert.False(agent.SupportsJitCompilation);
        Assert.Throws<NotSupportedException>(() => agent.ExportComputationGraph(new List<ComputationNode<double>>()));
    }

    [Fact]
    public void ReinforcementLearningAgentBase_DefaultsAndStateRoundTrip_Work()
    {
        var agent = new TestBaseAgent(CreateOptions());
        var state = new Vector<double>(agent.FeatureCount);

        var action = agent.Predict(state);
        Assert.Equal(agent.FeatureCount, action.Length);

        Assert.False(agent.SupportsJitCompilation);
        Assert.Throws<NotSupportedException>(() => agent.ExportComputationGraph(new List<ComputationNode<double>>()));
        Assert.Throws<NotSupportedException>(() => agent.Train(state, action));

        var names = agent.FeatureNames;
        Assert.Equal(agent.FeatureCount, names.Length);
        Assert.True(agent.IsFeatureUsed(0));
        Assert.False(agent.IsFeatureUsed(agent.FeatureCount));

        var importance = agent.GetFeatureImportance();
        Assert.Equal(agent.FeatureCount, importance.Count);

        var metrics = agent.GetMetrics();
        Assert.True(metrics.ContainsKey("TrainingSteps"));
        Assert.True(metrics.ContainsKey("Episodes"));
        Assert.True(metrics.ContainsKey("AverageLoss"));
        Assert.True(metrics.ContainsKey("AverageReward"));

        using var stream = new System.IO.MemoryStream();
        agent.SaveState(stream);
        stream.Position = 0;
        agent.LoadState(stream);
        Assert.True(agent.DeserializeCalled);
    }

    private static ReinforcementLearningOptions<double> CreateOptions()
    {
        return new ReinforcementLearningOptions<double>
        {
            LearningRate = 0.01,
            DiscountFactor = 0.9,
            LossFunction = new MeanSquaredErrorLoss<double>(),
            Seed = 3
        };
    }

    private sealed class TestPolicy : PolicyBase<double>
    {
        public TestPolicy() : base(new Random(1))
        {
        }

        public bool IsDisposed => _disposed;

        public void InvokeValidateState(Vector<double> state)
        {
            ValidateState(state, nameof(state));
        }

        public void InvokeValidateActionSize(int expected, int actual)
        {
            ValidateActionSize(expected, actual, nameof(actual));
        }

        public override Vector<double> SelectAction(Vector<double> state, bool training = true)
        {
            return state;
        }

        public override double ComputeLogProb(Vector<double> state, Vector<double> action)
        {
            return 0.0;
        }

        public override IReadOnlyList<INeuralNetwork<double>> GetNetworks()
        {
            return Array.Empty<INeuralNetwork<double>>();
        }
    }

    private sealed class TestExplorationStrategy : ExplorationStrategyBase<double>
    {
        public override Vector<double> GetExplorationAction(
            Vector<double> state,
            Vector<double> policyAction,
            int actionSpaceSize,
            Random random)
        {
            return policyAction;
        }

        public override void Update()
        {
        }

        public double SampleNormal(Random random)
        {
            return NumOps.ToDouble(BoxMullerSample(random));
        }

        public Vector<double> Clamp(Vector<double> action, double min, double max)
        {
            return ClampAction(action, min, max);
        }

        public void ValidateSize(int expected, int actual)
        {
            ValidateActionSize(expected, actual, nameof(actual));
        }
    }

    private sealed class TestDeepAgent : DeepReinforcementLearningAgentBase<double>
    {
        private readonly Vector<double> _parameters;
        private readonly IJitCompilable<double>? _jitPolicy;

        public TestDeepAgent(ReinforcementLearningOptions<double> options, IJitCompilable<double>? jitPolicy)
            : base(options)
        {
            _jitPolicy = jitPolicy;
            Networks.Add(CreateNetwork());
            _parameters = new Vector<double>(1);
            _parameters[0] = 0.1;
        }

        public int NetworkParameterCount => Networks.Sum(network => network.ParameterCount);

        public override int FeatureCount => 2;

        protected override IJitCompilable<double>? GetPolicyNetworkForJit()
        {
            return _jitPolicy;
        }

        public override Vector<double> SelectAction(Vector<double> state, bool training = true)
        {
            return new Vector<double>(2);
        }

        public override void StoreExperience(Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)
        {
        }

        public override double Train()
        {
            return 0.0;
        }

        public override ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                ModelType = ModelType.ReinforcementLearning,
                FeatureCount = FeatureCount
            };
        }

        public override byte[] Serialize()
        {
            return Array.Empty<byte>();
        }

        public override void Deserialize(byte[] data)
        {
        }

        public override Vector<double> GetParameters()
        {
            return _parameters.Clone();
        }

        public override void SetParameters(Vector<double> parameters)
        {
            if (parameters.Length > 0)
            {
                _parameters[0] = parameters[0];
            }
        }

        public override IFullModel<double, Vector<double>, Vector<double>> Clone()
        {
            return new TestDeepAgent(Options, _jitPolicy);
        }

        public override Vector<double> ComputeGradients(
            Vector<double> input,
            Vector<double> target,
            ILossFunction<double>? lossFunction = null)
        {
            return new Vector<double>(1);
        }

        public override void ApplyGradients(Vector<double> gradients, double learningRate)
        {
        }

        public override void SaveModel(string filepath)
        {
        }

        public override void LoadModel(string filepath)
        {
        }

        private static NeuralNetwork<double> CreateNetwork()
        {
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.ReinforcementLearning,
                complexity: NetworkComplexity.Simple,
                inputSize: 2,
                outputSize: 2);

            return new NeuralNetwork<double>(architecture);
        }
    }

    private sealed class TestBaseAgent : ReinforcementLearningAgentBase<double>
    {
        private Vector<double> _parameters;
        private bool _deserializeCalled;

        public TestBaseAgent(ReinforcementLearningOptions<double> options)
            : base(options)
        {
            _parameters = new Vector<double>(1);
            _parameters[0] = 0.1;
        }

        public bool DeserializeCalled => _deserializeCalled;

        public override Vector<double> SelectAction(Vector<double> state, bool training = true)
        {
            return new Vector<double>(FeatureCount);
        }

        public override void StoreExperience(Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)
        {
        }

        public override double Train()
        {
            TrainingSteps++;
            Episodes++;
            LossHistory.Add(NumOps.One);
            RewardHistory.Add(NumOps.One);
            return NumOps.One;
        }

        public override ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                ModelType = ModelType.ReinforcementLearning,
                FeatureCount = FeatureCount
            };
        }

        public override byte[] Serialize()
        {
            return new[] { (byte)1, (byte)2, (byte)3 };
        }

        public override void Deserialize(byte[] data)
        {
            _deserializeCalled = true;
        }

        public override Vector<double> GetParameters()
        {
            return _parameters.Clone();
        }

        public override void SetParameters(Vector<double> parameters)
        {
            if (parameters.Length > 0)
            {
                _parameters[0] = parameters[0];
            }
        }

        public override int ParameterCount => _parameters.Length;

        public override int FeatureCount => 2;

        public override IFullModel<double, Vector<double>, Vector<double>> Clone()
        {
            var clone = new TestBaseAgent(Options);
            clone.SetParameters(GetParameters());
            return clone;
        }

        public override Vector<double> ComputeGradients(
            Vector<double> input,
            Vector<double> target,
            ILossFunction<double>? lossFunction = null)
        {
            return new Vector<double>(ParameterCount);
        }

        public override void ApplyGradients(Vector<double> gradients, double learningRate)
        {
        }

        public override void SaveModel(string filepath)
        {
            var data = Serialize();
            System.IO.File.WriteAllBytes(filepath, data);
        }

        public override void LoadModel(string filepath)
        {
            var data = System.IO.File.ReadAllBytes(filepath);
            Deserialize(data);
        }
    }

    private sealed class StubJitPolicy : IJitCompilable<double>
    {
        public bool SupportsJitCompilation { get; set; }

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            var node = new ComputationNode<double>(Tensor<double>.FromScalar(1.0));
            inputNodes.Add(node);
            return node;
        }
    }
}
