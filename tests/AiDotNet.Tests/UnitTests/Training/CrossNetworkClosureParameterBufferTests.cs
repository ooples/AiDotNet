using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// An actor whose custom loss forwards a critic (SAC's actor step) reads the critic's weights as external leaves of
/// its compiled plan. The critic training on the eager tape between two actor steps must not change how those
/// weights are stored: the eager step used to move them into views of one contiguous parameter buffer, after which
/// every weight and bias of the critic shared a storage, the actor's next step could not be signed ("aliases another
/// external leaf"), and a committed actor plan threw.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public sealed class CrossNetworkClosureParameterBufferTests
{
    private const int StateDim = 3;
    private const int ActionDim = 2;
    private const int Batch = 4;

    // Built the way SACAgent builds its networks: dense layers that carry their own activations, so every trainable
    // layer holds parameters and the eager step is eligible for the contiguous parameter buffer.
    private static NeuralNetwork<float> Net(int inputSize, int outputSize, IActivationFunction<float> tail, int seed)
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: new List<ILayer<float>>
            {
                new DenseLayer<float>(8, (IActivationFunction<float>)new ReLUActivation<float>()),
                new DenseLayer<float>(outputSize, tail),
            })
        {
            RandomSeed = seed,
        };
        return new NeuralNetwork<float>(architecture);
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 2.0 - 1.0));
        return t;
    }

    [Fact]
    public void CriticEagerStep_BetweenActorSteps_KeepsTheActorPlanReplayable()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var rng = RandomHelper.CreateSeededRandom(7);
            using var actor = Net(StateDim, ActionDim, new TanhActivation<float>(), seed: 11);
            using var critic = Net(StateDim + ActionDim, 1, new IdentityActivation<float>(), seed: 13);
            var engine = AiDotNetEngine.Current;

            float ActorStep(Tensor<float> states) => actor.TrainWithCustomLoss(states, output =>
            {
                var joined = engine.TensorConcatenate([states, output], axis: 1);
                var q = critic.ForwardForTraining(joined);
                var flat = engine.ReduceSum(q, new[] { 1 }, keepDims: false);
                return engine.ReduceMean(engine.TensorNegate(flat), new[] { 0 }, keepDims: false);
            });

            void CriticEagerStep()
            {
                // The critic's own step on the eager tape (its fused path declined), as any decline does.
                var options = TensorCodecOptions.Current;
                bool previous = options.EnableCompilation;
                options.EnableCompilation = false;
                try { critic.Train(Random(rng, Batch, StateDim + ActionDim), Random(rng, Batch, 1)); }
                finally { options.EnableCompilation = previous; }
            }

            // The actor's first step compiles its plan while the critic's weights are standalone tensors.
            ActorStep(Random(rng, Batch, StateDim));
            Assert.True(AiDotNet.Training.CompiledTapeTrainingStep<float>.LastClosureCompileCount > 0,
                "the actor's first step did not compile a closure plan, so this test cannot observe the regression");
            for (int step = 0; step < 3; step++)
            {
                CriticEagerStep();
                float loss = ActorStep(Random(rng, Batch, StateDim));
                Assert.False(float.IsNaN(loss), $"actor step {step + 2} produced NaN");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }
}
