using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Graph networks that keep their graph in their layers (rather than packing it into each step's data) and change it
/// between compiled training steps: SetAdjacencyMatrix with another graph, or a write into the installed matrix. The
/// compiled plan holds the graph it traced, so each change must re-trace the step, and the re-traced plan must carry
/// the optimizer's moments on: the run has to match eager training step for step.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class GraphConvolutionalChangingGraphTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; training on a stale graph, or with restarted moments, moves
    // weights by far more.
    private const double RelativeTolerance = 1e-5;
    private const int Nodes = 6;
    private const int Features = 3;
    private const int Steps = 5;

    public static TheoryData<string> Models => new() { "GraphNeuralNetwork", "GraphAttentionNetwork", "GraphSAGENetwork", "GraphIsomorphismNetwork" };

    private sealed record RunResult(Vector<double> Parameters, long FusedSteps, string? Fallback);

    private static NeuralNetworkBase<double> Build(string kind)
    {
        var architecture = new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, complexity: NetworkComplexity.Simple,
            inputSize: Features, outputSize: 2);
        NeuralNetworkBase<double> model = kind switch
        {
            // Two GCN layers, no dropout (dropout draws are covered by GraphTaskFusedTrainingTests). Plain Adam: the
            // default optimizer's first-layer L2 (Kipf & Welling) has no fused form, so it always trains eagerly.
            "GraphNeuralNetwork" => new GraphNeuralNetwork<double>(
                new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.MultiClassClassification, complexity: NetworkComplexity.Simple,
                    inputSize: Features, outputSize: 2,
                    layers: new List<ILayer<double>>
                    {
                        new GraphConvolutionalLayer<double>(Features, 4, new ReLUActivation<double>()),
                        new GraphConvolutionalLayer<double>(4, 2, new IdentityActivation<double>()),
                    }),
                optimizer: new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                    new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01 }),
                lossFunction: new CrossEntropyWithLogitsLoss<double>(), graphConvolutionalVectorActivation: null),
            "GraphAttentionNetwork" => new GraphAttentionNetwork<double>(architecture,
                options: new GraphAttentionNetworkOptions { NumHeads = 2, HiddenSize = 4, NumLayers = 2, DropoutRate = 0.0 }),
            "GraphSAGENetwork" => new GraphSAGENetwork<double>(architecture),
            "GraphIsomorphismNetwork" => new GraphIsomorphismNetwork<double>(architecture),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        SetAdjacency(model, Ring(0));
        model.Predict(Features_(0));
        return model;
    }

    private static void SetAdjacency(NeuralNetworkBase<double> model, Tensor<double> adjacency)
    {
        switch (model)
        {
            case GraphNeuralNetwork<double> m: m.SetAdjacencyMatrix(adjacency); break;
            case GraphAttentionNetwork<double> m: m.SetAdjacencyMatrix(adjacency); break;
            case GraphSAGENetwork<double> m: m.SetAdjacencyMatrix(adjacency); break;
            case GraphIsomorphismNetwork<double> m: m.SetAdjacencyMatrix(adjacency); break;
            default: throw new ArgumentOutOfRangeException(nameof(model));
        }
    }

    // A ring with self-loops plus one chord that moves with the graph: same node count, different structure.
    private static Tensor<double> Ring(int chord)
    {
        var a = new Tensor<double>([Nodes, Nodes]);
        for (int i = 0; i < Nodes; i++)
        {
            a[i, i] = 1.0;
            a[i, (i + 1) % Nodes] = 1.0;
            a[(i + 1) % Nodes, i] = 1.0;
        }
        a[chord % Nodes, (chord + 3) % Nodes] = 1.0;
        a[(chord + 3) % Nodes, chord % Nodes] = 1.0;
        return a;
    }

    private static Tensor<double> Features_(int step)
    {
        var x = new Tensor<double>([Nodes, Features]);
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.41 * i + step);
        return x;
    }

    private static Tensor<double> Target(NeuralNetworkBase<double> model, int step)
    {
        var target = new Tensor<double>(model.Predict(Features_(step))._shape);
        target[step % target.Length] = 1.0;
        return target;
    }

    // Steps 1 and 3 install another graph; step 2 writes an extra chord into the graph installed at step 1; step 4
    // trains on the graph of step 3 again (a plain replay).
    private static RunResult Run(NeuralNetworkBase<double> model, bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate(model);
            Tensor<double>? installed = null;
            for (int step = 0; step < Steps; step++)
            {
                switch (step)
                {
                    case 0:
                    case 1:
                    case 3:
                        installed = Ring(step == 3 ? 2 : step);
                        SetAdjacency(model, installed);
                        break;
                    case 2:
                        installed![0, 2] = 1.0;
                        installed[2, 0] = 1.0;
                        break;
                }
                var target = Target(model, step);
                model.Train(Features_(step), target);
            }
            return new RunResult(model.GetParameters(), CompiledTapeTrainingStep<double>.GetFusedStepCount(),
                CompiledTapeTrainingStep<double>.GetLastFallbackException()?.Message ?? model.LastFusedGateMissReason);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate(model);
        }
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Models))]
    public async Task TrainingOnChangingGraphs_RetracesAndKeepsTheOptimizer(string kind)
    {
        await Task.Yield();
        var model = Build(kind);
        var twin = Build(kind);
        twin.SetParameters(model.GetParameters());
        var before = model.GetParameters();

        var eager = Run(model, compile: false);
        var fused = Run(twin, compile: true);

        Assert.True(fused.FusedSteps >= Steps - 1,
            $"{kind}: only {fused.FusedSteps} of {Steps} steps ran on the compiled plan, so the test does not exercise it (last fallback: {fused.Fallback ?? "none"}).");

        double scale = 0.0, worst = 0.0, moved = 0.0;
        for (int i = 0; i < eager.Parameters.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(eager.Parameters[i]));
            moved = Math.Max(moved, Math.Abs(eager.Parameters[i] - before[i]));
            worst = Math.Max(worst, Math.Abs(eager.Parameters[i] - fused.Parameters[i]));
        }

        Assert.True(moved > 100 * RelativeTolerance, $"{kind}: training barely moved the weights ({moved:G6}).");
        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"{kind}: fused training diverged from eager by {worst:G6} (parameter scale {scale:G6}).");
    }
}
