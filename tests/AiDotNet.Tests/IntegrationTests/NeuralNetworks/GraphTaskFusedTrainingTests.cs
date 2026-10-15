using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Graph task models train on the shared fused session (#1804) with the adjacency packed into each step's input.
/// Every step here trains on a different graph of the same size, the way graph classification walks a dataset: a
/// plan that kept the first graph's adjacency would train every later graph on the wrong structure.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class GraphTaskFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step on another graph moves weights by far more.
    private const double RelativeTolerance = 1e-5;
    private const int Nodes = 5;
    private const int Features = 3;

    public static TheoryData<string> Models => new() { "GraphClassification", "LinkPrediction", "NodeClassification" };

    private sealed record RunResult(Vector<double> Parameters, int FusedSteps, int EagerSteps);

    private static NeuralNetworkBase<double> Build(string kind)
    {
        NeuralNetworkBase<double> model = kind switch
        {
            "GraphClassification" => new GraphClassificationModel<double>(
                new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.MultiClassClassification, complexity: NetworkComplexity.Simple,
                    inputSize: Features, outputSize: 2),
                options: new GraphClassificationOptions { HiddenDim = 4, EmbeddingDim = 4, NumGnnLayers = 2, DropoutRate = 0.0 }),
            "LinkPrediction" => new LinkPredictionModel<double>(
                new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.BinaryClassification, complexity: NetworkComplexity.Simple,
                    inputSize: Features, outputSize: 1),
                options: new LinkPredictionOptions { HiddenDim = 4, EmbeddingDim = 3, NumLayers = 2, DropoutRate = 0.0 }),
            "NodeClassification" => new NodeClassificationModel<double>(
                new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.MultiClassClassification, complexity: NetworkComplexity.Simple,
                    inputSize: Features, outputSize: 2)),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        // Resolve lazily shaped layers so the twin can take these exact weights.
        SetAdjacency(model, Adjacency(0));
        model.Predict(Features_(0));
        return model;
    }

    private static void SetAdjacency(NeuralNetworkBase<double> model, Tensor<double> adjacency)
    {
        switch (model)
        {
            case GraphClassificationModel<double> g: g.SetAdjacencyMatrix(adjacency); break;
            case LinkPredictionModel<double> l: l.SetAdjacencyMatrix(adjacency); break;
            case NodeClassificationModel<double> n: n.SetAdjacencyMatrix(adjacency); break;
        }
    }

    // A ring plus one chord that moves with the step: same node count, different structure.
    private static Tensor<double> Adjacency(int step)
    {
        var adjacency = new Tensor<double>([Nodes, Nodes]);
        for (int i = 0; i < Nodes; i++)
        {
            adjacency[i, i] = 1.0;
            adjacency[i, (i + 1) % Nodes] = 1.0;
            adjacency[(i + 1) % Nodes, i] = 1.0;
        }
        int a = step % Nodes, b = (step + 2) % Nodes;
        adjacency[a, b] = 1.0;
        adjacency[b, a] = 1.0;
        return adjacency;
    }

    private static Tensor<double> Features_(int step)
    {
        var x = new Tensor<double>([Nodes, Features]);
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.41 * i + step);
        return x;
    }

    private static Tensor<double> Target(NeuralNetworkBase<double> model, int step)
    {
        var shape = model.Predict(Features_(step))._shape;
        var target = new Tensor<double>(shape);
        target[step % target.Length] = 1.0;
        return target;
    }

    private static RunResult Run(NeuralNetworkBase<double> model, bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            for (int step = 0; step < 4; step++)
            {
                SetAdjacency(model, Adjacency(step));
                model.Train(Features_(step), Target(model, step));
            }
            return new RunResult(model.GetParameters(), model.CustomFusedStepCount, model.CustomEagerStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Models))]
    public async Task TrainingOnChangingGraphs_RunsFused_AndTakesTheEagerStep(string kind)
    {
        await Task.Yield();
        var model = Build(kind);
        var twin = Build(kind);
        twin.SetParameters(model.GetParameters());
        var before = model.GetParameters();

        var eager = Run(model, compile: false);
        var fused = Run(twin, compile: true);

        Assert.Equal(0, eager.FusedSteps);
        Assert.Equal(4, eager.EagerSteps);
        Assert.Equal(0, fused.EagerSteps);
        Assert.Equal(4, fused.FusedSteps);

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
