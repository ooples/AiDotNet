using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// A compiled inference plan of a graph network holds the graph it was traced on: as a constant (the GCN's resident
/// tiled adjacency, the GAT's CSR pattern) or as the tensor object it read. Writing into the adjacency matrix, or
/// installing another one with SetAdjacencyMatrix, must reach the next compiled prediction, through both the network's
/// own compiled path (CompileForward / PredictCompiled) and the facade's compiled predictor.
/// </summary>
[Collection("NonParallelIntegration")]
public class GraphConvolutionalCompiledBindingTests : IDisposable
{
    private const int Nodes = 6;
    private const int Features = 3;
    private const double Tolerance = 1e-6;

    private readonly TensorCodecOptions _originalOptions;

    public GraphConvolutionalCompiledBindingTests()
    {
        AiDotNet.Tests.TestInfrastructure.TestEngines.EnsureCpu();
        _originalOptions = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
    }

    public void Dispose() => TensorCodecOptions.SetCurrent(_originalOptions);

    public static TheoryData<string, int, bool> Cases()
    {
        var data = new TheoryData<string, int, bool>();
        foreach (var layer in new[] { "gcn", "gat" })
            foreach (var batch in new[] { 1, 4 })
                foreach (var facade in new[] { false, true })
                    data.Add(layer, batch, facade);
        return data;
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Cases))]
    public async System.Threading.Tasks.Task CompiledPrediction_FollowsInPlaceEdits_AndNewGraphs(string layer, int batch, bool facade)
    {
        await System.Threading.Tasks.Task.Yield();
        var network = Build(layer);
        var adjacency = Ring(chord: 0);
        network.SetAdjacencyMatrix(adjacency);
        var input = Input(batch);
        Func<Tensor<double>> compiled = Compiled(network, input, facade);

        var eagerBefore = Eager(network, input);
        AssertClose(eagerBefore, compiled(), $"{layer} b{batch}: compiled vs eager on the first graph");
        AssertClose(eagerBefore, compiled(), $"{layer} b{batch}: compiled replay on the first graph");

        // 1. An in-place edit of the installed matrix: another chord.
        adjacency[1, 4] = 1.0;
        adjacency[4, 1] = 1.0;
        var eagerEdited = Eager(network, input);
        AssertDiffer(eagerBefore, eagerEdited, $"{layer} b{batch}: the edit must change the eager output");
        AssertClose(eagerEdited, compiled(), $"{layer} b{batch}: compiled prediction after an in-place edit of the adjacency");

        // 2. A different graph installed through SetAdjacencyMatrix.
        network.SetAdjacencyMatrix(Ring(chord: 2));
        var eagerNew = Eager(network, input);
        AssertDiffer(eagerEdited, eagerNew, $"{layer} b{batch}: the new graph must change the eager output");
        AssertClose(eagerNew, compiled(), $"{layer} b{batch}: compiled prediction after SetAdjacencyMatrix");
        AssertClose(eagerNew, compiled(), $"{layer} b{batch}: compiled replay after SetAdjacencyMatrix");
    }

    private static Func<Tensor<double>> Compiled(GraphNeuralNetwork<double> network, Tensor<double> input, bool facade)
    {
        if (facade)
        {
            var predictor = new AiDotNet.Inference.CompiledNeuralPredictor<double>(network, JitCompilationConfig.Default);
            return () => Copy(predictor.Predict(new[] { input })[0]);
        }

        Assert.True(network.CompileForward(input), "CompileForward did not produce a plan for the graph network.");
        return () => Copy(network.PredictCompiled(input));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SuppliedPackage_InvalidatesGraphChangeBeforeFirstUse(bool exportFirst)
    {
        using var network = Build("gcn");
        network.SetAdjacencyMatrix(Ring(0));
        var input = Input(2);
        var original = new AiDotNet.Inference.CompiledNeuralPredictor<double>(network, JitCompilationConfig.Default).Export(input);
        Assert.NotNull(original);
        var predictor = new AiDotNet.Inference.CompiledNeuralPredictor<double>(network, JitCompilationConfig.Default, original);
        network.SetAdjacencyMatrix(Ring(2));
        var expected = Eager(network, input);
        if (exportFirst) Assert.NotSame(original, predictor.Export(input));
        AssertClose(expected, predictor.Predict(new[] { input })[0], "supplied package after a graph change");
    }

    [Fact]
    public void TiledAdjacency_RebuildsAfterSameVersionStorageReplacement()
    {
        using var network = Build("gcn");
        var adjacency = Ring(0);
        adjacency[0, 0] = 1;
        network.SetAdjacencyMatrix(adjacency);
        var input = Input(2);
        var before = Eager(network, input);
        int version = adjacency.StorageMutationVersion;
        adjacency.AdoptFrom(Ring(2));
        Assert.Equal(version, adjacency.StorageMutationVersion);
        var after = Eager(network, input);
        AssertDiffer(before, after, "storage replacement must rebuild the cached graph");
        using var reference = Build("gcn");
        reference.SetAdjacencyMatrix(Ring(2));
        AssertClose(Eager(reference, input), after, "rebound storage vs a fresh graph");
    }

    [Fact]
    public void CompiledPackageKeepsUnchangedSourcesAndRetracesEqualVersionReplacements()
    {
        using var network = Build("gcn");
        network.SetAdjacencyMatrix(Ring(0));
        var input = Input(2);
        var predictor = new AiDotNet.Inference.CompiledNeuralPredictor<double>(network, JitCompilationConfig.Default);
        var original = predictor.Export(input);
        Assert.NotNull(original);
        Assert.Same(original, predictor.Export(input));
        var before = Eager(network, input);
        using var replacement = Build("gcn");
        replacement.SetAdjacencyMatrix(Ring(2));
        _ = replacement.CompiledBindingVersion;
        var expected = Eager(replacement, input);
        AssertDiffer(before, expected, "replacement graph must change the output");
        var layers = (IList<ILayer<double>>)network.LayersReadOnly;
        for (int i = 0; i < layers.Count; i++) layers[i] = replacement.LayersReadOnly[i];
        Assert.NotSame(original, predictor.Export(input));
        AssertClose(expected, predictor.Predict(new[] { input })[0], "replacement source after compiled export");
    }

    // The forward with compilation off, so the reference never routes through a compiled plan.
    private static Tensor<double> Eager(GraphNeuralNetwork<double> network, Tensor<double> input)
    {
        var previous = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
        try
        {
            return Copy(network.Predict(input));
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previous);
        }
    }

    private static GraphNeuralNetwork<double> Build(string layer)
    {
        var layers = new List<ILayer<double>>();
        if (layer == "gcn")
        {
            layers.Add(new GraphConvolutionalLayer<double>(Features, 5, new ReLUActivation<double>()));
            layers.Add(new GraphConvolutionalLayer<double>(5, 4, new ReLUActivation<double>()));
        }
        else
        {
            layers.Add(new GraphAttentionLayer<double>(Features, 5, numHeads: 2, activationFunction: new ReLUActivation<double>()));
            layers.Add(new GraphAttentionLayer<double>(5, 4, numHeads: 1, activationFunction: new IdentityActivation<double>()));
        }

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: Features,
            outputSize: 4,
            layers: layers);
        var network = new GraphNeuralNetwork<double>(architecture, lossFunction: new CrossEntropyWithLogitsLoss<double>(),
            graphConvolutionalVectorActivation: null);
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 0.5 * Math.Sin(0.9 * i + 0.3);
        network.SetParameters(parameters);
        return network;
    }

    // A ring with self-loops plus one chord from node 0: every row has an edge, and the chord moves with the graph.
    private static Tensor<double> Ring(int chord)
    {
        var a = new Tensor<double>([Nodes, Nodes]);
        for (int i = 0; i < Nodes; i++)
        {
            a[i, i] = 1.0;
            a[i, (i + 1) % Nodes] = 1.0;
            a[(i + 1) % Nodes, i] = 1.0;
        }
        a[chord, (chord + 3) % Nodes] = 1.0;
        a[(chord + 3) % Nodes, chord] = 1.0;
        return a;
    }

    private static Tensor<double> Input(int batch)
    {
        var x = new Tensor<double>([batch, Nodes, Features]);
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.37 * i) + 0.2 * Math.Cos(1.3 * i);
        return x;
    }

    private static Tensor<double> Copy(Tensor<double> t) => new(t.AsSpan().ToArray(), t._shape);

    private static void AssertClose(Tensor<double> expected, Tensor<double> actual, string what)
    {
        var e = expected.AsSpan();
        var a = actual.AsSpan();
        Assert.Equal(e.Length, a.Length);
        double worst = 0;
        for (int i = 0; i < e.Length; i++) worst = Math.Max(worst, Math.Abs(e[i] - a[i]) / Math.Max(1.0, Math.Abs(e[i])));
        Assert.True(worst <= Tolerance, $"{what}: off by {worst:G6}.");
    }

    private static void AssertDiffer(Tensor<double> a, Tensor<double> b, string what)
    {
        var x = a.AsSpan();
        var y = b.AsSpan();
        double most = 0;
        for (int i = 0; i < x.Length; i++) most = Math.Max(most, Math.Abs(x[i] - y[i]));
        Assert.True(most > 1e-4, $"{what} (largest change {most:G6}).");
    }
}
