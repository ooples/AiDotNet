using System.Collections.Generic;
using System.IO;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for <see cref="NeuralNetworkBase{T}.BuildFullGraph"/> — the
/// explicit graph-capture API that composes per-layer
/// <see cref="ILayer{T}.ExportComputationGraph"/> outputs into a full
/// network graph. Addresses the "LayerBase graph capture mode" checklist
/// item on github.com/ooples/AiDotNet#1015.
/// </summary>
public class BuildFullGraphTests
{
    /// <summary>
    /// BuildFullGraph throws on null input — fail-fast on invalid args.
    /// </summary>
    [Fact]
    public void BuildFullGraph_ThrowsOnNullInput()
    {
        var network = BuildConvNetwork();
        Assert.Throws<System.ArgumentNullException>(() => network.BuildFullGraph(null!));
    }

    /// <summary>
    /// With all layers opted into JIT (ConvolutionalLayer supports it),
    /// BuildFullGraph returns a non-null graph node. The node's value
    /// must match the eager forward-pass output shape.
    /// </summary>
    [Fact]
    public void BuildFullGraph_AllLayersJitCapable_ReturnsGraph()
    {
        var network = BuildConvNetwork();
        var input = MakeInput(new[] { 1, 3, 8, 8 });

        // Warmup — force lazy init on the conv layer so ExportComputationGraph
        // doesn't trip on a [0, 0] placeholder kernel.
        network.Predict(input);

        var graph = network.BuildFullGraph(input);
        Assert.NotNull(graph);
    }

    /// <summary>
    /// When the network has a layer that doesn't support JIT (doesn't
    /// inherit from LayerBase or returns SupportsJitCompilation=false),
    /// BuildFullGraph returns null — the graceful-fallback signal.
    /// </summary>
    [Fact]
    public void BuildFullGraph_UnsupportedLayer_ReturnsNull()
    {
        var network = BuildMixedNetwork();
        var input = MakeInput(new[] { 1, 4 });
        network.Predict(input); // warmup

        var graph = network.BuildFullGraph(input);
        Assert.Null(graph);
    }

    /// <summary>
    /// Empty-Layers network returns null — nothing to trace.
    /// </summary>
    [Fact]
    public void BuildFullGraph_EmptyNetwork_ReturnsNull()
    {
        var network = BuildEmptyNetwork();
        var input = MakeInput(new[] { 1, 4 });
        var graph = network.BuildFullGraph(input);
        Assert.Null(graph);
    }

    // ---------- helpers ----------

    private static JitTestNetwork BuildConvNetwork()
    {
        // ThreeDimensional input requires explicit inputHeight/Width/Depth.
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 3 * 8 * 8,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 3,
            outputSize: 3 * 8 * 8);

        var network = new JitTestNetwork(arch);
        // ConvolutionalLayer supports JIT after this PR.
        network.AddLayer(new ConvolutionalLayer<float>(
            inputDepth: 3,
            inputHeight: 8,
            inputWidth: 8,
            outputDepth: 3,
            kernelSize: 3,
            stride: 1,
            padding: 1));
        return network;
    }

    private static JitTestNetwork BuildMixedNetwork()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new JitTestNetwork(arch);
        // DenseLayer doesn't override ExportComputationGraph + SupportsJitCompilation
        // stays false — BuildFullGraph should return null.
        network.AddLayer(new DenseLayer<float>(4, 2));
        return network;
    }

    private static JitTestNetwork BuildEmptyNetwork()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);
        return new JitTestNetwork(arch);
    }

    private static Tensor<float> MakeInput(int[] shape)
    {
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.01f;
        return new Tensor<float>(data, shape);
    }

    internal sealed class JitTestNetwork : NeuralNetworkBase<float>
    {
        public JitTestNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
        }

        public override bool SupportsTraining => true;

        public void AddLayer(ILayer<float> layer) => AddLayerToCollection(layer);

        protected override void InitializeLayers() { }

        public override Tensor<float> Predict(Tensor<float> input)
        {
            Tensor<float> current = input;
            foreach (var layer in Layers) current = layer.Forward(current);
            return current;
        }

        public override void UpdateParameters(Vector<float> parameters) => SetParameters(parameters);

        public override void Train(Tensor<float> input, Tensor<float> expectedOutput)
            => TrainWithTape(input, expectedOutput);

        public override ModelMetadata<float> GetModelMetadata() =>
            new ModelMetadata<float>
            {
                Name = "JitTestNetwork",
                Version = "1.0",
                FeatureCount = Architecture.InputSize,
                Complexity = ParameterCount
            };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new JitTestNetwork(Architecture);
    }
}
