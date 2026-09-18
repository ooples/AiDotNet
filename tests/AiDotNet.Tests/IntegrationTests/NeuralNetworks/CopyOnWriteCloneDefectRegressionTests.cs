using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression coverage for two defects on the copy-on-write <c>DeepCopy</c> path, both of which
/// surfaced as a clone of a VALID model being rejected or refused.
/// </summary>
/// <remarks>
/// <para>
/// These are deliberately independent of the two tests that first exposed the defects
/// (<c>SequenceTokenSliceLayerShapeContractTests</c> and <c>HrePaperAChainShapeValidatorTests</c>):
/// different architectures, a different custom layer, and direct assertions on the property that
/// actually broke, so a regression cannot hide behind the other suites' assertions.
/// </para>
/// </remarks>
public partial class CopyOnWriteCloneDefectRegressionTests
{
    private const int Height = 4;
    private const int Width = 6;
    private const int Classes = 3;

    /// <summary>
    /// DEFECT 1: a multi-dimensional entry shape must survive the clone.
    /// </summary>
    /// <remarks>
    /// <c>InputLayer([4, 6])</c> was rebuilt through the <c>(int)</c> constructor with
    /// <c>inputShape[0]</c>, so the clone's entry layer declared <c>[4]</c> instead of
    /// <c>[4, 6]</c>. The architecture that accepted 4 x 6 = 24 then rejected its own clone with
    /// "The first layer's input size (4) must match the input size (24)". Pre-fix this test throws
    /// from <c>DeepCopy</c>; the shape assertion below is what pins the actual contract.
    /// </remarks>
    [Fact]
    public void DeepCopy_MultiDimensionalInputLayer_PreservesEntryShape()
    {
        var network = BuildMultiDimensionalNetwork();
        var input = new Tensor<float>([2, Height, Width]);
        FillDeterministically(input);
        Tensor<float> before = network.Predict(input);

        var copy = (FeedForwardNeuralNetwork<float>)network.DeepCopy();

        // The entry layer must declare the SAME multi-dimensional contract, not a collapsed rank-1
        // one. This is the assertion that fails if the lossy rebuild ever comes back.
        Assert.Equal([Height, Width], copy.Layers[0].GetInputShape());
        Assert.Equal([Height, Width], copy.Layers[0].GetOutputShape());
        Assert.Equal(network.Layers.Count, copy.Layers.Count);
        Assert.NotSame(network, copy);

        // A clone must also still compute the same function.
        Tensor<float> after = copy.Predict(input);
        Assert.Equal(before.Shape.ToArray(), after.Shape.ToArray());
        AssertSameValues(before, after);
    }

    /// <summary>
    /// DEFECT 2: a layer whose type lives outside AiDotNet must be cloneable.
    /// </summary>
    /// <remarks>
    /// <c>CloneForModelConstruction</c> rebuilt architecture layers through a name-keyed table
    /// populated by scanning AiDotNet's own assembly, so any consumer-defined layer failed with
    /// "Layer type ... is not supported for deserialization". Both the copy-on-write path and the
    /// eager path reach that code through <c>CreateNewInstance()</c>, so the pre-existing
    /// custom-layer guard did not avoid it.
    /// </remarks>
    [Fact]
    public void DeepCopy_LayerFromConsumerAssembly_IsCloned()
    {
        var network = BuildNetworkWithExternalLayer();
        var input = new Tensor<float>([2, Height * Width]);
        FillDeterministically(input);
        Tensor<float> before = network.Predict(input);

        var copy = (FeedForwardNeuralNetwork<float>)network.DeepCopy();

        Assert.NotSame(network, copy);
        Assert.Equal(network.Layers.Count, copy.Layers.Count);
        Assert.Equal(
            network.Layers.Select(layer => layer.GetType()),
            copy.Layers.Select(layer => layer.GetType()));

        // The consumer layer must be an INDEPENDENT object, not the original shared into the clone.
        int externalIndex = network.Layers
            .Select((layer, index) => (layer, index))
            .First(pair => pair.layer is ExternalIdentityLayer).index;
        Assert.IsType<ExternalIdentityLayer>(copy.Layers[externalIndex]);
        Assert.NotSame(network.Layers[externalIndex], copy.Layers[externalIndex]);

        Tensor<float> after = copy.Predict(input);
        AssertSameValues(before, after);
    }

    /// <summary>
    /// Guards the models that ALREADY cloned successfully: neither fix may change a clone's
    /// parameter count or its predictions.
    /// </summary>
    [Fact]
    public void DeepCopy_OrdinaryModel_KeepsParameterCountAndPredictions()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(Height * Width),
            new DenseLayer<float>(8, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: Height * Width,
            outputSize: Classes,
            layers: layers);
        var network = new FeedForwardNeuralNetwork<float>(architecture);

        var input = new Tensor<float>([2, Height * Width]);
        FillDeterministically(input);
        Tensor<float> before = network.Predict(input);
        long parameterCountBefore = network.ParameterCount;

        var copy = (FeedForwardNeuralNetwork<float>)network.DeepCopy();

        Assert.Equal(parameterCountBefore, copy.ParameterCount);
        AssertSameValues(before, copy.Predict(input));
    }

    // ====================================================================
    // Fixtures
    // ====================================================================

    private static FeedForwardNeuralNetwork<float> BuildMultiDimensionalNetwork()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>([Height, Width]),
            new FlattenLayer<float>(),
            new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Height,
            inputWidth: Width,
            outputSize: Classes,
            layers: layers);

        return new FeedForwardNeuralNetwork<float>(architecture);
    }

    private static FeedForwardNeuralNetwork<float> BuildNetworkWithExternalLayer()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(Height * Width),
            new ExternalIdentityLayer(),
            new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: Height * Width,
            outputSize: Classes,
            layers: layers);

        return new FeedForwardNeuralNetwork<float>(architecture);
    }

    private static void FillDeterministically(Tensor<float> tensor)
    {
        Span<float> values = tensor.AsWritableSpan();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = ((i % 13) - 6) / 6f;
        }
    }

    private static void AssertSameValues(Tensor<float> expected, Tensor<float> actual)
    {
        float[] left = expected.ToArray();
        float[] right = actual.ToArray();
        Assert.Equal(left.Length, right.Length);
        for (int i = 0; i < left.Length; i++)
        {
            Assert.Equal(left[i], right[i], 6);
        }
    }

    /// <summary>
    /// A minimal layer standing in for one defined in a CONSUMER assembly: it lives in the test
    /// assembly, so AiDotNet's name-keyed layer table cannot resolve it. Identity forward, no
    /// parameters — the clone path, not the math, is what is under test.
    /// </summary>
    [TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed partial class ExternalIdentityLayer : LayerBase<float>, IShapeContract
    {
        public ExternalIdentityLayer()
            : base([Height * Width], [Height * Width])
        {
        }

        public override long ParameterCount => 0;

        public override bool SupportsTraining => false;

        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        {
            if (inputRank != 1) return null;

            return [new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Height * Width))];
        }

        protected override Tensor<float> ForwardTraced(Tensor<float> input) => input;

        public override void UpdateParameters(float learningRate) { }

        public override Vector<float> GetParameters() => new Vector<float>(0);

        public override void SetParameters(Vector<float> parameters) { }

        public override Vector<float> GetParameterGradients() => new Vector<float>(0);

        public override void ResetState() { }
    }
}
