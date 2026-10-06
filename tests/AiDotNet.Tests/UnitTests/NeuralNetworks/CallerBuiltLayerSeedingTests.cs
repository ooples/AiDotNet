using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A caller who builds the layer stack (<c>Architecture.Layers</c>) constructs every layer before the model exists,
/// so the model's seed cannot reach a layer that draws its weights in its constructor. MambaBlock and RWKV7Block
/// did exactly that; they now draw on first use, and the model seeds caller-built layers at construction, so equal
/// model seeds reproduce the stack's initial weights.
/// </summary>
public class CallerBuiltLayerSeedingTests
{
    private const int Sequence = 4;
    private const int Width = 8;

    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class SuppliedStack : NeuralNetworkBase<double>
    {
        public SuppliedStack(NeuralNetworkArchitecture<double> architecture)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            InitializeLayers();
        }

        protected override void InitializeLayers() => Layers.AddRange(Architecture.Layers ?? new List<ILayer<double>>());

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy() => new SuppliedStack(Architecture);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(SuppliedStack),
            Description = "Test double that runs exactly the layers its caller supplied.",
        };
    }

    private static double[] InitialWeights(int seed)
    {
        // A fresh stack per model, built by the caller before any model or seed exists.
        var layers = new List<ILayer<double>>
        {
            new MambaBlock<double>(Sequence, modelDimension: Width, stateDimension: 4),
            new RWKV7Block<double>(Sequence, modelDimension: Width, numHeads: 2),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: Sequence, inputWidth: Width, outputSize: Width, layers: layers)
        {
            RandomSeed = seed,
        };

        var model = new SuppliedStack(architecture);
        model.MaterializeParameters();
        return model.GetParameters().ToArray();
    }

    [Fact]
    public void EqualModelSeeds_GiveEqualInitialWeights_ForACallerBuiltSsmStack()
    {
        var first = InitialWeights(2290);

        Assert.NotEmpty(first);
        Assert.Equal(first, InitialWeights(2290));
    }

    [Fact]
    public void DifferentModelSeeds_GiveDifferentInitialWeights_ForACallerBuiltSsmStack()
    {
        Assert.NotEqual(InitialWeights(2290), InitialWeights(2291));
    }
}
