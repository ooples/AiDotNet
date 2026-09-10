using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Pins that a model whose layer stack ignores its architecture is REPORTED (#2149).
/// </summary>
/// <remarks>
/// <para>
/// <c>ValidateCustomLayersInternal</c> has always carried the architecture-vs-first-layer check, but
/// all 205 of its call sites sit inside an <c>Architecture.Layers</c> guard — so it ran only when
/// the caller supplied the layers, never on the default path where a model builds its own stack
/// from a <c>LayerHelper</c> factory. 104 of those factories take a <c>NeuralNetworkArchitecture</c>
/// and never read it, so those models were free to contradict their own architecture silently.
/// </para>
/// <para>
/// These tests drive the FACTORY-BUILT path specifically: neither fixture supplies
/// <c>Architecture.Layers</c>, so before the fix neither could produce a finding no matter how badly
/// the two disagreed.
/// </para>
/// </remarks>
public class ArchitectureLayerAgreementTests
{
    /// <summary>A model that builds its own stack, sized from a field rather than the architecture.</summary>
    /// <remarks>
    /// The layout declarations are required by ADNSHAPE007 - every concrete model must publish the
    /// caller-facing ranks it supports - and [Batch, Features] is what this two-dense-layer stack
    /// genuinely consumes and emits.
    /// </remarks>
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class StackIgnoresArchitecture : NeuralNetworkBase<double>
    {
        private readonly int _firstLayerInput;

        public StackIgnoresArchitecture(NeuralNetworkArchitecture<double> architecture, int firstLayerInput)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            _firstLayerInput = firstLayerInput;

            // Real models build their stack from their own constructor (see
            // FeedForwardNeuralNetwork), because EnsureArchitectureInitialized has no caller in the
            // base. Without this the double's Layers stayed empty and there was nothing to compare.
            InitializeLayers();
        }

        /// <summary>Escalate the report so the test can observe it without a trace listener.</summary>
        protected override bool ThrowOnLayerContractMismatch => true;

        protected override void InitializeLayers()
        {
            // The defect shape under test: layers sized from something other than the architecture,
            // and Architecture.Layers deliberately not supplied, so this is the default path.
            // FullyConnectedLayer's two-int overload declares a CONCRETE input shape. The lazy
            // single-int layers report [-1], which IsFirstLayerShapeCompatible treats as compatible
            // by design, so they could never express the disagreement under test.
            Layers.Add(new FullyConnectedLayer<double>(_firstLayerInput, 8));
            Layers.Add(new FullyConnectedLayer<double>(8, 4));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
            => new StackIgnoresArchitecture(Architecture, _firstLayerInput);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(StackIgnoresArchitecture),
            Description = "Test double whose layer stack is sized without consulting its architecture.",
        };
    }

    private static NeuralNetworkArchitecture<double> Architecture(int inputSize)
        => new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputSize,
            outputSize: 4);

    [Fact]
    public void FactoryBuiltStack_ThatContradictsItsArchitecture_IsReported()
    {
        // Architecture says 32 features; the stack was built for 16 and never consulted it.
        //
        // The input fed matches the LAYERS, not the architecture, which is the case worth catching:
        // the forward succeeds, so nothing surfaces on its own, while the architecture quietly
        // misdescribes the model and ResolveLazyLayerShapes would resolve lazy weights against it.
        // Feeding the architecture's shape instead just makes the forward throw a shape error, which
        // is a different (and self-announcing) situation.
        using var model = new StackIgnoresArchitecture(Architecture(32), firstLayerInput: 16);

        var error = Assert.Throws<InvalidOperationException>(() => model.Predict(
            new Tensor<double>([1, 16])));

        Assert.Contains("does not describe the input its first layer accepts", error.Message);
        Assert.Contains("architecture input shape", error.Message);
    }

    [Fact]
    public void FactoryBuiltStack_ThatAgreesWithItsArchitecture_IsSilent()
    {
        // The control arm. Same code path, same escalation, only the disagreement removed — so a
        // pass here shows the check discriminates rather than firing on everything it sees.
        using var model = new StackIgnoresArchitecture(Architecture(16), firstLayerInput: 16);

        var output = model.Predict(new Tensor<double>([1, 16]));

        Assert.NotNull(output);
    }
}
