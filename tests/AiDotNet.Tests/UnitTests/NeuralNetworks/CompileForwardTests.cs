using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for the <see cref="NeuralNetworkBase{T}.CompileForward(Tensor{T})"/>
/// public method — addresses the
/// "NeuralNetworkBase.CompileForward()" checklist item on
/// github.com/ooples/AiDotNet#1015. Verifies the pre-warm contract:
/// call with a sample input, compilation succeeds (or fails gracefully),
/// and subsequent Predict calls replay the compiled plan.
/// </summary>
public class CompileForwardTests
{
    /// <summary>
    /// CompileForward must throw ArgumentNullException when called with a
    /// null sample input. Fail-fast on invalid arguments rather than
    /// silent no-op, matching the AiDotNet convention for constructor
    /// and factory parameters.
    /// </summary>
    [Fact]
    public void CompileForward_ThrowsOnNullInput()
    {
        var network = BuildNetwork();
        Assert.Throws<ArgumentNullException>(() => network.CompileForward(null!));
    }

    /// <summary>
    /// When compilation is disabled via
    /// <see cref="TensorCodecOptions.EnableCompilation"/>, CompileForward
    /// must return false without attempting to trace. This lets apps
    /// detect "compile is off" from the return value without exceptions.
    /// </summary>
    [Fact]
    public void CompileForward_ReturnsFalse_WhenCompilationDisabled()
    {
        var network = BuildNetwork();
        var sample = MakeInput(new[] { 1, 4 });

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            Assert.False(network.CompileForward(sample));
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    /// <summary>
    /// With compilation enabled and a well-formed sample input,
    /// CompileForward either succeeds (returns true) or fails gracefully
    /// (returns false). Either way, a subsequent <see cref="NeuralNetworkBase{T}.Predict"/>
    /// call must produce a correctly-shaped output — the compiled plan
    /// is either replayed (true) or the eager fallback is used (false).
    /// </summary>
    [Fact]
    public void CompileForward_SuccessOrGracefulFallback_ThenPredictWorks()
    {
        var network = BuildNetwork();
        var sample = MakeInput(new[] { 1, 4 });

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            bool compiled = network.CompileForward(sample);
            // compiled may be true or false depending on whether the layers
            // produced a traceable graph — both outcomes are contract-
            // compliant as long as Predict still works.

            var output = network.Predict(sample);
            Assert.NotNull(output);
            Assert.Equal(2, output.Shape[^1]); // output has 2 features (the network's output size)
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    /// <summary>
    /// Multiple calls to CompileForward with the SAME shape should be
    /// idempotent — the plan is cached after the first successful compile,
    /// and subsequent calls reuse it without re-tracing.
    /// </summary>
    [Fact]
    public void CompileForward_Idempotent_SameShape()
    {
        var network = BuildNetwork();
        var sample = MakeInput(new[] { 1, 4 });

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            bool first = network.CompileForward(sample);
            bool second = network.CompileForward(sample);
            // Both return the same result — either both true or both false.
            Assert.Equal(first, second);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    /// <summary>
    /// Calling CompileForward with DIFFERENT shapes pre-warms multiple
    /// plans in the same cache. Subsequent Predict calls at each shape
    /// should work without re-compile overhead.
    /// </summary>
    [Fact]
    public void CompileForward_PreWarms_MultipleShapes()
    {
        var network = BuildNetwork();
        var sample1 = MakeInput(new[] { 1, 4 });
        var sample2 = MakeInput(new[] { 2, 4 });

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            network.CompileForward(sample1);
            network.CompileForward(sample2);

            // Both shapes predict without error.
            var output1 = network.Predict(sample1);
            var output2 = network.Predict(sample2);
            Assert.Equal(sample1.Shape[0], output1.Shape[0]);
            Assert.Equal(sample2.Shape[0], output2.Shape[0]);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    // ---------- helpers ----------

    private static SimpleTestNetwork BuildNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new SimpleTestNetwork(architecture);
        network.AddLayer(new DenseLayer<float>(8));
        network.AddLayer(new DenseLayer<float>(2));
        return network;
    }

    private static Tensor<float> MakeInput(int[] shape)
    {
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.1f;
        return new Tensor<float>(data, shape);
    }

    /// <summary>
    /// Minimal NeuralNetworkBase subclass for CompileForward testing.
    /// </summary>
    private sealed class SimpleTestNetwork : NeuralNetworkBase<float>
    {
        public SimpleTestNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
        }

        public override bool SupportsTraining => true;

        public void AddLayer(ILayer<float> layer) => AddLayerToCollection(layer);

        protected override void InitializeLayers() { }

        public override Tensor<float> Predict(Tensor<float> input)
        {
            // Use the compiled path so CompileForward's cached plan is
            // exercised on subsequent calls. Falls back to eager if
            // compilation is off or failed.
            return PredictCompiled(input);
        }

        public override void UpdateParameters(Vector<float> parameters) => SetParameters(parameters);

        public override void Train(Tensor<float> input, Tensor<float> expectedOutput)
            => TrainWithTape(input, expectedOutput);

        public override ModelMetadata<float> GetModelMetadata() =>
            new ModelMetadata<float>
            {
                Name = "SimpleTestNetwork",
                Version = "1.0",
                FeatureCount = Architecture.InputSize,
                Complexity = ParameterCount
            };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new SimpleTestNetwork(Architecture);
    }
}
