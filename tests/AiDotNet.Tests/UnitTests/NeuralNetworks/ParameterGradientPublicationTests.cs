using System.Collections.Generic;
using System.IO;
using System.Reflection;
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

public class ParameterGradientPublicationTests
{
    [Fact]
    public void ExtraTrainableLayer_ParametersArePublished_WhenItIsAnSsmLayer()
    {
        // REPLACES FusedCapability_IncludesEagerOnlyExtraTrainableLayer, which asserted that a
        // private LayersSupportFusedCompiledTraining() existed and returned false for a network
        // carrying an RG-LRU extra layer. Commit 816384a6a2 ("fix(training): record RG-LRU in
        // compiled plans") made RG-LRU record as ordinary graph operations and deliberately deleted
        // that whole capability check, updating FusedOptimizerIntegrationTests and
        // RecurrentGemmaTrainingRegressionTests but not this file. The old assertion could only
        // fail: it reflected for a method that no longer exists, and its premise -- that an RG-LRU
        // layer forces the network eager-only -- is no longer true. 13 tests now cover the new
        // contract directly.
        //
        // What is still worth asserting here, and belongs in THIS file, is that such a layer's
        // parameters actually reach the publication surface rather than being silently dropped.
        using var network = new PublicationNetwork(useEagerOnlyExtra: true);

        var parameters = network.TrainableTensors;
        Assert.NotEmpty(parameters);

        var gradients = new Dictionary<Tensor<float>, Tensor<float>>();
        int expectedLength = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            gradients[parameters[i]] = FilledLike(parameters[i], i + 1);
            expectedLength += parameters[i].Length;
        }

        network.Publish(gradients);

        Assert.Equal(expectedLength, network.GetParameterGradients().Length);
    }

    [Fact]
    public void Publication_UsesCanonicalLayerExtraLayerAndRawTensorOrder()
    {
        using var network = new PublicationNetwork();
        var parameters = network.TrainableTensors;
        var gradients = new Dictionary<Tensor<float>, Tensor<float>>();
        int expectedLength = 0;

        for (int i = 0; i < parameters.Count; i++)
        {
            gradients[parameters[i]] = FilledLike(parameters[i], i + 1);
            expectedLength += parameters[i].Length;
        }

        network.Publish(gradients);
        var flat = network.GetParameterGradients();

        Assert.Equal(expectedLength, flat.Length);
        int offset = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            for (int j = 0; j < parameters[i].Length; j++)
                Assert.Equal(i + 1, flat[offset++]);
        }
    }

    [Fact]
    public void Publication_ReplacesPreviousSnapshotInsteadOfRetainingStaleLayerSlices()
    {
        using var network = new PublicationNetwork();
        var parameters = network.TrainableTensors;
        network.Publish(parameters.ToDictionary(
            parameter => parameter,
            parameter => FilledLike(parameter, 7)));

        var rawOnly = new Dictionary<Tensor<float>, Tensor<float>>
        {
            [parameters[^1]] = FilledLike(parameters[^1], 11)
        };
        network.Publish(rawOnly);
        var flat = network.GetParameterGradients();

        int rawOffset = flat.Length - parameters[^1].Length;
        for (int i = 0; i < rawOffset; i++) Assert.Equal(0, flat[i]);
        for (int i = rawOffset; i < flat.Length; i++) Assert.Equal(11, flat[i]);
    }

    // A tape step's device gradients are published as a snapshot: the optimizer that runs after publication scales
    // them in place (AdamOptimizer's own global-norm clip), and the surface must keep the values that were published,
    // as the CPU path's copy does.
    [SkippableFact]
    public void DevicePublication_IsASnapshot_NotAnAliasOfGradientsScaledInPlaceAfterwards()
    {
        using var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        if (System.Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpu;
        try
        {
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            using var network = new PublicationNetwork();
            var parameters = network.TrainableTensors;
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var grads = parameters.ToDictionary(p => p, p => engine.TensorMultiplyScalar(FilledLike(p, 1), 7f));
            network.Publish(grads);
            foreach (var g in grads.Values) engine.TensorMultiplyScalarInPlace(g, 0.5f);
            var flat = network.GetParameterGradients();
            int rawOffset = flat.Length - parameters[^1].Length;
            Assert.True(rawOffset > 0);
            for (int i = 0; i < rawOffset; i++) Assert.Equal(7f, flat[i]);
        }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Publication_ZeroesAReusedLayerSliceThatGetsNoGradientThisStep()
    {
        // Two publishes with no read in between: the second reuses the layer's buffer from the first. The layer's
        // bias gets no gradient the second time, so it must read as zero, not the first publish's 7.
        using var network = new PublicationNetwork();
        var parameters = network.TrainableTensors;
        network.Publish(parameters.ToDictionary(
            parameter => parameter,
            parameter => FilledLike(parameter, 7)));

        network.Publish(new Dictionary<Tensor<float>, Tensor<float>>
        {
            [parameters[0]] = FilledLike(parameters[0], 11)
        });
        var flat = network.GetParameterGradients();

        int weights = parameters[0].Length;
        int bias = parameters[1].Length;
        for (int i = 0; i < weights; i++) Assert.Equal(11, flat[i]);
        for (int i = weights; i < weights + bias; i++) Assert.Equal(0, flat[i]);
        for (int i = weights + bias; i < flat.Length; i++) Assert.Equal(0, flat[i]);
    }

    [Fact]
    public void FlatPublication_RejectsAParameterMisalignedVector()
    {
        using var network = new PublicationNetwork();

        var error = Assert.Throws<ArgumentException>(() =>
            network.PublishFlat(new Vector<float>((int)network.ParameterCount - 1)));

        Assert.Contains("must match ParameterCount", error.Message);
    }

    [Fact]
    public void GradientClippingSelection_ExcludesExtraLayersAndRawTensors()
    {
        using var network = new PublicationNetwork();

        var selected = network.TrainableTensors;
        var clipped = network.LayerOwnedTensorsForClipping;

        Assert.NotEmpty(clipped);
        Assert.True(clipped.Count < selected.Count);
        Assert.All(clipped, parameter =>
            Assert.Contains(selected, candidate => ReferenceEquals(candidate, parameter)));
        Assert.All(network.ExtraLayerTensors, parameter =>
            Assert.DoesNotContain(clipped, candidate => ReferenceEquals(candidate, parameter)));
        Assert.DoesNotContain(clipped, candidate => ReferenceEquals(candidate, network.RawParameter));
    }

    [Fact]
    public void StreamingRetention_MidStepReadOnUnreadSurface_DoesNotMixSteps()
    {
        var probe = new MidStepGradientProbeRegularization();
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        arch.RandomSeed = 19;

        var sgdProbe = new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AiDotNet.Models.Options.StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
                Regularization = probe,
            });
        var sgdTwin = new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AiDotNet.Models.Options.StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });

        using var net = new FeedForwardNeuralNetwork<float>(arch, sgdProbe);
        using var twin = new FeedForwardNeuralNetwork<float>(arch, sgdTwin);
        net.StreamingTraining = StreamingTrainingMode.ForceOn;
        twin.StreamingTraining = StreamingTrainingMode.ForceOn;
        var initial = net.GetParameters();
        twin.UpdateParameters(initial);
        probe.Target = net;

        var x = new Tensor<float>(new[] { 2, 4 });
        var y1 = new Tensor<float>(new[] { 2, 2 });
        var y2 = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < x.Length; i++) x[i] = (i + 1) * 0.25f;
        for (int i = 0; i < y1.Length; i++) { y1[i] = 1.5f * (i + 1); y2[i] = -3.0f * (i + 1); }

        // Step 1: leave net's published surface unread so its layers hold references to the published snapshot bank.
        net.Train(x, y1);
        twin.Train(x, y1);
        float[] expectedStep1 = twin.GetParameterGradients().ToArray();

        // Step 2: probe reads net.GetParameterGradients() right after the first parameter's RetainStreamingGradient
        // runs in backward, before other parameters or EndStreamingGradientRetention have run.
        probe.CaptureNextCall = true;
        net.Train(x, y2);
        twin.Train(x, y2);
        float[] expectedStep2 = twin.GetParameterGradients().ToArray();

        Assert.NotNull(probe.CapturedMidStep);
        if (probe.CapturedMidStep is { } midStep)
        {
            Assert.Equal(expectedStep1.Length, midStep.Length);
            bool stepsDiffer = false;
            for (int i = 0; i < expectedStep1.Length; i++)
            {
                if (System.Math.Abs(expectedStep1[i] - expectedStep2[i]) > 1e-4f) stepsDiffer = true;
                Assert.Equal(expectedStep1[i], midStep[i], 5);
            }
            Assert.True(stepsDiffer, "Step 1 and Step 2 gradients were identical, making the mid-step check vacuous.");
        }

        float[] actualStep2 = net.GetParameterGradients().ToArray();
        for (int i = 0; i < expectedStep2.Length; i++)
            Assert.Equal(expectedStep2[i], actualStep2[i], 5);
    }

    private sealed class MidStepGradientProbeRegularization : IRegularization<float, Tensor<float>, Tensor<float>>
    {
        public FeedForwardNeuralNetwork<float>? Target { get; set; }
        public bool CaptureNextCall { get; set; }
        public float[]? CapturedMidStep { get; private set; }

        public Matrix<float> Regularize(Matrix<float> data) => data;
        public Vector<float> Regularize(Vector<float> data) => data;

        public Tensor<float> Regularize(Tensor<float> gradient, Tensor<float> coefficients)
        {
            if (CaptureNextCall && Target is { } target)
            {
                CaptureNextCall = false;
                CapturedMidStep = target.GetParameterGradients().ToArray();
            }
            return gradient;
        }

        public RegularizationOptions GetOptions()
            => new RegularizationOptions { Type = RegularizationType.None, Strength = 0.0 };
    }

    private static Tensor<float> FilledLike(Tensor<float> parameter, float value)
    {
        var result = new Tensor<float>(parameter.Shape.ToArray());
        for (int i = 0; i < result.Length; i++) result[i] = value;
        return result;
    }

    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed class PublicationNetwork : NeuralNetworkBase<float>
    {
        private readonly LayerBase<float> _extraLayer;
        private readonly Tensor<float> _rawParameter;

        public PublicationNetwork(bool useEagerOnlyExtra = false)
            : base(CreateArchitecture(), new MeanSquaredErrorLoss<float>())
        {
            var sample = new Tensor<float>(new[] { 1, 3 });
            var primaryLayer = new DenseLayer<float>(2);
            _ = primaryLayer.Forward(sample);
            Layers.Add(primaryLayer);

            if (useEagerOnlyExtra)
            {
                _extraLayer = new RealGatedLinearRecurrenceLayer<float>(
                    sequenceLength: 1, modelDimension: 3);
            }
            else
            {
                _extraLayer = new DenseLayer<float>(2);
                _ = _extraLayer.Forward(sample);
            }
            _rawParameter = new Tensor<float>(new[] { 2 });
        }

        public IReadOnlyList<Tensor<float>> TrainableTensors => CollectModelTrainableTensors();

        public IReadOnlyList<Tensor<float>> LayerOwnedTensorsForClipping
            => CollectLayerOwnedTrainableTensorsForClipping(TrainableTensors);

        public IReadOnlyList<Tensor<float>> ExtraLayerTensors => _extraLayer.GetTrainableParameters();

        public Tensor<float> RawParameter => _rawParameter;

        public void Publish(IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients)
            => PublishParameterGradients(gradients);

        public void PublishFlat(Vector<float> gradients) => PublishFlatParameterGradients(gradients);

        protected override IEnumerable<LayerBase<float>?> GetExtraTrainableLayers()
        {
            yield return _extraLayer;
        }

        protected override IEnumerable<Tensor<float>> GetExtraTrainableTensors()
        {
            yield return _rawParameter;
        }

        protected override void InitializeLayers()
        {
        }

        public override void UpdateParameters(Vector<float> parameters) => SetParameters(parameters);

        public override ModelMetadata<float> GetModelMetadata() => new() { Name = "PublicationNetwork" };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new PublicationNetwork();

        private static NeuralNetworkArchitecture<float> CreateArchitecture()
            => new(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 3,
                outputSize: 2);
    }
}
