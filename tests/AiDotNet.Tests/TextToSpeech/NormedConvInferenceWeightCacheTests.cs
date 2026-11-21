using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class NormedConvInferenceWeightCacheTests
{
    private static NormedConv1DLayer<float> Create()
    {
        var layer = new NormedConv1DLayer<float>(2, 3, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Weight);
        layer.LoadTorchWeights(Enumerable.Range(0, 18).Select(i => .1 + .2 * Math.Sin(i * .7)).ToArray(),
            new[] { 1.0, 1.2, .9 }, new double[3]);
        layer.SetTrainingMode(false);
        return layer;
    }

    [Fact]
    public void CachedWeightsSurviveThePredictionArenaAndScope()
    {
        using var layer = Create();
        Tensor<float> kernel;
        float[] expected;
        using (TensorArena.Create())
        using (TensorScope.Begin())
        {
            kernel = layer.Kernel();
            expected = kernel.ToArray();
        }
        using (TensorArena.Create())
        using (TensorScope.Begin())
        {
            var overwrite = TensorAllocator.RentUninitialized<float>(new[] { 3, 6 });
            overwrite.Fill(123);
            Assert.Same(kernel, layer.Kernel());
            Assert.Equal(expected, layer.Kernel().ToArray());
        }
        Assert.Equal(expected, layer.Kernel().ToArray());
    }

    [Fact]
    public void InPlaceGainUpdatesAndCheckpointLoadsRebuildTheWeights()
    {
        using var layer = Create();
        var first = layer.Kernel();
        var values = first.ToArray();
        var gain = layer.GetTrainableParameters().ToArray()[1];
        AiDotNetEngine.Current.TensorMultiplyScalarInPlace(gain, 1.5f);
        var updated = layer.Kernel();
        Assert.NotSame(first, updated);
        var actual = updated.ToArray();
        for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - values[i] * 1.5f), 0, 1e-6);
        var checkpoint = layer.GetParameters();
        checkpoint[0] *= 2;
        layer.SetParameters(checkpoint);
        Assert.NotSame(updated, layer.Kernel());
    }

    [Fact]
    public void InferenceModeWritesThroughGainViewsInvalidateTheCachedWeights()
    {
        using var layer = Create();
        var first = layer.Kernel();
        var before = first.ToArray();
        var gain = layer.GetTrainableParameters()[1];
        int version = gain.Version;
        using (new InferenceModeScope<float>())
        {
            var alias = AiDotNetEngine.Current.Reshape(gain, new[] { 1, gain.Length });
            AiDotNetEngine.Current.TensorMultiplyScalarInPlace(alias, 1.5f);
            Assert.Equal(version, gain.Version);
            var current = layer.Kernel();
            Assert.NotSame(first, current);
            var actual = current.ToArray();
            for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - before[i] * 1.5f), 0, 1e-6);
        }
    }

    [Fact]
    public void PredictAfterSetParametersInInferenceModeMatchesAFreshModel()
    {
        using var candidate = HiFiGanFusedExecutionTests.Create();
        using var reference = HiFiGanFusedExecutionTests.Create();
        var mel = new Tensor<float>(new[] { 1, 8, 4 }, new Vector<float>(Enumerable.Range(0, 32).Select(i => .1f * (i + 1))));
        candidate.Predict(mel);
        var parameters = candidate.GetParameters();
        parameters[0] += .5f;
        using (new InferenceModeScope<float>()) candidate.SetParameters(parameters);
        reference.SetParameters(parameters);
        var expected = reference.Predict(mel).ToArray();
        var actual = candidate.Predict(mel).ToArray();
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PredictAfterTrainingMatchesAFreshModelWithTheUpdatedParameters(bool compile)
    {
        bool previous = TensorCodecOptions.Current.EnableCompilation;
        try
        {
            TensorCodecOptions.Current.EnableCompilation = compile;
            using var candidate = HiFiGanFusedExecutionTests.Create();
            using var reference = HiFiGanFusedExecutionTests.Create();
            var audio = new Tensor<float>(new[] { 512 }, new Vector<float>(
                Enumerable.Range(0, 512).Select(i => (float)(.5 * Math.Sin(i * .31)))));
            var mel = candidate.ComputeMel(audio);
            candidate.Predict(mel);
            candidate.Train(mel, audio);
            reference.SetParameters(candidate.GetParameters());
            var expected = reference.Predict(mel).ToArray();
            var actual = candidate.Predict(mel).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-5);
        }
        finally { TensorCodecOptions.Current.EnableCompilation = previous; }
    }

    private static float[] Gradients(NormedConv1DLayer<float> layer)
    {
        var parameters = layer.GetTrainableParameters().Take(2).ToArray();
        using var tape = new GradientTape<float>();
        var loss = AiDotNetEngine.Current.ReduceSum(layer.Kernel(), null);
        var gradients = tape.ComputeGradients(loss, sources: parameters);
        return parameters.SelectMany(parameter => gradients[parameter].ToArray()).ToArray();
    }

    [Fact]
    public void ARecordingTapeComputesTheSameGradientsAfterTheEvalCacheWasWarmed()
    {
        using var candidate = Create();
        using var reference = Create();
        candidate.Kernel();
        reference.SetTrainingMode(true);
        var expected = Gradients(reference);
        var actual = Gradients(candidate);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6);
    }

    [Fact]
    public void ATrainingGraphRetainsParameterEdgesAfterTheEvalCacheWasWarmed()
    {
        using var candidate = Create();
        using var reference = Create();
        candidate.Kernel();
        reference.SetTrainingMode(true);
        var parameters = candidate.GetTrainableParameters().Take(2).ToArray();
        var expectedGradients = Gradients(reference);
        var before = parameters.SelectMany(parameter => parameter.ToArray()).ToArray();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, AiDotNetEngine.Current.ReduceSum(candidate.Kernel(), null));
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .01);
            plan.Step();
        }
        var after = parameters.SelectMany(parameter => parameter.ToArray()).ToArray();
        for (int i = 0; i < after.Length; i++)
            Assert.InRange(Math.Abs(after[i] - (before[i] - .01f * expectedGradients[i])), 0, 1e-6);
    }
}
