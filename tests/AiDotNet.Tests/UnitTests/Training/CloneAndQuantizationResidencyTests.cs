using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using System.Reflection;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("NonParallelIntegration")]
public class CloneAndQuantizationResidencyTests
{
    private static void RequireDevice(OpenClBackend backend)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(backend.IsAvailable, "The required OpenCL lane has no usable device.");
        Skip.IfNot(backend.IsAvailable, "No OpenCL device is available.");
    }

    [SkippableFact]
    public void HostTensorClone_WithGpuEngineRetainsCopyOnWrite()
    {
        var prior = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        AiDotNetEngine.Current = gpu;
        try
        {
            var source = new Tensor<float>(new float[] { 2, 4 }, new[] { 2 });
            Assert.False(source.HasPendingGpuData);
            var clone = CopyOnWriteCloneHelper.ShareTensor(source);
            Assert.True(CopyOnWriteCloneHelper.AreLiveCowPeers(source, clone));
            clone[0] = 9;
            Assert.Equal(2f, source[0]);
            Assert.Equal(9f, clone[0]);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [SkippableTheory] [InlineData(false, false)] [InlineData(true, false)] [InlineData(false, true)] [InlineData(true, true)]
    [Trait("Category", "GPU")]
    public void LayerClone_PreservesDeviceWeightsAndBuffers(bool batchNorm, bool copyOnWrite)
    {
        using var backend = new OpenClBackend(); RequireDevice(backend);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            LayerBase<float> source = batchNorm ? new BatchNormalizationLayer<float>(3) : new DenseLayer<float>(2);
            var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
            source.SetTrainingMode(false); source.Forward(input);
            var chunks = source.GetParameterStateChunks("$").ToArray();
            foreach (var chunk in chunks) ((IEngine)gpu).TensorCopy(gpu.TensorAddScalar(chunk.Tensor, .2f), chunk.Tensor);
            var expected = source.Forward(input).ToArray();
            LayerBase<float> clone; Tensor<float> output;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                clone = (LayerBase<float>)LayerCloning.Clone(source, copyOnWrite ? CloneOptions.CopyOnWrite : CloneOptions.Full);
                clone.SetTrainingMode(false); output = clone.Forward(input);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            Assert.Equal(expected, output.ToArray());
            var copied = clone.GetParameterStateChunks("$").ToArray();
            for (int i = 0; i < chunks.Length; i++)
            {
                Assert.NotSame(chunks[i].Tensor, copied[i].Tensor);
                var original = chunks[i].Tensor.ToArray();
                Assert.Equal(original, copied[i].Tensor.ToArray());
                ((IEngine)gpu).TensorCopy(gpu.TensorAddScalar(copied[i].Tensor, 1f), copied[i].Tensor);
                Assert.Equal(original, chunks[i].Tensor.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void SpectralNormalization_DevicePowerIterationMatchesCpuForwardAndGradients()
    {
        using var backend = new OpenClBackend(); RequireDevice(backend);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine(); AiDotNetEngine.Current = cpu;
            var input = new Tensor<float>(new float[] { .2f, -.4f, .7f, .6f, .1f, -.3f }, new[] { 2, 3 });
            var dense = new DenseLayer<float>(2, (IActivationFunction<float>)new IdentityActivation<float>());
            dense.Forward(input);
            var parameters = dense.GetTrainableParameters();
            for (int pIndex = 0; pIndex < parameters.Count; pIndex++)
                for (int i = 0; i < parameters[pIndex].Length; i++) parameters[pIndex][i] = MathF.Sin(i + .2f) * .3f;
            var layer = new SpectralNormalizationLayer<float>(dense, powerIterations: 2);
            layer.SetTrainingMode(true);
            var fields = new[] { "_u", "_v" }.Select(name => typeof(SpectralNormalizationLayer<float>)
                .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!).ToArray();
            var saved = fields.Select(field => ((Tensor<float>)field.GetValue(layer)!).Clone()).ToArray();
            var leaves = new[] { input }.Concat(parameters).ToArray();
            Tensor<float> expected; Dictionary<Tensor<float>, Tensor<float>> reference;
            using (var tape = new GradientTape<float>())
            {
                expected = layer.Forward(input);
                reference = tape.ComputeGradients(cpu.ReduceSum(expected, null), leaves);
            }
            for (int i = 0; i < fields.Length; i++) fields[i].SetValue(layer, saved[i]);
            AiDotNetEngine.Current = gpu;
            Tensor<float> actual; Dictionary<Tensor<float>, Tensor<float>> gradients;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                using var tape = new GradientTape<float>();
                actual = layer.Forward(input);
                gradients = tape.ComputeGradients(gpu.ReduceSum(actual, null), leaves);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            void Near(Tensor<float> a, Tensor<float> b)
            { for (int i = 0; i < a.Length; i++) Assert.InRange(MathF.Abs(a[i] - b[i]), 0, 2e-4f); }
            Near(expected, actual);
            foreach (var leaf in leaves) Near(reference[leaf], gradients[leaf]);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    public void GraFPrint_LossEvaluationRestoresBatchNormStatistics()
    {
        var bn = new BatchNormalizationLayer<float>(8);
        var architecture = new NeuralNetworkArchitecture<float>(
            AiDotNet.Enums.InputType.ThreeDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 2, inputWidth: 2, inputDepth: 2, outputSize: 8,
            layers: new List<ILayer<float>> { new FlattenLayer<float>(), bn });
        using var model = new AiDotNet.Audio.Fingerprinting.GraFPrint<float>(architecture,
            new AiDotNet.Audio.Fingerprinting.GraFPrintOptions { EmbeddingDim = 8, DropoutRate = 0 });
        var first = new Tensor<float>(Enumerable.Range(0, 16).Select(i => .1f + i * .07f).ToArray(), new[] { 2, 2, 2, 2 });
        var second = new Tensor<float>(Enumerable.Range(0, 16).Select(i => MathF.Cos(i) * .3f).ToArray(), new[] { 2, 2, 2, 2 });
        var mean = bn.GetRunningMean().ToArray(); var variance = bn.GetRunningVariance().ToArray();
        float a = model.ComputeContrastiveLoss(first, second);
        float b = model.ComputeContrastiveLoss(first, second);
        Assert.True(!float.IsNaN(a) && !float.IsInfinity(a)); Assert.Equal(a, b);
        Assert.Equal(mean, bn.GetRunningMean().ToArray()); Assert.Equal(variance, bn.GetRunningVariance().ToArray());
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void GeneratedTensorClonesAndRetainedGradients_AreIndependentResidentSnapshots()
    {
        using var backend = new OpenClBackend(); RequireDevice(backend);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            using var network = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 3, outputSize: 2, layers: new List<ILayer<float>> { new DenseLayer<float>(2) }));
            var type = typeof(NeuralNetworkBase<float>);
            var source = gpu.TensorAddScalar(new Tensor<float>(new[] { 2, 3 }), 2f);
            var destination = new Tensor<float>(new[] { 2, 3 });
            var copies = new List<Tensor<float>>();
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                foreach (var method in new[] { "CloneGeneratedTrainableTensor", "CloneRequiredGeneratedTrainableTensor" })
                    copies.Add((Tensor<float>)type.GetMethod(method, BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, new object[] { source })!);
                type.GetMethod("CopyGeneratedTrainableTensorValues", BindingFlags.NonPublic | BindingFlags.Static)!
                    .Invoke(null, new object[] { source, destination, "weight" });
                copies.Add(destination);
                type.GetMethod("BeginStreamingGradientRetention", BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(network, null);
                type.GetMethod("RetainStreamingGradient", BindingFlags.NonPublic | BindingFlags.Instance)!
                    .Invoke(network, new object[] { destination, source });
                var retained = (Dictionary<Tensor<float>, Tensor<float>>)type.GetField("_retainedStreamingGrads", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(network)!;
                copies.Add(retained[destination]);
                // Optimizer scratch may be reused immediately after retention.
                ((IEngine)gpu).TensorCopy(gpu.TensorAddScalar(source, 5f), source);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            foreach (var copy in copies) Assert.All(copy.ToArray(), value => Assert.Equal(2f, value));
            Assert.All(source.ToArray(), value => Assert.Equal(7f, value));
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void BidirectionalConstruction_ClonesResidentRecurrentWeightsWithoutFlattening()
    {
        using var backend = new OpenClBackend(); RequireDevice(backend);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            var source = new LSTMLayer<float>(2); var input = new Tensor<float>(new[] { 1, 3, 2 });
            source.Forward(input);
            foreach (var tensor in source.GetTrainableParameters())
                ((IEngine)gpu).TensorCopy(gpu.TensorAddScalar(tensor, .1f), tensor);
            using var scope = GpuResidencyScope.Begin(captureOperations: true);
            var bidirectional = new BidirectionalLayer<float>(source, activationFunction: (IActivationFunction<float>?)null);
            bidirectional.Forward(input);
            Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableTheory] [InlineData(1, false)] [InlineData(4, false)] [InlineData(8, false)]
    [InlineData(1, true)] [InlineData(4, true)] [InlineData(8, true)]
    [Trait("Category", "GPU")]
    public void QaLora_QuantizedForwardAndStraightThroughGradientsStayOnDevice(int bits, bool fullyConnected)
    {
        using var backend = new OpenClBackend(); RequireDevice(backend);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine(); AiDotNetEngine.Current = cpu;
            var input = new Tensor<float>(new float[] { .2f, -.3f, .4f, .6f, .1f, -.2f }, new[] { 2, 3 });
            LayerBase<float> baseLayer = fullyConnected
                ? new FullyConnectedLayer<float>(2, (IActivationFunction<float>)new IdentityActivation<float>())
                : new DenseLayer<float>(2, (IActivationFunction<float>)new IdentityActivation<float>());
            baseLayer.Forward(input);
            var adapter = new QALoRAAdapter<float>(baseLayer, rank: 2, quantizationBits: bits, groupSize: 4);
            var factors = adapter.LoRALayer.GetTrainableParameters();
            int cursor = 0;
            foreach (var factor in factors) for (int i = 0; i < factor.Length; i++, cursor++)
                factor[i] = cursor < 4 ? 0 : MathF.Sin(cursor + .1f) * .3f;
            var originals = factors.Select(t => t.ToArray()).ToArray();
            var values = originals.SelectMany(x => x).ToArray(); var quantized = new float[values.Length];
            double high = Math.Pow(2, bits - 1) - 1, low = -Math.Pow(2, bits - 1);
            for (int start = 0; start < values.Length; start += 4)
            {
                int end = Math.Min(start + 4, values.Length);
                float maximum = values.Skip(start).Take(end - start).Max(MathF.Abs);
                float scale = (maximum == 0 ? 1e-8f : maximum) / (float)Math.Max(1, high);
                for (int i = start; i < end; i++) quantized[i] = (float)Math.Max(low, Math.Min(high, Math.Round(values[i] / scale))) * scale;
            }
            var qa = new Tensor<float>(quantized.Take(factors[0].Length).ToArray(), factors[0].Shape.ToArray());
            var qb = new Tensor<float>(quantized.Skip(factors[0].Length).ToArray(), factors[1].Shape.ToArray());
            Tensor<float> expected; Dictionary<Tensor<float>, Tensor<float>> reference;
            using (var tape = new GradientTape<float>())
            {
                expected = cpu.TensorAdd(baseLayer.Forward(input), adapter.LoRALayer.ForwardWithFactors(input, qa, qb));
                reference = tape.ComputeGradients(cpu.ReduceSum(expected, null), new[] { input, qa, qb });
            }
            AiDotNetEngine.Current = gpu;
            Tensor<float> actual; Dictionary<Tensor<float>, Tensor<float>> gradients;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                using var tape = new GradientTape<float>(); actual = adapter.Forward(input);
                gradients = tape.ComputeGradients(gpu.ReduceSum(actual, null), new[] { input, factors[0], factors[1] });
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            void Near(Tensor<float> a, Tensor<float> b)
            { for (int i = 0; i < a.Length; i++) Assert.InRange(MathF.Abs(a[i] - b[i]), 0, 5e-5f); }
            Near(expected, actual); Near(reference[input], gradients[input]);
            Near(reference[qa], gradients[factors[0]]); Near(reference[qb], gradients[factors[1]]);
            for (int i = 0; i < factors.Count; i++) Assert.Equal(originals[i], factors[i].ToArray());
            var stats = adapter.GetQuantizationStats();
            Assert.Equal(3, stats.numGroups); Assert.True(!double.IsNaN(stats.averageError) && !double.IsInfinity(stats.averageError));
            ILayer<float> merged;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                merged = adapter.MergeToOriginalLayer();
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            var baseWeights = baseLayer.GetTrainableParameters()[0].ToArray();
            var mergedValues = new float[baseWeights.Length];
            for (int row = 0; row < 3; row++) for (int col = 0; col < 2; col++)
            {
                float delta = 0;
                for (int rank = 0; rank < 2; rank++) delta += originals[0][row * 2 + rank] * originals[1][rank * 2 + col];
                int index = fullyConnected ? col * 3 + row : row * 2 + col;
                mergedValues[index] = baseWeights[index] + delta * adapter.LoRALayer.Scaling;
            }
            for (int start = 0; start < mergedValues.Length; start += 4)
            {
                int end = Math.Min(start + 4, mergedValues.Length);
                float maximum = mergedValues.Skip(start).Take(end - start).Max(MathF.Abs);
                float scale = (maximum == 0 ? 1e-8f : maximum) / (float)Math.Max(1, high);
                for (int i = start; i < end; i++) mergedValues[i] = (float)Math.Max(low, Math.Min(high, Math.Round(mergedValues[i] / scale))) * scale;
            }
            var mergedParameters = ((LayerBase<float>)merged).GetTrainableParameters();
            Near(new Tensor<float>(mergedValues, mergedParameters[0].Shape.ToArray()), mergedParameters[0]);
            Assert.Equal(baseLayer.GetTrainableParameters()[1].ToArray(), mergedParameters[1].ToArray());
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
