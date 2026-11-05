// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// What a warm GPU <c>Predict</c> costs beyond its kernels. #2385: a b1 MLP <c>Predict</c> took 0.9-1.5 ms on CUDA
/// against PyTorch's 0.2 ms, a fixed per-call floor that a result download or per-call host work explains and the
/// kernels do not. #2384: Transformer and ViT inference grew linearly with batch size (85x PyTorch at b128), the
/// signature of a forward that issues more kernels for a bigger batch. Both are counted here through
/// <see cref="GpuLaunchProbe"/>, which every GPU backend increments at its dispatch point, so they run on any GPU
/// backend; the CUDA timings themselves are the parity harness's (#2381). Skips with no GPU backend.
/// </summary>
// Sets AiDotNetEngine.Current and reads the process-wide GpuLaunchProbe: run apart from every other test class.
[Collection("EngineCurrentGlobalState")]
[Trait("Category", "GPU")]
public class GpuPredictTransferAndLaunchTests
{
    private const int WarmCalls = 3, MeasuredCalls = 10;
    private readonly ITestOutputHelper _output;
    public GpuPredictTransferAndLaunchTests(ITestOutputHelper output) => _output = output;

    [SkippableFact(Timeout = 300000)]
    public async Task WarmMlpPredict_WhoseResultIsNotRead_DownloadsNothing_AndUploadsOnlyItsInput()
    {
        await Task.Yield();
        await OnGpu(() =>
        {
            var network = BuildMlp();
            var input = RandomInput(new[] { 1, 784 }, 2385);
            for (int i = 0; i < WarmCalls; i++) network.Predict(input);

            var (launches, readbacks, readbackBytes, uploads, uploadBytes) = Measure(() => network.Predict(input));
            // Host time per call, for the record (#2385's target is CUDA timing, measured by the parity harness).
            var watch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < 200; i++) network.Predict(input);
            double unreadMs = watch.Elapsed.TotalMilliseconds / 200;
            watch.Restart();
            for (int i = 0; i < 200; i++) network.Predict(input).ToArray();
            _output.WriteLine($"warm b1 Predict: {unreadMs:F3} ms unread, {watch.Elapsed.TotalMilliseconds / 200:F3} ms read");
            _output.WriteLine($"per call: {launches} launches, {readbacks} readbacks ({readbackBytes} B), " +
                $"{uploads} uploads ({uploadBytes} B)");
            Assert.True(readbackBytes == 0,
                $"a warm Predict whose result was not read downloaded {readbackBytes} B per call in {readbacks} readbacks: " +
                string.Join("; ", GpuLaunchProbe.ReadbackSites));
            // The input is a host tensor, so its bytes go up once; the weights are already on the device.
            long inputBytes = input.Length * sizeof(float);
            Assert.True(uploadBytes <= inputBytes,
                $"a warm Predict uploaded {uploadBytes} B per call, more than its {inputBytes} B input: " +
                string.Join("; ", GpuLaunchProbe.UploadSites));
        });
    }

    public enum BatchedModel { Transformer, VisionTransformer }

    public static IEnumerable<object[]> BatchedModels()
    {
        yield return new object[] { BatchedModel.Transformer };
        yield return new object[] { BatchedModel.VisionTransformer };
    }

    [SkippableTheory(Timeout = 600000)]
    [MemberData(nameof(BatchedModels))]
    public async Task BatchedPredict_LaunchCountDoesNotGrowWithTheBatch(BatchedModel model)
    {
        await Task.Yield();
        await OnGpu(() =>
        {
            bool transformer = model == BatchedModel.Transformer;
            NeuralNetworkBase<float> network = transformer ? BuildTransformer() : BuildVisionTransformer();
            int[] sample = transformer ? new[] { 32, 32 } : new[] { 3, 32, 32 };
            long Launches(int batch)
            {
                var input = RandomInput(Prepend(batch, sample), 2384 + batch);
                for (int i = 0; i < WarmCalls; i++) network.Predict(input);
                return Measure(() => network.Predict(input)).Launches;
            }

            long single = Launches(1), batched = Launches(128);
            _output.WriteLine($"{model}: {single} launches per b1 Predict, {batched} per b128 Predict");
            Assert.True(single > 0, $"{model}: a b1 Predict launched no GPU kernel, so it did not run on the GPU");
            // The same layers run whatever the batch, each as one batched kernel. The count still moves a little with
            // the batch (measured on OpenCL: Transformer 29-35 launches from b1 to b128, ViT 81-107) because a bigger
            // GEMM can pick a kernel with another launch shape; a part of the forward looped per sample would add at
            // least 127 launches per looped op at b128, so twice the b1 count separates the two with room to spare.
            Assert.True(batched < 2 * single,
                $"{model}: a b128 Predict launched {batched} kernels against {single} for b1, so part of the forward " +
                "runs once per sample");
        });
    }

    private async Task OnGpu(Action body)
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) { _output.WriteLine($"No GPU engine: {ex.GetType().Name}: {ex.Message}"); }
        var previous = AiDotNetEngine.Current;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable, "No GPU backend available.");
            AiDotNetEngine.Current = gpu;
            body();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }
    }

    private static (long Launches, long Readbacks, long ReadbackBytes, long Uploads, long UploadBytes) Measure(
        Action call)
    {
        bool priorCapture = GpuLaunchProbe.CaptureReadbackSites;
        GpuLaunchProbe.CaptureReadbackSites = true;
        try
        {
            GpuLaunchProbe.Reset();
            long before = GpuLaunchProbe.Count;
            for (int i = 0; i < MeasuredCalls; i++) call();
            return ((GpuLaunchProbe.Count - before) / MeasuredCalls,
                GpuLaunchProbe.Readbacks / MeasuredCalls,
                GpuLaunchProbe.ReadbackBytes / MeasuredCalls,
                GpuLaunchProbe.Uploads / MeasuredCalls,
                GpuLaunchProbe.UploadBytes / MeasuredCalls);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = priorCapture;
        }
    }

    private static int[] Prepend(int batch, int[] sample)
    {
        var shape = new int[sample.Length + 1];
        shape[0] = batch;
        Array.Copy(sample, 0, shape, 1, sample.Length);
        return shape;
    }

    private static Tensor<float> RandomInput(int[] shape, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var input = new Tensor<float>(shape);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return input;
    }

    // The parity harness's rows (benchmarks/AiDotNet.PyTorchParity), so the counts describe the measured models.
    private static FeedForwardNeuralNetwork<float> BuildMlp()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(512, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(128, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 784,
            outputSize: 10,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch, lossFunction: new CrossEntropyWithLogitsLoss<float>());
    }

    private static Transformer<float> BuildTransformer()
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 128,
            inputSize: 32,
            outputSize: 10,
            dropoutRate: 0.0,
            maxSequenceLength: 32,
            vocabularySize: 0,
            usePositionalEncoding: false,
            sequencePooling: SequencePoolingMode.MeanPool);
        return new Transformer<float>(arch, lossFunction: new CrossEntropyWithLogitsLoss<float>());
    }

    private static VisionTransformer<float> BuildVisionTransformer()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 10);
        return new VisionTransformer<float>(arch, 32, 32, 3, patchSize: 4, numClasses: 10,
            lossFunction: new CrossEntropyWithLogitsLoss<float>(),
            options: new VisionTransformerOptions { HiddenDim = 192, NumLayers = 6, NumHeads = 3, MlpDim = 768 });
    }
}
