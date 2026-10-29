using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// #2382: MobileNetV2 opted out of the fused compiled training step because its loss differed from the eager tape's,
/// which read as the plan not preserving the inverted-residual block. The block was fine: the eager path froze the
/// top-level batch-norm statistics during training (an override added in #2295; Sandler et al. 2018 and torchvision
/// train them) and the compiled step did not. One fused step must match the eager tape: same loss, same update.
/// </summary>
[Collection("NonParallelIntegration")]
public class MobileNetV2FusedParityTests
{
    private const int Batch = 4;
    private const int Size = 8;
    private const int Classes = 4;
    private readonly ITestOutputHelper _output;

    public MobileNetV2FusedParityTests(ITestOutputHelper output) => _output = output;

    private static List<ILayer<float>> Layers(string variant)
    {
        var layers = new List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1,
                activationFunction: new IdentityActivation<float>()),
        };
        if (variant != "conv-relu6") layers.Add(new BatchNormalizationLayer<float>());
        if (variant != "conv-bn") layers.Add(new ActivationLayer<float>(activationFunction: new ReLU6Activation<float>()));
        switch (variant)
        {
            case "stem":
            case "conv-bn":
            case "conv-relu6":
                break;
            case "residual":
                layers.Add(new InvertedResidualBlock<float>(outChannels: 8, expansionRatio: 2, stride: 1,
                    activationFunction: new ReLU6Activation<float>()));
                break;
            case "stride2":
                layers.Add(new InvertedResidualBlock<float>(outChannels: 12, expansionRatio: 2, stride: 2,
                    activationFunction: new ReLU6Activation<float>()));
                break;
            case "noexpand":
                layers.Add(new InvertedResidualBlock<float>(outChannels: 8, expansionRatio: 1, stride: 1,
                    activationFunction: new ReLU6Activation<float>()));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(variant));
        }
        layers.Add(new AdaptiveAveragePoolingLayer<float>(1, 1));
        layers.Add(new FlattenLayer<float>());
        layers.Add(new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()));
        return layers;
    }

    private static NeuralNetworkBase<float> Build(string variant)
    {

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Size, inputWidth: Size, inputDepth: 3, outputSize: Classes,
            layers: Layers(variant));

        return new MobileNetV2Network<float>(arch,
            new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha100, Classes, Size, Size),
            options: new MobileNetV2Options { DisableFusedOptimizerStep = false });
    }

    private static Tensor<float> Seeded(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Theory]
    [InlineData("stem")]
    [InlineData("conv-bn")]
    [InlineData("conv-relu6")]
    [InlineData("residual")]
    [InlineData("stride2")]
    [InlineData("noexpand")]
    public void One_fused_step_matches_the_eager_tape(string variant)
    {
        var prior = AiDotNetEngine.Current;
        var priorOptions = TensorCodecOptions.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var x = Seeded(11, Batch, 3, Size, Size);
            var y = new Tensor<float>(new[] { Batch, Classes });
            for (int b = 0; b < Batch; b++) y[b, b % Classes] = 1f;

            var fused = Build(variant);
            var eager = Build(variant);
            fused.Predict(x);   // materialise the lazy weights
            eager.Predict(x);
            var init = fused.GetParameters();
            eager.UpdateParameters(init);
            var before = fused.Predict(x).ToArray();
            var beforeEager = eager.Predict(x).ToArray();
            double predDiff = 0;
            for (int i = 0; i < before.Length; i++) predDiff = Math.Max(predDiff, Math.Abs(before[i] - beforeEager[i]));
            Assert.True(predDiff == 0, $"{variant}: the parameter copy left the two models {predDiff:E3} apart");
            fused.SetTrainingMode(true);
            eager.SetTrainingMode(true);

            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            fused.Train(x, y);
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            eager.Train(x, y);

            double lossFused = Convert.ToDouble(fused.GetLastLoss());
            double lossEager = Convert.ToDouble(eager.GetLastLoss());
            var p0 = init.ToArray();
            var pf = fused.GetParameters().ToArray();
            var pe = eager.GetParameters().ToArray();
            double dot = 0, nf = 0, ne = 0, maxAbs = 0;
            for (int i = 0; i < p0.Length; i++)
            {
                double df = pf[i] - p0[i], de = pe[i] - p0[i];
                dot += df * de; nf += df * df; ne += de * de;
                maxAbs = Math.Max(maxAbs, Math.Abs(pf[i] - pe[i]));
            }
            double cosine = dot / Math.Sqrt(nf * ne);
            _output.WriteLine($"{variant}: loss fused {lossFused:R} eager {lossEager:R}; update cosine {cosine:R}; " +
                              $"max |param diff| {maxAbs:E3}; params {p0.Length}");

            Assert.True(Math.Abs(lossFused - lossEager) <= 1e-4 * Math.Max(1.0, Math.Abs(lossEager)),
                $"{variant}: fused loss {lossFused} vs eager {lossEager}");
            Assert.True(cosine >= 0.9999, $"{variant}: fused vs eager update cosine {cosine}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(priorOptions);
            AiDotNetEngine.Current = prior;
        }
    }
}