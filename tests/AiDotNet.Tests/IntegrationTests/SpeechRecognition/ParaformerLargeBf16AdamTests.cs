using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SpeechRecognition;

/// <summary>
/// #1767: ParaformerLarge at its full default scale (661M parameters under the generic template, the release's ~220M
/// since #2397 built it on its SAN-M encoder and decoder), trained eagerly with the BF16-moment 8-bit Adam,
/// threw "Source array was not long enough" on the heavy lane. Nothing exercised that configuration afterwards, so a
/// recurrence would have gone unseen. This runs it for real, at full scale, not a reduced stand-in.
/// </summary>
/// <remarks>
/// Measured on 2026-10-20 (master 5752a0fab7) under the lane's 11 GiB heap cap: 48 s setup, then 87-105 s per step,
/// heap 4.9-9.4 GB. Hence the HeavyTimeout lane and a timeout well above three steps.
/// </remarks>
[Trait("Category", "HeavyTimeout")]
[Collection("FoundationScaleSerial")]
public sealed class ParaformerLargeBf16AdamTests
{
    [Fact(Timeout = 1_500_000)]
    public async Task FullScale_EagerBf16EightBitAdam_TrainsWithFiniteChangingLoss()
    {
        await Task.Yield();
        var previousCodec = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
        try
        {
            const int samples = 16000;                                                       // 1 s of 16 kHz audio
            var architecture = new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
                NeuralNetworkTaskType.SequenceToSequence, inputSize: samples, outputSize: 8404);
            var model = new ParaformerLarge<float>(architecture);

            model.SetBaseTrainOptimizer(new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(model,
                new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    UseBFloat16MomentStorage = true,
                    UseAMSGrad = false,
                }));

            var rng = RandomHelper.CreateSeededRandom(1767);
            var input = new Tensor<float>(new[] { samples });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            var shape = model.Predict(input).Shape.ToArray();
            // Lazy layers report their size once a forward has resolved them; the release is about 220M parameters.
            Assert.True(model.GetParameterCount() > 200_000_000,
                $"#1767 needs the full default scale; got {model.GetParameterCount():N0} parameters.");
            var target = new Tensor<float>(shape);
            for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();

            var losses = new float[3];
            for (int step = 0; step < losses.Length; step++)
            {
                model.Train(input, target);
                losses[step] = model.GetLastLoss();
                Assert.True(!float.IsNaN(losses[step]) && !float.IsInfinity(losses[step]), $"Step {step + 1} loss is {losses[step]}.");
            }

            // The optimizer applied updates: the same sample's loss moved.
            Assert.NotEqual(losses[0], losses[losses.Length - 1]);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousCodec);
        }
    }
}
