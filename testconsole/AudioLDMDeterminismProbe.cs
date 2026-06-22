using System;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.LinearAlgebra;

namespace AiDotNetTestConsole;

// Regression diagnostic for the AudioLDM Predict_ShouldBeDeterministic failure (PR #1633).
// Predicts the SAME input THREE times at two levels and prints the first/last output elements:
//   Stage A — UNet.PredictNoise (the noise-predictor forward) in isolation.
//   Stage B — AudioLDM.Predict (the full Generate pipeline: scheduler loop + VAE).
// Root cause (now fixed): the verify-then-trust gate's one compiled-plan execution dirtied the
// AiDotNet.Tensors process-global eager/compiled scratch, so the eager fallback for a REJECTED
// shape oscillated with period 2 (call #1 == call #3 != call #2) — non-deterministic and wrong on
// half the calls. NoisePredictorBase now defaults the compiled inference path OFF (pure eager,
// bit-identical across calls); opt back in with AIDOTNET_ENABLE_AUTO_COMPILE=1. With the fix, all
// three calls of both stages print identical values.
internal static class AudioLDMDeterminismProbe
{
    public static void Run()
    {
        const int latentCh = 8;

        var unet = new UNetNoisePredictor<double>(
            inputChannels: latentCh, outputChannels: latentCh,
            baseChannels: 64, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocks: 1, attentionResolutions: new[] { 1, 2 },
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        var sample = new Tensor<double>(new[] { 1, latentCh, 16, 16 });
        for (int i = 0; i < sample.Length; i++) sample[i] = (i % 7) * 0.1 - 0.3;

        Console.WriteLine("=== Stage A: UNet.PredictNoise 3x (timestep=500) ===");
        for (int call = 1; call <= 3; call++)
        {
            var outp = unet.PredictNoise(sample, timestep: 500);
            Console.WriteLine($"  A call #{call}: out[0]={outp[0]:R}  out[last]={outp[outp.Length - 1]:R}");
        }

        var model = new AudioLDMModel<double>(
            unet: new UNetNoisePredictor<double>(
                inputChannels: latentCh, outputChannels: latentCh,
                baseChannels: 64, channelMultipliers: new[] { 1, 2, 4 },
                numResBlocks: 1, attentionResolutions: new[] { 1, 2 },
                contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42),
            seed: 42);

        var input = new Tensor<double>(new[] { 1, latentCh, 16, 16 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 7) * 0.1 - 0.3;

        Console.WriteLine("=== Stage B: AudioLDM.Predict 3x (same input) ===");
        for (int call = 1; call <= 3; call++)
        {
            var outp = model.Predict(input);
            Console.WriteLine($"  B call #{call}: out[0]={outp[0]:R}  out[last]={outp[outp.Length - 1]:R}");
        }
        Console.WriteLine("All three calls of each stage must be identical (fixed: compiled inference defaults off).");
    }
}
