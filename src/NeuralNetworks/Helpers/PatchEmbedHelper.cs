using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Helpers;

/// <summary>
/// ViT-style patch embedding (Dosovitskiy et al. 2020 §3.1): turns raw NCHW
/// image input into a BSC token sequence for transformer-backed vision
/// encoders (PaLM-E, BiomedCLIP, PaLI-3, OpenCLIP, ViT). Already-tokenized
/// inputs pass through unchanged.
/// </summary>
public static class PatchEmbedHelper
{
    public static Tensor<T> TokenizeImageNCHWToBSC<T>(
        Tensor<T> input,
        int visionDim,
        int imageSize,
        ref ConvolutionalLayer<T>? patchEmbed,
        IEngine engine)
    {
        if (input is null) throw new System.ArgumentNullException(nameof(input));
        if (engine is null) throw new System.ArgumentNullException(nameof(engine));

        bool isImage = (input.Rank == 3 && input.Shape[0] == 3) ||
                       (input.Rank == 4 && input.Shape[1] == 3);
        if (!isImage) return input;

        int patchSize = System.Math.Max(1, imageSize / 16);
        if (patchEmbed is null)
        {
            patchEmbed = new ConvolutionalLayer<T>(
                outputDepth: visionDim,
                kernelSize: patchSize,
                stride: patchSize,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
        }

        var patched = patchEmbed.Forward(input);
        int b, ch, h, w;
        if (patched.Rank == 4)
        {
            b = patched.Shape[0]; ch = patched.Shape[1]; h = patched.Shape[2]; w = patched.Shape[3];
        }
        else
        {
            b = 1; ch = patched.Shape[0]; h = patched.Shape[1]; w = patched.Shape[2];
            patched = engine.Reshape(patched, new[] { 1, ch, h, w });
        }
        var bhwc = engine.TensorPermute(patched, new[] { 0, 2, 3, 1 }).Contiguous();
        return engine.Reshape(bhwc, new[] { b, h * w, ch });
    }
}
