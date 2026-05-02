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
        if (visionDim <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(visionDim), "visionDim must be positive.");
        if (imageSize <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(imageSize), "imageSize must be positive.");

        // 3-channel images can be either [3,H,W] (rank-3, no batch) or [B,3,H,W].
        // Rank-3 with first dim 3 is ambiguous with already-tokenized
        // [B=3, S, C], so additionally require Shape[1]==Shape[2] (square H==W
        // is the universal ViT input convention) AND Shape[1]==imageSize
        // before treating it as an image.
        //
        // Rank-4 with C=3 is unambiguous as far as channel count goes, but
        // we must still validate H/W against imageSize: patchSize is derived
        // from imageSize (imageSize/16), so a [B,3,H,W] tensor at a
        // different resolution would silently get tokenized with a patch
        // size that doesn't match the actual frame geometry, producing
        // wrong-grid embeddings. Require H==W==imageSize for the rank-4
        // path too — callers passing a different resolution must reshape /
        // resize first.
        bool isImage =
            (input.Rank == 4 && input.Shape[1] == 3
                && input.Shape[2] == imageSize && input.Shape[3] == imageSize) ||
            (input.Rank == 3 && input.Shape[0] == 3
                && input.Shape[1] == input.Shape[2] && input.Shape[1] == imageSize);
        if (!isImage)
        {
            // Surface mismatched-resolution NCHW images explicitly rather
            // than silently routing them through as already-tokenized
            // [B,S,C] — the rank-3 [B,3,...] case below already does this
            // via the Shape[1]==Shape[2] gate, but rank-4 with C=3 was
            // previously a silent miscompute.
            if (input.Rank == 4 && input.Shape[1] == 3)
            {
                throw new System.ArgumentException(
                    $"PatchEmbedHelper expects rank-4 image input at resolution " +
                    $"[B, 3, {imageSize}, {imageSize}] (matching the imageSize this " +
                    $"helper was built for); got [{string.Join(",", input._shape)}]. " +
                    $"Resize the image to {imageSize}x{imageSize} or rebuild the " +
                    $"helper with the matching imageSize before passing it in.",
                    nameof(input));
            }
            return input;
        }

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
