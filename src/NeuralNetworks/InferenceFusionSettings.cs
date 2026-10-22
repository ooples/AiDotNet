using System;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Inference-time graph fusions applied by the layers themselves.
/// </summary>
/// <remarks>
/// <para>
/// <b>Batch-norm folding.</b> In inference a batch normalization after a convolution is a fixed per-channel affine map,
/// so it is folded into the convolution: <c>W' = W * gamma / sqrt(var + eps)</c> and
/// <c>b' = (b - mean) * gamma / sqrt(var + eps) + beta</c>, and a ReLU that follows runs in the same kernel. This is
/// what PyTorch's frozen models (<c>torch.jit.freeze</c>), ONNX export and TensorRT do. It saves the batch norm's pass
/// over every activation (and the ReLU's), at the cost of a different rounding: outputs differ from the unfused
/// convolution, batch norm and ReLU by about 1e-6 relative in float. Training, a recording gradient tape and graph
/// capture never fold.
/// </para>
/// <para>
/// Default <c>ON</c>. Set <c>AIDOTNET_FOLD_BATCHNORM=0</c>, or <see cref="FoldBatchNormIntoConvolution"/> to
/// <c>false</c>, to run the layers unfused (bit-for-bit the separate operations).
/// </para>
/// </remarks>
public static class InferenceFusionSettings
{
    /// <summary>
    /// Whether inference folds a batch normalization (and a ReLU after it) into the preceding convolution. Default
    /// <c>true</c>; set <c>AIDOTNET_FOLD_BATCHNORM=0</c> to disable.
    /// </summary>
    public static bool FoldBatchNormIntoConvolution { get; set; } =
        !string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_FOLD_BATCHNORM"),
            "0",
            StringComparison.Ordinal);
}
