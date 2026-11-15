namespace AiDotNet.Enums;

/// <summary>
/// How a <see cref="AiDotNet.Diffusion.NoisePredictors.DiffusionResBlock{T}"/> initializes its convolutions.
/// </summary>
/// <remarks>
/// <para>
/// Diffusion U-Nets (DDPM's <c>diffusion_tf</c>, guided-diffusion, LDM) zero the block's second convolution so the
/// block starts as its skip path. Models whose reference code never re-initializes its modules, such as LGM's
/// asymmetric U-Net, start from PyTorch's default layer initialization instead.
/// </para>
/// <para><b>For Beginners:</b> This chooses the starting weights of the block before training or loading a
/// checkpoint. Use the default for diffusion denoisers, and <see cref="TorchDefault"/> when reproducing a PyTorch
/// model that relies on PyTorch's own defaults.</para>
/// </remarks>
public enum DiffusionResBlockInitialization
{
    /// <summary>
    /// The diffusion convention: the second convolution starts at zero; the first convolution and the skip
    /// projection use the convolution layer's default (Kaiming-uniform weights, zero biases).
    /// </summary>
    ZeroResidual = 0,

    /// <summary>
    /// PyTorch's default <c>nn.Conv2d</c> initialization for every convolution of the block:
    /// <c>kaiming_uniform_(a = sqrt(5))</c> weights, which is U(-1/sqrt(fan_in), 1/sqrt(fan_in)), and biases drawn
    /// from the same bound.
    /// </summary>
    TorchDefault = 1
}
