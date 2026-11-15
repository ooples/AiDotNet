using AiDotNet.Diffusion.ThreeD.Meshing;

namespace AiDotNet.NeuralRadianceFields.Models;

public partial class GaussianSplatting<T>
{
    /// <summary>
    /// Exports the Gaussians in activated form, for mesh extraction with <see cref="GaussianMeshExtractor"/>.
    /// </summary>
    /// <returns>
    /// Centres [N, 3], standard deviations [N, 3] (the absolute stored scales floored at 1e-6, as the renderer's
    /// covariance uses them), rotation quaternions (w, x, y, z) [N, 4] and opacities [N] (the sigmoid of the stored
    /// logits).
    /// </returns>
    public GaussianCloud<T> ToGaussianCloud()
    {
        int count = _gaussians.Count;
        var positions = new T[3 * count];
        var scales = new T[3 * count];
        var rotations = new T[4 * count];
        var logits = new T[count];
        for (int i = 0; i < count; i++)
        {
            var gaussian = _gaussians[i];
            Array.Copy(gaussian.Position.ToArray(), 0, positions, 3 * i, 3);
            Array.Copy(gaussian.Scale.ToArray(), 0, scales, 3 * i, 3);
            Array.Copy(gaussian.Rotation.ToArray(), 0, rotations, 4 * i, 4);
            logits[i] = gaussian.Opacity;
        }

        var scaleTensor = Engine.TensorClamp(
            Engine.TensorAbs(new Tensor<T>(scales, new[] { count, 3 })),
            NumOps.FromDouble(1e-6),
            NumOps.FromDouble(double.MaxValue));
        return new GaussianCloud<T>(
            new Tensor<T>(positions, new[] { count, 3 }),
            scaleTensor,
            new Tensor<T>(rotations, new[] { count, 4 }),
            Engine.Sigmoid(new Tensor<T>(logits, new[] { count })));
    }
}
