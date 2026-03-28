using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Abstract base class for image safety modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for all image safety modules. Concrete modules implement
/// <see cref="EvaluateImage(Tensor{T})"/> and this base class handles the
/// <see cref="ISafetyModule{T}.Evaluate(Vector{T})"/> bridge by reshaping vectors into tensors.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class handles the plumbing so that each image safety
/// module only needs to implement one method: <c>EvaluateImage(Tensor&lt;T&gt;)</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ImageSafetyModuleBase<T> : IImageSafetyModule<T>
{
    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image);

    /// <inheritdoc />
    /// <remarks>
    /// The base implementation wraps the vector in a 1D tensor and delegates to
    /// <see cref="EvaluateImage(Tensor{T})"/>. Subclasses that expect specific
    /// tensor shapes (e.g., [C,H,W]) should validate accordingly.
    /// </remarks>
    public virtual IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        // Wrap the flat vector as a 1D tensor
        var tensor = new Tensor<T>(content.ToArray(), new[] { content.Length });
        return EvaluateImage(tensor);
    }
}
