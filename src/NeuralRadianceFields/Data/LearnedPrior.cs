using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Optional single-image / few-shot reconstruction prior attached to an <see cref="ImageView{T}"/>.
/// #1834 excellence goal — reference impls (Zero123, TripoSR, SyncDreamer) all live in separate
/// codebases with their own inference stacks. Attaching a prior directly to the view unifies
/// single-image reconstruction into the same NeRF/GS training loop: the model composes the
/// prior's hallucinated novel views with the caller's actual photos.
/// </summary>
/// <remarks>
/// The default prior is <see langword="null"/> — normal multi-view reconstruction with no
/// hallucination. Users opt in by supplying a prior on the single-photo view. Callers can
/// implement <see cref="ILearnedPrior{T}"/> to plug in their own diffusion-based novel-view
/// synthesizers; the reference impl <see cref="IsotropicPrior{T}"/> ships as a paper-standard
/// fallback that produces spatially-isotropic (view-independent) predictions.
/// </remarks>
public abstract class LearnedPrior<T>
{
    /// <summary>
    /// Synthesizes a novel-view prediction for the given camera pose. The radiance-field
    /// model blends this into its rendering loss so the density field is pushed toward the
    /// prior's guess in regions unseen by real photos.
    /// </summary>
    /// <param name="cameraPosition">Camera origin [3].</param>
    /// <param name="cameraRotation">Camera-to-world rotation [3, 3].</param>
    /// <param name="height">Requested image height.</param>
    /// <param name="width">Requested image width.</param>
    /// <returns>Hallucinated novel-view image [H, W, 3].</returns>
    public abstract Tensor<T> SynthesizeNovelView(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int height,
        int width);

    /// <summary>
    /// Prior weight in [0, 1] — how strongly the model should trust this hallucination vs
    /// its own current estimate. Reference impls hard-code 0.5; here it's per-prior configurable.
    /// </summary>
    public virtual double Confidence { get; init; } = 0.5;
}

/// <summary>
/// Paper-standard fallback prior — synthesizes the anchor view's mean color at every pixel
/// (spatially uniform). Simple enough to ship as a working default; sophisticated learned
/// priors (diffusion-based novel-view synthesis) plug in as subclasses.
/// </summary>
public sealed class IsotropicPrior<T> : LearnedPrior<T>
{
    private readonly Tensor<T> _anchorPhoto;
    private readonly T _rMean;
    private readonly T _gMean;
    private readonly T _bMean;

    public IsotropicPrior(Tensor<T> anchorPhoto)
    {
        _anchorPhoto = anchorPhoto ?? throw new System.ArgumentNullException(nameof(anchorPhoto));

        // Shape validation matches ImageView's constructor (rank-3 with 3 or 4 channels).
        if (_anchorPhoto.Shape.Length != 3
            || (_anchorPhoto.Shape[2] != 3 && _anchorPhoto.Shape[2] != 4))
        {
            var dims = new int[_anchorPhoto.Shape.Length];
            for (int i = 0; i < dims.Length; i++) dims[i] = _anchorPhoto.Shape[i];
            throw new System.ArgumentException(
                $"anchorPhoto must be [H, W, 3] or [H, W, 4]; got [{string.Join(",", dims)}].",
                nameof(anchorPhoto));
        }

        // Cache the anchor mean at construction — SynthesizeNovelView is called every
        // training iteration, so recomputing an O(H*W) sum on every call would dominate
        // training time on large anchors.
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int aH = _anchorPhoto.Shape[0];
        int aW = _anchorPhoto.Shape[1];
        T rSum = numOps.Zero, gSum = numOps.Zero, bSum = numOps.Zero;
        for (int y = 0; y < aH; y++)
        {
            for (int x = 0; x < aW; x++)
            {
                rSum = numOps.Add(rSum, _anchorPhoto[y, x, 0]);
                gSum = numOps.Add(gSum, _anchorPhoto[y, x, 1]);
                bSum = numOps.Add(bSum, _anchorPhoto[y, x, 2]);
            }
        }
        T count = numOps.FromDouble(aH * aW);
        _rMean = numOps.Divide(rSum, count);
        _gMean = numOps.Divide(gSum, count);
        _bMean = numOps.Divide(bSum, count);
    }

    public override Tensor<T> SynthesizeNovelView(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int height,
        int width)
    {
        var data = new T[height * width * 3];
        for (int i = 0; i < height * width; i++)
        {
            data[i * 3 + 0] = _rMean;
            data[i * 3 + 1] = _gMean;
            data[i * 3 + 2] = _bMean;
        }
        return new Tensor<T>(new[] { height, width, 3 }, new Vector<T>(data));
    }
}
