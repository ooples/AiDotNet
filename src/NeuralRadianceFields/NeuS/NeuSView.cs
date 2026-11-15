namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// One posed input view for NeuS or SparseNeuS: an image, its camera, and optionally a foreground mask and a normal map.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The image is <c>[H, W, 3]</c> with colors in [0, 1]. The mask, when present, is <c>[H, W]</c> with 1 on the object
/// (NeuS's <c>mask &gt; 0.5</c> test binarizes it when the mask loss is on). The normals, when present, are
/// <c>[H, W, 3]</c> unit vectors in <b>world</b> space, the normal supervision Wonder3D's cross-domain diffusion
/// produces; rotate camera-space normals by the camera-to-world rotation before passing them.
/// </para>
/// <para><b>For Beginners:</b> A view is one photo plus where the camera was. Several views of the same object from
/// different angles let NeuS work out its 3D shape.</para>
/// </remarks>
internal sealed class NeuSView<T>
{
    /// <summary>Creates a view.</summary>
    /// <param name="image">The image, <c>[H, W, 3]</c>, colors in [0, 1].</param>
    /// <param name="camera">The camera.</param>
    /// <param name="mask">The foreground mask, <c>[H, W]</c>, or null.</param>
    /// <param name="normals">World-space normals, <c>[H, W, 3]</c>, or null.</param>
    /// <param name="weight">The view's weight in the mask and normal terms (Wonder3D's <c>view_weights</c>; 1 in NeuS).</param>
    public NeuSView(Tensor<T> image, NeuSCamera camera, Tensor<T>? mask = null, Tensor<T>? normals = null,
        double weight = 1.0)
    {
        if (!(weight >= 0) || double.IsInfinity(weight))
            throw new ArgumentOutOfRangeException(nameof(weight), "The view weight must be non-negative and finite.");
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (camera is null) throw new ArgumentNullException(nameof(camera));
        if (image.Shape.Length != 3 || image.Shape[2] != 3)
            throw new ArgumentException($"The image must be [H, W, 3]; got [{string.Join(",", image.Shape.ToArray())}].", nameof(image));
        int h = image.Shape[0], w = image.Shape[1];
        if (mask is not null && (mask.Shape.Length != 2 || mask.Shape[0] != h || mask.Shape[1] != w))
            throw new ArgumentException("The mask must be [H, W] with the image's size.", nameof(mask));
        if (normals is not null && (normals.Shape.Length != 3 || normals.Shape[0] != h || normals.Shape[1] != w
            || normals.Shape[2] != 3))
            throw new ArgumentException("The normals must be [H, W, 3] with the image's size.", nameof(normals));
        Image = image;
        Camera = camera;
        Mask = mask;
        Normals = normals;
        Weight = weight;
    }

    /// <summary>The view's weight in the mask and normal terms.</summary>
    public double Weight { get; }

    /// <summary>The image, <c>[H, W, 3]</c>.</summary>
    public Tensor<T> Image { get; }

    /// <summary>The camera.</summary>
    public NeuSCamera Camera { get; }

    /// <summary>The foreground mask, <c>[H, W]</c>, or null.</summary>
    public Tensor<T>? Mask { get; }

    /// <summary>World-space normals, <c>[H, W, 3]</c>, or null.</summary>
    public Tensor<T>? Normals { get; }

    /// <summary>The image height.</summary>
    public int Height => Image.Shape[0];

    /// <summary>The image width.</summary>
    public int Width => Image.Shape[1];
}
