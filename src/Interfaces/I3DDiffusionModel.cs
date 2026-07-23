namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// 3D diffusion models extend diffusion to generate three-dimensional content,
/// including point clouds, meshes, textured models, and full 3D scenes. They
/// are used in computer graphics, game development, and virtual reality.
/// </para>
/// <para>
/// <b>For Beginners:</b> 3D diffusion models create 3D objects instead of flat images.
///
/// Types of 3D generation:
/// - Point Clouds: Sets of 3D points that form a shape
/// - Meshes: Surfaces made of triangles (like in games)
/// - Textured Models: Meshes with colors and materials
/// - Novel Views: New angles of an object from one image
///
/// How it works:
/// 1. Text-to-3D: Describe what you want → 3D model
/// 2. Image-to-3D: Single image → full 3D model
/// 3. Score Distillation: Use 2D diffusion to guide 3D optimization
///
/// Applications:
/// - Game asset creation
/// - Product design visualization
/// - VR/AR content generation
/// - Architectural modeling
/// </para>
/// <para>
/// This interface extends <see cref="IDiffusionModel{T}"/> with 3D-specific operations.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ThreeDDiffusionModel")]
public interface I3DDiffusionModel<T> : IDiffusionModel<T>
{
    /// <summary>
    /// Gets the default number of points in generated point clouds.
    /// </summary>
    int DefaultPointCount { get; }

    /// <summary>
    /// Gets whether this model supports point cloud generation.
    /// </summary>
    bool SupportsPointCloud { get; }

    /// <summary>
    /// Gets whether this model supports mesh generation.
    /// </summary>
    bool SupportsMesh { get; }

    /// <summary>
    /// Gets whether this model supports texture generation.
    /// </summary>
    bool SupportsTexture { get; }

    /// <summary>
    /// Gets whether this model supports novel view synthesis.
    /// </summary>
    bool SupportsNovelView { get; }

    /// <summary>
    /// Gets whether this model supports score distillation sampling (SDS).
    /// </summary>
    /// <remarks>
    /// <para>
    /// SDS uses gradients from a 2D diffusion model to optimize a 3D representation.
    /// This is the technique behind DreamFusion and similar text-to-3D methods.
    /// </para>
    /// </remarks>
    bool SupportsScoreDistillation { get; }

    /// <summary>
    /// Generates a point cloud from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired 3D object.</param>
    /// <param name="negativePrompt">What to avoid.</param>
    /// <param name="numPoints">Number of points in the cloud.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Point cloud tensor [batch, numPoints, 3] for XYZ coordinates.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a cloud of 3D points that form a shape:
    /// - prompt: "A chair" → 4096 points arranged in a chair shape
    /// - The points define the surface of the object
    /// - Can be converted to a mesh for rendering
    /// </para>
    /// </remarks>
    Tensor<T> GeneratePointCloud(
        string prompt,
        string? negativePrompt = null,
        int? numPoints = null,
        int numInferenceSteps = 64,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Generates a mesh from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired 3D object.</param>
    /// <param name="negativePrompt">What to avoid.</param>
    /// <param name="resolution">Mesh resolution (higher = more detail).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Mesh data containing vertices and faces.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a 3D surface you can render:
    /// - Vertices: 3D points that define corners
    /// - Faces: Triangles connecting vertices to form surfaces
    /// - Can be exported to common 3D formats (OBJ, STL, etc.)
    /// </para>
    /// </remarks>
    Mesh3D<T> GenerateMesh(
        string prompt,
        string? negativePrompt = null,
        int resolution = 128,
        int numInferenceSteps = 64,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Generates a 3D model from a single input image.
    /// </summary>
    /// <param name="inputImage">The input image showing the object.</param>
    /// <param name="numViews">Number of views to generate for reconstruction.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Reconstructed mesh with optional texture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This turns a flat picture into a 3D model:
    /// - Input: Photo of a shoe from the front
    /// - Output: Full 3D model you can view from any angle
    /// - The model infers what the hidden parts look like
    /// </para>
    /// </remarks>
    Mesh3D<T> ImageTo3D(
        Tensor<T> inputImage,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Synthesizes novel views of an object from a reference image.
    /// </summary>
    /// <param name="inputImage">The reference image.</param>
    /// <param name="targetAngles">Target viewing angles (azimuth, elevation) in radians.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Array of images from the requested viewpoints.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This shows an object from different angles:
    /// - Input: Front view of a car
    /// - Target: 45°, 90°, 135° rotations
    /// - Output: Images showing the car from those angles
    /// - Useful for product visualization
    /// </para>
    /// </remarks>
    Tensor<T>[] SynthesizeNovelViews(
        Tensor<T> inputImage,
        (double azimuth, double elevation)[] targetAngles,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Computes score distillation gradients for 3D optimization.
    /// </summary>
    /// <param name="renderedViews">2D renders from the current 3D representation.</param>
    /// <param name="prompt">Text prompt guiding the optimization.</param>
    /// <param name="timestep">Diffusion timestep for noise level.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <returns>Gradients to apply to the 3D representation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helps create 3D from text using 2D knowledge:
    /// 1. Render your 3D object from multiple angles
    /// 2. Ask a 2D diffusion model "does this look like [prompt]?"
    /// 3. Get gradients that tell you how to improve the 3D
    /// 4. Repeat until the 3D looks right from all angles
    ///
    /// This is how DreamFusion works - it uses 2D diffusion to guide 3D creation.
    /// </para>
    /// </remarks>
    Tensor<T> ComputeScoreDistillationGradients(
        Tensor<T> renderedViews,
        string prompt,
        int timestep,
        double guidanceScale = 100.0);

    /// <summary>
    /// Converts a point cloud to a mesh.
    /// </summary>
    /// <param name="pointCloud">Point cloud [batch, numPoints, 3].</param>
    /// <param name="method">Surface reconstruction method.</param>
    /// <returns>Reconstructed mesh.</returns>
    Mesh3D<T> PointCloudToMesh(Tensor<T> pointCloud, SurfaceReconstructionMethod method = SurfaceReconstructionMethod.Poisson);

    /// <summary>
    /// Adds colors/normals to a point cloud.
    /// </summary>
    /// <param name="pointCloud">Point cloud positions [batch, numPoints, 3].</param>
    /// <param name="prompt">Text prompt for coloring.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Point cloud with colors [batch, numPoints, 6] (XYZ + RGB).</returns>
    Tensor<T> ColorizePointCloud(
        Tensor<T> pointCloud,
        string prompt,
        int numInferenceSteps = 50,
        int? seed = null);
}

/// <summary>
/// Represents a 3D mesh with vertices, faces, and optional textures.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class Mesh3D<T>
{
    /// <summary>
    /// Gets or sets the vertex positions [numVertices, 3].
    /// </summary>
    public Tensor<T> Vertices { get; set; }

    /// <summary>
    /// Gets or sets the face indices [numFaces, 3] for triangular faces.
    /// </summary>
    public int[,] Faces { get; set; }

    /// <summary>
    /// Gets or sets the vertex normals [numVertices, 3] (optional).
    /// </summary>
    public Tensor<T>? Normals { get; set; }

    /// <summary>
    /// Gets or sets the vertex colors [numVertices, 3] (optional).
    /// </summary>
    public Tensor<T>? Colors { get; set; }

    /// <summary>
    /// Gets or sets the UV texture coordinates [numVertices, 2] (optional).
    /// </summary>
    public Tensor<T>? UVCoordinates { get; set; }

    /// <summary>
    /// Gets or sets the texture image [height, width, channels] (optional).
    /// </summary>
    public Tensor<T>? TextureImage { get; set; }

    /// <summary>
    /// Initializes a new empty mesh.
    /// </summary>
    public Mesh3D()
    {
        Vertices = new Tensor<T>(new[] { 0, 3 });
        Faces = new int[0, 3];
    }

    /// <summary>
    /// Initializes a mesh with vertices and faces.
    /// </summary>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3].</param>
    public Mesh3D(Tensor<T> vertices, int[,] faces)
    {
        Vertices = vertices;
        Faces = faces;
    }
}

/// <summary>
/// Methods for reconstructing surfaces from point clouds.
/// </summary>
public enum SurfaceReconstructionMethod
{
    /// <summary>
    /// Poisson surface reconstruction (smooth, watertight meshes).
    /// </summary>
    Poisson,

    /// <summary>
    /// Ball pivoting algorithm (preserves sharp features).
    /// </summary>
    BallPivoting,

    /// <summary>
    /// Marching cubes on a voxel grid.
    /// </summary>
    MarchingCubes,

    /// <summary>
    /// Alpha shape algorithm.
    /// </summary>
    AlphaShape
}
