using AiDotNet.Interfaces;

namespace AiDotNet.NeuralRadianceFields.Interfaces;

/// <summary>
/// Defines the core functionality for neural radiance field models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A neural radiance field represents a 3D scene as a continuous function.
///
/// Traditional 3D representations:
/// - Meshes: Surfaces made of triangles
/// - Voxels: 3D grid of cubes (like 3D pixels)
/// - Point clouds: Collection of 3D points
///
/// Neural Radiance Fields (NeRF):
/// - Represents scene as a neural network
/// - Input: 3D position (X, Y, Z) and viewing direction
/// - Output: Color (RGB) and density (opacity/volume density)
///
/// Think of it like this:
/// - The neural network "knows" what the scene looks like from any position
/// - Ask "What's at position (x, y, z) when viewed from direction (θ, φ)?"
/// - Network responds "Color is (r, g, b) and density is σ"
///
/// Why this is powerful:
/// - Continuous representation (query any position, not limited to discrete grid)
/// - View-dependent effects (reflections, specularities)
/// - Compact storage (just network weights, not millions of voxels)
/// - Novel view synthesis (render from any camera angle)
///
/// Applications:
/// - Virtual reality and AR: Create photorealistic 3D scenes
/// - Film and gaming: Capture real locations and render from any angle
/// - Robotics: Build 3D maps of environments
/// - Cultural heritage: Digitally preserve historical sites
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("RadianceField")]
public interface IRadianceField<T> : INeuralNetwork<T>
{
    /// <summary>
    /// Queries the radiance field at specific 3D positions and viewing directions.
    /// </summary>
    /// <param name="positions">Tensor of 3D positions [N, 3] where N is number of query points.</param>
    /// <param name="viewingDirections">Tensor of viewing directions [N, 3] (unit vectors).</param>
    /// <returns>A tuple containing RGB colors [N, 3] and density values [N, 1].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the core operation of a radiance field.
    ///
    /// For each query point:
    /// - Position (x, y, z): Where in 3D space are we looking?
    /// - Direction (dx, dy, dz): Which direction are we looking from?
    ///
    /// The network returns:
    /// - RGB (r, g, b): The color at that point from that direction
    /// - Density σ: How "solid" or "opaque" that point is
    ///
    /// Example query:
    /// - Position: (2.5, 1.0, -3.0) - a point in space
    /// - Direction: (0.0, 0.0, -1.0) - looking straight down negative Z axis
    /// - Result: (Red: 0.8, Green: 0.3, Blue: 0.1, Density: 5.2)
    ///   This means the point appears orange when viewed from that direction,
    ///   and it's fairly opaque (high density)
    ///
    /// Density interpretation:
    /// - Density = 0: Completely transparent (empty space)
    /// - Density > 0: Increasingly opaque (solid material)
    /// - Higher density = light is more likely to stop at this point
    /// </remarks>
    (Tensor<T> rgb, Tensor<T> density) QueryField(Tensor<T> positions, Tensor<T> viewingDirections);

    /// <summary>
    /// Renders an image from a specific camera position and orientation.
    /// </summary>
    /// <param name="cameraPosition">3D position of the camera [3].</param>
    /// <param name="cameraRotation">Rotation matrix of the camera [3, 3].</param>
    /// <param name="imageWidth">Width of the output image in pixels.</param>
    /// <param name="imageHeight">Height of the output image in pixels.</param>
    /// <param name="focalLength">Camera focal length.</param>
    /// <returns>Rendered RGB image tensor [height, width, 3].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This renders a 2D image from the 3D scene representation.
    ///
    /// The rendering process:
    /// 1. For each pixel in the output image:
    ///    - Cast a ray from camera through that pixel
    ///    - Sample points along the ray
    ///    - Query radiance field at each sample point
    ///    - Combine colors using volume rendering (accumulate with alpha blending)
    ///
    /// Volume rendering equation:
    /// - For each ray, accumulate: Color = Σ(transmittance × color × alpha)
    /// - Transmittance: How much light passes through previous points
    /// - Alpha: How much light is absorbed at this point (based on density)
    ///
    /// Example:
    /// - Camera at (0, 0, 5) looking at origin
    /// - Image 512×512 pixels
    /// - Cast 512×512 = 262,144 rays
    /// - Sample 64 points per ray = 16.8 million queries
    /// - Blend results to get final image
    ///
    /// This is why NeRF rendering can be slow - many network queries!
    /// Optimizations like Instant-NGP speed this up significantly.
    /// </remarks>
    Tensor<T> RenderImage(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength);

    /// <summary>
    /// Renders rays through the radiance field using volume rendering.
    /// </summary>
    /// <param name="rayOrigins">Origins of rays to render [N, 3].</param>
    /// <param name="rayDirections">Directions of rays (unit vectors) [N, 3].</param>
    /// <param name="numSamples">Number of samples per ray.</param>
    /// <param name="nearBound">Near clipping distance.</param>
    /// <param name="farBound">Far clipping distance.</param>
    /// <returns>Rendered RGB colors for each ray [N, 3].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Volume rendering is how we convert the radiance field into images.
    ///
    /// For each ray:
    /// 1. Sample points: Generate sample positions between near and far bounds
    /// 2. Query field: Get RGB and density at each sample
    /// 3. Compute alpha: Convert density to opacity for each segment
    /// 4. Accumulate color: Blend colors front-to-back
    ///
    /// The algorithm:
    /// ```
    /// For each sample i along ray:
    ///   alpha_i = 1 - exp(-density_i * distance_i)
    ///   transmittance_i = exp(-sum of all previous densities)
    ///   color_contribution_i = transmittance_i * alpha_i * color_i
    ///   total_color += color_contribution_i
    /// ```
    ///
    /// Example with 4 samples:
    /// - Sample 0: Empty space (density ≈ 0) → contributes little
    /// - Sample 1: Empty space (density ≈ 0) → contributes little
    /// - Sample 2: Surface (density high) → contributes most of the color
    /// - Sample 3: Behind surface → mostly blocked by sample 2
    ///
    /// Parameters:
    /// - numSamples: More samples = better quality but slower (typical: 64-192)
    /// - nearBound/farBound: Define region to sample (e.g., 0.1 to 10.0 meters)
    /// </remarks>
    Tensor<T> RenderRays(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        int numSamples,
        T nearBound,
        T farBound);
}
