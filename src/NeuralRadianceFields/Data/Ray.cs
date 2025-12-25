namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Represents a ray in 3D space for rendering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A ray is a half-line starting from a point and extending in a direction.
///
/// In computer graphics and rendering:
/// - Rays are cast from camera through each pixel
/// - They represent the path light travels
/// - We sample points along rays to determine what we see
///
/// Ray equation:
/// - Point on ray = Origin + t * Direction
/// - t is the distance along the ray (t >= 0)
/// - Origin: Starting point (usually camera position)
/// - Direction: Which way the ray points (unit vector)
///
/// Example:
/// - Camera at origin (0, 0, 0)
/// - Looking down negative Z axis
/// - Ray for center pixel: Origin = (0, 0, 0), Direction = (0, 0, -1)
/// - Point at distance 5: (0, 0, 0) + 5 * (0, 0, -1) = (0, 0, -5)
///
/// In NeRF:
/// - Each pixel corresponds to one ray
/// - We sample many points along each ray
/// - Query the neural network at each sample point
/// - Blend results to get final pixel color
/// </remarks>
public class Ray<T>
{
    /// <summary>
    /// Gets or sets the origin point of the ray.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Where the ray starts (typically the camera position).
    ///
    /// Format: Vector of length 3 representing (X, Y, Z) coordinates
    ///
    /// Example:
    /// - Camera ray: Origin might be (0, 0, 5) if camera is 5 units away
    /// - Shadow ray: Origin might be surface point being tested for shadows
    /// - Reflection ray: Origin is the point where light bounced
    /// </remarks>
    public Vector<T> Origin { get; set; }

    /// <summary>
    /// Gets or sets the direction vector of the ray (should be normalized).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Which direction the ray travels (should be a unit vector).
    ///
    /// Format: Vector of length 3 representing (dX, dY, dZ) direction
    /// Important: Should be normalized (length = 1) for correct distance calculations
    ///
    /// Example directions:
    /// - (0, 0, -1): Straight down negative Z axis
    /// - (1, 0, 0): Straight along positive X axis
    /// - (0.707, 0.707, 0): 45 degrees between X and Y axes (√2/2 ≈ 0.707)
    ///
    /// Normalization:
    /// - Unnormalized: (3, 4, 0) has length 5
    /// - Normalized: (0.6, 0.8, 0) has length 1
    /// - Formula: direction / ||direction||
    /// </remarks>
    public Vector<T> Direction { get; set; }

    /// <summary>
    /// Gets or sets the near bound for sampling along the ray.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Closest distance from origin where we start sampling.
    ///
    /// Why we need this:
    /// - Avoid sampling too close to camera (numerical issues)
    /// - Skip empty space we know is in front of the scene
    /// - Optimize rendering by not wasting samples
    ///
    /// Example:
    /// - Scene is all beyond 2 units from camera
    /// - Set NearBound = 2.0
    /// - Don't waste samples on empty space from 0 to 2
    ///
    /// Typical values:
    /// - Indoor scenes: 0.1 to 1.0
    /// - Outdoor scenes: 1.0 to 10.0
    /// - Depends on scene scale
    /// </remarks>
    public T NearBound { get; set; }

    /// <summary>
    /// Gets or sets the far bound for sampling along the ray.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Farthest distance from origin where we stop sampling.
    ///
    /// Why we need this:
    /// - Avoid sampling infinitely far away
    /// - Skip empty space we know is behind the scene
    /// - Limit computation to relevant region
    ///
    /// Example:
    /// - Scene is contained within 10 units
    /// - Set FarBound = 10.0
    /// - Don't waste samples beyond where scene ends
    ///
    /// Typical values:
    /// - Indoor scenes: 5.0 to 20.0
    /// - Outdoor scenes: 50.0 to 1000.0
    /// - Infinity can be approximated with very large value
    ///
    /// Together with NearBound:
    /// - Samples distributed between NearBound and FarBound
    /// - Example: Near=2, Far=10, NumSamples=8
    /// - Sample at: 2, 3.14, 4.29, 5.43, 6.57, 7.71, 8.86, 10
    /// </remarks>
    public T FarBound { get; set; }

    /// <summary>
    /// Initializes a new instance of the Ray class.
    /// </summary>
    /// <param name="origin">The origin point of the ray.</param>
    /// <param name="direction">The direction vector of the ray (will be normalized).</param>
    /// <param name="nearBound">The near clipping distance.</param>
    /// <param name="farBound">The far clipping distance.</param>
    public Ray(Vector<T> origin, Vector<T> direction, T nearBound, T farBound)
    {
        Origin = origin;
        Direction = direction.Normalize();
        NearBound = nearBound;
        FarBound = farBound;
    }

    /// <summary>
    /// Computes a point along the ray at a specific distance.
    /// </summary>
    /// <param name="t">The distance along the ray from the origin.</param>
    /// <returns>The 3D point at distance t along the ray.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Calculate a specific point along the ray.
    ///
    /// Formula: Point = Origin + t * Direction
    ///
    /// Example:
    /// - Origin = (0, 0, 0)
    /// - Direction = (0, 0, -1)
    /// - t = 5.0
    /// - Result: (0, 0, 0) + 5.0 * (0, 0, -1) = (0, 0, -5)
    ///
    /// Uses:
    /// - Generate sample points for volume rendering
    /// - Find intersection points with surfaces
    /// - Determine positions to query the radiance field
    ///
    /// Note: t should typically be between NearBound and FarBound
    /// </remarks>
    public Vector<T> PointAt(T t)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var point = new Vector<T>(3);

        for (int i = 0; i < 3; i++)
        {
            point[i] = numOps.Add(Origin[i], numOps.Multiply(t, Direction[i]));
        }

        return point;
    }
}
