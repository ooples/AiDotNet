using AiDotNet.Interfaces;

namespace AiDotNet.PointCloud.Interfaces;

/// <summary>
/// Defines the core functionality for point cloud processing models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A point cloud is a collection of 3D points that represent the surface of an object or scene.
///
/// Think of a point cloud as a 3D scan of the real world:
/// - Each point has X, Y, Z coordinates representing its position in 3D space
/// - Points can also have additional features like color, intensity, or surface normals
/// - Point clouds are commonly collected by LIDAR sensors, depth cameras, or 3D scanners
///
/// Common applications:
/// - Autonomous vehicles use LIDAR to create point clouds of their surroundings
/// - Robotics uses point clouds for object recognition and manipulation
/// - AR/VR applications use point clouds for 3D reconstruction
/// - Architecture and construction use point clouds for building modeling
///
/// This interface defines operations for processing point cloud data with neural networks.
/// </remarks>
public interface IPointCloudModel<T> : INeuralNetwork<T>
{
    /// <summary>
    /// Extracts global features from a point cloud.
    /// </summary>
    /// <param name="pointCloud">Input point cloud tensor of shape [N, 3+F] where N is number of points,
    /// and 3+F represents XYZ coordinates plus F additional features.</param>
    /// <returns>A feature vector representing the global characteristics of the point cloud.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts a compact representation of the entire point cloud.
    ///
    /// It's like creating a summary or "fingerprint" of the 3D object:
    /// - Input: Many individual 3D points (could be thousands or millions)
    /// - Output: A single feature vector that captures the essential characteristics
    /// - This summary can be used for classification, detection, or comparison
    ///
    /// For example, extracting global features from point clouds of chairs would produce
    /// similar feature vectors for all chairs, despite differences in specific details.
    /// </remarks>
    Vector<T> ExtractGlobalFeatures(Tensor<T> pointCloud);

    /// <summary>
    /// Extracts per-point features from a point cloud.
    /// </summary>
    /// <param name="pointCloud">Input point cloud tensor of shape [N, 3+F].</param>
    /// <returns>A tensor of shape [N, D] where D is the feature dimension for each point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts features for each individual point.
    ///
    /// Unlike global features which summarize the entire cloud, per-point features describe each point:
    /// - Input: N points with XYZ coordinates
    /// - Output: N feature vectors, one for each point
    /// - Each feature vector captures local and contextual information about that point
    ///
    /// This is useful for tasks like:
    /// - Point cloud segmentation (labeling each point)
    /// - Finding specific features or parts in the 3D data
    /// - Understanding the local geometry around each point
    /// </remarks>
    Tensor<T> ExtractPointFeatures(Tensor<T> pointCloud);
}
