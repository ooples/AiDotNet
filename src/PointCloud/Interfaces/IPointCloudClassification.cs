namespace AiDotNet.PointCloud.Interfaces;

/// <summary>
/// Defines functionality for point cloud classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Point cloud classification determines what object or category an entire point cloud represents.
///
/// Think of classification as recognizing what an object is:
/// - Input: A complete point cloud of an object
/// - Output: The category the object belongs to
/// - It's like looking at a 3D scan and saying "this is a chair" or "this is a table"
///
/// Common classification benchmarks:
/// - ModelNet40: 40 categories of 3D objects (chair, table, car, airplane, etc.)
/// - ShapeNet: Large-scale dataset with many object categories
/// - ScanNet: Real-world scanned objects and scenes
///
/// Applications:
/// - Object recognition in 3D scans
/// - Quality control in manufacturing (identify defective parts)
/// - Archaeological artifact classification
/// - Medical imaging (classify anatomical structures)
/// </remarks>
public interface IPointCloudClassification<T> : IPointCloudModel<T>
{
    /// <summary>
    /// Classifies a point cloud into one of the predefined categories.
    /// </summary>
    /// <param name="pointCloud">Input point cloud tensor of shape [N, 3+F].</param>
    /// <returns>A vector of class probabilities of length C, where C is the number of classes.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method determines what object the point cloud represents.
    ///
    /// The classification process:
    /// - Takes the entire point cloud as input
    /// - Processes all points together to understand the overall shape
    /// - Returns probabilities for each possible category
    ///
    /// Example with furniture classification:
    /// - Input: Point cloud of a furniture piece
    /// - Output: [0.85 chair, 0.10 stool, 0.03 table, 0.02 bench]
    /// - The model predicts it's most likely a chair (85% confidence)
    ///
    /// The returned probabilities sum to 1.0, allowing you to:
    /// - Pick the most likely category (highest probability)
    /// - Understand the model's confidence in its prediction
    /// - Consider alternative possibilities if the top prediction has low confidence
    /// </remarks>
    Vector<T> ClassifyPointCloud(Tensor<T> pointCloud);
}
