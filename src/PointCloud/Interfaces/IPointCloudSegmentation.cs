namespace AiDotNet.PointCloud.Interfaces;

/// <summary>
/// Defines functionality for point cloud segmentation tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Point cloud segmentation assigns a label to each point in a 3D point cloud.
///
/// Think of segmentation as coloring a 3D model:
/// - Each point in the cloud gets assigned to a category
/// - Points belonging to the same object or part get the same label
/// - This allows you to identify and separate different components
///
/// Common segmentation tasks:
/// - Semantic segmentation: Label each point by object type (car, road, building, etc.)
/// - Instance segmentation: Separate individual objects (this car vs that car)
/// - Part segmentation: Identify parts of an object (chair leg, chair back, seat)
///
/// Applications:
/// - Autonomous driving: Identify pedestrians, vehicles, road surfaces
/// - Robotics: Recognize and grasp specific parts of objects
/// - 3D scene understanding: Parse indoor/outdoor environments
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("PointCloudSegmentation")]
public interface IPointCloudSegmentation<T> : IPointCloudModel<T>
{
    /// <summary>
    /// Performs semantic segmentation on a point cloud.
    /// </summary>
    /// <param name="pointCloud">Input point cloud tensor of shape [N, 3+F].</param>
    /// <returns>A tensor of shape [N, C] containing class probabilities for each point,
    /// where C is the number of classes.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method assigns a category label to each point.
    ///
    /// For each point, it predicts which category it belongs to:
    /// - Input: Point cloud with N points
    /// - Output: For each point, probabilities for each possible category
    /// - The category with highest probability is the predicted label
    ///
    /// Example for indoor scene segmentation:
    /// - Categories might be: floor, wall, ceiling, furniture, etc.
    /// - Each point gets probabilities: [0.01 floor, 0.95 wall, 0.02 ceiling, 0.02 furniture]
    /// - The point would be labeled as "wall" (highest probability)
    /// </remarks>
    Tensor<T> SegmentPointCloud(Tensor<T> pointCloud);
}
