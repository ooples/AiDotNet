namespace AiDotNet.PointCloud.Data;

/// <summary>
/// Represents a point cloud data structure with coordinates and optional features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This class stores 3D point cloud data in an organized way.
///
/// A point cloud consists of:
/// - Points: XYZ coordinates representing positions in 3D space
/// - Features: Optional additional information per point (color, intensity, normals, etc.)
/// - Labels: Optional category or class information for each point
///
/// Think of it as a spreadsheet where:
/// - Each row is a point
/// - First 3 columns are X, Y, Z coordinates
/// - Additional columns can store colors, surface properties, etc.
/// - Another column can store what category each point belongs to
///
/// This structure makes it easy to:
/// - Load point cloud data from sensors or files
/// - Pass data to neural networks for processing
/// - Store results from classification or segmentation
/// </remarks>
public class PointCloudData<T>
{
    /// <summary>
    /// Gets or sets the tensor containing point coordinates and features.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This stores all the point data.
    ///
    /// The tensor shape is [N, 3+F] where:
    /// - N = number of points (could be thousands or millions)
    /// - 3 = XYZ coordinates (required for every point)
    /// - F = number of additional features (optional, could be 0)
    ///
    /// Example with RGB colors:
    /// - Shape would be [N, 6]
    /// - First 3 values per point: X, Y, Z position
    /// - Last 3 values per point: R, G, B color values
    ///
    /// Example without features:
    /// - Shape would be [N, 3]
    /// - Only XYZ coordinates
    /// </remarks>
    public Tensor<T> Points { get; set; }

    /// <summary>
    /// Gets or sets the number of points in the cloud.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is how many individual points are in the point cloud.
    ///
    /// Point clouds can vary greatly in size:
    /// - Small objects: hundreds to thousands of points
    /// - Detailed scans: hundreds of thousands of points
    /// - Large environments: millions of points
    ///
    /// More points provide more detail but require more memory and processing time.
    /// </remarks>
    public int NumPoints { get; set; }

    /// <summary>
    /// Gets or sets the number of features per point (including XYZ coordinates).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells how much information we have for each point.
    ///
    /// Common feature counts:
    /// - 3: Just XYZ coordinates
    /// - 6: XYZ + RGB color
    /// - 7: XYZ + RGBA (color with transparency)
    /// - 9: XYZ + RGB + surface normal (3 values)
    /// - Custom: Any combination of features you need
    ///
    /// The features help the neural network understand more about each point
    /// than just its position in space.
    /// </remarks>
    public int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets optional labels for classification or segmentation tasks.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This stores the correct answers for training or the predictions for testing.
    ///
    /// Depending on the task:
    /// - Classification: Single label for the entire point cloud (e.g., "chair")
    /// - Segmentation: One label per point (e.g., point 1 is "leg", point 2 is "seat")
    ///
    /// During training:
    /// - Labels contain the ground truth (correct answers)
    /// - The model learns by comparing its predictions to these labels
    ///
    /// During inference:
    /// - Labels can store the model's predictions
    /// - Or be null if we just want the raw output probabilities
    /// </remarks>
    public Vector<T>? Labels { get; set; }

    /// <summary>
    /// Initializes a new instance of the PointCloudData class.
    /// </summary>
    /// <param name="points">Tensor of shape [N, 3+F] containing point coordinates and features.</param>
    /// <param name="labels">Optional labels for the points.</param>
    public PointCloudData(Tensor<T> points, Vector<T>? labels = null)
    {
        Points = points;
        NumPoints = points.Shape[0];
        NumFeatures = points.Shape[1];
        Labels = labels;
    }

    /// <summary>
    /// Creates a point cloud from coordinates only (no additional features).
    /// </summary>
    /// <param name="coordinates">Matrix of shape [N, 3] with XYZ coordinates.</param>
    /// <param name="labels">Optional labels for the points.</param>
    /// <returns>A new PointCloudData instance.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A convenient way to create a basic point cloud with just positions.
    ///
    /// Use this when:
    /// - You only have position data (no colors, normals, etc.)
    /// - You want to process purely geometric information
    /// - Your data source only provides XYZ coordinates
    ///
    /// This automatically converts the coordinate matrix to the tensor format needed internally.
    /// </remarks>
    public static PointCloudData<T> FromCoordinates(Matrix<T> coordinates, Vector<T>? labels = null)
    {
        // Convert matrix to tensor [N, 3]
        var tensor = Tensor<T>.FromMatrix(coordinates);
        return new PointCloudData<T>(tensor, labels);
    }

    /// <summary>
    /// Extracts only the XYZ coordinates from the point cloud.
    /// </summary>
    /// <returns>A tensor of shape [N, 3] containing only position information.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Gets just the position data, ignoring any additional features.
    ///
    /// Useful when:
    /// - You need only geometric information
    /// - Performing operations that work on positions only
    /// - Visualizing the basic 3D structure
    ///
    /// If the point cloud has colors or other features, this method discards them
    /// and returns only the X, Y, Z coordinates.
    /// </remarks>
    public Tensor<T> GetCoordinates()
    {
        if (NumFeatures == 3)
        {
            return Points;
        }

        // Extract first 3 channels (XYZ)
        var coords = new T[NumPoints * 3];
        for (int i = 0; i < NumPoints; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                coords[i * 3 + j] = Points.Data.Span[i * NumFeatures + j];
            }
        }

        return new Tensor<T>(coords, [NumPoints, 3]);
    }

    /// <summary>
    /// Extracts additional features (excluding XYZ coordinates).
    /// </summary>
    /// <returns>A tensor of shape [N, F] containing feature data, or null if no features exist.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Gets the extra information (like colors) without the positions.
    ///
    /// Useful when:
    /// - Processing features separately from geometry
    /// - Analyzing color or intensity patterns
    /// - Applying feature-specific transformations
    ///
    /// Returns null if the point cloud only has XYZ coordinates (NumFeatures == 3).
    /// </remarks>
    public Tensor<T>? GetFeatures()
    {
        if (NumFeatures < 3)
        {
            throw new InvalidOperationException("NumFeatures must be at least 3 (XYZ coordinates).");
        }

        if (NumFeatures == 3)
        {
            return null;
        }

        int featureDim = NumFeatures - 3;
        var features = new T[NumPoints * featureDim];

        for (int i = 0; i < NumPoints; i++)
        {
            for (int j = 0; j < featureDim; j++)
            {
                features[i * featureDim + j] = Points.Data.Span[i * NumFeatures + 3 + j];
            }
        }

        return new Tensor<T>(features, [NumPoints, featureDim]);
    }
}
