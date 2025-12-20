namespace AiDotNet.Enums;

/// <summary>
/// Specifies different methods for interpolating 2D data points to create a continuous surface.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Interpolation is like "filling in the blanks" between known data points. 
/// Imagine you have temperature readings from several weather stations across a city, and you 
/// want to estimate the temperature at locations between these stations. Interpolation methods 
/// are different mathematical techniques to make these estimates.
/// 
/// Each method has different strengths:
/// - Some are faster but less accurate
/// - Some preserve certain properties of your data better than others
/// - Some work better for smooth data, others for data with sharp changes
/// 
/// The right choice depends on your specific data and what properties you want to preserve.
/// </para>
/// </remarks>
public enum Interpolation2DType
{
    /// <summary>
    /// A simple, fast interpolation method that uses linear interpolation in both x and y directions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bilinear interpolation is like drawing straight lines between your known points 
    /// and using those lines to estimate values in between.
    /// 
    /// Think of it as:
    /// - The simplest and fastest method
    /// - Like stretching a flat sheet over your data points
    /// - Good for when you need quick results and don't need perfect smoothness
    /// - Similar to how a low-resolution image looks when zoomed in
    /// 
    /// Best used when:
    /// - Speed is more important than accuracy
    /// - Your data is already fairly smooth
    /// - You're working with grid-like data (like images)
    /// - You need a simple, predictable result
    /// </para>
    /// </remarks>
    Bilinear,

    /// <summary>
    /// A smoother interpolation method that uses cubic polynomials in both x and y directions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bicubic interpolation creates a smoother surface than bilinear by using curved lines 
    /// instead of straight lines between points.
    /// 
    /// Think of it as:
    /// - Like bilinear, but with curves that create smoother transitions
    /// - Similar to how high-quality image resizing works
    /// - Preserves smoothness better than bilinear
    /// - Still relatively fast, but more accurate
    /// 
    /// Best used when:
    /// - You need smoother results than bilinear provides
    /// - Working with data that should be continuous and smooth
    /// - Resizing images or other grid data with better quality
    /// - You want a good balance between speed and smoothness
    /// </para>
    /// </remarks>
    Bicubic,

    /// <summary>
    /// A flexible interpolation method that minimizes the bending energy of a thin metal plate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Thin Plate Spline interpolation creates a smooth surface that passes through all your data points 
    /// while minimizing the overall "bending" of the surface.
    /// 
    /// Think of it as:
    /// - Like placing a thin, flexible metal sheet over your data points
    /// - The sheet bends to touch all points but stays as flat as possible elsewhere
    /// - Creates very natural-looking smooth surfaces
    /// - Good for scattered (non-grid) data points
    /// 
    /// Best used when:
    /// - Your data points aren't arranged in a grid
    /// - You need a smooth surface that passes exactly through your data points
    /// - You're working with geographic or spatial data
    /// - You want natural-looking results for physical phenomena
    /// </para>
    /// </remarks>
    ThinPlateSpline,

    /// <summary>
    /// A geostatistical method that uses spatial correlation between data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kriging is an advanced method that considers how data points relate to each other 
    /// based on distance and direction. It was originally developed for mining and geology applications.
    /// 
    /// Think of it as:
    /// - A "smart" method that learns patterns from your data
    /// - Takes into account how values tend to vary with distance
    /// - Provides both estimated values AND uncertainty estimates
    /// - More computationally intensive but potentially more accurate
    /// 
    /// Best used when:
    /// - Your data has spatial patterns or trends
    /// - You need to know how confident you can be in the interpolated values
    /// - Working with geospatial data like elevation, rainfall, or mineral concentrations
    /// - You have enough data points to establish reliable spatial relationships
    /// </para>
    /// </remarks>
    Kriging,

    /// <summary>
    /// A distance-weighted interpolation method that gives more influence to nearby points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Shepard's Method calculates values based on the idea that nearby points 
    /// should have more influence than distant points.
    /// 
    /// Think of it as:
    /// - Like a weighted average where closer points count more
    /// - The influence of each point decreases with distance
    /// - Simple to understand and implement
    /// - Works with irregularly spaced data points
    /// 
    /// Best used when:
    /// - You have scattered data points (not in a grid)
    /// - Closer points should logically have more influence
    /// - You need a method that's intuitive and relatively simple
    /// - You want to avoid complex mathematical models
    /// </para>
    /// </remarks>
    ShepardsMethod,

    /// <summary>
    /// A flexible method that fits local polynomial functions to nearby data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Moving Least Squares creates a smooth surface by fitting small, simple 
    /// mathematical functions to groups of nearby points.
    /// 
    /// Think of it as:
    /// - Like having many small "patches" that blend together
    /// - Each area is influenced mainly by nearby points
    /// - Creates a smooth surface that adapts to local patterns
    /// - More flexible than simpler methods
    /// 
    /// Best used when:
    /// - Your data has different patterns in different regions
    /// - You need a smooth result that adapts to local features
    /// - Working with complex surfaces like terrain or 3D models
    /// - You need better quality than simple methods but don't want the complexity of Kriging
    /// </para>
    /// </remarks>
    MovingLeastSquares,

    /// <summary>
    /// An interpolation method using radial basis functions with multiquadratic form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MultiQuadratic interpolation uses special mathematical functions that 
    /// create smooth hills and valleys centered at each data point.
    /// 
    /// Think of it as:
    /// - Placing a smooth bump or dip at each data point
    /// - These bumps blend together to form a continuous surface
    /// - Creates very smooth results even with scattered data
    /// - Good for capturing both local and global patterns
    /// 
    /// Best used when:
    /// - You need very smooth interpolation
    /// - Working with scattered data points
    /// - Your data represents a physical phenomenon that should be smooth
    /// - You need accurate results and smoothness is important
    /// </para>
    /// </remarks>
    MultiQuadratic,

    /// <summary>
    /// An interpolation method that preserves the sharpness of edges while providing smooth results elsewhere.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cubic Convolution is similar to bicubic interpolation but uses a wider range 
    /// of neighboring points to calculate each value.
    /// 
    /// Think of it as:
    /// - An enhanced version of bicubic interpolation
    /// - Better at preserving edges and details
    /// - Creates smooth results without excessive blurring
    /// - Commonly used in image processing and remote sensing
    /// 
    /// Best used when:
    /// - You need to preserve edges and details
    /// - Working with images or grid data
    /// - You want better quality than bicubic but without artifacts
    /// - Your data contains both smooth regions and sharp transitions
    /// </para>
    /// </remarks>
    CubicConvolution
}
