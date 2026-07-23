namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates 1D interpolation functions from 2D interpolation methods.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Interpolation is a method of finding new data points within the range 
/// of a discrete set of known data points. It's like filling in the gaps between dots on a graph.
/// </para>
/// <para>
/// This factory helps you create interpolation functions that can estimate values between known data points.
/// Think of it like predicting what happens between measurements - if you know the temperature at 1pm and 3pm,
/// interpolation helps you estimate what it was at 2pm.
/// </para>
/// </remarks>
public static class Interpolation2DFactory<T>
{
    /// <summary>
    /// Creates a 1D interpolation function from a 2D interpolation by fixing one coordinate.
    /// </summary>
    /// <param name="x">The x-coordinate values.</param>
    /// <param name="y">The y-coordinate values.</param>
    /// <param name="zMatrix">The matrix of z-values corresponding to the x and y coordinates.</param>
    /// <param name="zVector">A vector of z-values for point-based interpolation methods.</param>
    /// <param name="type">The type of 2D interpolation to use.</param>
    /// <param name="fixedCoordinate">The value of the coordinate to fix (either x or y).</param>
    /// <param name="isXFixed">If true, the x-coordinate is fixed; otherwise, the y-coordinate is fixed.</param>
    /// <param name="kernelFunction">Optional kernel function for methods like Kriging.</param>
    /// <param name="matrixDecomposition">Optional matrix decomposition method for solving linear systems.</param>
    /// <returns>A 1D interpolation function that can be used to estimate values along the non-fixed coordinate.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported interpolation type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a 2D interpolation (which works with a surface) and creates a 1D interpolation 
    /// (which works with a line) by fixing one of the coordinates. It's like taking a slice through a 3D surface.
    /// </para>
    /// <para>
    /// Available interpolation types include:
    /// <list type="bullet">
    /// <item><description>Bilinear: A simple method that performs linear interpolation in both directions.</description></item>
    /// <item><description>Bicubic: A more advanced method that uses cubic polynomials for smoother results.</description></item>
    /// <item><description>ThinPlateSpline: Creates a smooth surface that passes through all data points.</description></item>
    /// <item><description>Kriging: A geostatistical method that estimates values based on spatial correlation.</description></item>
    /// <item><description>ShepardsMethod: Uses weighted averages where closer points have more influence.</description></item>
    /// <item><description>MovingLeastSquares: Fits local polynomials to create a smooth interpolation.</description></item>
    /// <item><description>MultiQuadratic: Uses radial basis functions for interpolation.</description></item>
    /// <item><description>CubicConvolution: Provides smooth interpolation with continuous first derivatives.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The parameters <paramref name="kernelFunction"/> and <paramref name="matrixDecomposition"/> are advanced options 
    /// that most beginners won't need to modify. They provide customization for specific mathematical aspects of certain 
    /// interpolation methods.
    /// </para>
    /// </remarks>
    public static IInterpolation<T> Create1DFromSlice(
        Vector<T> x, Vector<T> y, Matrix<T> zMatrix, Vector<T> zVector, Interpolation2DType type, T fixedCoordinate, bool isXFixed, IKernelFunction<T>? kernelFunction = null,
        IMatrixDecomposition<T>? matrixDecomposition = null)
    {
        I2DInterpolation<T> interpolation2D = type switch
        {
            Interpolation2DType.Bilinear => new BilinearInterpolation<T>(x, y, zMatrix),
            Interpolation2DType.Bicubic => new BicubicInterpolation<T>(x, y, zMatrix, matrixDecomposition),
            Interpolation2DType.ThinPlateSpline => new ThinPlateSplineInterpolation<T>(x, y, zVector, matrixDecomposition),
            Interpolation2DType.Kriging => new KrigingInterpolation<T>(x, y, zVector, kernelFunction, matrixDecomposition),
            Interpolation2DType.ShepardsMethod => new ShepardsMethodInterpolation<T>(x, y, zVector),
            Interpolation2DType.MovingLeastSquares => new MovingLeastSquaresInterpolation<T>(x, y, zVector, decomposition: matrixDecomposition),
            Interpolation2DType.MultiQuadratic => new MultiquadricInterpolation<T>(x, y, zVector, decomposition: matrixDecomposition),
            Interpolation2DType.CubicConvolution => new CubicConvolutionInterpolation<T>(x, y, zMatrix),
            _ => throw new ArgumentException("Unsupported 2D interpolation type", nameof(type))
        };

        return new Interpolation2DTo1DAdapter<T>(interpolation2D, fixedCoordinate, isXFixed);
    }
}
