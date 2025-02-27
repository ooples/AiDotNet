namespace AiDotNet.Factories;

public static class Interpolation2DFactory<T>
{
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