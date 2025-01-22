namespace AiDotNet.Interpolation;

public class Interpolation2DTo1DAdapter<T> : IInterpolation<T>
{
    private readonly I2DInterpolation<T> _interpolation2D;
    private readonly T _fixedCoordinate;
    private readonly bool _isXFixed;

    public string Description { get; }

    public Interpolation2DTo1DAdapter(I2DInterpolation<T> interpolation2D, T fixedCoordinate, bool isXFixed)
    {
        _interpolation2D = interpolation2D;
        _fixedCoordinate = fixedCoordinate;
        _isXFixed = isXFixed;
        Description = $"2D interpolation slice with {(_isXFixed ? "X" : "Y")} fixed at {_fixedCoordinate}";
    }

    public T Interpolate(T point)
    {
        return _isXFixed
            ? _interpolation2D.Interpolate(_fixedCoordinate, point)
            : _interpolation2D.Interpolate(point, _fixedCoordinate);
    }
}