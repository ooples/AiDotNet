namespace AiDotNet.Interpolation;

public class GaussianProcessInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly GaussianProcessRegression<T> _gpr;
    private readonly INumericOperations<T> _numOps;

    public GaussianProcessInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        var options = new GaussianProcessRegressionOptions
        {
            OptimizeHyperparameters = true,
            MaxIterations = 100,
            Tolerance = 1e-6,
            NoiseLevel = 1e-8
        };

        _gpr = new GaussianProcessRegression<T>(options);
        
        // Train the GPR model
        Matrix<T> xMatrix = new Matrix<T>(_x.Length, 1);
        for (int i = 0; i < _x.Length; i++)
        {
            xMatrix[i, 0] = _x[i];
        }

        _gpr.Train(xMatrix, _y);
    }

    public T Interpolate(T x)
    {
        Vector<T> xVector = new Vector<T>([x]);
        Matrix<T> xMatrix = new Matrix<T>(1, 1);
        xMatrix[0, 0] = x;

        Vector<T> prediction = _gpr.Predict(xMatrix);
        return prediction[0];
    }
}