using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class SSADecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _windowSize;
    private readonly int _numberOfComponents;
    private readonly SSAAlgorithmType _algorithmType;

    public SSADecomposition(Vector<T> timeSeries, int windowSize, int numberOfComponents, SSAAlgorithmType algorithmType = SSAAlgorithmType.Basic)
        : base(timeSeries)
    {
        if (windowSize <= 0 || windowSize > timeSeries.Length / 2)
        {
            throw new ArgumentException("Window size must be positive and not greater than half the time series length.", nameof(windowSize));
        }

        if (numberOfComponents <= 0 || numberOfComponents > windowSize)
        {
            throw new ArgumentException("Number of components must be positive and not greater than the window size.", nameof(numberOfComponents));
        }

        _windowSize = windowSize;
        _numberOfComponents = numberOfComponents;
        _algorithmType = algorithmType;
        Decompose();
    }

    protected override void Decompose()
    {
        Matrix<T> trajectoryMatrix;
        Matrix<T> U;
        Vector<T> S;
        Matrix<T> V;

        switch (_algorithmType)
        {
            case SSAAlgorithmType.Basic:
                trajectoryMatrix = CreateTrajectoryMatrix();
                (U, S, V) = PerformSVD(trajectoryMatrix);
                break;
            case SSAAlgorithmType.Sequential:
                (U, S, V) = PerformSequentialSSA();
                break;
            case SSAAlgorithmType.Toeplitz:
                (U, S, V) = PerformToeplitzSSA();
                break;
            default:
                throw new ArgumentException("Invalid SSA algorithm type.");
        }

        var groupedComponents = GroupComponents(U, S, V);
        var reconstructedComponents = ReconstructComponents(groupedComponents);

        AssignComponents(reconstructedComponents);
    }

    private (Matrix<T> U, Vector<T> S, Matrix<T> V) PerformSVD(Matrix<T> trajectoryMatrix)
    {
        var svdDecomposition = new SvdDecomposition<T>(trajectoryMatrix);
        return (svdDecomposition.U, svdDecomposition.S, svdDecomposition.Vt);
    }

    private (Matrix<T> U, Vector<T> S, Matrix<T> V) PerformSequentialSSA()
    {
        int N = TimeSeries.Length;
        int K = N - _windowSize + 1;
    
        Matrix<T> X = CreateTrajectoryMatrix();
        Matrix<T> U = new Matrix<T>(_windowSize, _numberOfComponents);
        Vector<T> S = new Vector<T>(_numberOfComponents);
        Matrix<T> V = new Matrix<T>(K, _numberOfComponents);

        for (int i = 0; i < _numberOfComponents; i++)
        {
            Vector<T> u = Vector<T>.CreateRandom(_windowSize);
            u = u.Normalize();

            for (int iter = 0; iter < 10; iter++) // Number of iterations for power method
            {
                Vector<T> v1 = X.Transpose().Multiply(u);
                T s1 = v1.Norm();
                v1 = v1.Divide(s1);

                u = X.Multiply(v1);
                u = u.Normalize();
            }

            Vector<T> v = X.Transpose().Multiply(u);
            T s = v.Norm();
            v = v.Divide(s);

            U.SetColumn(i, u);
            S[i] = s;
            V.SetColumn(i, v);

            // Deflate X
            X = X.Subtract(u.OuterProduct(v).Multiply(s));
        }

        return (U, S, V);
    }

    private (Matrix<T> U, Vector<T> S, Matrix<T> V) PerformToeplitzSSA()
    {
        int N = TimeSeries.Length;
        int K = N - _windowSize + 1;

        // Create the Toeplitz matrix
        Matrix<T> C = new Matrix<T>(_windowSize, _windowSize);
        for (int i = 0; i < _windowSize; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < N - i + j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(TimeSeries[k], TimeSeries[k + i - j]));
                }
                C[i, j] = C[j, i] = NumOps.Divide(sum, NumOps.FromDouble(N - i + j));
            }
        }

        // Perform eigendecomposition
        var eigenDecomposition = new EigenDecomposition<T>(C);
        Matrix<T> U = eigenDecomposition.EigenVectors;
        Vector<T> S = eigenDecomposition.EigenValues.Transform(x => NumOps.Sqrt(x));

        // Compute V
        Matrix<T> V = new Matrix<T>(K, _numberOfComponents);
        Matrix<T> X = CreateTrajectoryMatrix();
        for (int i = 0; i < _numberOfComponents; i++)
        {
            Vector<T> v = X.Transpose().Multiply(U.GetColumn(i));
            v = v.Divide(S[i]);
            V.SetColumn(i, v);
        }

        // Truncate to the specified number of components
        U = U.Submatrix(0, _windowSize - 1, 0, _numberOfComponents - 1);
        S = S.Subvector(0, _numberOfComponents - 1);
        V = V.Submatrix(0, K - 1, 0, _numberOfComponents - 1);

        return (U, S, V);
    }

    private void AssignComponents(Vector<T>[] reconstructedComponents)
    {
        AddComponent(DecompositionComponentType.Trend, reconstructedComponents[0]);
        
        if (reconstructedComponents.Length > 2)
        {
            AddComponent(DecompositionComponentType.Seasonal, reconstructedComponents[1]);
            
            var residual = new Vector<T>(TimeSeries.Length);
            for (int i = 2; i < reconstructedComponents.Length; i++)
            {
                residual = residual.Add(reconstructedComponents[i]);
            }
            AddComponent(DecompositionComponentType.Residual, residual);
        }
        else if (reconstructedComponents.Length == 2)
        {
            AddComponent(DecompositionComponentType.Residual, reconstructedComponents[1]);
        }
    }

    private Matrix<T> CreateTrajectoryMatrix()
    {
        int K = TimeSeries.Length - _windowSize + 1;
        var trajectoryMatrix = new Matrix<T>(_windowSize, K);

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < _windowSize; j++)
            {
                trajectoryMatrix[j, i] = TimeSeries[i + j];
            }
        }

        return trajectoryMatrix;
    }

    private Matrix<T>[] GroupComponents(Matrix<T> U, Vector<T> S, Matrix<T> V)
    {
        var groupedComponents = new Matrix<T>[_numberOfComponents];

        for (int i = 0; i < _numberOfComponents; i++)
        {
            var Ui = U.GetColumn(i);
            var Vi = V.GetColumn(i);
            groupedComponents[i] = Ui.OuterProduct(Vi).Multiply(S[i]);
        }

        return groupedComponents;
    }

    private Vector<T>[] ReconstructComponents(Matrix<T>[] groupedComponents)
    {
        var reconstructedComponents = new Vector<T>[_numberOfComponents];

        for (int k = 0; k < _numberOfComponents; k++)
        {
            var component = new Vector<T>(TimeSeries.Length);
            var X = groupedComponents[k];

            for (int i = 0; i < TimeSeries.Length; i++)
            {
                T sum = NumOps.Zero;
                int count = 0;

                for (int j = Math.Max(0, i - X.Columns + 1); j <= Math.Min(i, X.Rows - 1); j++)
                {
                    sum = NumOps.Add(sum, X[j, i - j]);
                    count++;
                }

                component[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
            }

            reconstructedComponents[k] = component;
        }

        return reconstructedComponents;
    }
}