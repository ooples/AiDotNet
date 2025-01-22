namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class SSADecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _windowSize;
    private readonly int _numberOfComponents;

    public SSADecomposition(Vector<T> timeSeries, int windowSize, int numberOfComponents)
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
    }

    public void Decompose()
    {
        // Step 1: Embedding
        var trajectoryMatrix = CreateTrajectoryMatrix();

        // Step 2: SVD
        var svdDecomposition = new SvdDecomposition<T>(trajectoryMatrix);

        Matrix<T> U = svdDecomposition.U;
        Vector<T> S = svdDecomposition.S;
        Matrix<T> V = svdDecomposition.Vt;

        // Step 3: Grouping
        var groupedComponents = GroupComponents(U, S, V);

        // Step 4: Diagonal Averaging (Reconstruction)
        var reconstructedComponents = ReconstructComponents(groupedComponents);

        // Assign components
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