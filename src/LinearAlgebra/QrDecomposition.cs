namespace AiDotNet.LinearAlgebra;

public class QrDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> QMatrix { get; set; }
    private Matrix<double> RMatrix { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public QrDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        QMatrix = new Matrix<double>(AMatrix.RowCount, AMatrix.RowCount);
        RMatrix = new Matrix<double>(AMatrix.RowCount, AMatrix.RowCount);
        Decompose(AMatrix);
        SolutionVector = Solve(QMatrix, BVector);
    }

    public void Decompose(Matrix<double> matrix)
    {
        // Householder method qr algo
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        var columns = matrix.ColumnCount;
        var rows = matrix.RowCount;
        if (rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        if (columns <= 0)
        {
            throw new ArgumentException($"{nameof(columns)} needs to be an integer larger than 0");
        }

        if (rows != columns)
        {
            throw new ArgumentException($"You need to have a square matrix to calculate the determinant value so the length of {nameof(rows)} {nameof(columns)} must be equal");
        }

        var qTemp = MatrixHelper.CreateIdentityMatrix<double>(rows);
        var rTemp = matrix.Duplicate();

        for (int i = 0; i < rows - 1; i++)
        {
            var hMatrix = MatrixHelper.CreateIdentityMatrix<double>(rows);
            var aVector = new Vector<double>(rows - i);
            int k = 0;
            for (int j = i; j < rows; j++)
            {
                aVector[k++] = rTemp[j, i];
            }

            double aNorm = aVector.Norm();
            if (aVector[0] < 0.0)
            { 
                aNorm = -aNorm; 
            }

            var vector = new Vector<double>(aVector.Count);
            for (int j = 0; j < vector.Count; j++)
            {
                vector[j] = aVector[j] / (aVector[0] + aNorm);
                vector[0] = 1.0;
            }

            var hMatrix2 = MatrixHelper.CreateIdentityMatrix<double>(aVector.Count);
            double vvDot = vector.DotProduct(vector);
            var alpha = vector.Reshape(vector.Count, 1);
            var beta = vector.Reshape(1, vector.Count);
            var aMultB = alpha.DotProduct(beta);

            for (int i2 = 0; i2 < hMatrix2.RowCount; i2++)
            {
                for (int j2 = 0; j2 < hMatrix2.ColumnCount; j2++)
                {
                    hMatrix2[i2, j2] -= 2.0 / vvDot * aMultB[i2, j2];
                }
            }

            int d = rows - hMatrix2.RowCount;
            for (int i2 = 0; i2 < hMatrix2.RowCount; i2++)
            {
                for (int j2 = 0; j2 < hMatrix2.ColumnCount; j2++)
                {
                    hMatrix[i2 + d, j2 + d] = hMatrix2[i2, j2];
                }
            }

            qTemp = qTemp.DotProduct(hMatrix);
            rTemp = hMatrix.DotProduct(rTemp);
        }

        QMatrix = qTemp;
        RMatrix = rTemp;
    }

    public Matrix<double> Invert()
    {
        var rMatrixInverted = Inverse(RMatrix);
        var qMatrixTransposed = QMatrix.Transpose();

        return qMatrixTransposed.DotProduct(rMatrixInverted);
    }

    private Matrix<double> Inverse(Matrix<double> matrix)
    {
        int n = matrix.RowCount;
        var inv = new Matrix<double>(n, n);
        var eye = MatrixHelper.CreateIdentityMatrix<double>(n);

        for (int i = 0; i < n; i++)
        {
            var x = Solve(matrix, eye.GetColumn(i));
            for (int j = 0; j < n; ++j)
            {
                inv[j, i] = x[j];
            }
        }

        return inv;
    }

    public Vector<double> Solve(Matrix<double> qMatrix, Vector<double> bVector)
    {
        var Qtb = qMatrix.Transpose().DotProduct(bVector);

        return RMatrix.BackwardSubstitution(Qtb);
    }
}