using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class PolarDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> A { get; }
    public Matrix<T> U { get; private set; }
    public Matrix<T> P { get; private set; }

    public PolarDecomposition(Matrix<T> matrix, PolarAlgorithmType algorithm = PolarAlgorithmType.SVD)
    {
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
        U = new Matrix<T>(matrix.Rows, matrix.Columns, NumOps);
        P = new Matrix<T>(matrix.Rows, matrix.Columns, NumOps);
        Decompose(algorithm);
    }

    public void Decompose(PolarAlgorithmType algorithm = PolarAlgorithmType.SVD)
    {
        switch (algorithm)
        {
            case PolarAlgorithmType.SVD:
                DecomposeSVD();
                break;
            case PolarAlgorithmType.NewtonSchulz:
                DecomposeNewtonSchulz();
                break;
            case PolarAlgorithmType.HalleyIteration:
                DecomposeHalleyIteration();
                break;
            case PolarAlgorithmType.QRIteration:
                DecomposeQRIteration();
                break;
            case PolarAlgorithmType.ScalingAndSquaring:
                DecomposeScalingAndSquaring();
                break;
            default:
                throw new ArgumentException("Unsupported Polar decomposition algorithm.");
        }
    }

    private void DecomposeSVD()
    {
        var svd = new SvdDecomposition<T>(A);

        U = svd.U.Multiply(svd.Vt.Transpose());
        var sigma = Matrix<T>.CreateDiagonal(svd.S, NumOps);
        P = svd.Vt.Transpose().Multiply(sigma).Multiply(svd.Vt);
    }

    private void DecomposeNewtonSchulz()
    {
        Matrix<T> X = A.Copy();
        Matrix<T> Y = Matrix<T>.CreateIdentity(A.Rows, NumOps);
        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            Matrix<T> XtX = X.Transpose().Multiply(X);
            Matrix<T> YtY = Y.Transpose().Multiply(Y);

            // Check for numerical stability
            if (!MatrixHelper.IsInvertible(XtX) || !MatrixHelper.IsInvertible(YtY))
            {
                throw new InvalidOperationException("Matrix became singular during Newton-Schulz iteration.");
            }

            Matrix<T> nextX = X.Multiply(NumOps.FromDouble(0.5)).Add(Y.Transpose().Multiply(NumOps.FromDouble(0.5)));
            Matrix<T> nextY = Y.Multiply(NumOps.FromDouble(0.5)).Add(XtX.Inverse().Multiply(X.Transpose()).Multiply(NumOps.FromDouble(0.5)));

            T errorX = nextX.Subtract(X).FrobeniusNorm();
            T errorY = nextY.Subtract(Y).FrobeniusNorm();

            if (NumOps.LessThan(errorX, tolerance) && NumOps.LessThan(errorY, tolerance))
            {
                break;
            }

            X = nextX;
            Y = nextY;

            // Check for divergence
            if (NumOps.GreaterThan(errorX, NumOps.FromDouble(1e6)) || NumOps.GreaterThan(errorY, NumOps.FromDouble(1e6)))
            {
                throw new InvalidOperationException("Newton-Schulz iteration diverged.");
            }
        }

        U = X;
        P = X.Transpose().Multiply(A);

        // Ensure orthogonality of U
        U = MatrixHelper.OrthogonalizeColumns(U);
    }

    private void DecomposeHalleyIteration()
    {
        Matrix<T> X = A.Copy();
        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            if (!MatrixHelper.IsInvertible(X))
            {
                throw new InvalidOperationException("Matrix became singular during Halley iteration.");
            }

            Matrix<T> Y = X.Inverse();
            Matrix<T> Z = Y.Transpose();
            Matrix<T> nextX = X.Multiply(NumOps.FromDouble(3)).Add(Z).Multiply(NumOps.FromDouble(0.25))
                .Add(X.Multiply(NumOps.FromDouble(3)).Multiply(Y).Multiply(Z).Multiply(NumOps.FromDouble(0.25)));

            T error = nextX.Subtract(X).FrobeniusNorm();

            if (NumOps.LessThan(error, tolerance))
            {
                break;
            }

            X = nextX;

            // Check for divergence
            if (NumOps.GreaterThan(error, NumOps.FromDouble(1e6)))
            {
                throw new InvalidOperationException("Halley iteration diverged.");
            }
        }

        U = X;
        P = X.Transpose().Multiply(A);

        // Ensure orthogonality of U
        U = MatrixHelper.OrthogonalizeColumns(U);
    }

    private void DecomposeQRIteration()
    {
        Matrix<T> X = A.Copy();
        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            var qr = new QrDecomposition<T>(X);
            Matrix<T> Q = qr.Q;
            Matrix<T> R = qr.R;

            Matrix<T> nextX = Q.Multiply(R.Add(R.Transpose())).Multiply(NumOps.FromDouble(0.5));

            T error = nextX.Subtract(X).FrobeniusNorm();

            if (NumOps.LessThan(error, tolerance))
            {
                break;
            }

            X = nextX;

            // Check for divergence
            if (NumOps.GreaterThan(error, NumOps.FromDouble(1e6)))
            {
                throw new InvalidOperationException("QR iteration diverged.");
            }
        }

        U = X;
        P = X.Transpose().Multiply(A);

        // Ensure orthogonality of U
        U = MatrixHelper.OrthogonalizeColumns(U);
    }

    private void DecomposeScalingAndSquaring()
    {
        Matrix<T> X = A.Copy();
        T norm = MatrixHelper.SpectralNorm(X);
        int scalingFactor = (int)Math.Ceiling(MathHelper.Log2(Convert.ToDouble(norm)));

        if (scalingFactor > 0)
        {
            X = X.Multiply(NumOps.FromDouble(Math.Pow(2, -scalingFactor)));
        }

        Matrix<T> Y = Matrix<T>.CreateIdentity(A.Rows, NumOps);
        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 20;

        for (int i = 0; i < maxIterations; i++)
        {
            if (!MatrixHelper.IsInvertible(Y))
            {
                throw new InvalidOperationException("Matrix became singular during Scaling and Squaring iteration.");
            }

            Matrix<T> Z = X.Subtract(Y.Inverse());
            Y = Y.Add(Y.Multiply(Z).Multiply(NumOps.FromDouble(0.5)));
            X = X.Subtract(Z.Multiply(X).Multiply(NumOps.FromDouble(0.5)));

            T error = Z.FrobeniusNorm();

            if (NumOps.LessThan(error, tolerance))
            {
                break;
            }

            // Check for divergence
            if (NumOps.GreaterThan(error, NumOps.FromDouble(1e6)))
            {
                throw new InvalidOperationException("Scaling and Squaring iteration diverged.");
            }
        }

        for (int i = 0; i < scalingFactor; i++)
        {
            Y = Y.Multiply(Y);
        }

        U = Y;
        P = Y.Transpose().Multiply(A);

        // Ensure orthogonality of U
        U = MatrixHelper.OrthogonalizeColumns(U);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        // Solve Px = b
        var x = MatrixSolutionHelper.SolveLinearSystem(P, b, MatrixDecompositionType.Polar);

        // Compute y = U^T * x (equivalent to solving Uy = x)
        return U.Transpose().Multiply(x);
    }

    public Matrix<T> Invert()
    {
        var invP = P.Inverse();
        var invU = U.Transpose();

        return invP.Multiply(invU);
    }

    public (Matrix<T> U, Matrix<T> P) GetFactors()
    {
        return (U, P);
    }
}