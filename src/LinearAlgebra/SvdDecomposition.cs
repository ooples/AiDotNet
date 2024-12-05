namespace AiDotNet.LinearAlgebra;

public class SvdDecomposition : IMatrixDecomposition<double>
{
    private Matrix<double> UMatrix { get; set; }
    private Matrix<double> VtMatrix { get; set; }
    private Vector<double> SVector { get; set; }
    private Matrix<double> AMatrix { get; set; }
    private Vector<double> BVector { get; set; }

    public Vector<double> SolutionVector { get; private set; }

    public SvdDecomposition(IEnumerable<IEnumerable<double>> expectedValues, IEnumerable<double> actualValues)
    {
        AMatrix = new Matrix<double>(expectedValues);
        BVector = new Vector<double>(actualValues);
        UMatrix = new Matrix<double>(1, 1);
        VtMatrix = new Matrix<double>(1, 1);
        SVector = new Vector<double>(1);
        Decompose(AMatrix);
        SolutionVector = Solve(AMatrix, BVector);
    }

    public void Decompose(Matrix<double> aMatrix)
    {
        // Golub-Reinsch algorithm
        int m = aMatrix.Rows;
        int n = aMatrix.Columns;
        
        var a = aMatrix.Duplicate();
        UMatrix = new Matrix<double>(m, m);
        SVector = new Vector<double>(Math.Min(m, n));
        VtMatrix = new Matrix<double>(n, n);
      
        // Bidiagonalization phase
        var e = new Vector<double>(n);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                UMatrix[i, j] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                VtMatrix[i, j] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < n; i++)
        {
            // Compute the left singular vectors
            double nrm = 0;
            for (int k = i; k < m; k++)
            {
                nrm = MatrixHelper.Hypotenuse(nrm, a[k, i]);
            }

            if (nrm != 0.0)
            {
                if (a[i, i] < 0)
                {
                    nrm = -nrm;
                }

                for (int k = i; k < m; k++)
                {
                    a[k, i] /= nrm;
                }

                a[i, i] += 1.0;

                for (int j = i + 1; j < n; j++)
                {
                    double s = 0.0;
                    for (int k = i; k < m; k++)
                    {
                        s += a[k, i] * a[k, j];
                    }

                    s = -s / a[i, i];

                    for (int k = i; k < m; k++)
                    {
                        a[k, j] += s * a[k, i];
                    }
                }
            }

            SVector[i] = nrm;
            nrm = 0;

            for (int j = i + 1; j < n; j++)
            {
                nrm = MatrixHelper.Hypotenuse(nrm, a[i, j]);
            }

            if (nrm != 0.0)
            {
                if (a[i, i + 1] < 0)
                {
                    nrm = -nrm;
                }

                for (int k = i + 1; k < n; k++)
                {
                    a[i, k] /= nrm;
                }

                a[i, i + 1] += 1.0;

                for (int k = i + 1; k < m; k++)
                {
                    double s = 0.0;
                    for (int j = i + 1; j < n; j++)
                    {
                        s += a[k, j] * a[i, j];
                    }

                    s = -s / a[i, i + 1];

                    for (int j = i + 1; j < n; j++)
                    {
                        a[k, j] += s * a[i, j];
                    }
                }
            }

            e[i] = nrm;
        }

        // Accumulation of the right-hand transformations
        for (int i = n - 1; i >= 0; i--)
        {
            if (e[i] != 0.0)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double t = 0;
                    for (int k = i + 1; k < n; k++)
                    {
                        t += VtMatrix[k, i] * VtMatrix[k, j];
                    }

                    t = -t / VtMatrix[i + 1, i];

                    for (int k = i + 1; k < n; k++)
                    {
                        VtMatrix[k, j] += t * VtMatrix[k, i];
                    }
                }
            }

            for (int j = 0; j < n; j++)
            {
                VtMatrix[i, j] = 0.0;
            }

            VtMatrix[i, i] = 1.0;
        }

        // Accumulation of the left-hand transformations
        for (int i = n - 1; i >= 0; i--)
        {
            if (SVector[i] != 0.0)
            {
                for (int j = i + 1; j < m; j++)
                {
                    double t = 0;
                    for (int k = i; k < m; k++)
                    {
                        t += UMatrix[k, i] * UMatrix[k, j];
                    }

                    t = -t / UMatrix[i, i];

                    for (int k = i; k < m; k++)
                    {
                        UMatrix[k, j] += t * UMatrix[k, i];
                    }
                }

                for (int j = 0; j < m; j++)
                {
                    UMatrix[j, i] = 0.0;
                }

                UMatrix[i, i] = 1.0;
            }
        }

        // Diagonalization of the bidiagonal form
        for (int k = n - 1; k >= 0; k--)
        {
            for (int iter = 0; iter < 30; iter++)
            {
                bool flag = true;

                int nm = 0;
                for (int l = k; l >= 0; l--)
                {
                    nm = l - 1;
                    if (Math.Abs(e[l]) <= double.Epsilon + double.Epsilon * Math.Abs(SVector[l]) + double.Epsilon * Math.Abs(e[l + 1]))
                    {
                        e[l] = 0.0;
                        flag = false;
                        break;
                    }
                }

                double f = 0, g = 0, h = 0, c = 0, s = 1, y = 0, z = 0;
                if (flag)
                {
                    for (int i = nm; i <= k; i++)
                    {
                        f = s * e[i];
                        e[i] = c * e[i];

                        if (Math.Abs(f) <= double.Epsilon + double.Epsilon * Math.Abs(SVector[i]) + double.Epsilon * Math.Abs(e[i + 1]))
                        {
                            break;
                        }

                        g = SVector[i];
                        h = MatrixHelper.Hypotenuse(f, g);
                        SVector[i] = h;
                        c = g / h;
                        s = -f / h;

                        for (int j = 0; j < m; j++)
                        {
                            y = UMatrix[j, nm];
                            z = UMatrix[j, i];
                            UMatrix[j, nm] = y * c + z * s;
                            UMatrix[j, i] = -y * s + z * c;
                        }
                    }
                }

                z = SVector[k];
                if (nm == k)
                {
                    if (z < 0.0)
                    {
                        SVector[k] = -z;
                        for (int j = 0; j < n; j++)
                        {
                            VtMatrix[j, k] = -VtMatrix[j, k];
                        }
                    }
                    break;
                }

                if (iter == 29)
                {
                    throw new Exception("No convergence in 30 iterations");
                }

                double x = SVector[nm];
                y = SVector[k - 1];
                g = e[k - 1];
                h = e[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = MatrixHelper.Hypotenuse(f, 1.0);

                if (f < 0.0)
                {
                    g = -g;
                }

                f = ((x - z) * (x + z) + h * (y / (f + g) - h)) / x;
                c = s = 1.0;

                for (int j = nm; j <= k - 1; j++)
                {
                    int i = j + 1;
                    g = e[i];
                    y = SVector[i];
                    h = s * g;
                    g = c * g;
                    z = MatrixHelper.Hypotenuse(f, h);
                    e[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = -x * s + g * c;
                    h = y * s;
                    y = y * c;

                    for (int l = 0; l < n; l++)
                    {
                        x = VtMatrix[l, j];
                        z = VtMatrix[l, i];
                        VtMatrix[l, j] = x * c + z * s;
                        VtMatrix[l, i] = -x * s + z * c;
                    }

                    z = MatrixHelper.Hypotenuse(f, h);
                    SVector[j] = z;

                    if (z != 0.0)
                    {
                        c = f / z;
                        s = h / z;
                    }

                    f = c * g + s * y;
                    x = -s * g + c * y;

                    for (int l = 0; l < m; l++)
                    {
                        y = UMatrix[l, j];
                        z = UMatrix[l, i];
                        UMatrix[l, j] = y * c + z * s;
                        UMatrix[l, i] = -y * s + z * c;
                    }
                }

                e[nm] = 0.0;
                e[k] = f;
                SVector[k] = x;
            }
        }
    }

    public Matrix<double> Invert()
    {
        int m = UMatrix.Rows;
        int n = VtMatrix.Columns;

        var sInv = new Matrix<double>(n, m);

        for (int i = 0; i < SVector.Count; i++)
        {
            if (SVector[i] != 0)
            {
                sInv[i, i] = 1.0 / SVector[i];
            }
        }

        var uTranspose = UMatrix.Transpose();
        var vtTranspose = VtMatrix.Transpose();

        return vtTranspose.DotProduct(sInv).DotProduct(uTranspose);
    }

    public Vector<double> Solve(Matrix<double> aMatrix, Vector<double> bVector)
    {
        int m = UMatrix.Rows;
        int n = VtMatrix.Columns;

        var sInv = new Matrix<double>(n, m);
        for (int i = 0; i < SVector.Count; i++)
        {
            if (SVector[i] != 0)
            {
                sInv[i, i] = 1.0 / SVector[i];
            }
        }

        var uTransposeB = UMatrix.Transpose().DotProduct(bVector);
        var sInvUTransposeB = sInv.DotProduct(uTransposeB);

        return VtMatrix.Transpose().DotProduct(sInvUTransposeB);
    }
}