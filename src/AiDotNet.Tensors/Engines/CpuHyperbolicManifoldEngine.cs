using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// CPU implementation of hyperbolic manifold operations for hyperbolic neural networks.
/// </summary>
/// <remarks>
/// <para>
/// CpuHyperbolicManifoldEngine provides numerically stable implementations of operations
/// on hyperbolic spaces. These are used in hyperbolic neural networks for hierarchical
/// data representation.
/// </para>
/// <para><b>For Beginners:</b> This implements the math for hyperbolic neural networks.
/// Hyperbolic space is good for representing tree-like data (hierarchies, taxonomies, graphs).
/// The operations here are the building blocks for hyperbolic layers and models.
/// </para>
/// </remarks>
public sealed class CpuHyperbolicManifoldEngine : IHyperbolicManifoldEngine
{
    /// <summary>
    /// Singleton instance for convenience.
    /// </summary>
    public static CpuHyperbolicManifoldEngine Instance { get; } = new CpuHyperbolicManifoldEngine();

    /// <summary>
    /// Small epsilon for numerical stability (prevents division by zero, overflow).
    /// </summary>
    private const double Epsilon = 1e-15;

    /// <summary>
    /// Maximum norm for points in the Poincare ball (slightly less than 1/sqrt(c)).
    /// </summary>
    private const double MaxNormFactor = 1.0 - 1e-5;

    #region Poincare Ball Model Operations

    /// <inheritdoc/>
    public Vector<T> PoincareExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature)
    {
        if (basePoint is null) throw new ArgumentNullException(nameof(basePoint));
        if (tangentVector is null) throw new ArgumentNullException(nameof(tangentVector));
        if (basePoint.Length != tangentVector.Length)
        {
            throw new ArgumentException("Base point and tangent vector must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0; // Default curvature

        int dim = basePoint.Length;
        double[] x = ToDoubleArray(basePoint, ops);
        double[] v = ToDoubleArray(tangentVector, ops);

        // Compute ||v||
        double vNorm = Norm(v);
        if (vNorm < Epsilon)
        {
            // Zero tangent vector, return base point
            return basePoint;
        }

        // Compute lambda_x = 2 / (1 - c * ||x||^2)
        double xNormSq = DotProduct(x, x);
        double lambdaX = 2.0 / Math.Max(1.0 - c * xNormSq, Epsilon);

        // Compute tanh(sqrt(c) * lambda_x * ||v|| / 2) / (sqrt(c) * ||v||)
        double sqrtC = Math.Sqrt(c);
        double scaledNorm = sqrtC * lambdaX * vNorm / 2.0;

        // Use clamp to prevent overflow in tanh
        scaledNorm = Math.Min(scaledNorm, 20.0); // tanh(20) ≈ 1
        double tanhVal = Math.Tanh(scaledNorm);
        double coeff = tanhVal / (sqrtC * vNorm + Epsilon);

        // Compute direction = v / ||v||
        double[] direction = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            direction[i] = v[i] / (vNorm + Epsilon);
        }

        // Compute scaled direction for Möbius addition
        double[] scaledV = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            scaledV[i] = coeff * v[i];
        }

        // Result = x ⊕ scaledV (Möbius addition)
        double[] result = MobiusAddInternal(x, scaledV, c);

        // Project to ensure we stay inside the ball
        result = ProjectToBall(result, c);

        return FromDoubleArray<T>(result, ops);
    }

    /// <inheritdoc/>
    public Vector<T> PoincareLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature)
    {
        if (basePoint is null) throw new ArgumentNullException(nameof(basePoint));
        if (targetPoint is null) throw new ArgumentNullException(nameof(targetPoint));
        if (basePoint.Length != targetPoint.Length)
        {
            throw new ArgumentException("Base point and target point must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = basePoint.Length;
        double[] x = ToDoubleArray(basePoint, ops);
        double[] y = ToDoubleArray(targetPoint, ops);

        // Compute -x ⊕ y
        double[] negX = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            negX[i] = -x[i];
        }
        double[] diff = MobiusAddInternal(negX, y, c);

        double diffNorm = Norm(diff);
        if (diffNorm < Epsilon)
        {
            // Points are the same, return zero vector
            return new Vector<T>(new T[dim]);
        }

        // Compute lambda_x
        double xNormSq = DotProduct(x, x);
        double lambdaX = 2.0 / Math.Max(1.0 - c * xNormSq, Epsilon);

        // Compute 2 / (sqrt(c) * lambda_x) * arctanh(sqrt(c) * ||diff||)
        double sqrtC = Math.Sqrt(c);
        double arg = sqrtC * diffNorm;
        arg = Math.Min(arg, 1.0 - Epsilon); // Clamp to avoid arctanh(1) = inf
        double atanhVal = Arctanh(arg);
        double coeff = (2.0 / (sqrtC * lambdaX + Epsilon)) * atanhVal / (diffNorm + Epsilon);

        double[] result = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            result[i] = coeff * diff[i];
        }

        return FromDoubleArray<T>(result, ops);
    }

    /// <inheritdoc/>
    public Vector<T> MobiusAdd<T>(Vector<T> x, Vector<T> y, T curvature)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Vectors must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        double[] xArr = ToDoubleArray(x, ops);
        double[] yArr = ToDoubleArray(y, ops);

        double[] result = MobiusAddInternal(xArr, yArr, c);
        result = ProjectToBall(result, c);

        return FromDoubleArray<T>(result, ops);
    }

    /// <inheritdoc/>
    public T PoincareDistance<T>(Vector<T> x, Vector<T> y, T curvature)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Vectors must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = x.Length;
        double[] xArr = ToDoubleArray(x, ops);
        double[] yArr = ToDoubleArray(y, ops);

        // d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕ y||)
        double[] negX = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            negX[i] = -xArr[i];
        }
        double[] diff = MobiusAddInternal(negX, yArr, c);

        double diffNorm = Norm(diff);
        double sqrtC = Math.Sqrt(c);
        double arg = sqrtC * diffNorm;
        arg = Math.Min(arg, 1.0 - Epsilon);
        double distance = (2.0 / sqrtC) * Arctanh(arg);

        return ops.FromDouble(distance);
    }

    /// <inheritdoc/>
    public Vector<T> PoincareParallelTransport<T>(Vector<T> x, Vector<T> y, Vector<T> v, T curvature)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (x.Length != y.Length || x.Length != v.Length)
        {
            throw new ArgumentException("All vectors must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = x.Length;
        double[] xArr = ToDoubleArray(x, ops);
        double[] yArr = ToDoubleArray(y, ops);
        double[] vArr = ToDoubleArray(v, ops);

        // Compute conformal factors
        double xNormSq = DotProduct(xArr, xArr);
        double yNormSq = DotProduct(yArr, yArr);
        double lambdaX = 2.0 / Math.Max(1.0 - c * xNormSq, Epsilon);
        double lambdaY = 2.0 / Math.Max(1.0 - c * yNormSq, Epsilon);

        // Full gyration-based parallel transport formula:
        // PT_{x→y}(v) = (lambda_x / lambda_y) * gyr[y, -x](v)
        // where gyr[a, b](v) is the gyration operator

        // Compute -x ⊕ y for gyration
        double[] negX = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            negX[i] = -xArr[i];
        }

        // Gyration formula: gyr[y, -x](v) = -(-x ⊕ y) ⊕ (-x ⊕ (y ⊕ v))
        // For numerical stability, use the explicit gyration formula:
        // gyr[a, b](v) = v + 2c * (<a, v> * b + <b, v> * a) / (1 + 2c<a,b> + c²||a||²||b||²)
        //              + 2c² * <a, v> * ||b||² * a / denominator
        // Simplified for parallel transport using conformal factor scaling
        double xyDot = DotProduct(xArr, yArr);
        double xvDot = DotProduct(xArr, vArr);
        double yvDot = DotProduct(yArr, vArr);

        // Gyration coefficient calculation
        double denom = 1.0 + 2.0 * c * xyDot + c * c * xNormSq * yNormSq;
        denom = Math.Max(Math.Abs(denom), Epsilon);

        double coeff1 = 2.0 * c * (1.0 + c * yNormSq) * xvDot / denom;
        double coeff2 = -2.0 * c * (1.0 + c * xNormSq) * yvDot / denom;

        // Apply gyration and conformal scaling
        double scale = lambdaX / lambdaY;
        double[] result = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            double gyrated = vArr[i] + coeff1 * yArr[i] + coeff2 * xArr[i];
            result[i] = scale * gyrated;
        }

        return FromDoubleArray<T>(result, ops);
    }

    /// <inheritdoc/>
    public Vector<T> PoincareProject<T>(Vector<T> point, T curvature, T epsilon)
    {
        if (point is null) throw new ArgumentNullException(nameof(point));

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        double eps = ops.ToDouble(epsilon);
        if (eps < Epsilon) eps = 1e-5;

        double[] arr = ToDoubleArray(point, ops);
        double[] result = ProjectToBall(arr, c, 1.0 - eps);

        return FromDoubleArray<T>(result, ops);
    }

    #endregion

    #region Hyperboloid Model Operations

    /// <inheritdoc/>
    public Vector<T> HyperboloidExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature)
    {
        if (basePoint is null) throw new ArgumentNullException(nameof(basePoint));
        if (tangentVector is null) throw new ArgumentNullException(nameof(tangentVector));
        if (basePoint.Length != tangentVector.Length)
        {
            throw new ArgumentException("Base point and tangent vector must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = basePoint.Length;
        double[] x = ToDoubleArray(basePoint, ops);
        double[] v = ToDoubleArray(tangentVector, ops);

        // Compute Lorentzian norm of v: ||v||_L = sqrt(-<v,v>_L) where <a,b>_L = -a0*b0 + sum(ai*bi)
        double vNormL = LorentzianNorm(v);
        if (vNormL < Epsilon)
        {
            return basePoint;
        }

        // exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * (v / ||v||_L)
        double coshVal = Math.Cosh(vNormL);
        double sinhVal = Math.Sinh(vNormL);

        double[] result = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            result[i] = coshVal * x[i] + sinhVal * v[i] / vNormL;
        }

        // Project to hyperboloid
        result = ProjectToHyperboloid(result, c);

        return FromDoubleArray<T>(result, ops);
    }

    /// <inheritdoc/>
    public Vector<T> HyperboloidLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature)
    {
        if (basePoint is null) throw new ArgumentNullException(nameof(basePoint));
        if (targetPoint is null) throw new ArgumentNullException(nameof(targetPoint));
        if (basePoint.Length != targetPoint.Length)
        {
            throw new ArgumentException("Base point and target point must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = basePoint.Length;
        double[] x = ToDoubleArray(basePoint, ops);
        double[] y = ToDoubleArray(targetPoint, ops);

        // Compute Minkowski inner product <x,y>_L
        double innerProduct = MinkowskiInnerProduct(x, y);

        // Clamp for numerical stability (should be <= -1/c for valid hyperboloid points)
        innerProduct = Math.Min(innerProduct, -1.0 / c - Epsilon);

        // Distance: d = (1/sqrt(c)) * arcosh(-c * <x,y>_L)
        double arg = -c * innerProduct;
        arg = Math.Max(arg, 1.0 + Epsilon); // arcosh domain is [1, inf)
        double dist = Arcosh(arg) / Math.Sqrt(c);

        if (dist < Epsilon)
        {
            return new Vector<T>(new T[dim]);
        }

        // log_x(y) = (d / sinh(d)) * (y - cosh(d) * x)
        double coshD = Math.Cosh(dist);
        double sinhD = Math.Sinh(dist);
        double coeff = dist / (sinhD + Epsilon);

        double[] result = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            result[i] = coeff * (y[i] - coshD * x[i]);
        }

        return FromDoubleArray<T>(result, ops);
    }

    /// <inheritdoc/>
    public T HyperboloidDistance<T>(Vector<T> x, Vector<T> y, T curvature)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Vectors must have the same dimension.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        double[] xArr = ToDoubleArray(x, ops);
        double[] yArr = ToDoubleArray(y, ops);

        double innerProduct = MinkowskiInnerProduct(xArr, yArr);
        double arg = -c * innerProduct;
        arg = Math.Max(arg, 1.0 + Epsilon);
        double distance = Arcosh(arg) / Math.Sqrt(c);

        return ops.FromDouble(distance);
    }

    /// <inheritdoc/>
    public Vector<T> HyperboloidProject<T>(Vector<T> point, T curvature)
    {
        if (point is null) throw new ArgumentNullException(nameof(point));

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        double[] arr = ToDoubleArray(point, ops);
        double[] result = ProjectToHyperboloid(arr, c);

        return FromDoubleArray<T>(result, ops);
    }

    #endregion

    #region Model Conversions

    /// <inheritdoc/>
    public Vector<T> PoincareToHyperboloid<T>(Vector<T> poincarePoint, T curvature)
    {
        if (poincarePoint is null) throw new ArgumentNullException(nameof(poincarePoint));

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = poincarePoint.Length;
        double[] p = ToDoubleArray(poincarePoint, ops);
        double pNormSq = DotProduct(p, p);

        // Hyperboloid has one extra dimension (time-like)
        double[] h = new double[dim + 1];

        double sqrtC = Math.Sqrt(c);
        double denom = 1.0 - c * pNormSq;
        if (Math.Abs(denom) < Epsilon) denom = Epsilon;

        // h_0 = (1 + c*||p||^2) / (1 - c*||p||^2) / sqrt(c)
        h[0] = (1.0 + c * pNormSq) / denom / sqrtC;

        // h_i = 2*sqrt(c)*p_i / (1 - c*||p||^2) / sqrt(c) = 2*p_i / (1 - c*||p||^2)
        for (int i = 0; i < dim; i++)
        {
            h[i + 1] = 2.0 * p[i] / denom;
        }

        return FromDoubleArray<T>(h, ops);
    }

    /// <inheritdoc/>
    public Vector<T> HyperboloidToPoincare<T>(Vector<T> hyperboloidPoint, T curvature)
    {
        if (hyperboloidPoint is null) throw new ArgumentNullException(nameof(hyperboloidPoint));
        if (hyperboloidPoint.Length < 2)
        {
            throw new ArgumentException("Hyperboloid point must have at least 2 dimensions.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        double c = Math.Abs(ops.ToDouble(curvature));
        if (c < Epsilon) c = 1.0;

        int dim = hyperboloidPoint.Length - 1;
        double[] h = ToDoubleArray(hyperboloidPoint, ops);

        double sqrtC = Math.Sqrt(c);
        double denom = h[0] * sqrtC + 1.0;
        if (Math.Abs(denom) < Epsilon) denom = Epsilon;

        double[] p = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            p[i] = h[i + 1] / denom;
        }

        // Project to ensure valid Poincare point
        p = ProjectToBall(p, c);

        return FromDoubleArray<T>(p, ops);
    }

    #endregion

    #region Batch Operations

    /// <inheritdoc/>
    public Matrix<T> PoincareExpMapBatch<T>(Matrix<T> basePoints, Matrix<T> tangentVectors, T curvature)
    {
        if (basePoints is null) throw new ArgumentNullException(nameof(basePoints));
        if (tangentVectors is null) throw new ArgumentNullException(nameof(tangentVectors));
        if (basePoints.Rows != tangentVectors.Rows || basePoints.Columns != tangentVectors.Columns)
        {
            throw new ArgumentException("Base points and tangent vectors matrices must have the same shape.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(basePoints.Rows, basePoints.Columns);

        for (int i = 0; i < basePoints.Rows; i++)
        {
            var basePoint = basePoints.GetRow(i);
            var tangent = tangentVectors.GetRow(i);
            var expResult = PoincareExpMap(basePoint, tangent, curvature);

            for (int j = 0; j < basePoints.Columns; j++)
            {
                result[i, j] = expResult[j];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> PoincareDistanceBatch<T>(Matrix<T> x, Matrix<T> y, T curvature)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (x.Rows != y.Rows || x.Columns != y.Columns)
        {
            throw new ArgumentException("Matrices must have the same shape.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new T[x.Rows];

        for (int i = 0; i < x.Rows; i++)
        {
            var xRow = x.GetRow(i);
            var yRow = y.GetRow(i);
            result[i] = PoincareDistance(xRow, yRow, curvature);
        }

        return new Vector<T>(result);
    }

    #endregion

    #region Helper Methods

    private static double[] ToDoubleArray<T>(Vector<T> v, INumericOperations<T> ops)
    {
        double[] result = new double[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = ops.ToDouble(v[i]);
        }
        return result;
    }

    private static Vector<T> FromDoubleArray<T>(double[] arr, INumericOperations<T> ops)
    {
        T[] result = new T[arr.Length];
        for (int i = 0; i < arr.Length; i++)
        {
            result[i] = ops.FromDouble(arr[i]);
        }
        return new Vector<T>(result);
    }

    private static double Norm(double[] v)
    {
        double sum = 0.0;
        for (int i = 0; i < v.Length; i++)
        {
            sum += v[i] * v[i];
        }
        return Math.Sqrt(sum);
    }

    private static double DotProduct(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double[] MobiusAddInternal(double[] x, double[] y, double c)
    {
        int dim = x.Length;
        double xNormSq = DotProduct(x, x);
        double yNormSq = DotProduct(y, y);
        double xyDot = DotProduct(x, y);

        double denom = 1.0 + 2.0 * c * xyDot + c * c * xNormSq * yNormSq;
        if (Math.Abs(denom) < Epsilon) denom = Epsilon;

        double[] result = new double[dim];
        double coeff1 = 1.0 + 2.0 * c * xyDot + c * yNormSq;
        double coeff2 = 1.0 - c * xNormSq;

        for (int i = 0; i < dim; i++)
        {
            result[i] = (coeff1 * x[i] + coeff2 * y[i]) / denom;
        }

        return result;
    }

    private static double[] ProjectToBall(double[] point, double c, double maxNormFactor = MaxNormFactor)
    {
        double maxNorm = maxNormFactor / Math.Sqrt(c);
        double norm = Norm(point);

        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            double[] result = new double[point.Length];
            for (int i = 0; i < point.Length; i++)
            {
                result[i] = point[i] * scale;
            }
            return result;
        }

        return point;
    }

    private static double[] ProjectToHyperboloid(double[] point, double c)
    {
        // Hyperboloid constraint: -x0^2 + x1^2 + ... = -1/c
        // Adjust x0 to satisfy the constraint
        int dim = point.Length;
        if (dim < 1) return point;

        double spatialNormSq = 0.0;
        for (int i = 1; i < dim; i++)
        {
            spatialNormSq += point[i] * point[i];
        }

        // x0 = sqrt(1/c + spatial_norm^2)
        double x0 = Math.Sqrt(1.0 / c + spatialNormSq);

        double[] result = new double[dim];
        result[0] = x0;
        for (int i = 1; i < dim; i++)
        {
            result[i] = point[i];
        }

        return result;
    }

    private static double MinkowskiInnerProduct(double[] a, double[] b)
    {
        // <a,b>_L = -a0*b0 + sum(ai*bi) for i > 0
        if (a.Length < 1 || b.Length < 1) return 0.0;

        double result = -a[0] * b[0];
        for (int i = 1; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    private static double LorentzianNorm(double[] v)
    {
        // ||v||_L = sqrt(|<v,v>_L|)
        double innerProd = MinkowskiInnerProduct(v, v);
        return Math.Sqrt(Math.Abs(innerProd));
    }

    private static double Arctanh(double x)
    {
        // arctanh(x) = 0.5 * log((1+x)/(1-x))
        x = Math.Max(-1.0 + Epsilon, Math.Min(1.0 - Epsilon, x));
        return 0.5 * Math.Log((1.0 + x) / (1.0 - x));
    }

    private static double Arcosh(double x)
    {
        // arcosh(x) = log(x + sqrt(x^2 - 1))
        x = Math.Max(1.0, x);
        return Math.Log(x + Math.Sqrt(x * x - 1.0));
    }

    #endregion
}
