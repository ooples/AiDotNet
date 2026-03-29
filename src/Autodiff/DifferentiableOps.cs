using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff;

/// <summary>
/// Differentiable tensor operations that record to the active <see cref="GradientTape{T}"/>.
/// Each method performs the forward computation via <see cref="IEngine"/> and registers a backward
/// function for reverse-mode automatic differentiation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>When no tape is active, these are zero-overhead wrappers around Engine ops.</para>
/// <para>
/// Backward functions follow standard autodiff rules:
/// <list type="bullet">
/// <item>Add: d(a+b)/da = 1, d(a+b)/db = 1</item>
/// <item>Multiply: d(a*b)/da = b, d(a*b)/db = a</item>
/// <item>MatMul(A,B): dA = grad @ B^T, dB = A^T @ grad</item>
/// <item>Sigmoid: grad * σ(x) * (1 - σ(x))</item>
/// <item>Tanh: grad * (1 - tanh²(x))</item>
/// <item>ReLU: grad * (x > 0)</item>
/// </list>
/// </para>
/// <para><b>Reference:</b> Baydin et al. "Automatic Differentiation in Machine Learning: a Survey" (2018)</para>
/// </remarks>
public static class DifferentiableOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    // ─── Elementwise ops ─────────────────────────────────────────────

    /// <summary>Elementwise addition: c = a + b.</summary>
    public static Tensor<T> Add(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorAdd(a, b);
        GradientTape<T>.Current?.RecordOp("Add", [a, b], result,
            grad => [grad, grad]);
        return result;
    }

    /// <summary>Elementwise subtraction: c = a - b.</summary>
    public static Tensor<T> Subtract(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorSubtract(a, b);
        GradientTape<T>.Current?.RecordOp("Subtract", [a, b], result,
            grad =>
            {
                var negGrad = Engine.TensorNegate(grad);
                return [grad, negGrad];
            });
        return result;
    }

    /// <summary>Elementwise multiplication (Hadamard product): c = a ⊙ b.</summary>
    public static Tensor<T> Multiply(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorMultiply(a, b);
        GradientTape<T>.Current?.RecordOp("Multiply", [a, b], result,
            grad =>
            [
                Engine.TensorMultiply(grad, b),
                Engine.TensorMultiply(grad, a)
            ]);
        return result;
    }

    /// <summary>Elementwise division: c = a / b.</summary>
    public static Tensor<T> Divide(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorDivide(a, b);
        GradientTape<T>.Current?.RecordOp("Divide", [a, b], result,
            grad =>
            {
                // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
                var gradA = Engine.TensorDivide(grad, b);
                var bSquared = Engine.TensorMultiply(b, b);
                var gradB = Engine.TensorNegate(Engine.TensorDivide(
                    Engine.TensorMultiply(grad, a), bSquared));
                return [gradA, gradB];
            });
        return result;
    }

    /// <summary>Elementwise negation: c = -a.</summary>
    public static Tensor<T> Negate(Tensor<T> a)
    {
        var result = Engine.TensorNegate(a);
        GradientTape<T>.Current?.RecordOp("Negate", [a], result,
            grad => [Engine.TensorNegate(grad)]);
        return result;
    }

    // ─── Scalar ops ──────────────────────────────────────────────────

    /// <summary>Multiply all elements by a scalar: c = a * scalar.</summary>
    public static Tensor<T> MultiplyScalar(Tensor<T> a, T scalar)
    {
        var result = Engine.TensorMultiplyScalar(a, scalar);
        GradientTape<T>.Current?.RecordOp("MultiplyScalar", [a], result,
            grad => [Engine.TensorMultiplyScalar(grad, scalar)]);
        return result;
    }

    /// <summary>Add a scalar to all elements: c = a + scalar.</summary>
    public static Tensor<T> AddScalar(Tensor<T> a, T scalar)
    {
        var result = Engine.TensorAddScalar(a, scalar);
        GradientTape<T>.Current?.RecordOp("AddScalar", [a], result,
            grad => [grad]); // d(a+c)/da = 1
        return result;
    }

    // ─── Matrix ops ──────────────────────────────────────────────────

    /// <summary>Matrix multiplication: C = A @ B.</summary>
    public static Tensor<T> MatMul(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorMatMul(a, b);
        GradientTape<T>.Current?.RecordOp("MatMul", [a, b], result,
            grad =>
            {
                // dL/dA = grad @ B^T
                var bT = Engine.TensorTranspose(b);
                var gradA = Engine.TensorMatMul(grad, bT);
                // dL/dB = A^T @ grad
                var aT = Engine.TensorTranspose(a);
                var gradB = Engine.TensorMatMul(aT, grad);
                return [gradA, gradB];
            });
        return result;
    }

    /// <summary>Transpose the last two dimensions.</summary>
    public static Tensor<T> Transpose(Tensor<T> a)
    {
        var result = Engine.TensorTranspose(a);
        GradientTape<T>.Current?.RecordOp("Transpose", [a], result,
            grad => [Engine.TensorTranspose(grad)]);
        return result;
    }

    // ─── Reduction ops ───────────────────────────────────────────────

    /// <summary>Sum all elements to a scalar.</summary>
    public static Tensor<T> Sum(Tensor<T> a)
    {
        var result = new Tensor<T>([1]);
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
            sum = NumOps.Add(sum, a[i]);
        result[0] = sum;

        GradientTape<T>.Current?.RecordOp("Sum", [a], result,
            grad =>
            {
                // dSum/da_i = 1 for all i, broadcast scalar grad
                var expanded = Tensor<T>.CreateDefault(a.Shape.ToArray(), grad[0]);
                return [expanded];
            });
        return result;
    }

    /// <summary>Mean of all elements to a scalar.</summary>
    public static Tensor<T> Mean(Tensor<T> a)
    {
        var result = new Tensor<T>([1]);
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
            sum = NumOps.Add(sum, a[i]);
        result[0] = NumOps.Divide(sum, NumOps.FromDouble(a.Length));

        GradientTape<T>.Current?.RecordOp("Mean", [a], result,
            grad =>
            {
                // dMean/da_i = 1/n for all i
                T scale = NumOps.Divide(grad[0], NumOps.FromDouble(a.Length));
                var expanded = Tensor<T>.CreateDefault(a.Shape.ToArray(), scale);
                return [expanded];
            });
        return result;
    }

    // ─── Activation ops ──────────────────────────────────────────────

    /// <summary>Elementwise sigmoid: σ(x) = 1/(1+exp(-x)).</summary>
    public static Tensor<T> Sigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }

        GradientTape<T>.Current?.RecordOp("Sigmoid", [x], result,
            grad =>
            {
                // σ'(x) = σ(x) * (1 - σ(x))
                var ones = Tensor<T>.CreateDefault(result.Shape.ToArray(), NumOps.One);
                var oneMinusSigmoid = Engine.TensorSubtract(ones, result);
                var derivative = Engine.TensorMultiply(result, oneMinusSigmoid);
                return [Engine.TensorMultiply(grad, derivative)];
            });
        return result;
    }

    /// <summary>Elementwise tanh.</summary>
    public static Tensor<T> Tanh(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(x[i])));

        GradientTape<T>.Current?.RecordOp("Tanh", [x], result,
            grad =>
            {
                // tanh'(x) = 1 - tanh²(x)
                var tanhSquared = Engine.TensorMultiply(result, result);
                var ones = Tensor<T>.CreateDefault(result.Shape.ToArray(), NumOps.One);
                var derivative = Engine.TensorSubtract(ones, tanhSquared);
                return [Engine.TensorMultiply(grad, derivative)];
            });
        return result;
    }

    /// <summary>Elementwise ReLU: max(0, x).</summary>
    public static Tensor<T> ReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = val > 0 ? x[i] : NumOps.Zero;
        }

        GradientTape<T>.Current?.RecordOp("ReLU", [x], result,
            grad =>
            {
                // ReLU'(x) = 1 if x > 0, else 0
                var mask = new Tensor<T>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++)
                    mask[i] = NumOps.ToDouble(x[i]) > 0 ? NumOps.One : NumOps.Zero;
                return [Engine.TensorMultiply(grad, mask)];
            });
        return result;
    }

    /// <summary>Elementwise GELU (Gaussian Error Linear Unit).</summary>
    public static Tensor<T> GELU(Tensor<T> x)
    {
        // GELU(x) = x * Φ(x) where Φ is the standard normal CDF
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        const double sqrt2OverPi = 0.7978845608028654; // sqrt(2/π)
        const double coeff = 0.044715;

        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            double inner = sqrt2OverPi * (xi + coeff * xi * xi * xi);
            double tanhInner = Math.Tanh(inner);
            result[i] = NumOps.FromDouble(0.5 * xi * (1.0 + tanhInner));
        }

        GradientTape<T>.Current?.RecordOp("GELU", [x], result,
            grad =>
            {
                // GELU'(x) = 0.5 * (1 + tanh(c)) + 0.5 * x * sech²(c) * c'
                // where c = sqrt(2/π) * (x + 0.044715x³), c' = sqrt(2/π) * (1 + 3*0.044715*x²)
                var derivative = new Tensor<T>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++)
                {
                    double xi = NumOps.ToDouble(x[i]);
                    double c = sqrt2OverPi * (xi + coeff * xi * xi * xi);
                    double tanhC = Math.Tanh(c);
                    double sech2C = 1.0 - tanhC * tanhC;
                    double cPrime = sqrt2OverPi * (1.0 + 3.0 * coeff * xi * xi);
                    double geluPrime = 0.5 * (1.0 + tanhC) + 0.5 * xi * sech2C * cPrime;
                    derivative[i] = NumOps.FromDouble(geluPrime);
                }
                return [Engine.TensorMultiply(grad, derivative)];
            });
        return result;
    }

    /// <summary>Elementwise Swish/SiLU: x * σ(x).</summary>
    public static Tensor<T> Swish(Tensor<T> x)
    {
        var sig = new Tensor<T>(x.Shape.ToArray());
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            double s = 1.0 / (1.0 + Math.Exp(-xi));
            sig[i] = NumOps.FromDouble(s);
            result[i] = NumOps.FromDouble(xi * s);
        }

        GradientTape<T>.Current?.RecordOp("Swish", [x], result,
            grad =>
            {
                // Swish'(x) = σ(x) + x * σ(x) * (1 - σ(x)) = σ(x) * (1 + x * (1 - σ(x)))
                var derivative = new Tensor<T>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++)
                {
                    double xi = NumOps.ToDouble(x[i]);
                    double s = NumOps.ToDouble(sig[i]);
                    derivative[i] = NumOps.FromDouble(s * (1.0 + xi * (1.0 - s)));
                }
                return [Engine.TensorMultiply(grad, derivative)];
            });
        return result;
    }

    // ─── Shape ops ───────────────────────────────────────────────────

    /// <summary>Reshape tensor (no data copy — gradient just reshapes back).</summary>
    public static Tensor<T> Reshape(Tensor<T> a, int[] newShape)
    {
        var originalShape = a.Shape.ToArray();
        var result = a.Reshape(newShape);

        GradientTape<T>.Current?.RecordOp("Reshape", [a], result,
            grad => [grad.Reshape(originalShape)]);
        return result;
    }
}
