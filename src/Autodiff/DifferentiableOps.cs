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

    // ─── Broadcast gradient reduction ────────────────────────────────

    /// <summary>
    /// Reduces a gradient tensor to match the original input shape by summing
    /// along dimensions that were broadcast during the forward pass.
    /// Required for correct autodiff when operands have different shapes.
    /// </summary>
    private static Tensor<T> ReduceBroadcastGrad(Tensor<T> grad, int[] targetShape)
    {
        var gradShape = grad.Shape.ToArray();

        // Same shape — no reduction needed
        if (gradShape.SequenceEqual(targetShape))
            return grad;

        // Same total length but different shape — just reshape
        int gradLen = grad.Length;
        int targetLen = 1;
        foreach (int d in targetShape) targetLen *= d;
        if (gradLen == targetLen)
            return grad.Reshape(targetShape);

        // Scalar target — sum everything
        if (targetShape.Length == 0 || (targetShape.Length == 1 && targetShape[0] == 1))
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < grad.Length; i++)
                sum = NumOps.Add(sum, grad[i]);
            var scalar = new Tensor<T>([1]);
            scalar[0] = sum;
            return scalar;
        }

        // General case: pad targetShape with leading 1s to match grad rank,
        // then sum along dimensions where target has size 1
        int rank = gradShape.Length;
        var padded = new int[rank];
        int offset = rank - targetShape.Length;
        for (int i = 0; i < rank; i++)
            padded[i] = i < offset ? 1 : targetShape[i - offset];

        // Find axes to reduce (where padded == 1 but grad > 1)
        var result = grad;
        for (int axis = 0; axis < rank; axis++)
        {
            if (padded[axis] == 1 && gradShape[axis] > 1)
            {
                result = Engine.ReduceSum(result, new[] { axis }, keepDims: true);
            }
        }

        return result.Reshape(targetShape);
    }

    // ─── Elementwise ops ─────────────────────────────────────────────

    /// <summary>Elementwise addition: c = a + b. Supports broadcasting.</summary>
    public static Tensor<T> Add(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorAdd(a, b);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var aShape = a.Shape.ToArray();
            var bShape = b.Shape.ToArray();
            tape.RecordOp("Add", [a, b], result,
                grad =>
                [
                    ReduceBroadcastGrad(grad, aShape),
                    ReduceBroadcastGrad(grad, bShape)
                ]);
        }
        return result;
    }

    /// <summary>Elementwise subtraction: c = a - b. Supports broadcasting.</summary>
    public static Tensor<T> Subtract(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorSubtract(a, b);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var aShape = a.Shape.ToArray();
            var bShape = b.Shape.ToArray();
            tape.RecordOp("Subtract", [a, b], result,
                grad =>
                [
                    ReduceBroadcastGrad(grad, aShape),
                    ReduceBroadcastGrad(Negate(grad), bShape)
                ]);
        }
        return result;
    }

    /// <summary>Elementwise multiplication (Hadamard product): c = a ⊙ b. Supports broadcasting.</summary>
    public static Tensor<T> Multiply(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorMultiply(a, b);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var aShape = a.Shape.ToArray();
            var bShape = b.Shape.ToArray();
            tape.RecordOp("Multiply", [a, b], result,
                grad =>
                [
                    ReduceBroadcastGrad(Multiply(grad, b), aShape),
                    ReduceBroadcastGrad(Multiply(grad, a), bShape)
                ]);
        }
        return result;
    }

    /// <summary>Elementwise division: c = a / b. Supports broadcasting. Guards against b=0.</summary>
    public static Tensor<T> Divide(Tensor<T> a, Tensor<T> b)
    {
        var result = Engine.TensorDivide(a, b);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var aShape = a.Shape.ToArray();
            var bShape = b.Shape.ToArray();
            tape.RecordOp("Divide", [a, b], result,
                grad =>
                {
                    var gradA = Divide(grad, b);
                    var bSquared = Multiply(b, b);
                    var eps = NumOps.FromDouble(1e-12);
                    var safeBSquared = new Tensor<T>(bSquared.Shape.ToArray());
                    for (int i = 0; i < bSquared.Length; i++)
                        safeBSquared[i] = NumOps.GreaterThan(
                            NumOps.Abs(bSquared[i]), eps) ? bSquared[i] : eps;
                    var gradB = Negate(Divide(Multiply(grad, a), safeBSquared));
                    return [ReduceBroadcastGrad(gradA, aShape), ReduceBroadcastGrad(gradB, bShape)];
                });
        }
        return result;
    }

    /// <summary>Elementwise negation: c = -a.</summary>
    public static Tensor<T> Negate(Tensor<T> a)
    {
        var result = Engine.TensorNegate(a);
        GradientTape<T>.Current?.RecordOp("Negate", [a], result,
            grad => [Negate(grad)]);
        return result;
    }

    // ─── Scalar ops ──────────────────────────────────────────────────

    /// <summary>Multiply all elements by a scalar: c = a * scalar.</summary>
    public static Tensor<T> MultiplyScalar(Tensor<T> a, T scalar)
    {
        var result = Engine.TensorMultiplyScalar(a, scalar);
        GradientTape<T>.Current?.RecordOp("MultiplyScalar", [a], result,
            grad => [MultiplyScalar(grad, scalar)]);
        return result;
    }

    /// <summary>Add a scalar to all elements: c = a + scalar.</summary>
    public static Tensor<T> AddScalar(Tensor<T> a, T scalar)
    {
        var result = Engine.TensorAddScalar(a, scalar);
        GradientTape<T>.Current?.RecordOp("AddScalar", [a], result,
            grad => [grad]);
        return result;
    }

    /// <summary>Divide all elements by a scalar: c = a / scalar.</summary>
    public static Tensor<T> DivideScalar(Tensor<T> a, T scalar)
    {
        var result = Engine.TensorDivideScalar(a, scalar);
        GradientTape<T>.Current?.RecordOp("DivideScalar", [a], result,
            grad => [DivideScalar(grad, scalar)]);
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
                var bT = Transpose(b);
                var gradA = MatMul(grad, bT);
                var aT = Transpose(a);
                var gradB = MatMul(aT, grad);
                return [gradA, gradB];
            });
        return result;
    }

    /// <summary>Transpose the last two dimensions.</summary>
    public static Tensor<T> Transpose(Tensor<T> a)
    {
        var result = Engine.TensorTranspose(a);
        GradientTape<T>.Current?.RecordOp("Transpose", [a], result,
            grad => [Transpose(grad)]);
        return result;
    }

    // ─── Reduction ops ───────────────────────────────────────────────

    /// <summary>Sum all elements to a scalar.</summary>
    public static Tensor<T> Sum(Tensor<T> a)
    {
        // Use Engine's SIMD-accelerated sum when available
        var result = new Tensor<T>([1]);
        result[0] = Engine.TensorSum(a);

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
        if (a.Length == 0)
        {
            throw new ArgumentException("Cannot compute mean of a zero-length tensor.", nameof(a));
        }

        // Use Engine's SIMD-accelerated sum then divide by count
        var result = new Tensor<T>([1]);
        result[0] = NumOps.Divide(Engine.TensorSum(a), NumOps.FromDouble(a.Length));

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

    // ─── Math ops (Log, Exp, Pow, Sqrt, Abs, Clamp) ───────────────────

    /// <summary>Elementwise natural logarithm. Needed for NLLLoss, KL divergence.</summary>
    public static Tensor<T> Log(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Log(Math.Max(NumOps.ToDouble(x[i]), 1e-12)));

        GradientTape<T>.Current?.RecordOp("Log", [x], result,
            grad =>
            {
                // d(log(x))/dx = 1/x, clamped for safety
                var dx = new Tensor<T>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++)
                {
                    double xi = Math.Max(NumOps.ToDouble(x[i]), 1e-12);
                    dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) / xi);
                }
                return [dx];
            });
        return result;
    }

    /// <summary>Elementwise exponential. Needed for log-space computations.</summary>
    public static Tensor<T> Exp(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(x[i])));

        GradientTape<T>.Current?.RecordOp("Exp", [x], result,
            grad =>
            {
                // d(exp(x))/dx = exp(x) = result
                return [Multiply(grad, result)];
            });
        return result;
    }

    /// <summary>Elementwise power: x^n. Needed for Lp norms, polynomial features.</summary>
    public static Tensor<T> Pow(Tensor<T> x, double n)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Pow(NumOps.ToDouble(x[i]), n));

        GradientTape<T>.Current?.RecordOp("Pow", [x], result,
            grad =>
            {
                // d(x^n)/dx = n * x^(n-1)
                var dx = new Tensor<T>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++)
                {
                    double xi = NumOps.ToDouble(x[i]);
                    dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * n * Math.Pow(xi, n - 1));
                }
                return [dx];
            });
        return result;
    }

    /// <summary>Elementwise square root. Needed for RMSNorm, distance metrics.</summary>
    public static Tensor<T> Sqrt(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Sqrt(Math.Max(NumOps.ToDouble(x[i]), 0)));

        GradientTape<T>.Current?.RecordOp("Sqrt", [x], result,
            grad =>
            {
                // d(sqrt(x))/dx = 1/(2*sqrt(x)), clamped for safety
                var dx = new Tensor<T>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++)
                {
                    double sqrtX = Math.Max(NumOps.ToDouble(result[i]), 1e-12);
                    dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) / (2.0 * sqrtX));
                }
                return [dx];
            });
        return result;
    }

    /// <summary>Elementwise absolute value. Needed for L1 loss.</summary>
    public static Tensor<T> Abs(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(x[i])));

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var signs = new Tensor<T>(x.Shape.ToArray());
            for (int i = 0; i < x.Length; i++)
                signs[i] = NumOps.FromDouble(NumOps.ToDouble(x[i]) >= 0 ? 1.0 : -1.0);

            tape.RecordOp("Abs", [x], result,
                grad => [Multiply(grad, signs)]);
        }
        return result;
    }

    /// <summary>Elementwise clamp to [min, max]. Needed for gradient clipping, ReLU6.</summary>
    public static Tensor<T> Clamp(Tensor<T> x, double min, double max)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        var tape = GradientTape<T>.Current;
        Tensor<T>? mask = null;
        if (tape is not null)
            mask = new Tensor<T>(x.Shape.ToArray());

        for (int i = 0; i < x.Length; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(Math.Max(min, Math.Min(max, xi)));
            if (mask is not null)
                mask[i] = (xi >= min && xi <= max) ? NumOps.One : NumOps.Zero;
        }

        if (tape is not null)
        {
            var capturedMask = mask;
            tape.RecordOp("Clamp", [x], result,
                grad => [Multiply(grad, capturedMask ?? grad)]);
        }
        return result;
    }

    // ─── Activation ops ──────────────────────────────────────────────

    /// <summary>Elementwise sigmoid: σ(x) = 1/(1+exp(-x)). Uses Engine vectorized path.</summary>
    public static Tensor<T> Sigmoid(Tensor<T> x)
    {
        var result = Engine.Sigmoid(x);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Sigmoid", [x], result,
                grad =>
                {
                    // σ'(x) = σ(x) * (1 - σ(x))
                    return [Engine.SigmoidBackward(grad, result)];
                });
        }
        return result;
    }

    /// <summary>Elementwise tanh. Uses Engine vectorized path.</summary>
    public static Tensor<T> Tanh(Tensor<T> x)
    {
        var result = Engine.Tanh(x);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Tanh", [x], result,
                grad =>
                {
                    // tanh'(x) = 1 - tanh²(x)
                    return [Engine.TanhBackward(grad, result)];
                });
        }
        return result;
    }

    /// <summary>Elementwise ReLU: max(0, x). Uses byte mask (1 byte vs 4-8 bytes per element).</summary>
    public static Tensor<T> ReLU(Tensor<T> x)
    {
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            // When recording, compute with mask for backward
            var result = new Tensor<T>(x.Shape.ToArray());
            var mask = new byte[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                double val = NumOps.ToDouble(x[i]);
                result[i] = val > 0 ? x[i] : NumOps.Zero;
                mask[i] = val > 0 ? (byte)1 : (byte)0;
            }

            var shape = x.Shape.ToArray();
            tape.RecordOp("ReLU", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(shape);
                    for (int i = 0; i < grad.Length; i++)
                        dx[i] = mask[i] == 1 ? grad[i] : NumOps.Zero;
                    return [dx];
                });
            return result;
        }

        // No tape — just compute forward (Engine ReLU or manual)
        var fwd = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            fwd[i] = val > 0 ? x[i] : NumOps.Zero;
        }
        return fwd;
    }

    /// <summary>Elementwise GELU (Gaussian Error Linear Unit).</summary>
    public static Tensor<T> GELU(Tensor<T> x)
    {
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;

        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            double inner = sqrt2OverPi * (xi + coeff * xi * xi * xi);
            result[i] = NumOps.FromDouble(0.5 * xi * (1.0 + Math.Tanh(inner)));
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("GELU", [x], result,
                grad =>
                {
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
                    return [Multiply(grad, derivative)];
                });
        }
        return result;
    }

    /// <summary>Elementwise Swish/SiLU: x * σ(x). Uses Engine vectorized sigmoid.</summary>
    public static Tensor<T> Swish(Tensor<T> x)
    {
        // Forward: Swish(x) = x * σ(x) — use Engine.Sigmoid for vectorized σ
        var sig = Engine.Sigmoid(x);
        var result = Engine.TensorMultiply(x, sig);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Swish", [x], result,
                grad =>
                {
                    // Swish'(x) = σ(x) * (1 + x * (1 - σ(x)))
                    var derivative = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double xi = NumOps.ToDouble(x[i]);
                        double s = NumOps.ToDouble(sig[i]);
                        derivative[i] = NumOps.FromDouble(s * (1.0 + xi * (1.0 - s)));
                    }
                    return [Multiply(grad, derivative)];
                });
        }
        return result;
    }

    // ─── Shape ops ───────────────────────────────────────────────────

    /// <summary>Reshape tensor (no data copy — gradient just reshapes back).</summary>
    public static Tensor<T> Reshape(Tensor<T> a, int[] newShape)
    {
        var result = a.Reshape(newShape);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var originalShape = a.Shape.ToArray();
            tape.RecordOp("Reshape", [a], result,
                grad => [grad.Reshape(originalShape)]);
        }
        return result;
    }

    /// <summary>
    /// Concatenates tensors along the specified axis.
    /// Backward splits the gradient back into per-input chunks.
    /// Essential for U-Net skip connections and multi-head attention.
    /// </summary>
    public static Tensor<T> Concatenate(Tensor<T>[] tensors, int axis = 0)
    {
        if (tensors.Length == 0)
            throw new ArgumentException("At least one tensor required.", nameof(tensors));
        if (tensors.Length == 1)
            return tensors[0];

        // Compute output shape
        var firstShape = tensors[0].Shape.ToArray();
        int rank = firstShape.Length;
        if (axis < 0) axis += rank;
        var outShape = (int[])firstShape.Clone();
        var splitSizes = new int[tensors.Length];
        splitSizes[0] = firstShape[axis];
        for (int t = 1; t < tensors.Length; t++)
        {
            splitSizes[t] = tensors[t].Shape[axis];
            outShape[axis] += splitSizes[t];
        }

        // Forward: concatenate along axis
        var result = new Tensor<T>(outShape);
        int offset = 0;
        int outerSize = 1, innerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= outShape[d];
        for (int d = axis + 1; d < rank; d++) innerSize *= outShape[d];

        for (int t = 0; t < tensors.Length; t++)
        {
            int dimSize = splitSizes[t];
            for (int o = 0; o < outerSize; o++)
            {
                for (int d = 0; d < dimSize; d++)
                {
                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        int srcIdx = (o * dimSize + d) * innerSize + inner;
                        int dstIdx = (o * outShape[axis] + offset + d) * innerSize + inner;
                        result[dstIdx] = tensors[t][srcIdx];
                    }
                }
            }
            offset += dimSize;
        }

        var capturedShapes = tensors.Select(t => t.Shape.ToArray()).ToArray();
        GradientTape<T>.Current?.RecordOp("Concatenate", tensors, result,
            grad =>
            {
                // Backward: split gradient along the concat axis
                var grads = new Tensor<T>[capturedShapes.Length];
                int off = 0;
                for (int t = 0; t < capturedShapes.Length; t++)
                {
                    int dimSize = capturedShapes[t][axis];
                    grads[t] = new Tensor<T>(capturedShapes[t]);
                    for (int o = 0; o < outerSize; o++)
                    {
                        for (int d = 0; d < dimSize; d++)
                        {
                            for (int inner = 0; inner < innerSize; inner++)
                            {
                                int srcIdx = (o * outShape[axis] + off + d) * innerSize + inner;
                                int dstIdx = (o * dimSize + d) * innerSize + inner;
                                grads[t][dstIdx] = grad[srcIdx];
                            }
                        }
                    }
                    off += dimSize;
                }
                return grads;
            });
        return result;
    }

    /// <summary>
    /// Splits a tensor into chunks along the specified axis.
    /// Backward concatenates the gradient chunks back together.
    /// </summary>
    public static Tensor<T>[] Split(Tensor<T> x, int[] splitSizes, int axis = 0)
    {
        var shape = x.Shape.ToArray();
        int rank = shape.Length;
        if (axis < 0) axis += rank;

        int outerSize = 1, innerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= shape[d];
        for (int d = axis + 1; d < rank; d++) innerSize *= shape[d];

        var results = new Tensor<T>[splitSizes.Length];
        int offset = 0;
        for (int t = 0; t < splitSizes.Length; t++)
        {
            int dimSize = splitSizes[t];
            var chunkShape = (int[])shape.Clone();
            chunkShape[axis] = dimSize;
            results[t] = new Tensor<T>(chunkShape);
            for (int o = 0; o < outerSize; o++)
            {
                for (int d = 0; d < dimSize; d++)
                {
                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        int srcIdx = (o * shape[axis] + offset + d) * innerSize + inner;
                        int dstIdx = (o * dimSize + d) * innerSize + inner;
                        results[t][dstIdx] = x[srcIdx];
                    }
                }
            }
            offset += dimSize;
        }

        // Record one op per output chunk, with backward concatenating gradients
        var capturedAxis = axis;
        var capturedShape = shape;
        for (int t = 0; t < results.Length; t++)
        {
            int chunkIdx = t;
            GradientTape<T>.Current?.RecordOp($"Split[{t}]", [x], results[t],
                grad =>
                {
                    // Backward: place this chunk's gradient at the right position in the full gradient
                    var fullGrad = new Tensor<T>(capturedShape);
                    int off = 0;
                    for (int k = 0; k < chunkIdx; k++) off += splitSizes[k];
                    int ds = splitSizes[chunkIdx];
                    for (int o = 0; o < outerSize; o++)
                    {
                        for (int d = 0; d < ds; d++)
                        {
                            for (int inner = 0; inner < innerSize; inner++)
                            {
                                int dstIdx = (o * capturedShape[capturedAxis] + off + d) * innerSize + inner;
                                int srcIdx = (o * ds + d) * innerSize + inner;
                                fullGrad[dstIdx] = grad[srcIdx];
                            }
                        }
                    }
                    return [fullGrad];
                });
        }

        return results;
    }

    // ─── Dropout / Pooling / Embedding ops ─────────────────────────────

    /// <summary>
    /// Dropout: randomly zeros elements with probability p during training.
    /// Backward: gradient is masked by the same pattern used in forward.
    /// </summary>
    public static Tensor<T> Dropout(Tensor<T> x, double p, bool training, Random? rng = null)
    {
        if (!training || p <= 0)
            return x;

        rng ??= new Random();
        var result = new Tensor<T>(x.Shape.ToArray());
        var mask = new byte[x.Length];
        double scale = 1.0 / (1.0 - p);

        for (int i = 0; i < x.Length; i++)
        {
            if (rng.NextDouble() >= p)
            {
                mask[i] = 1;
                result[i] = NumOps.FromDouble(NumOps.ToDouble(x[i]) * scale);
            }
            else
            {
                mask[i] = 0;
                result[i] = NumOps.Zero;
            }
        }

        var shape = x.Shape.ToArray();
        GradientTape<T>.Current?.RecordOp("Dropout", [x], result,
            grad =>
            {
                var dx = new Tensor<T>(shape);
                for (int i = 0; i < grad.Length; i++)
                    dx[i] = mask[i] == 1 ? NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale) : NumOps.Zero;
                return [dx];
            });
        return result;
    }

    /// <summary>
    /// 2D average pooling. Backward distributes gradient evenly to each pooled region.
    /// </summary>
    public static Tensor<T> AvgPool2D(Tensor<T> x, int poolSize, int stride)
    {
        var shape = x.Shape.ToArray();
        // Expect [N, C, H, W] or [C, H, W]
        int rank = shape.Length;
        int h = shape[rank - 2];
        int w = shape[rank - 1];
        int outH = (h - poolSize) / stride + 1;
        int outW = (w - poolSize) / stride + 1;

        var outShape = (int[])shape.Clone();
        outShape[rank - 2] = outH;
        outShape[rank - 1] = outW;
        var result = new Tensor<T>(outShape);

        int channels = x.Length / (h * w);
        double invPoolArea = 1.0 / (poolSize * poolSize);

        for (int c = 0; c < channels; c++)
        {
            for (int oh = 0; oh < outH; oh++)
            {
                for (int ow = 0; ow < outW; ow++)
                {
                    double sum = 0;
                    for (int ph = 0; ph < poolSize; ph++)
                        for (int pw = 0; pw < poolSize; pw++)
                            sum += NumOps.ToDouble(x[c * h * w + (oh * stride + ph) * w + ow * stride + pw]);
                    result[c * outH * outW + oh * outW + ow] = NumOps.FromDouble(sum * invPoolArea);
                }
            }
        }

        GradientTape<T>.Current?.RecordOp("AvgPool2D", [x], result,
            grad =>
            {
                var dx = new Tensor<T>(shape);
                double invArea = 1.0 / (poolSize * poolSize);
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            double g = NumOps.ToDouble(grad[c * outH * outW + oh * outW + ow]) * invArea;
                            for (int ph = 0; ph < poolSize; ph++)
                                for (int pw = 0; pw < poolSize; pw++)
                                {
                                    int idx = c * h * w + (oh * stride + ph) * w + ow * stride + pw;
                                    dx[idx] = NumOps.FromDouble(NumOps.ToDouble(dx[idx]) + g);
                                }
                        }
                    }
                }
                return [dx];
            });
        return result;
    }

    /// <summary>
    /// 2D max pooling. Stores argmax indices for backward pass.
    /// </summary>
    public static Tensor<T> MaxPool2D(Tensor<T> x, int poolSize, int stride)
    {
        var shape = x.Shape.ToArray();
        int rank = shape.Length;
        int h = shape[rank - 2];
        int w = shape[rank - 1];
        int outH = (h - poolSize) / stride + 1;
        int outW = (w - poolSize) / stride + 1;

        var outShape = (int[])shape.Clone();
        outShape[rank - 2] = outH;
        outShape[rank - 1] = outW;
        var result = new Tensor<T>(outShape);
        var argmax = new int[result.Length]; // Store which input index was max

        int channels = x.Length / (h * w);

        for (int c = 0; c < channels; c++)
        {
            for (int oh = 0; oh < outH; oh++)
            {
                for (int ow = 0; ow < outW; ow++)
                {
                    double maxVal = double.MinValue;
                    int maxIdx = 0;
                    for (int ph = 0; ph < poolSize; ph++)
                    {
                        for (int pw = 0; pw < poolSize; pw++)
                        {
                            int idx = c * h * w + (oh * stride + ph) * w + ow * stride + pw;
                            double val = NumOps.ToDouble(x[idx]);
                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }
                    int outIdx = c * outH * outW + oh * outW + ow;
                    result[outIdx] = NumOps.FromDouble(maxVal);
                    argmax[outIdx] = maxIdx;
                }
            }
        }

        GradientTape<T>.Current?.RecordOp("MaxPool2D", [x], result,
            grad =>
            {
                // Gradient flows only to the max element in each pool window
                var dx = new Tensor<T>(shape);
                for (int i = 0; i < grad.Length; i++)
                    dx[argmax[i]] = NumOps.Add(dx[argmax[i]], grad[i]);
                return [dx];
            });
        return result;
    }

    /// <summary>
    /// Embedding lookup: maps integer indices to dense vectors.
    /// Backward: sparse gradient accumulation into the embedding table.
    /// </summary>
    public static Tensor<T> Embedding(Tensor<T> table, int[] indices)
    {
        int embDim = table.Shape[1];
        var result = new Tensor<T>([indices.Length, embDim]);
        for (int i = 0; i < indices.Length; i++)
            for (int d = 0; d < embDim; d++)
                result[i * embDim + d] = table[indices[i] * embDim + d];

        GradientTape<T>.Current?.RecordOp("Embedding", [table], result,
            grad =>
            {
                // Sparse gradient: accumulate into rows that were looked up
                var tableGrad = new Tensor<T>(table.Shape.ToArray());
                for (int i = 0; i < indices.Length; i++)
                    for (int d = 0; d < embDim; d++)
                        tableGrad[indices[i] * embDim + d] = NumOps.Add(
                            tableGrad[indices[i] * embDim + d], grad[i * embDim + d]);
                return [tableGrad];
            });
        return result;
    }

    // ─── Conv ops ────────────────────────────────────────────────────

    /// <summary>
    /// 2D convolution: output = Conv2D(input, kernel, stride, padding, dilation).
    /// Backward uses Engine.Conv2DBackwardInput and Conv2DBackwardKernel.
    /// </summary>
    /// <param name="input">Input tensor [C_in, H, W] or [N, C_in, H, W].</param>
    /// <param name="kernel">Kernel tensor [C_out, C_in, kH, kW].</param>
    /// <param name="stride">Stride array [sH, sW].</param>
    /// <param name="padding">Padding array [pH, pW].</param>
    /// <param name="dilation">Dilation array [dH, dW].</param>
    public static Tensor<T> Conv2D(
        Tensor<T> input, Tensor<T> kernel,
        int[] stride, int[] padding, int[] dilation)
    {
        var result = Engine.Conv2D(input, kernel, stride, padding, dilation);

        GradientTape<T>.Current?.RecordOp("Conv2D", [input, kernel], result,
            grad =>
            {
                var inputGrad = Engine.Conv2DBackwardInput(
                    grad, kernel, input.Shape.ToArray(), stride, padding, dilation);
                var kernelGrad = Engine.Conv2DBackwardKernel(
                    grad, input, kernel.Shape.ToArray(), stride, padding, dilation);
                return [inputGrad, kernelGrad];
            });
        return result;
    }

    // ─── Normalization ops ───────────────────────────────────────────

    /// <summary>
    /// Layer normalization over the last dimension.
    /// Per Ba et al. (2016) "Layer Normalization".
    /// </summary>
    /// <param name="input">Input tensor of any shape. Normalization is over the last dim.</param>
    /// <param name="gamma">Scale parameter [lastDim].</param>
    /// <param name="beta">Shift parameter [lastDim].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    public static Tensor<T> LayerNorm(
        Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon = 1e-5)
    {
        int lastDim = input.Shape[^1];
        int outerSize = input.Length / lastDim;
        var result = new Tensor<T>(input.Shape.ToArray());

        // Forward: for each "row" (outer index), normalize over lastDim
        var means = new double[outerSize];
        var vars = new double[outerSize];

        for (int o = 0; o < outerSize; o++)
        {
            double sum = 0;
            for (int d = 0; d < lastDim; d++)
                sum += NumOps.ToDouble(input[o * lastDim + d]);
            means[o] = sum / lastDim;

            double varSum = 0;
            for (int d = 0; d < lastDim; d++)
            {
                double diff = NumOps.ToDouble(input[o * lastDim + d]) - means[o];
                varSum += diff * diff;
            }
            vars[o] = varSum / lastDim;

            double invStd = 1.0 / Math.Sqrt(vars[o] + epsilon);
            for (int d = 0; d < lastDim; d++)
            {
                double normalized = (NumOps.ToDouble(input[o * lastDim + d]) - means[o]) * invStd;
                double scaled = normalized * NumOps.ToDouble(gamma[d]) + NumOps.ToDouble(beta[d]);
                result[o * lastDim + d] = NumOps.FromDouble(scaled);
            }
        }

        GradientTape<T>.Current?.RecordOp("LayerNorm", [input, gamma, beta], result,
            grad =>
            {
                // Per Ba et al. 2016, LayerNorm backward:
                // dx = (1/σ) * (dout*γ - mean(dout*γ) - x_hat*mean(dout*γ*x_hat))
                var inputGrad = new Tensor<T>(input.Shape.ToArray());
                var gammaGrad = new Tensor<T>(gamma.Shape.ToArray());
                var betaGrad = new Tensor<T>(beta.Shape.ToArray());

                for (int o = 0; o < outerSize; o++)
                {
                    double invStd = 1.0 / Math.Sqrt(vars[o] + epsilon);

                    // Compute x_hat and dout*gamma for this row
                    double sumDoutGamma = 0;
                    double sumDoutGammaXhat = 0;
                    for (int d = 0; d < lastDim; d++)
                    {
                        double xHat = (NumOps.ToDouble(input[o * lastDim + d]) - means[o]) * invStd;
                        double doutGamma = NumOps.ToDouble(grad[o * lastDim + d]) * NumOps.ToDouble(gamma[d]);
                        sumDoutGamma += doutGamma;
                        sumDoutGammaXhat += doutGamma * xHat;
                    }

                    for (int d = 0; d < lastDim; d++)
                    {
                        double xHat = (NumOps.ToDouble(input[o * lastDim + d]) - means[o]) * invStd;
                        double doutGamma = NumOps.ToDouble(grad[o * lastDim + d]) * NumOps.ToDouble(gamma[d]);

                        // dx = invStd * (dout*gamma - mean(dout*gamma) - xhat*mean(dout*gamma*xhat))
                        double dx = invStd * (doutGamma - sumDoutGamma / lastDim - xHat * sumDoutGammaXhat / lastDim);
                        inputGrad[o * lastDim + d] = NumOps.FromDouble(dx);

                        // dgamma += dout * x_hat, dbeta += dout
                        gammaGrad[d] = NumOps.Add(gammaGrad[d],
                            NumOps.FromDouble(NumOps.ToDouble(grad[o * lastDim + d]) * xHat));
                        betaGrad[d] = NumOps.Add(betaGrad[d], grad[o * lastDim + d]);
                    }
                }

                return [inputGrad, gammaGrad, betaGrad];
            });
        return result;
    }

    /// <summary>
    /// Batch normalization over the batch dimension (dim 0).
    /// Per Ioffe &amp; Szegedy (2015) "Batch Normalization".
    /// </summary>
    /// <param name="input">Input tensor [N, D] or [N, C, H, W].</param>
    /// <param name="gamma">Scale parameter [D] or [C].</param>
    /// <param name="beta">Shift parameter [D] or [C].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    public static Tensor<T> BatchNorm(
        Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon = 1e-5)
    {
        // Support 2D [N,D] and 4D [N,C,H,W] (per Ioffe & Szegedy 2015).
        // For 4D: reshape to [N, C, H*W], normalize per-channel, reshape back.
        bool is4D = input.Shape.Length == 4;
        Tensor<T> flatInput;
        int n, features;

        if (is4D)
        {
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int spatial = input.Shape[2] * input.Shape[3];
            flatInput = input.Reshape([batchSize, channels * spatial]);
            n = batchSize;
            features = channels * spatial;
        }
        else if (input.Shape.Length == 2)
        {
            flatInput = input;
            n = input.Shape[0];
            features = input.Length / n;
        }
        else
        {
            throw new NotSupportedException(
                $"BatchNorm supports [N,D] or [N,C,H,W] inputs. Got rank {input.Shape.Length}.");
        }

        // gamma/beta may be [C] for 4D or [features] for 2D — use modulo for broadcasting
        var flatResult = new Tensor<T>(flatInput.Shape.ToArray());

        var means = new double[features];
        var vars = new double[features];

        // Compute per-feature mean and variance over batch
        for (int f = 0; f < features; f++)
        {
            double sum = 0;
            for (int b = 0; b < n; b++)
                sum += NumOps.ToDouble(flatInput[b * features + f]);
            means[f] = sum / n;

            double varSum = 0;
            for (int b = 0; b < n; b++)
            {
                double diff = NumOps.ToDouble(flatInput[b * features + f]) - means[f];
                varSum += diff * diff;
            }
            vars[f] = varSum / n;
        }

        // Normalize
        for (int b = 0; b < n; b++)
        {
            for (int f = 0; f < features; f++)
            {
                double invStd = 1.0 / Math.Sqrt(vars[f] + epsilon);
                double xHat = (NumOps.ToDouble(flatInput[b * features + f]) - means[f]) * invStd;
                int gammaIdx = f % gamma.Length;
                double scaled = xHat * NumOps.ToDouble(gamma[gammaIdx]) + NumOps.ToDouble(beta[gammaIdx]);
                flatResult[b * features + f] = NumOps.FromDouble(scaled);
            }
        }

        // Reshape back to original shape for 4D inputs
        var result = is4D ? flatResult.Reshape(input.Shape.ToArray()) : flatResult;

        var originalShape = input.Shape.ToArray();
        GradientTape<T>.Current?.RecordOp("BatchNorm", [input, gamma, beta], result,
            grad =>
            {
                // Per Ioffe & Szegedy 2015, BatchNorm backward:
                var flatGrad = is4D ? grad.Reshape([n, features]) : grad;
                var flatInputGrad = new Tensor<T>([n, features]);
                var gammaGrad = new Tensor<T>(gamma.Shape.ToArray());
                var betaGrad = new Tensor<T>(beta.Shape.ToArray());

                for (int f = 0; f < features; f++)
                {
                    double invStd = 1.0 / Math.Sqrt(vars[f] + epsilon);
                    int gIdx = f % gamma.Length;

                    double sumDoutGamma = 0;
                    double sumDoutGammaXhat = 0;
                    for (int b = 0; b < n; b++)
                    {
                        double xHat = (NumOps.ToDouble(flatInput[b * features + f]) - means[f]) * invStd;
                        double dg = NumOps.ToDouble(flatGrad[b * features + f]) * NumOps.ToDouble(gamma[gIdx]);
                        sumDoutGamma += dg;
                        sumDoutGammaXhat += dg * xHat;
                    }

                    for (int b = 0; b < n; b++)
                    {
                        double xHat = (NumOps.ToDouble(flatInput[b * features + f]) - means[f]) * invStd;
                        double dg = NumOps.ToDouble(flatGrad[b * features + f]) * NumOps.ToDouble(gamma[gIdx]);
                        double dx = invStd * (dg - sumDoutGamma / n - xHat * sumDoutGammaXhat / n);
                        flatInputGrad[b * features + f] = NumOps.FromDouble(dx);

                        gammaGrad[gIdx] = NumOps.Add(gammaGrad[gIdx],
                            NumOps.FromDouble(NumOps.ToDouble(flatGrad[b * features + f]) * xHat));
                        betaGrad[gIdx] = NumOps.Add(betaGrad[gIdx], flatGrad[b * features + f]);
                    }
                }

                var inputGrad = is4D ? flatInputGrad.Reshape(originalShape) : flatInputGrad;
                return [inputGrad, gammaGrad, betaGrad];
            });
        return result;
    }

    /// <summary>
    /// Group normalization. Normalizes within groups of channels.
    /// Per Wu &amp; He (2018) "Group Normalization".
    /// </summary>
    /// <param name="input">Input [N, C, ...] or [C, ...] where C is divisible by numGroups.</param>
    /// <param name="gamma">Scale [C].</param>
    /// <param name="beta">Shift [C].</param>
    /// <param name="numGroups">Number of groups to divide channels into.</param>
    /// <param name="epsilon">Numerical stability constant.</param>
    public static Tensor<T> GroupNorm(
        Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, int numGroups, double epsilon = 1e-5)
    {
        var shape = input.Shape.ToArray();
        int channels = shape.Length >= 2 ? shape[1] : shape[0];
        if (channels % numGroups != 0)
            throw new ArgumentException($"Channels ({channels}) must be divisible by numGroups ({numGroups}).");

        int channelsPerGroup = channels / numGroups;
        int n = shape.Length >= 2 ? shape[0] : 1;
        int spatialSize = input.Length / (n * channels);

        var result = new Tensor<T>(shape);
        var groupMeans = new double[n * numGroups];
        var groupVars = new double[n * numGroups];

        // Forward: compute per-group mean/var, normalize, scale+shift
        for (int b = 0; b < n; b++)
        {
            for (int g = 0; g < numGroups; g++)
            {
                int gIdx = b * numGroups + g;
                double sum = 0;
                int count = channelsPerGroup * spatialSize;
                for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                    for (int s = 0; s < spatialSize; s++)
                        sum += NumOps.ToDouble(input[(b * channels + c) * spatialSize + s]);
                groupMeans[gIdx] = sum / count;

                double varSum = 0;
                for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                    for (int s = 0; s < spatialSize; s++)
                    {
                        double diff = NumOps.ToDouble(input[(b * channels + c) * spatialSize + s]) - groupMeans[gIdx];
                        varSum += diff * diff;
                    }
                groupVars[gIdx] = varSum / count;

                double invStd = 1.0 / Math.Sqrt(groupVars[gIdx] + epsilon);
                for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = (b * channels + c) * spatialSize + s;
                        double xHat = (NumOps.ToDouble(input[idx]) - groupMeans[gIdx]) * invStd;
                        result[idx] = NumOps.FromDouble(xHat * NumOps.ToDouble(gamma[c]) + NumOps.ToDouble(beta[c]));
                    }
            }
        }

        GradientTape<T>.Current?.RecordOp("GroupNorm", [input, gamma, beta], result,
            grad =>
            {
                var inputGrad = new Tensor<T>(shape);
                var gammaGrad = new Tensor<T>(gamma.Shape.ToArray());
                var betaGrad = new Tensor<T>(beta.Shape.ToArray());

                for (int b = 0; b < n; b++)
                {
                    for (int g = 0; g < numGroups; g++)
                    {
                        int gIdx = b * numGroups + g;
                        double invStd = 1.0 / Math.Sqrt(groupVars[gIdx] + epsilon);
                        int count = channelsPerGroup * spatialSize;

                        double sumDG = 0, sumDGXhat = 0;
                        for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                            for (int s = 0; s < spatialSize; s++)
                            {
                                int idx = (b * channels + c) * spatialSize + s;
                                double xHat = (NumOps.ToDouble(input[idx]) - groupMeans[gIdx]) * invStd;
                                double dg = NumOps.ToDouble(grad[idx]) * NumOps.ToDouble(gamma[c]);
                                sumDG += dg;
                                sumDGXhat += dg * xHat;
                            }

                        for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++)
                            for (int s = 0; s < spatialSize; s++)
                            {
                                int idx = (b * channels + c) * spatialSize + s;
                                double xHat = (NumOps.ToDouble(input[idx]) - groupMeans[gIdx]) * invStd;
                                double dg = NumOps.ToDouble(grad[idx]) * NumOps.ToDouble(gamma[c]);
                                double dx = invStd * (dg - sumDG / count - xHat * sumDGXhat / count);
                                inputGrad[idx] = NumOps.FromDouble(dx);

                                gammaGrad[c] = NumOps.Add(gammaGrad[c],
                                    NumOps.FromDouble(NumOps.ToDouble(grad[idx]) * xHat));
                                betaGrad[c] = NumOps.Add(betaGrad[c], grad[idx]);
                            }
                    }
                }

                return [inputGrad, gammaGrad, betaGrad];
            });
        return result;
    }

    // ─── Softmax / CrossEntropy / LogSoftmax ─────────────────────────

    /// <summary>
    /// Softmax over the last dimension.
    /// Backward: Jacobian-vector product using softmax output.
    /// </summary>
    public static Tensor<T> Softmax(Tensor<T> x)
    {
        int lastDim = x.Shape[^1];
        int outerSize = x.Length / lastDim;
        var result = new Tensor<T>(x.Shape.ToArray());

        for (int o = 0; o < outerSize; o++)
        {
            // Numerically stable softmax: subtract max
            double maxVal = double.MinValue;
            for (int d = 0; d < lastDim; d++)
                maxVal = Math.Max(maxVal, NumOps.ToDouble(x[o * lastDim + d]));

            double expSum = 0;
            for (int d = 0; d < lastDim; d++)
            {
                double expVal = Math.Exp(NumOps.ToDouble(x[o * lastDim + d]) - maxVal);
                result[o * lastDim + d] = NumOps.FromDouble(expVal);
                expSum += expVal;
            }
            for (int d = 0; d < lastDim; d++)
                result[o * lastDim + d] = NumOps.FromDouble(
                    NumOps.ToDouble(result[o * lastDim + d]) / expSum);
        }

        GradientTape<T>.Current?.RecordOp("Softmax", [x], result,
            grad =>
            {
                // Softmax backward: dx_i = s_i * (dout_i - sum(dout * s))
                var inputGrad = new Tensor<T>(x.Shape.ToArray());
                for (int o = 0; o < outerSize; o++)
                {
                    double dotGradSoftmax = 0;
                    for (int d = 0; d < lastDim; d++)
                        dotGradSoftmax += NumOps.ToDouble(grad[o * lastDim + d]) *
                                          NumOps.ToDouble(result[o * lastDim + d]);

                    for (int d = 0; d < lastDim; d++)
                    {
                        double si = NumOps.ToDouble(result[o * lastDim + d]);
                        double dx = si * (NumOps.ToDouble(grad[o * lastDim + d]) - dotGradSoftmax);
                        inputGrad[o * lastDim + d] = NumOps.FromDouble(dx);
                    }
                }
                return [inputGrad];
            });
        return result;
    }

    /// <summary>
    /// Numerically stable log-softmax: log(softmax(x)) = x - log(sum(exp(x))).
    /// Uses log-sum-exp trick to avoid overflow.
    /// </summary>
    public static Tensor<T> LogSoftmax(Tensor<T> x)
    {
        int lastDim = x.Shape[^1];
        int outerSize = x.Length / lastDim;
        var result = new Tensor<T>(x.Shape.ToArray());

        for (int o = 0; o < outerSize; o++)
        {
            // Log-sum-exp trick: log(sum(exp(x_i))) = max + log(sum(exp(x_i - max)))
            double maxVal = double.MinValue;
            for (int d = 0; d < lastDim; d++)
                maxVal = Math.Max(maxVal, NumOps.ToDouble(x[o * lastDim + d]));

            double logSumExp = 0;
            for (int d = 0; d < lastDim; d++)
                logSumExp += Math.Exp(NumOps.ToDouble(x[o * lastDim + d]) - maxVal);
            logSumExp = maxVal + Math.Log(logSumExp);

            for (int d = 0; d < lastDim; d++)
                result[o * lastDim + d] = NumOps.FromDouble(NumOps.ToDouble(x[o * lastDim + d]) - logSumExp);
        }

        GradientTape<T>.Current?.RecordOp("LogSoftmax", [x], result,
            grad =>
            {
                // d(logSoftmax)/dx = I - softmax(x)
                // dx_i = grad_i - softmax_i * sum(grad)
                var inputGrad = new Tensor<T>(x.Shape.ToArray());
                for (int o = 0; o < outerSize; o++)
                {
                    double sumGrad = 0;
                    for (int d = 0; d < lastDim; d++)
                        sumGrad += NumOps.ToDouble(grad[o * lastDim + d]);

                    for (int d = 0; d < lastDim; d++)
                    {
                        double softmax_d = Math.Exp(NumOps.ToDouble(result[o * lastDim + d]));
                        inputGrad[o * lastDim + d] = NumOps.FromDouble(
                            NumOps.ToDouble(grad[o * lastDim + d]) - softmax_d * sumGrad);
                    }
                }
                return [inputGrad];
            });
        return result;
    }

    /// <summary>
    /// Fused cross-entropy loss: -sum(targets * log_softmax(logits)) / batch_size.
    /// Numerically stable — uses log-sum-exp internally.
    /// </summary>
    /// <param name="logits">Raw logits [N, C] (NOT softmax output).</param>
    /// <param name="targets">Target probabilities [N, C] (one-hot or soft labels).</param>
    public static Tensor<T> CrossEntropyLoss(Tensor<T> logits, Tensor<T> targets)
    {
        int n = logits.Shape[0];
        int c = logits.Length / n;

        // Compute log_softmax
        var logSoftmax = new double[logits.Length];
        for (int b = 0; b < n; b++)
        {
            double maxVal = double.MinValue;
            for (int j = 0; j < c; j++)
                maxVal = Math.Max(maxVal, NumOps.ToDouble(logits[b * c + j]));

            double logSumExp = 0;
            for (int j = 0; j < c; j++)
                logSumExp += Math.Exp(NumOps.ToDouble(logits[b * c + j]) - maxVal);
            logSumExp = maxVal + Math.Log(logSumExp);

            for (int j = 0; j < c; j++)
                logSoftmax[b * c + j] = NumOps.ToDouble(logits[b * c + j]) - logSumExp;
        }

        // Loss = -sum(targets * log_softmax) / N
        double loss = 0;
        for (int i = 0; i < logits.Length; i++)
            loss -= NumOps.ToDouble(targets[i]) * logSoftmax[i];
        loss /= n;

        var result = new Tensor<T>([1]);
        result[0] = NumOps.FromDouble(loss);

        GradientTape<T>.Current?.RecordOp("CrossEntropyLoss", [logits], result,
            grad =>
            {
                // d(CE)/d(logits) = (softmax - targets) / N * upstream_grad
                var logitGrad = new Tensor<T>(logits.Shape.ToArray());
                double scale = NumOps.ToDouble(grad[0]) / n;
                for (int b = 0; b < n; b++)
                {
                    for (int j = 0; j < c; j++)
                    {
                        double softmax_j = Math.Exp(logSoftmax[b * c + j]);
                        logitGrad[b * c + j] = NumOps.FromDouble(
                            (softmax_j - NumOps.ToDouble(targets[b * c + j])) * scale);
                    }
                }
                return [logitGrad];
            });
        return result;
    }
}
