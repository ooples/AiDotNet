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

    // ─── Attention ops ───────────────────────────────────────────────

    /// <summary>
    /// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
    /// Equivalent to PyTorch's F.scaled_dot_product_attention.
    /// </summary>
    /// <param name="query">Query tensor [batch, seq_q, d_k].</param>
    /// <param name="key">Key tensor [batch, seq_kv, d_k].</param>
    /// <param name="value">Value tensor [batch, seq_kv, d_v].</param>
    /// <param name="mask">Optional attention mask (additive, -inf for masked positions).</param>
    /// <returns>Attention output [batch, seq_q, d_v].</returns>
    public static Tensor<T> ScaledDotProductAttention(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<T>? mask = null)
    {
        int dk = query.Shape[^1];
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(dk));

        // scores = Q @ K^T / sqrt(d_k)
        var keyT = Transpose(key);
        var scores = MatMul(query, keyT);
        scores = MultiplyScalar(scores, scale);

        // Apply mask if provided
        if (mask is not null)
            scores = Add(scores, mask);

        // Attention weights = softmax(scores)
        var attnWeights = Softmax(scores);

        // Output = attn_weights @ V
        return MatMul(attnWeights, value);
    }

    // ─── Additional ops (PyTorch parity) ────────────────────────────

    /// <summary>Elementwise softplus: log(1 + exp(x)). Needed for VAE variance, smooth ReLU.</summary>
    public static Tensor<T> Softplus(Tensor<T> x, double beta = 1.0, double threshold = 20.0)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double xi = NumOps.ToDouble(x[i]) * beta;
            result[i] = xi > threshold
                ? x[i] // Linear for large values (numerical stability)
                : NumOps.FromDouble(Math.Log(1.0 + Math.Exp(xi)) / beta);
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Softplus", [x], result,
                grad =>
                {
                    // d(softplus)/dx = sigmoid(beta * x)
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double xi = NumOps.ToDouble(x[i]) * beta;
                        double sig = xi > threshold ? 1.0 : 1.0 / (1.0 + Math.Exp(-xi));
                        dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * sig);
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>Log-softmax with numerical stability: log(softmax(x)) = x - log(sum(exp(x))).</summary>
    public static Tensor<T> LogSoftmax(Tensor<T> x, int axis = -1)
    {
        int rank = x.Shape.Length;
        if (axis < 0) axis += rank;
        int axisSize = x.Shape[axis];

        // Compute along axis: logsoftmax = x - max(x) - log(sum(exp(x - max(x))))
        var result = new Tensor<T>(x.Shape.ToArray());
        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= x.Shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= x.Shape[i];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Find max for stability
                double maxVal = double.NegativeInfinity;
                for (int j = 0; j < axisSize; j++)
                {
                    int idx = (outer * axisSize + j) * innerSize + inner;
                    double v = NumOps.ToDouble(x[idx]);
                    if (v > maxVal) maxVal = v;
                }

                // Compute log-sum-exp
                double logSumExp = 0;
                for (int j = 0; j < axisSize; j++)
                {
                    int idx = (outer * axisSize + j) * innerSize + inner;
                    logSumExp += Math.Exp(NumOps.ToDouble(x[idx]) - maxVal);
                }
                logSumExp = maxVal + Math.Log(logSumExp);

                // LogSoftmax = x - logSumExp
                for (int j = 0; j < axisSize; j++)
                {
                    int idx = (outer * axisSize + j) * innerSize + inner;
                    result[idx] = NumOps.FromDouble(NumOps.ToDouble(x[idx]) - logSumExp);
                }
            }
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("LogSoftmax", [x], result,
                grad =>
                {
                    // d(logsoftmax)/dx = grad - softmax * sum(grad, axis)
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int outer = 0; outer < outerSize; outer++)
                    {
                        for (int inner = 0; inner < innerSize; inner++)
                        {
                            double sumGrad = 0;
                            for (int j = 0; j < axisSize; j++)
                            {
                                int idx = (outer * axisSize + j) * innerSize + inner;
                                sumGrad += NumOps.ToDouble(grad[idx]);
                            }
                            for (int j = 0; j < axisSize; j++)
                            {
                                int idx = (outer * axisSize + j) * innerSize + inner;
                                double softmax_j = Math.Exp(NumOps.ToDouble(result[idx]));
                                dx[idx] = NumOps.FromDouble(
                                    NumOps.ToDouble(grad[idx]) - softmax_j * sumGrad);
                            }
                        }
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>Elementwise LeakyReLU: max(alpha*x, x).</summary>
    public static Tensor<T> LeakyReLU(Tensor<T> x, double alpha = 0.01)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        var tape = GradientTape<T>.Current;
        byte[]? mask = tape is not null ? new byte[x.Length] : null;

        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = val >= 0 ? x[i] : NumOps.FromDouble(val * alpha);
            if (mask is not null) mask[i] = val >= 0 ? (byte)1 : (byte)0;
        }

        if (tape is not null)
        {
            var capturedMask = mask;
            tape.RecordOp("LeakyReLU", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(grad.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                        dx[i] = capturedMask![i] == 1
                            ? grad[i]
                            : NumOps.FromDouble(NumOps.ToDouble(grad[i]) * alpha);
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>Elementwise ELU: x if x >= 0, alpha*(exp(x)-1) otherwise.</summary>
    public static Tensor<T> ELU(Tensor<T> x, double alpha = 1.0)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = val >= 0 ? x[i] : NumOps.FromDouble(alpha * (Math.Exp(val) - 1.0));
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("ELU", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double val = NumOps.ToDouble(x[i]);
                        double deriv = val >= 0 ? 1.0 : NumOps.ToDouble(result[i]) + alpha;
                        dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * deriv);
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>Elementwise SELU (scaled ELU for self-normalizing networks).</summary>
    public static Tensor<T> SELU(Tensor<T> x)
    {
        const double lambdaSelu = 1.0507009873554804934193349852946;
        const double alphaSelu = 1.6732632423543772848170429916717;

        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = val >= 0
                ? NumOps.FromDouble(lambdaSelu * val)
                : NumOps.FromDouble(lambdaSelu * alphaSelu * (Math.Exp(val) - 1.0));
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("SELU", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double val = NumOps.ToDouble(x[i]);
                        double deriv = val >= 0 ? lambdaSelu : lambdaSelu * alphaSelu * Math.Exp(val);
                        dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * deriv);
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>Permute tensor dimensions (generalized transpose for any rank).</summary>
    public static Tensor<T> Permute(Tensor<T> a, int[] axes)
    {
        var result = a.Transpose(axes);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            // Inverse permutation: if forward is perm[i] = j, backward is inv[j] = i
            var inverse = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++) inverse[axes[i]] = i;
            tape.RecordOp("Permute", [a], result,
                grad => [grad.Transpose(inverse)]);
        }
        return result;
    }

    /// <summary>Where/select: out[i] = condition[i] ? x[i] : y[i].</summary>
    public static Tensor<T> Where(Tensor<T> condition, Tensor<T> x, Tensor<T> y)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            bool cond = NumOps.ToDouble(condition[i]) != 0;
            result[i] = cond ? x[i] : y[i];
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Where", [x, y], result,
                grad =>
                {
                    var gradX = new Tensor<T>(x.Shape.ToArray());
                    var gradY = new Tensor<T>(y.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                    {
                        bool cond = NumOps.ToDouble(condition[i]) != 0;
                        gradX[i] = cond ? grad[i] : NumOps.Zero;
                        gradY[i] = cond ? NumOps.Zero : grad[i];
                    }
                    return [gradX, gradY];
                });
        }
        return result;
    }

    /// <summary>Mish activation: x * tanh(softplus(x)).</summary>
    public static Tensor<T> Mish(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            double sp = Math.Log(1.0 + Math.Exp(xi));
            result[i] = NumOps.FromDouble(xi * Math.Tanh(sp));
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Mish", [x], result,
                grad =>
                {
                    // Mish'(x) = tanh(sp) + x * sech²(sp) * sigmoid(x)
                    // where sp = softplus(x) = log(1 + exp(x))
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double xi = NumOps.ToDouble(x[i]);
                        double sp = Math.Log(1.0 + Math.Exp(xi));
                        double tanhSp = Math.Tanh(sp);
                        double sech2Sp = 1.0 - tanhSp * tanhSp;
                        double sig = 1.0 / (1.0 + Math.Exp(-xi));
                        double deriv = tanhSp + xi * sech2Sp * sig;
                        dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * deriv);
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>HardSigmoid: clamp((x + 3) / 6, 0, 1). Mobile-optimized activation.</summary>
    public static Tensor<T> HardSigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        var tape = GradientTape<T>.Current;
        byte[]? mask = tape is not null ? new byte[x.Length] : null;

        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double hs = Math.Max(0, Math.Min(1, (val + 3.0) / 6.0));
            result[i] = NumOps.FromDouble(hs);
            if (mask is not null)
                mask[i] = (val > -3.0 && val < 3.0) ? (byte)1 : (byte)0;
        }

        if (tape is not null)
        {
            tape.RecordOp("HardSigmoid", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(grad.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                        dx[i] = mask![i] == 1
                            ? NumOps.FromDouble(NumOps.ToDouble(grad[i]) / 6.0)
                            : NumOps.Zero;
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>HardSwish: x * HardSigmoid(x). Mobile-optimized activation.</summary>
    public static Tensor<T> HardSwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double hs = Math.Max(0, Math.Min(1, (val + 3.0) / 6.0));
            result[i] = NumOps.FromDouble(val * hs);
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("HardSwish", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double val = NumOps.ToDouble(x[i]);
                        double deriv;
                        if (val <= -3.0) deriv = 0;
                        else if (val >= 3.0) deriv = 1;
                        else deriv = (2.0 * val + 3.0) / 6.0;
                        dx[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * deriv);
                    }
                    return [dx];
                });
        }
        return result;
    }

    // ─── Axis-specific reductions (289 uses in codebase) ────────────

    /// <summary>Sum along specified axes. Like torch.sum(x, dim).</summary>
    public static Tensor<T> SumAxis(Tensor<T> a, int[] axes, bool keepDims = false)
    {
        var result = Engine.ReduceSum(a, axes, keepDims);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var origShape = a.Shape.ToArray();
            tape.RecordOp("SumAxis", [a], result,
                grad =>
                {
                    // Expand grad back to original shape by broadcasting along reduced axes
                    var expanded = keepDims ? grad : ExpandDims(grad, origShape, axes);
                    return [BroadcastTo(expanded, origShape)];
                });
        }
        return result;
    }

    /// <summary>Mean along specified axes. Like torch.mean(x, dim).</summary>
    public static Tensor<T> MeanAxis(Tensor<T> a, int[] axes, bool keepDims = false)
    {
        int count = 1;
        foreach (int ax in axes) count *= a.Shape[ax];

        var result = Engine.ReduceSum(a, axes, keepDims);
        T scale = NumOps.FromDouble(1.0 / count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var origShape = a.Shape.ToArray();
            tape.RecordOp("MeanAxis", [a], result,
                grad =>
                {
                    var expanded = keepDims ? grad : ExpandDims(grad, origShape, axes);
                    var broadcast = BroadcastTo(expanded, origShape);
                    for (int i = 0; i < broadcast.Length; i++)
                        broadcast[i] = NumOps.Multiply(broadcast[i], scale);
                    return [broadcast];
                });
        }
        return result;
    }

    // ─── Variance / Standard Deviation (65 uses) ────────────────────

    /// <summary>Variance along specified axes. Like torch.var(x, dim).</summary>
    public static Tensor<T> Var(Tensor<T> a, int[] axes, bool keepDims = false, bool unbiased = true)
    {
        int count = 1;
        foreach (int ax in axes) count *= a.Shape[ax];

        var mean = MeanAxis(a, axes, keepDims: true);
        var centered = Subtract(a, BroadcastTo(mean, a.Shape.ToArray()));
        var squared = Multiply(centered, centered);
        var sumSq = SumAxis(squared, axes, keepDims);

        T divisor = NumOps.FromDouble(unbiased ? Math.Max(count - 1, 1) : count);
        for (int i = 0; i < sumSq.Length; i++)
            sumSq[i] = NumOps.Divide(sumSq[i], divisor);

        return sumSq; // Already on tape via composed ops
    }

    /// <summary>Standard deviation along specified axes. Like torch.std(x, dim).</summary>
    public static Tensor<T> Std(Tensor<T> a, int[] axes, bool keepDims = false, bool unbiased = true)
    {
        var variance = Var(a, axes, keepDims, unbiased);
        return Sqrt(variance); // Sqrt is already tape-aware
    }

    // ─── Batched Matrix Multiply (22 uses) ──────────────────────────

    /// <summary>Batched matrix multiply: C[b] = A[b] @ B[b]. Like torch.bmm.</summary>
    public static Tensor<T> Bmm(Tensor<T> a, Tensor<T> b)
    {
        // a: [batch, M, K], b: [batch, K, N] -> result: [batch, M, N]
        var result = Engine.TensorMatMul(a, b); // Engine handles batched matmul

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Bmm", [a, b], result,
                grad =>
                {
                    // dA = grad @ B^T, dB = A^T @ grad (per batch)
                    var bT = Transpose(b);
                    var gradA = MatMul(grad, bT);
                    var aT = Transpose(a);
                    var gradB = MatMul(aT, grad);
                    return [gradA, gradB];
                });
        }
        return result;
    }

    // ─── Elementwise min/max, sign, square, reciprocal ──────────────

    /// <summary>Elementwise minimum: out[i] = min(a[i], b[i]).</summary>
    public static Tensor<T> Min(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape.ToArray());
        for (int i = 0; i < a.Length; i++)
        {
            bool aIsMin = NumOps.ToDouble(a[i]) <= NumOps.ToDouble(b[i]);
            result[i] = aIsMin ? a[i] : b[i];
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Min", [a, b], result,
                grad =>
                {
                    var gradA = new Tensor<T>(a.Shape.ToArray());
                    var gradB = new Tensor<T>(b.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                    {
                        bool aIsMinB = NumOps.ToDouble(a[i]) <= NumOps.ToDouble(b[i]);
                        gradA[i] = aIsMinB ? grad[i] : NumOps.Zero;
                        gradB[i] = aIsMinB ? NumOps.Zero : grad[i];
                    }
                    return [gradA, gradB];
                });
        }
        return result;
    }

    /// <summary>Elementwise maximum: out[i] = max(a[i], b[i]).</summary>
    public static Tensor<T> Max(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape.ToArray());
        for (int i = 0; i < a.Length; i++)
        {
            bool aIsMax = NumOps.ToDouble(a[i]) >= NumOps.ToDouble(b[i]);
            result[i] = aIsMax ? a[i] : b[i];
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Max", [a, b], result,
                grad =>
                {
                    var gradA = new Tensor<T>(a.Shape.ToArray());
                    var gradB = new Tensor<T>(b.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                    {
                        bool aIsMaxB = NumOps.ToDouble(a[i]) >= NumOps.ToDouble(b[i]);
                        gradA[i] = aIsMaxB ? grad[i] : NumOps.Zero;
                        gradB[i] = aIsMaxB ? NumOps.Zero : grad[i];
                    }
                    return [gradA, gradB];
                });
        }
        return result;
    }

    /// <summary>Elementwise sign: -1, 0, or +1.</summary>
    public static Tensor<T> Sign(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(val > 0 ? 1.0 : (val < 0 ? -1.0 : 0.0));
        }
        // Sign has zero gradient everywhere (piecewise constant)
        GradientTape<T>.Current?.RecordOp("Sign", [x], result,
            grad =>
            {
                var zero = new Tensor<T>(grad.Shape.ToArray());
                return [zero];
            });
        return result;
    }

    /// <summary>Elementwise square: x^2 (optimized, avoids Pow overhead).</summary>
    public static Tensor<T> Square(Tensor<T> x)
    {
        return Multiply(x, x); // Already tape-aware, gradient = 2x via product rule
    }

    /// <summary>Elementwise reciprocal: 1/x.</summary>
    public static Tensor<T> Reciprocal(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(1.0 / Math.Max(Math.Abs(NumOps.ToDouble(x[i])), 1e-12)
                * Math.Sign(NumOps.ToDouble(x[i])));

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Reciprocal", [x], result,
                grad =>
                {
                    // d(1/x)/dx = -1/x^2
                    var dx = new Tensor<T>(x.Shape.ToArray());
                    for (int i = 0; i < x.Length; i++)
                    {
                        double ri = NumOps.ToDouble(result[i]);
                        dx[i] = NumOps.FromDouble(-NumOps.ToDouble(grad[i]) * ri * ri);
                    }
                    return [dx];
                });
        }
        return result;
    }

    // ─── Loss functions (12 uses) ───────────────────────────────────

    /// <summary>Mean squared error loss: mean((pred - target)^2).</summary>
    public static Tensor<T> MSELoss(Tensor<T> pred, Tensor<T> target)
    {
        var diff = Subtract(pred, target);
        var sq = Multiply(diff, diff);
        return Mean(sq); // Composed from tape-aware ops
    }

    /// <summary>L1 loss: mean(|pred - target|).</summary>
    public static Tensor<T> L1Loss(Tensor<T> pred, Tensor<T> target)
    {
        var diff = Subtract(pred, target);
        var absDiff = Abs(diff);
        return Mean(absDiff); // Composed from tape-aware ops
    }

    /// <summary>Binary cross-entropy loss with logits: -mean(target*log(σ(x)) + (1-target)*log(1-σ(x))).</summary>
    public static Tensor<T> BCEWithLogitsLoss(Tensor<T> logits, Tensor<T> targets)
    {
        // Numerically stable: max(x, 0) - x*t + log(1 + exp(-|x|))
        var result = new Tensor<T>([1]);
        double loss = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double x = NumOps.ToDouble(logits[i]);
            double t = NumOps.ToDouble(targets[i]);
            loss += Math.Max(x, 0) - x * t + Math.Log(1 + Math.Exp(-Math.Abs(x)));
        }
        result[0] = NumOps.FromDouble(loss / logits.Length);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            int n = logits.Length;
            tape.RecordOp("BCEWithLogitsLoss", [logits], result,
                grad =>
                {
                    var dx = new Tensor<T>(logits.Shape.ToArray());
                    double scale = NumOps.ToDouble(grad[0]) / n;
                    for (int i = 0; i < n; i++)
                    {
                        double x = NumOps.ToDouble(logits[i]);
                        double t = NumOps.ToDouble(targets[i]);
                        double sig = 1.0 / (1.0 + Math.Exp(-x));
                        dx[i] = NumOps.FromDouble((sig - t) * scale);
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>Huber loss: smooth L1, quadratic for small errors, linear for large.</summary>
    public static Tensor<T> HuberLoss(Tensor<T> pred, Tensor<T> target, double delta = 1.0)
    {
        var diff = Subtract(pred, target);
        var result = new Tensor<T>([1]);
        double loss = 0;
        for (int i = 0; i < diff.Length; i++)
        {
            double d = Math.Abs(NumOps.ToDouble(diff[i]));
            loss += d <= delta ? 0.5 * d * d : delta * (d - 0.5 * delta);
        }
        result[0] = NumOps.FromDouble(loss / diff.Length);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            int n = diff.Length;
            tape.RecordOp("HuberLoss", [pred], result,
                grad =>
                {
                    var dx = new Tensor<T>(pred.Shape.ToArray());
                    double scale = NumOps.ToDouble(grad[0]) / n;
                    for (int i = 0; i < n; i++)
                    {
                        double d = NumOps.ToDouble(diff[i]);
                        double g = Math.Abs(d) <= delta ? d : delta * Math.Sign(d);
                        dx[i] = NumOps.FromDouble(g * scale);
                    }
                    return [dx];
                });
        }
        return result;
    }

    /// <summary>KL divergence: sum(p * log(p/q)). Input is log-probabilities.</summary>
    public static Tensor<T> KLDivLoss(Tensor<T> logP, Tensor<T> target)
    {
        // KL(target || exp(logP)) = sum(target * (log(target) - logP))
        var result = new Tensor<T>([1]);
        double loss = 0;
        for (int i = 0; i < logP.Length; i++)
        {
            double t = NumOps.ToDouble(target[i]);
            if (t > 0)
            {
                double lp = NumOps.ToDouble(logP[i]);
                loss += t * (Math.Log(t) - lp);
            }
        }
        result[0] = NumOps.FromDouble(loss / logP.Length);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            int n = logP.Length;
            tape.RecordOp("KLDivLoss", [logP], result,
                grad =>
                {
                    var dx = new Tensor<T>(logP.Shape.ToArray());
                    double scale = NumOps.ToDouble(grad[0]) / n;
                    for (int i = 0; i < n; i++)
                    {
                        double t = NumOps.ToDouble(target[i]);
                        dx[i] = NumOps.FromDouble(-t * scale);
                    }
                    return [dx];
                });
        }
        return result;
    }

    // ─── Shape ops (squeeze, unsqueeze, stack, flatten, pad) ────────

    /// <summary>Remove dimensions of size 1. Like torch.squeeze.</summary>
    public static Tensor<T> Squeeze(Tensor<T> a, int? dim = null)
    {
        var origShape = a.Shape.ToArray();
        var newShape = dim.HasValue
            ? origShape.Where((s, i) => !(i == dim.Value && s == 1)).ToArray()
            : origShape.Where(s => s != 1).ToArray();
        if (newShape.Length == 0) newShape = [1];
        return Reshape(a, newShape); // Reshape is already tape-aware
    }

    /// <summary>Insert a dimension of size 1. Like torch.unsqueeze.</summary>
    public static Tensor<T> Unsqueeze(Tensor<T> a, int dim)
    {
        var origShape = a.Shape.ToArray();
        var newShape = new int[origShape.Length + 1];
        for (int i = 0; i < dim; i++) newShape[i] = origShape[i];
        newShape[dim] = 1;
        for (int i = dim; i < origShape.Length; i++) newShape[i + 1] = origShape[i];
        return Reshape(a, newShape); // Reshape is already tape-aware
    }

    /// <summary>Stack tensors along a new dimension. Like torch.stack.</summary>
    public static Tensor<T> Stack(Tensor<T>[] tensors, int axis = 0)
    {
        // Unsqueeze each tensor at the given axis, then concatenate
        var unsqueezed = new Tensor<T>[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
            unsqueezed[i] = Unsqueeze(tensors[i], axis);
        return Concatenate(unsqueezed, axis); // Already tape-aware
    }

    /// <summary>Flatten to 1D. Like torch.flatten.</summary>
    public static Tensor<T> Flatten(Tensor<T> a)
    {
        return Reshape(a, [a.Length]); // Already tape-aware
    }

    /// <summary>ReLU6: min(max(x, 0), 6). Mobile-optimized.</summary>
    public static Tensor<T> ReLU6(Tensor<T> x)
    {
        return Clamp(ReLU(x), 0, 6); // Composed from tape-aware ops
    }

    // ─── Helpers for axis-specific reductions ───────────────────────

    /// <summary>Expand dimensions that were removed by reduction back to size 1.</summary>
    private static Tensor<T> ExpandDims(Tensor<T> a, int[] targetShape, int[] reducedAxes)
    {
        var expandedShape = new int[targetShape.Length];
        Array.Copy(targetShape, expandedShape, targetShape.Length);
        foreach (int ax in reducedAxes) expandedShape[ax] = 1;
        return a.Reshape(expandedShape);
    }

    /// <summary>Broadcast tensor to target shape by repeating along dimensions of size 1.</summary>
    private static Tensor<T> BroadcastTo(Tensor<T> a, int[] targetShape)
    {
        if (a.Shape.ToArray().SequenceEqual(targetShape)) return a;

        var result = new Tensor<T>(targetShape);
        int resultLen = 1;
        foreach (int d in targetShape) resultLen *= d;

        var aShape = a.Shape.ToArray();
        // Pad aShape with leading 1s if needed
        while (aShape.Length < targetShape.Length)
            aShape = new[] { 1 }.Concat(aShape).ToArray();

        // Compute strides for source
        for (int i = 0; i < resultLen; i++)
        {
            // Convert flat index to multi-index in target shape
            int remaining = i;
            int srcIdx = 0;
            int srcStride = 1;
            for (int d = targetShape.Length - 1; d >= 0; d--)
            {
                int coord = remaining % targetShape[d];
                remaining /= targetShape[d];
                int srcCoord = aShape[d] == 1 ? 0 : coord;
                srcIdx += srcCoord * srcStride;
                srcStride *= aShape[d];
            }
            result[i] = a[Math.Min(srcIdx, a.Length - 1)];
        }
        return result;
    }

    // ─── Gather / Scatter (27 uses) ─────────────────────────────────

    /// <summary>Gather elements along axis using integer indices. Like torch.gather.</summary>
    public static Tensor<T> Gather(Tensor<T> input, int[] indices, int axis = 0)
    {
        var indicesTensor = new Tensor<int>(indices, [indices.Length]);
        var result = Engine.TensorGather(input, indicesTensor, axis: axis);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var inputShape = input.Shape.ToArray();
            tape.RecordOp("Gather", [input], result,
                grad =>
                {
                    var inputGrad = new Tensor<T>(inputShape);
                    for (int i = 0; i < grad.Length && i < indices.Length; i++)
                    {
                        int idx = indices[i];
                        if (idx >= 0 && idx < inputGrad.Length)
                            inputGrad[idx] = NumOps.Add(inputGrad[idx], grad[i]);
                    }
                    return [inputGrad];
                });
        }
        return result;
    }

    /// <summary>Scatter add: accumulate values at specified indices. Like torch.scatter_add.</summary>
    public static Tensor<T> ScatterAdd(Tensor<T> src, int[] indices, int dim, int outputSize)
    {
        var indicesTensor = new Tensor<int>(indices, [indices.Length]);
        var result = Engine.ScatterAdd(src, indicesTensor, dim: dim, outputSize: outputSize);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("ScatterAdd", [src], result,
                grad =>
                {
                    var srcGrad = new Tensor<T>(src.Shape.ToArray());
                    for (int i = 0; i < src.Length && i < indices.Length; i++)
                    {
                        int idx = indices[i];
                        if (idx >= 0 && idx < grad.Length)
                            srcGrad[i] = grad[idx];
                    }
                    return [srcGrad];
                });
        }
        return result;
    }

    // ─── Pad (6 uses) ───────────────────────────────────────────────

    /// <summary>Pad tensor with constant value. Like torch.nn.functional.pad.</summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="padding">Padding amounts: [before_dim0, after_dim0, before_dim1, after_dim1, ...].</param>
    /// <param name="value">Constant fill value (default 0).</param>
    public static Tensor<T> ConstantPad(Tensor<T> input, int[] padding, double value = 0)
    {
        var inShape = input.Shape.ToArray();
        int rank = inShape.Length;
        var outShape = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            int padBefore = d * 2 < padding.Length ? padding[d * 2] : 0;
            int padAfter = d * 2 + 1 < padding.Length ? padding[d * 2 + 1] : 0;
            outShape[d] = inShape[d] + padBefore + padAfter;
        }

        var result = Tensor<T>.CreateDefault(outShape, NumOps.FromDouble(value));

        // Copy input into padded region
        CopyRegion(input, result, inShape, outShape, padding);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("ConstantPad", [input], result,
                grad =>
                {
                    // Backward: extract the unpadded region from gradient
                    var inputGrad = new Tensor<T>(inShape);
                    CopyRegionReverse(grad, inputGrad, inShape, outShape, padding);
                    return [inputGrad];
                });
        }
        return result;
    }

    private static void CopyRegion(Tensor<T> src, Tensor<T> dst, int[] srcShape, int[] dstShape, int[] padding)
    {
        // Simple flat copy for 1D, loop copy for higher ranks
        if (srcShape.Length == 1)
        {
            int padBefore = padding.Length > 0 ? padding[0] : 0;
            for (int i = 0; i < srcShape[0]; i++)
                dst[padBefore + i] = src[i];
            return;
        }
        // General: iterate over all source elements
        int totalSrc = 1;
        foreach (int d in srcShape) totalSrc *= d;
        for (int flat = 0; flat < totalSrc; flat++)
        {
            int remaining = flat;
            int dstFlat = 0;
            int dstStride = 1;
            for (int d = srcShape.Length - 1; d >= 0; d--)
            {
                int coord = remaining % srcShape[d];
                remaining /= srcShape[d];
                int padBefore = d * 2 < padding.Length ? padding[d * 2] : 0;
                dstFlat += (coord + padBefore) * dstStride;
                dstStride *= dstShape[d];
            }
            // Fix: compute properly with correct stride order
            // Recompute with forward strides
            dstFlat = 0;
            remaining = flat;
            var coords = new int[srcShape.Length];
            for (int d = srcShape.Length - 1; d >= 0; d--)
            {
                coords[d] = remaining % srcShape[d];
                remaining /= srcShape[d];
            }
            int dstIdx = 0;
            int stride = 1;
            for (int d = srcShape.Length - 1; d >= 0; d--)
            {
                int padBefore = d * 2 < padding.Length ? padding[d * 2] : 0;
                dstIdx += (coords[d] + padBefore) * stride;
                stride *= dstShape[d];
            }
            if (dstIdx < dst.Length)
                dst[dstIdx] = src[flat];
        }
    }

    private static void CopyRegionReverse(Tensor<T> src, Tensor<T> dst, int[] dstShape, int[] srcShape, int[] padding)
    {
        int totalDst = 1;
        foreach (int d in dstShape) totalDst *= d;
        for (int flat = 0; flat < totalDst; flat++)
        {
            int remaining = flat;
            var coords = new int[dstShape.Length];
            for (int d = dstShape.Length - 1; d >= 0; d--)
            {
                coords[d] = remaining % dstShape[d];
                remaining /= dstShape[d];
            }
            int srcIdx = 0;
            int stride = 1;
            for (int d = dstShape.Length - 1; d >= 0; d--)
            {
                int padBefore = d * 2 < padding.Length ? padding[d * 2] : 0;
                srcIdx += (coords[d] + padBefore) * stride;
                stride *= srcShape[d];
            }
            if (srcIdx < src.Length)
                dst[flat] = src[srcIdx];
        }
    }

    // ─── Interpolate / Upsample (33 uses) ───────────────────────────

    /// <summary>Nearest-neighbor upsampling. Like F.interpolate(mode='nearest').</summary>
    public static Tensor<T> UpsampleNearest(Tensor<T> input, int scaleFactor)
    {
        // Input: [batch, channels, H, W] -> Output: [batch, channels, H*scale, W*scale]
        var inShape = input.Shape.ToArray();
        if (inShape.Length < 2) return input;

        int h = inShape.Length >= 3 ? inShape[^2] : 1;
        int w = inShape[^1];
        int newH = h * scaleFactor;
        int newW = w * scaleFactor;

        var outShape = (int[])inShape.Clone();
        if (outShape.Length >= 3) outShape[^2] = newH;
        outShape[^1] = newW;

        var result = new Tensor<T>(outShape);
        int batchChannels = 1;
        for (int i = 0; i < inShape.Length - 2; i++) batchChannels *= inShape[i];

        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int iy = 0; iy < newH; iy++)
            {
                int srcY = iy / scaleFactor;
                for (int ix = 0; ix < newW; ix++)
                {
                    int srcX = ix / scaleFactor;
                    result[bc * newH * newW + iy * newW + ix] =
                        input[bc * h * w + srcY * w + srcX];
                }
            }
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("UpsampleNearest", [input], result,
                grad =>
                {
                    // Backward: sum gradients from upsampled positions back to source
                    var inputGrad = new Tensor<T>(inShape);
                    for (int bc = 0; bc < batchChannels; bc++)
                    {
                        for (int iy = 0; iy < newH; iy++)
                        {
                            int srcY = iy / scaleFactor;
                            for (int ix = 0; ix < newW; ix++)
                            {
                                int srcX = ix / scaleFactor;
                                inputGrad[bc * h * w + srcY * w + srcX] = NumOps.Add(
                                    inputGrad[bc * h * w + srcY * w + srcX],
                                    grad[bc * newH * newW + iy * newW + ix]);
                            }
                        }
                    }
                    return [inputGrad];
                });
        }
        return result;
    }

    /// <summary>Bilinear upsampling. Like F.interpolate(mode='bilinear').</summary>
    public static Tensor<T> UpsampleBilinear(Tensor<T> input, int outputH, int outputW)
    {
        var inShape = input.Shape.ToArray();
        int h = inShape.Length >= 3 ? inShape[^2] : 1;
        int w = inShape[^1];

        var outShape = (int[])inShape.Clone();
        if (outShape.Length >= 3) outShape[^2] = outputH;
        outShape[^1] = outputW;

        var result = new Tensor<T>(outShape);
        int batchChannels = 1;
        for (int i = 0; i < inShape.Length - 2; i++) batchChannels *= inShape[i];

        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int oy = 0; oy < outputH; oy++)
            {
                double srcY = (double)oy * (h - 1) / Math.Max(outputH - 1, 1);
                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, h - 1);
                double fy = srcY - y0;

                for (int ox = 0; ox < outputW; ox++)
                {
                    double srcX = (double)ox * (w - 1) / Math.Max(outputW - 1, 1);
                    int x0 = (int)Math.Floor(srcX);
                    int x1 = Math.Min(x0 + 1, w - 1);
                    double fx = srcX - x0;

                    double v00 = NumOps.ToDouble(input[bc * h * w + y0 * w + x0]);
                    double v01 = NumOps.ToDouble(input[bc * h * w + y0 * w + x1]);
                    double v10 = NumOps.ToDouble(input[bc * h * w + y1 * w + x0]);
                    double v11 = NumOps.ToDouble(input[bc * h * w + y1 * w + x1]);

                    double interp = v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx
                                  + v10 * fy * (1 - fx) + v11 * fy * fx;
                    result[bc * outputH * outputW + oy * outputW + ox] = NumOps.FromDouble(interp);
                }
            }
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("UpsampleBilinear", [input], result,
                grad =>
                {
                    var inputGrad = new Tensor<T>(inShape);
                    for (int bc = 0; bc < batchChannels; bc++)
                    {
                        for (int oy = 0; oy < outputH; oy++)
                        {
                            double srcY = (double)oy * (h - 1) / Math.Max(outputH - 1, 1);
                            int y0 = (int)Math.Floor(srcY);
                            int y1 = Math.Min(y0 + 1, h - 1);
                            double fy = srcY - y0;

                            for (int ox = 0; ox < outputW; ox++)
                            {
                                double srcX = (double)ox * (w - 1) / Math.Max(outputW - 1, 1);
                                int x0 = (int)Math.Floor(srcX);
                                int x1 = Math.Min(x0 + 1, w - 1);
                                double fx = srcX - x0;

                                double g = NumOps.ToDouble(grad[bc * outputH * outputW + oy * outputW + ox]);
                                int baseIdx = bc * h * w;
                                inputGrad[baseIdx + y0 * w + x0] = NumOps.Add(inputGrad[baseIdx + y0 * w + x0], NumOps.FromDouble(g * (1 - fy) * (1 - fx)));
                                inputGrad[baseIdx + y0 * w + x1] = NumOps.Add(inputGrad[baseIdx + y0 * w + x1], NumOps.FromDouble(g * (1 - fy) * fx));
                                inputGrad[baseIdx + y1 * w + x0] = NumOps.Add(inputGrad[baseIdx + y1 * w + x0], NumOps.FromDouble(g * fy * (1 - fx)));
                                inputGrad[baseIdx + y1 * w + x1] = NumOps.Add(inputGrad[baseIdx + y1 * w + x1], NumOps.FromDouble(g * fy * fx));
                            }
                        }
                    }
                    return [inputGrad];
                });
        }
        return result;
    }

    // ─── ConvTranspose2D (14 uses) ──────────────────────────────────

    /// <summary>Transposed (deconvolution) 2D convolution. Like F.conv_transpose2d.</summary>
    public static Tensor<T> ConvTranspose2D(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int outputPadding = 0)
    {
        var strideArr = new[] { stride, stride };
        var paddingArr = new[] { padding, padding };
        var outputPaddingArr = new[] { outputPadding, outputPadding };
        var result = Engine.ConvTranspose2D(input, kernel, strideArr, paddingArr, outputPaddingArr);
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("ConvTranspose2D", [input, kernel], result,
                grad =>
                {
                    // Backward of ConvTranspose2D is Conv2D
                    var strideArr = new[] { stride, stride };
                    var paddingArr = new[] { padding, padding };
                    var dilationArr = new[] { 1, 1 };
                    var inputGrad = Engine.Conv2D(grad, kernel, strideArr, paddingArr, dilationArr);
                    var kernelGrad = Engine.Conv2DBackwardKernel(grad, input, kernel.Shape.ToArray(),
                        strideArr, paddingArr, dilationArr);
                    return [inputGrad, kernelGrad];
                });
        }
        return result;
    }

    // ─── InstanceNorm (6 uses) ──────────────────────────────────────

    /// <summary>Instance normalization. Like F.instance_norm.</summary>
    public static Tensor<T> InstanceNorm(Tensor<T> input, Tensor<T>? weight = null, Tensor<T>? bias = null, double eps = 1e-5)
    {
        // Input: [batch, channels, ...spatial]
        // Normalize each (batch, channel) independently over spatial dims
        var inShape = input.Shape.ToArray();
        int batch = inShape[0];
        int channels = inShape.Length > 1 ? inShape[1] : 1;
        int spatialSize = 1;
        for (int i = 2; i < inShape.Length; i++) spatialSize *= inShape[i];

        var result = new Tensor<T>(inShape);
        var means = new double[batch * channels];
        var invStds = new double[batch * channels];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * spatialSize;
                double sum = 0;
                for (int s = 0; s < spatialSize; s++)
                    sum += NumOps.ToDouble(input[offset + s]);
                double mean = sum / spatialSize;

                double varSum = 0;
                for (int s = 0; s < spatialSize; s++)
                {
                    double diff = NumOps.ToDouble(input[offset + s]) - mean;
                    varSum += diff * diff;
                }
                double invStd = 1.0 / Math.Sqrt(varSum / spatialSize + eps);

                means[b * channels + c] = mean;
                invStds[b * channels + c] = invStd;

                double w = weight is not null && c < weight.Length ? NumOps.ToDouble(weight[c]) : 1.0;
                double bi = bias is not null && c < bias.Length ? NumOps.ToDouble(bias[c]) : 0.0;

                for (int s = 0; s < spatialSize; s++)
                {
                    double normalized = (NumOps.ToDouble(input[offset + s]) - mean) * invStd;
                    result[offset + s] = NumOps.FromDouble(normalized * w + bi);
                }
            }
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("InstanceNorm", [input], result,
                grad =>
                {
                    var inputGrad = new Tensor<T>(inShape);
                    for (int b = 0; b < batch; b++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            int offset = (b * channels + c) * spatialSize;
                            double mean = means[b * channels + c];
                            double invStd2 = invStds[b * channels + c];
                            double w = weight is not null && c < weight.Length ? NumOps.ToDouble(weight[c]) : 1.0;

                            double sumGrad = 0, sumGradX = 0;
                            for (int s = 0; s < spatialSize; s++)
                            {
                                double g = NumOps.ToDouble(grad[offset + s]) * w;
                                sumGrad += g;
                                sumGradX += g * (NumOps.ToDouble(input[offset + s]) - mean);
                            }

                            for (int s = 0; s < spatialSize; s++)
                            {
                                double g = NumOps.ToDouble(grad[offset + s]) * w;
                                double xHat = (NumOps.ToDouble(input[offset + s]) - mean) * invStd2;
                                inputGrad[offset + s] = NumOps.FromDouble(
                                    invStd2 / spatialSize * (spatialSize * g - sumGrad - xHat * sumGradX * invStd2));
                            }
                        }
                    }
                    return [inputGrad];
                });
        }
        return result;
    }

    // ─── RMSNorm (used in LLM layers) ───────────────────────────────

    /// <summary>Root Mean Square Layer Normalization. Used in LLaMA, Gemma.</summary>
    public static Tensor<T> RMSNorm(Tensor<T> input, Tensor<T> weight, double eps = 1e-6)
    {
        int lastDim = input.Shape[^1];
        int outerSize = input.Length / lastDim;

        var result = new Tensor<T>(input.Shape.ToArray());
        var rmsValues = new double[outerSize]; // Store for backward

        for (int outer = 0; outer < outerSize; outer++)
        {
            int offset = outer * lastDim;
            double sumSq = 0;
            for (int d = 0; d < lastDim; d++)
            {
                double v = NumOps.ToDouble(input[offset + d]);
                sumSq += v * v;
            }
            double rms = Math.Sqrt(sumSq / lastDim + eps);
            rmsValues[outer] = rms;

            for (int d = 0; d < lastDim; d++)
            {
                double normalized = NumOps.ToDouble(input[offset + d]) / rms;
                double w = d < weight.Length ? NumOps.ToDouble(weight[d]) : 1.0;
                result[offset + d] = NumOps.FromDouble(normalized * w);
            }
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var inShape = input.Shape.ToArray();
            tape.RecordOp("RMSNorm", [input, weight], result,
                grad =>
                {
                    var inputGrad = new Tensor<T>(inShape);
                    var weightGrad = new Tensor<T>(weight.Shape.ToArray());

                    for (int outer = 0; outer < outerSize; outer++)
                    {
                        int offset = outer * lastDim;
                        double rms = rmsValues[outer];
                        double invRms = 1.0 / rms;

                        double sumGradNorm = 0;
                        for (int d = 0; d < lastDim; d++)
                        {
                            double w = d < weight.Length ? NumOps.ToDouble(weight[d]) : 1.0;
                            double g = NumOps.ToDouble(grad[offset + d]) * w;
                            double xNorm = NumOps.ToDouble(input[offset + d]) * invRms;
                            sumGradNorm += g * xNorm;

                            // Weight gradient
                            weightGrad[d] = NumOps.Add(weightGrad[d], NumOps.FromDouble(
                                NumOps.ToDouble(grad[offset + d]) * xNorm));
                        }

                        for (int d = 0; d < lastDim; d++)
                        {
                            double w = d < weight.Length ? NumOps.ToDouble(weight[d]) : 1.0;
                            double g = NumOps.ToDouble(grad[offset + d]) * w;
                            double x = NumOps.ToDouble(input[offset + d]);
                            inputGrad[offset + d] = NumOps.FromDouble(
                                invRms * (g - x * invRms * invRms * sumGradNorm / lastDim));
                        }
                    }
                    return [inputGrad, weightGrad];
                });
        }
        return result;
    }

    // ─── Tile / Repeat (used in attention, positional encoding) ─────

    /// <summary>Repeat tensor along specified dimensions. Like torch.tile.</summary>
    public static Tensor<T> Tile(Tensor<T> input, int[] repeats)
    {
        var inShape = input.Shape.ToArray();
        var outShape = new int[inShape.Length];
        for (int d = 0; d < inShape.Length; d++)
            outShape[d] = inShape[d] * (d < repeats.Length ? repeats[d] : 1);

        var result = new Tensor<T>(outShape);
        int totalOut = 1;
        foreach (int d in outShape) totalOut *= d;

        for (int flat = 0; flat < totalOut; flat++)
        {
            int remaining = flat;
            int srcIdx = 0;
            int srcStride = 1;
            for (int d = inShape.Length - 1; d >= 0; d--)
            {
                int coord = remaining % outShape[d];
                remaining /= outShape[d];
                srcIdx += (coord % inShape[d]) * srcStride;
                srcStride *= inShape[d];
            }
            result[flat] = input[Math.Min(srcIdx, input.Length - 1)];
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("Tile", [input], result,
                grad =>
                {
                    // Backward: sum gradient from all repeated positions back to source
                    var inputGrad = new Tensor<T>(inShape);
                    for (int flat = 0; flat < totalOut; flat++)
                    {
                        int remaining2 = flat;
                        int srcIdx2 = 0;
                        int srcStride2 = 1;
                        for (int d = inShape.Length - 1; d >= 0; d--)
                        {
                            int coord = remaining2 % outShape[d];
                            remaining2 /= outShape[d];
                            srcIdx2 += (coord % inShape[d]) * srcStride2;
                            srcStride2 *= inShape[d];
                        }
                        int si = Math.Min(srcIdx2, input.Length - 1);
                        inputGrad[si] = NumOps.Add(inputGrad[si], grad[flat]);
                    }
                    return [inputGrad];
                });
        }
        return result;
    }

    // ─── MaskedFill (attention masks) ───────────────────────────────

    /// <summary>Fill positions where mask is true with a value. Like torch.masked_fill.</summary>
    public static Tensor<T> MaskedFill(Tensor<T> input, Tensor<T> mask, double fillValue)
    {
        var result = new Tensor<T>(input.Shape.ToArray());
        for (int i = 0; i < input.Length; i++)
        {
            bool isMasked = i < mask.Length && NumOps.ToDouble(mask[i]) != 0;
            result[i] = isMasked ? NumOps.FromDouble(fillValue) : input[i];
        }

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            tape.RecordOp("MaskedFill", [input], result,
                grad =>
                {
                    // Gradient is zero where mask is true (constant fill), passthrough elsewhere
                    var inputGrad = new Tensor<T>(input.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                    {
                        bool isMasked = i < mask.Length && NumOps.ToDouble(mask[i]) != 0;
                        inputGrad[i] = isMasked ? NumOps.Zero : grad[i];
                    }
                    return [inputGrad];
                });
        }
        return result;
    }

    // ─── Cosine Similarity ──────────────────────────────────────────

    /// <summary>Cosine similarity between two tensors along last dimension.</summary>
    public static Tensor<T> CosineSimilarity(Tensor<T> a, Tensor<T> b, double eps = 1e-8)
    {
        // Composed from tape-aware ops
        var dotProduct = Multiply(a, b);
        var aNorm = Sqrt(AddScalar(Sum(Square(a)), NumOps.FromDouble(eps)));
        var bNorm = Sqrt(AddScalar(Sum(Square(b)), NumOps.FromDouble(eps)));
        var normProduct = Multiply(aNorm, bNorm);
        return Divide(Sum(dotProduct), normProduct);
    }

    // ─── OneHot (149 uses) ──────────────────────────────────────────

    /// <summary>Creates one-hot encoded tensor. Not differentiable (discrete).</summary>
    public static Tensor<T> OneHot(int[] indices, int numClasses)
    {
        int n = indices.Length;
        var result = new Tensor<T>([n, numClasses]);
        for (int i = 0; i < n; i++)
        {
            if (indices[i] >= 0 && indices[i] < numClasses)
                result[i * numClasses + indices[i]] = NumOps.One;
        }
        // No gradient — one-hot is not differentiable
        return result;
    }

    // ─── Lp Norm ────────────────────────────────────────────────────

    /// <summary>Lp norm along last dimension. Like torch.norm(x, p, dim).</summary>
    public static Tensor<T> Norm(Tensor<T> x, double p = 2.0)
    {
        // Composed from tape-aware ops: (sum(|x|^p))^(1/p)
        var absx = Abs(x);
        var powered = Pow(absx, p);
        var summed = Sum(powered);
        return Pow(summed, 1.0 / p);
    }

    // ─── Threshold ──────────────────────────────────────────────────

    /// <summary>Threshold activation: x if x > threshold, else value.</summary>
    public static Tensor<T> Threshold(Tensor<T> x, double threshold, double value)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        var tape = GradientTape<T>.Current;
        byte[]? mask = tape is not null ? new byte[x.Length] : null;

        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = val > threshold ? x[i] : NumOps.FromDouble(value);
            if (mask is not null) mask[i] = val > threshold ? (byte)1 : (byte)0;
        }

        if (tape is not null)
        {
            tape.RecordOp("Threshold", [x], result,
                grad =>
                {
                    var dx = new Tensor<T>(grad.Shape.ToArray());
                    for (int i = 0; i < grad.Length; i++)
                        dx[i] = mask![i] == 1 ? grad[i] : NumOps.Zero;
                    return [dx];
                });
        }
        return result;
    }

    // ─── Floor / Ceil / Round (piecewise constant — zero gradient) ──

    /// <summary>Elementwise floor. Gradient is zero (piecewise constant).</summary>
    public static Tensor<T> Floor(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Floor(NumOps.ToDouble(x[i])));
        // STE (straight-through estimator): pass gradient through unchanged
        GradientTape<T>.Current?.RecordOp("Floor", [x], result,
            grad => [grad]);
        return result;
    }

    /// <summary>Elementwise ceil. Gradient is zero (piecewise constant).</summary>
    public static Tensor<T> Ceil(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Ceiling(NumOps.ToDouble(x[i])));
        GradientTape<T>.Current?.RecordOp("Ceil", [x], result,
            grad => [grad]);
        return result;
    }

    /// <summary>Elementwise round. Gradient uses straight-through estimator.</summary>
    public static Tensor<T> Round(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = NumOps.FromDouble(Math.Round(NumOps.ToDouble(x[i])));
        GradientTape<T>.Current?.RecordOp("Round", [x], result,
            grad => [grad]);
        return result;
    }

    // ─── LogSumExp (numerically stable) ─────────────────────────────

    /// <summary>Log-sum-exp: log(sum(exp(x))). Numerically stable via max subtraction.</summary>
    public static Tensor<T> LogSumExp(Tensor<T> x)
    {
        // Composed from tape-aware ops for automatic gradient
        double maxVal = double.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
        {
            double v = NumOps.ToDouble(x[i]);
            if (v > maxVal) maxVal = v;
        }
        var shifted = AddScalar(x, NumOps.FromDouble(-maxVal));
        var expShifted = Exp(shifted);
        var sumExp = Sum(expShifted);
        var logSum = Log(sumExp);
        return AddScalar(logSum, NumOps.FromDouble(maxVal));
    }
}
