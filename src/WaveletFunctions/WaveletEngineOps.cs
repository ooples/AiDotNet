namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Whole-array filter-bank primitives shared by the wavelet implementations, expressed as engine operations.
/// </summary>
/// <typeparam name="T">The numeric type of the signal and filter values.</typeparam>
/// <remarks>
/// <para>
/// Every primitive reproduces one of the scalar loops the wavelets used before: zero-padded linear convolution,
/// "same"-length centred convolution, periodic (circular) correlation with a stride, zero-insertion upsampling and
/// decimation. They only differ from those loops in the order the products are summed.
/// </para>
/// </remarks>
internal static class WaveletEngineOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Full linear convolution with zero padding: <c>result[i] = sum_j input[i - j] * kernel[j]</c>,
    /// length <c>input.Length + kernel.Length - 1</c>.
    /// </summary>
    internal static Vector<T> FullConvolve(Vector<T> input, Vector<T> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        if (input.Length == 0 || kernel.Length == 0)
        {
            var zeros = new Vector<T>(Math.Max(resultLength, 0));
            zeros.Fill(NumOps.Zero);
            return zeros;
        }

        return FullConvolveTensor(Tensor<T>.FromVector(input), Tensor<T>.FromVector(kernel)).ToVector();
    }

    /// <summary>
    /// Full linear convolution of two rank-1 tensors (see <see cref="FullConvolve(Vector{T}, Vector{T})"/>).
    /// </summary>
    internal static Tensor<T> FullConvolveTensor(Tensor<T> input, Tensor<T> kernel)
    {
        int n = input.Length;
        int k = kernel.Length;

        // Convolution is cross-correlation with the reversed kernel over the input zero-padded by k - 1.
        var flipped = Engine.TensorFlip(kernel, new[] { 0 });
        var output = Engine.Conv1D(input.Reshape(1, 1, n), flipped.Reshape(1, 1, k), 1, k - 1, 1);
        return output.Reshape(n + k - 1);
    }

    /// <summary>
    /// Centred "same"-length convolution with zero padding:
    /// <c>result[i] = sum_j input[i - j + filter.Length / 2] * filter[j]</c>, length <c>input.Length</c>.
    /// </summary>
    internal static Vector<T> CenteredConvolve(Vector<T> input, Vector<T> filter)
    {
        if (input.Length == 0)
        {
            return new Vector<T>(0);
        }

        if (filter.Length == 0)
        {
            var zeros = new Vector<T>(input.Length);
            zeros.Fill(NumOps.Zero);
            return zeros;
        }

        return FullConvolve(input, filter).GetSubVector(filter.Length / 2, input.Length);
    }

    /// <summary>
    /// Reverses a vector: <c>result[i] = input[input.Length - 1 - i]</c>.
    /// </summary>
    internal static Vector<T> Reverse(Vector<T> input)
    {
        if (input.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorFlip(Tensor<T>.FromVector(input), new[] { 0 }).ToVector();
    }

    /// <summary>
    /// Keeps every <paramref name="factor"/>-th sample starting at 0, <paramref name="count"/> samples in total.
    /// </summary>
    internal static Vector<T> Decimate(Vector<T> input, int factor, int count)
    {
        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        return Engine.StridedGather(input, 0, factor, count);
    }

    /// <summary>
    /// Inserts <c>factor - 1</c> zeros after every sample: <c>result[i * factor] = input[i]</c>, length
    /// <c>input.Length * factor</c>.
    /// </summary>
    internal static Vector<T> Upsample(Vector<T> input, int factor)
    {
        var result = new Vector<T>(input.Length * factor);
        result.Fill(NumOps.Zero);
        if (input.Length > 0)
        {
            Engine.StridedScatter(result, input, 0, factor);
        }

        return result;
    }

    /// <summary>
    /// Periodic correlation with a stride:
    /// <c>result[i] = sum_j filter[j] * input[(stride * i + j + shift) mod input.Length]</c> for
    /// <c>i &lt; count</c>.
    /// </summary>
    internal static Vector<T> PeriodicCorrelate(Vector<T> input, Vector<T> filter, int stride, int count, int shift = 0)
    {
        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        if (filter.Length == 0)
        {
            var zeros = new Vector<T>(count);
            zeros.Fill(NumOps.Zero);
            return zeros;
        }

        int n = input.Length;
        int length = filter.Length;
        int extendedLength = stride * (count - 1) + length;

        // Rotate so the window of output 0 starts at index 0, then wrap the signal periodically far enough
        // for the last window.
        var signal = Tensor<T>.FromVector(input);
        int start = ((shift % n) + n) % n;
        if (start != 0)
        {
            signal = Engine.TensorRoll(signal, new[] { -start }, new[] { 0 });
        }

        int copies = (extendedLength + n - 1) / n;
        if (copies > 1)
        {
            signal = Engine.TensorTile(signal, new[] { copies });
        }

        var extended = Engine.TensorSlice(signal, new[] { 0 }, new[] { extendedLength });
        var output = Engine.Conv1D(
            extended.Reshape(1, 1, extendedLength),
            Tensor<T>.FromVector(filter).Reshape(1, 1, length),
            stride,
            0,
            1);
        return output.Reshape(count).ToVector();
    }

    /// <summary>
    /// Periodic two-channel synthesis, the adjoint of the periodic analysis
    /// <c>a[i] = sum_j h[j] x[(2i + j) mod n]</c>, <c>d[i] = sum_j g[j] x[(2i + j) mod n]</c>:
    /// <c>x[m] = sum_j h[j] up(a)[(m - j) mod n] + g[j] up(d)[(m - j) mod n]</c>, where <c>up</c> inserts a zero
    /// after every coefficient and <c>n = 2 * a.Length</c>. For orthonormal filters this inverts the analysis.
    /// </summary>
    internal static Vector<T> PeriodicSynthesis(Vector<T> approximation, Vector<T> detail, Vector<T> low, Vector<T> high)
    {
        int outputLength = approximation.Length * 2;
        if (outputLength == 0)
        {
            return new Vector<T>(0);
        }

        // sum_j f[j] u[(m - j) mod n] is a periodic correlation with the reversed filter whose window starts
        // L - 1 samples before m.
        var lowBranch = PeriodicCorrelate(Upsample(approximation, 2), Reverse(low), 1, outputLength, -(low.Length - 1));
        var highBranch = PeriodicCorrelate(Upsample(detail, 2), Reverse(high), 1, outputLength, -(high.Length - 1));
        return Engine.Add(lowBranch, highBranch);
    }

    /// <summary>
    /// The sign pattern <c>(-1)^i</c> for <c>i = 0 .. length - 1</c> (starting with +1).
    /// </summary>
    internal static Vector<T> AlternatingSigns(int length)
    {
        if (length <= 0)
        {
            return new Vector<T>(0);
        }

        // (-1)^i = 1 - 2 * (i mod 2)
        var index = Engine.TensorArange<T>(0, length, 1);
        var parity = Engine.TensorRemainder(index, Engine.TensorFullLike(index, NumOps.FromDouble(2)));
        return Engine.ScalarMinusTensor(NumOps.One, Engine.TensorMultiplyScalar(parity, NumOps.FromDouble(2))).ToVector();
    }

    /// <summary>
    /// Quadrature-mirror filter of <paramref name="filter"/>: <c>result[i] = (-1)^i * filter[L - 1 - i]</c>.
    /// </summary>
    internal static Vector<T> QuadratureMirror(Vector<T> filter)
    {
        if (filter.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.Multiply(AlternatingSigns(filter.Length), Reverse(filter));
    }

    /// <summary>
    /// The sample positions <c>i - size / 2</c> (integer division) for <c>i = 0 .. size - 1</c>.
    /// </summary>
    internal static Tensor<T> CenteredGrid(int size)
    {
        if (size <= 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        int half = size / 2;
        return Engine.TensorArange<T>(-half, size - half, 1);
    }

    /// <summary>
    /// The first <paramref name="length"/> entries of <paramref name="input"/> as a tensor (the whole vector when
    /// it already has that length).
    /// </summary>
    internal static Tensor<T> Prefix(Vector<T> input, int length)
    {
        return Tensor<T>.FromVector(input.Length == length ? input : input.GetSubVector(0, length));
    }

    /// <summary>
    /// Pointwise analysis used by the sampled continuous wavelets: <c>a[i] = w[i] * x[i]</c> and
    /// <c>d[i] = w'[i] * x[i]</c>.
    /// </summary>
    internal static (Vector<T> approximation, Vector<T> detail) PointwiseDecompose(
        Vector<T> input, Tensor<T> waveletValue, Tensor<T> derivativeValue)
    {
        var signal = Tensor<T>.FromVector(input);
        var approximation = Engine.TensorMultiply(waveletValue, signal).ToVector();
        var detail = Engine.TensorMultiply(derivativeValue, signal).ToVector();
        return (approximation, detail);
    }

    /// <summary>
    /// Least-squares inverse of <see cref="PointwiseDecompose"/>: <c>(a * w + d * w') / (w^2 + w'^2)</c>, or
    /// <c>a + d</c> where <c>|w^2 + w'^2| &lt;= 1e-10</c>.
    /// </summary>
    internal static Vector<T> PointwiseReconstruct(
        Vector<T> approximation, Vector<T> detail, Tensor<T> waveletValue, Tensor<T> derivativeValue)
    {
        int size = approximation.Length;
        var a = Tensor<T>.FromVector(approximation);
        var d = Prefix(detail, size);
        var denominator = Engine.TensorAdd(Engine.TensorSquare(waveletValue), Engine.TensorSquare(derivativeValue));
        var numerator = Engine.TensorAdd(Engine.TensorMultiply(a, waveletValue), Engine.TensorMultiply(d, derivativeValue));
        var usable = Engine.TensorGreaterThan(Engine.TensorAbs(denominator), NumOps.FromDouble(1e-10));
        return Engine.TensorWhere(usable, Engine.TensorDivide(numerator, denominator), Engine.TensorAdd(a, d)).ToVector();
    }

    /// <summary>
    /// Element-wise sum of the first <c>min(a.Length, b.Length)</c> entries of two vectors.
    /// </summary>
    internal static Vector<T> AddTruncated(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length == 0)
        {
            return new Vector<T>(0);
        }

        var left = a.Length == length ? a : a.GetSubVector(0, length);
        var right = b.Length == length ? b : b.GetSubVector(0, length);
        return Engine.Add(left, right);
    }
}

/// <summary>
/// Complex-valued counterparts of <see cref="WaveletEngineOps{T}"/>, computed on the real and imaginary planes.
/// </summary>
/// <typeparam name="T">The numeric type of the real and imaginary parts.</typeparam>
internal static class ComplexWaveletEngineOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Full linear convolution of complex vectors: <c>result[i] = sum_j input[i - j] * kernel[j]</c>.
    /// </summary>
    internal static Vector<Complex<T>> FullConvolve(Vector<Complex<T>> input, Vector<Complex<T>> kernel)
    {
        int n = input.Length;
        int k = kernel.Length;
        int resultLength = n + k - 1;
        if (n == 0 || k == 0)
        {
            return Zeros(Math.Max(resultLength, 0));
        }

        var x = Tensor<Complex<T>>.FromVector(input);
        var h = Engine.TensorFlip(Tensor<Complex<T>>.FromVector(kernel), new[] { 0 });
        var xr = Engine.TensorReal(x);
        var xi = Engine.TensorImag(x);
        var hr = Engine.TensorReal(h);
        var hi = Engine.TensorImag(h);

        // (xr + i xi) * (hr + i hi) = (xr hr - xi hi) + i (xr hi + xi hr): one two-channel convolution whose
        // output channel 0 is the real plane and channel 1 the imaginary plane.
        var signal = Engine.TensorStack(new[] { xr, xi }, 0).Reshape(1, 2, n);
        var realKernel = Engine.TensorStack(new[] { hr, Engine.TensorNegate(hi) }, 0);
        var imagKernel = Engine.TensorStack(new[] { hi, hr }, 0);
        var kernels = Engine.TensorStack(new[] { realKernel, imagKernel }, 0);
        var output = Engine.Conv1D(signal, kernels, 1, k - 1, 1).Reshape(2, resultLength);

        var real = Engine.TensorSlice(output, new[] { 0, 0 }, new[] { 1, resultLength }).Reshape(resultLength);
        var imag = Engine.TensorSlice(output, new[] { 1, 0 }, new[] { 1, resultLength }).Reshape(resultLength);
        return Engine.TensorComplex(real, imag).ToVector();
    }

    /// <summary>
    /// Reverses a complex vector.
    /// </summary>
    internal static Vector<Complex<T>> Reverse(Vector<Complex<T>> input)
    {
        if (input.Length == 0)
        {
            return new Vector<Complex<T>>(0);
        }

        return Engine.TensorFlip(Tensor<Complex<T>>.FromVector(input), new[] { 0 }).ToVector();
    }

    /// <summary>
    /// Keeps every <paramref name="factor"/>-th sample starting at 0, <paramref name="count"/> samples in total.
    /// </summary>
    internal static Vector<Complex<T>> Decimate(Vector<Complex<T>> input, int factor, int count)
    {
        if (count <= 0)
        {
            return new Vector<Complex<T>>(0);
        }

        return Engine.StridedGather(input, 0, factor, count);
    }

    /// <summary>
    /// Inserts <c>factor - 1</c> zeros after every sample.
    /// </summary>
    internal static Vector<Complex<T>> Upsample(Vector<Complex<T>> input, int factor)
    {
        var result = Zeros(input.Length * factor);
        if (input.Length > 0)
        {
            Engine.StridedScatter(result, input, 0, factor);
        }

        return result;
    }

    /// <summary>
    /// Element-wise sum of the first <c>min(a.Length, b.Length)</c> entries of two complex vectors.
    /// </summary>
    internal static Vector<Complex<T>> AddTruncated(Vector<Complex<T>> a, Vector<Complex<T>> b)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length == 0)
        {
            return new Vector<Complex<T>>(0);
        }

        var left = Tensor<Complex<T>>.FromVector(a.Length == length ? a : a.GetSubVector(0, length));
        var right = Tensor<Complex<T>>.FromVector(b.Length == length ? b : b.GetSubVector(0, length));
        var real = Engine.TensorAdd(Engine.TensorReal(left), Engine.TensorReal(right));
        var imag = Engine.TensorAdd(Engine.TensorImag(left), Engine.TensorImag(right));
        return Engine.TensorComplex(real, imag).ToVector();
    }

    /// <summary>
    /// Concatenates the non-empty parts into one vector (pure data movement).
    /// </summary>
    internal static Vector<Complex<T>> Concat(params Vector<Complex<T>>[] parts)
    {
        var nonEmpty = new List<Vector<Complex<T>>>();
        foreach (var part in parts)
        {
            if (part.Length > 0)
            {
                nonEmpty.Add(part);
            }
        }

        return nonEmpty.Count == 0 ? new Vector<Complex<T>>(0) : Vector<Complex<T>>.Concatenate(nonEmpty);
    }

    /// <summary>
    /// The first <paramref name="count"/> entries of <paramref name="input"/> (empty when <paramref name="count"/>
    /// is not positive).
    /// </summary>
    internal static Vector<Complex<T>> Head(Vector<Complex<T>> input, int count)
    {
        return Slice(input, 0, count);
    }

    /// <summary>
    /// <paramref name="count"/> entries of <paramref name="input"/> starting at <paramref name="start"/> (empty
    /// when <paramref name="count"/> is not positive).
    /// </summary>
    internal static Vector<Complex<T>> Slice(Vector<Complex<T>> input, int start, int count)
    {
        if (count <= 0)
        {
            return new Vector<Complex<T>>(0);
        }

        return start == 0 && count == input.Length ? input : input.GetSubVector(start, count);
    }

    /// <summary>
    /// A complex vector of zeros.
    /// </summary>
    internal static Vector<Complex<T>> Zeros(int length)
    {
        var zeros = new Vector<Complex<T>>(length);
        zeros.Fill(new Complex<T>(NumOps.Zero, NumOps.Zero));
        return zeros;
    }
}
