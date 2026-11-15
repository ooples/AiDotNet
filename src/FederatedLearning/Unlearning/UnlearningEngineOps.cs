namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Whole-tensor helpers shared by the federated unlearners.
/// </summary>
internal static class UnlearningEngineOps
{
    /// <summary>
    /// A new <c>[length]</c> tensor holding the first entries of <paramref name="source"/> (read flat,
    /// at most <c>Shape[0]</c> of them), zero padded when the source is shorter.
    /// </summary>
    internal static Tensor<T> FitLength<T>(Tensor<T> source, int length)
    {
        var fitted = new Tensor<T>(new[] { Math.Max(0, length) });
        int sourceRows = source.Shape.Length > 0 ? source.Shape[0] : 0;
        int available = Math.Max(0, Math.Min(length, Math.Min(sourceRows, source.Length)));
        if (available > 0)
        {
            source.AsSpan().Slice(0, available).CopyTo(fitted.AsWritableSpan());
        }

        return fitted;
    }

    /// <summary>
    /// Sum of <c>FitLength(update, length)</c> over <paramref name="updates"/>.
    /// </summary>
    internal static Tensor<T> SumFitted<T>(IEnumerable<Tensor<T>> updates, int length)
    {
        var engine = AiDotNetEngine.Current;
        var sum = new Tensor<T>(new[] { Math.Max(0, length) });
        foreach (var update in updates)
        {
            sum = engine.TensorAdd(sum, FitLength(update, length));
        }

        return sum;
    }

    /// <summary>Sum fitted updates without narrowing intermediate contributions.</summary>
    internal static Tensor<double> SumFittedWide<T>(IEnumerable<Tensor<T>> updates, int length)
    {
        var engine = AiDotNetEngine.Current;
        var sum = new Tensor<double>(new[] { Math.Max(0, length) });
        foreach (var update in updates)
            sum = engine.TensorAdd(sum, FitLength(update, length).Cast<double>());
        return sum;
    }

    /// <summary>Mean fitted updates, converting only the completed mean.</summary>
    internal static Tensor<T> MeanFitted<T>(IReadOnlyCollection<Tensor<T>> updates, int length)
    {
        var sum = SumFittedWide(updates, length);
        return (updates.Count > 0 ? AiDotNetEngine.Current.TensorDivideScalar(sum, (double)updates.Count) : sum).Cast<T>();
    }

    /// <summary>
    /// Euclidean distance between the first <c>min(a.Shape[0], b.Shape[0])</c> entries of the two tensors.
    /// </summary>
    internal static double L2Distance<T>(Tensor<T> a, Tensor<T> b)
    {
        int size = Math.Min(a.Shape[0], b.Shape[0]);
        if (size <= 0)
        {
            return 0.0;
        }

        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(FitLength(a, size).Cast<double>(), FitLength(b, size).Cast<double>());
        return Math.Sqrt(Dot(diff, diff));
    }

    /// <summary>
    /// sum_i a_i b_i, reduced in double. The certificate metrics are doubles, and in T the products and
    /// long sums overflow or saturate (Half: 65504, and integer spacing 2 above 2048).
    /// </summary>
    internal static double Dot<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSum(engine.TensorMultiply(a.Cast<double>(), b.Cast<double>()));
    }

    /// <summary>sum_i a_i, reduced in double (see <see cref="Dot{T}"/>).</summary>
    internal static double Sum<T>(Tensor<T> a)
    {
        return AiDotNetEngine.Current.TensorSum(a.Cast<double>());
    }

    /// <summary>Random bytes one standard normal consumes: two 53-bit uniforms, each read from two 32-bit words.</summary>
    internal const int RandomBytesPerNormal = 16;

    /// <summary>Samples <see cref="SecureStandardNormals{T}"/> draws and transforms at a time.</summary>
    internal const int NormalsPerChunk = 65536;

    /// <summary>
    /// The random bytes <paramref name="samples"/> standard normals consume, as a long: for more than 134 million
    /// samples the count is past <see cref="int.MaxValue"/>, and an int product wraps to a negative size.
    /// </summary>
    internal static long RandomBytesFor(long samples)
    {
        if (samples < 0) throw new ArgumentOutOfRangeException(nameof(samples));
        return checked(samples * RandomBytesPerNormal);
    }

    /// <summary>
    /// <paramref name="count"/> standard normal samples drawn from the operating system's cryptographic generator.
    /// Each sample consumes 16 fresh random bytes, so the noise cannot be reproduced from a small seed: an engine
    /// normal op seeded with one <c>Random.Next()</c> draw has at most 2^31 possible noise vectors, which an observer
    /// of the noised model can enumerate. The samples are drawn <see cref="NormalsPerChunk"/> at a time, so the
    /// memory used beyond the result does not grow with <paramref name="count"/>.
    /// </summary>
    internal static Tensor<T> SecureStandardNormals<T>(int count)
    {
        using (var generator = System.Security.Cryptography.RandomNumberGenerator.Create())
        {
            return StandardNormals<T>(count, bytes => generator.GetBytes(bytes), NormalsPerChunk);
        }
    }

    /// <summary>
    /// <paramref name="count"/> standard normals from the bytes <paramref name="fillRandomBytes"/> supplies, at most
    /// <paramref name="samplesPerChunk"/> samples at a time: each chunk asks for exactly the bytes its samples
    /// consume and is transformed by <see cref="StandardNormalsFromBytes{T}"/>, so the working memory is that of one
    /// chunk whatever the count.
    /// </summary>
    internal static Tensor<T> StandardNormals<T>(int count, Action<byte[]> fillRandomBytes, int samplesPerChunk)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (fillRandomBytes is null) throw new ArgumentNullException(nameof(fillRandomBytes));
        if (samplesPerChunk < 1) throw new ArgumentOutOfRangeException(nameof(samplesPerChunk));
        if (count == 0) return new Tensor<T>(new[] { 0 });

        var result = new T[count];
        var bytes = new byte[checked((int)RandomBytesFor(Math.Min(count, samplesPerChunk)))];
        int done = 0;
        while (done < count)
        {
            int take = Math.Min(samplesPerChunk, count - done);
            long needed = RandomBytesFor(take);
            if (bytes.Length != needed)
            {
                // The last chunk is shorter: the source is asked for no more bytes than are used.
                bytes = new byte[checked((int)needed)];
            }

            fillRandomBytes(bytes);
            Array.Copy(StandardNormalsFromBytes<T>(bytes, take).ToArray(), 0, result, done, take);
            done += take;
        }

        return new Tensor<T>(result, new[] { count });
    }

    /// <summary>
    /// Box-Muller standard normals from random bytes: sample i is sqrt(-2 ln u1_i) cos(2 pi u2_i). The bytes are read
    /// as 4 * <paramref name="count"/> little-endian 32-bit words; with c = count and w = k + 2^31 for a word read as the signed k,
    /// n1_i = w_i * 2^21 + floor(w_(c + i) / 2^11) and n2_i = w_(2c + i) * 2^21 + floor(w_(3c + i) / 2^11) are uniform
    /// 53-bit integers, u1 = (n1 + 1) / 2^53 lies in (0, 1] (the logarithm never sees 0) and u2 = n2 / 2^53 in [0, 1).
    /// The arithmetic runs in double, where every step up to the uniforms is exact, and is rounded to T once.
    /// </summary>
    /// <remarks>
    /// The smallest u1 is 2^-53, so |z| reaches sqrt(106 ln 2) = 8.57, the range of a double-precision uniform. A
    /// 32-bit uniform stops at 6.76: it cannot produce the 1.4e-11 of the normal's mass beyond that, a truncated tail
    /// that noise calibrated as Gaussian must not have.
    /// </remarks>
    internal static Tensor<T> StandardNormalsFromBytes<T>(byte[] bytes, int count)
    {
        if (bytes is null) throw new ArgumentNullException(nameof(bytes));
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (bytes.Length < RandomBytesFor(count)) throw new ArgumentException($"Each sample needs {RandomBytesPerNormal} bytes.", nameof(bytes));
        if (count == 0) return new Tensor<T>(new[] { 0 });

        var engine = AiDotNetEngine.Current;
        var words = new int[checked(4 * count)];
        Buffer.BlockCopy(bytes, 0, words, 0, checked(words.Length * 4));
        // A word read as a signed k is the unsigned word k + 2^31 with its top bit flipped: uniform over [0, 2^32).
        var unsigned = engine.TensorAddScalar(new Tensor<int>(words, new[] { words.Length }).Cast<double>(), 2147483648.0);

        Tensor<double> Integer53(int highAt, int lowAt)
        {
            var high = engine.TensorSlice(unsigned, new[] { highAt }, new[] { count });
            var low = engine.TensorSlice(unsigned, new[] { lowAt }, new[] { count });
            return engine.TensorAdd(engine.TensorMultiplyScalar(high, 2097152.0), engine.TensorFloor(engine.TensorMultiplyScalar(low, 1.0 / 2048.0)));
        }

        const double TwoToMinus53 = 1.0 / 9007199254740992.0;
        var u1 = engine.TensorMultiplyScalar(engine.TensorAddScalar(Integer53(0, count), 1.0), TwoToMinus53);
        var u2 = engine.TensorMultiplyScalar(Integer53(2 * count, 3 * count), TwoToMinus53);
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(engine.TensorLog(u1), -2.0));
        var angle = engine.TensorCos(engine.TensorMultiplyScalar(u2, 2.0 * Math.PI));
        return engine.TensorMultiply(radius, angle).Cast<T>();
    }

    /// <summary>
    /// Per-parameter influence of a client from its update history: |mean| * (1 + std) over the rounds, scaled to
    /// [0, 1] by the largest score (left as is when that is not positive or any score is NaN). The variance is the
    /// two-pass mean of squared deviations: E[x^2] - E[x]^2 cancels catastrophically in float and Half when the
    /// updates are large next to their spread.
    /// </summary>
    internal static Tensor<T> InfluenceScores<T>(IReadOnlyList<Tensor<T>> updates, int modelSize)
    {
        var engine = AiDotNetEngine.Current;
        var influence = new Tensor<double>(new[] { Math.Max(0, modelSize) });
        if (updates.Count > 0)
        {
            double rounds = updates.Count;
            var mean = engine.TensorDivideScalar(SumFittedWide(updates, modelSize), rounds);
            var squaredDeviation = new Tensor<double>(new[] { Math.Max(0, modelSize) });
            foreach (var update in updates)
            {
                squaredDeviation = engine.TensorAdd(squaredDeviation, engine.TensorSquare(engine.TensorSubtract(FitLength(update, modelSize).Cast<double>(), mean)));
            }

            var std = engine.TensorSqrt(engine.TensorDivideScalar(squaredDeviation, rounds));
            influence = engine.TensorMultiply(engine.TensorAbs(mean), engine.TensorAddScalar(std, 1.0));
        }

        // A NaN score leaves the scores unnormalized, as the Math.Max fold did.
        if (modelSize > 0 && !double.IsNaN(Sum(influence)))
        {
            double maxInfluence = engine.ReduceMax(influence, new[] { 0 }, false).GetFlat(0);
            if (maxInfluence > 0)
            {
                influence = engine.TensorDivideScalar(influence, maxInfluence);
            }
        }

        return influence.Cast<T>();
    }
}
