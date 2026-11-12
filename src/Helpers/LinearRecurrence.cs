namespace AiDotNet.Helpers;

/// <summary>
/// Evaluates the first-order linear recurrence <c>s_t = drive_t + decay * s_(t-1)</c> with whole-tensor engine ops.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The recurrence is the product of the drive with a lower-triangular Toeplitz matrix of decay powers. Evaluating it
/// in blocks keeps that matrix small and the powers bounded: each block is one matrix product plus the decayed carry
/// of the previous block's last value.
/// </remarks>
internal static class LinearRecurrence<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Block length of the blocked evaluation.</summary>
    private const int Block = 256;

    /// <summary>
    /// <c>s_t = drive_t + decay * s_(t-1)</c> for <c>t = 0..n-1</c> with <c>s_(-1) = initial</c>, over a 1-D drive.
    /// </summary>
    internal static Tensor<T> Solve(Tensor<T> drive, T decay, T initial)
    {
        var engine = AiDotNetEngine.Current;
        int n = drive.Length;
        if (n == 0)
        {
            return drive;
        }

        int block = Math.Min(Block, Math.Max(1, n));

        // q = (1, decay, decay^2, ...); toeplitz[i, j] = q[i - j] for i >= j; carry[i] = decay^(i + 1).
        var decays = new Tensor<T>(new[] { block });
        engine.TensorFill(decays, decay);
        var carryPowers = engine.TensorCumProd(decays, 0);
        var powers = engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { NumOps.One }, new[] { 1 }),
            engine.TensorSlice(carryPowers, new[] { 0 }, new[] { block - 1 })
        }, 0);
        var lag = Enumerable.Range(0, block * block).Select(k => Math.Max(0, (k / block) - (k % block))).ToArray();
        var toeplitz = engine.TensorTril(engine.Reshape(
            engine.TensorTake(powers, new Tensor<int>(lag, new[] { block * block })), new[] { block, block }));

        var carry = new Tensor<T>(new[] { initial }, new[] { 1 });
        var pieces = new List<Tensor<T>>();
        for (int start = 0; start < n; start += block)
        {
            int len = Math.Min(block, n - start);
            var segment = engine.Reshape(engine.TensorSlice(drive, new[] { start }, new[] { len }), new[] { len, 1 });
            var operatorBlock = len == block ? toeplitz : engine.TensorSlice(toeplitz, new[] { 0, 0 }, new[] { len, len });
            var homogeneous = engine.TensorMultiply(
                engine.TensorSlice(carryPowers, new[] { 0 }, new[] { len }),
                engine.TensorBroadcastTo(carry, new[] { len }));
            var piece = engine.TensorAdd(
                engine.Reshape(engine.TensorMatMul(operatorBlock, segment), new[] { len }), homogeneous);
            pieces.Add(piece);
            carry = engine.TensorSlice(piece, new[] { len - 1 }, new[] { 1 });
        }

        return pieces.Count == 1 ? pieces[0] : engine.TensorConcatenate(pieces.ToArray(), 0);
    }

    /// <summary>
    /// Discounted returns-to-go <c>G_t = r_t + gamma * G_(t+1)</c> with <c>G_n = 0</c>: the recurrence run backwards.
    /// </summary>
    internal static Tensor<T> DiscountedReturns(Tensor<T> rewards, T gamma)
    {
        var engine = AiDotNetEngine.Current;
        if (rewards.Length == 0)
        {
            return rewards;
        }

        var reversed = engine.TensorFlip(rewards, new[] { 0 });
        return engine.TensorFlip(Solve(reversed, gamma, NumOps.Zero), new[] { 0 });
    }
}
