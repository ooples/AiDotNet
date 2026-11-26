namespace AiDotNet.Helpers;

/// <summary>
/// Evaluates the backward episodic recurrence <c>s_t = drive_t + decay * (1 - done_t) * s_(t+1)</c> with whole-tensor
/// engine ops: the discounted returns-to-go and GAE(λ) recursions of on-policy reinforcement learning, where an
/// episode end stops the discounted carry.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Within a block the recurrence is one matrix product <c>s = M d</c> with <c>M[t, k] = decay^(k - t)</c> for
/// <c>k &gt;= t</c> when no episode ends in <c>[t, k)</c>, and 0 otherwise. An episode end inside <c>[t, k)</c> is
/// exactly a change in the exclusive running count of dones between <c>t</c> and <c>k</c>. The value after the block
/// enters as one more drive element with no episode end, so blocks are solved from the last one backwards, each
/// carrying its first value into the block before it; that keeps the matrices small and the powers bounded.
/// </remarks>
internal static class EpisodicRecurrence<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Block length of the blocked evaluation.</summary>
    private const int Block = 256;

    /// <summary>
    /// <c>s_t = drive_t + decay * (1 - done_t) * s_(t+1)</c> for <c>t = n-1..0</c> with <c>s_n = terminal</c>.
    /// </summary>
    /// <param name="drive">The per-step drive, 1-D of length n (rewards, or TD residuals for GAE).</param>
    /// <param name="dones">1 where step t ends its episode, 0 otherwise; 1-D of length n.</param>
    /// <param name="decay">The per-step decay (gamma, or gamma * lambda for GAE).</param>
    /// <param name="terminal">The value after the last step, shape [1] (a bootstrap such as V(s_n), or 0).</param>
    internal static Tensor<T> SolveBackward(Tensor<T> drive, Tensor<T> dones, T decay, Tensor<T> terminal)
    {
        var engine = AiDotNetEngine.Current;
        int n = drive.Length;
        if (n == 0)
        {
            return drive;
        }

        var flatDrive = engine.Reshape(drive, new[] { n });
        var flatDones = engine.Reshape(dones, new[] { n });
        var carry = engine.Reshape(terminal, new[] { 1 });
        var pieces = new List<Tensor<T>>();
        for (int end = n; end > 0; end -= Block)
        {
            int start = Math.Max(0, end - Block);
            int length = end - start;
            var piece = SolveBlock(
                engine.TensorSlice(flatDrive, new[] { start }, new[] { length }),
                engine.TensorSlice(flatDones, new[] { start }, new[] { length }),
                carry,
                decay);
            pieces.Insert(0, piece);
            carry = engine.TensorSlice(piece, new[] { 0 }, new[] { 1 });
        }

        return pieces.Count == 1 ? pieces[0] : engine.TensorConcatenate(pieces.ToArray(), 0);
    }

    private static Tensor<T> SolveBlock(Tensor<T> drive, Tensor<T> dones, Tensor<T> carry, T decay)
    {
        var engine = AiDotNetEngine.Current;
        int m = drive.Length;
        int length = m + 1;
        var square = new[] { length, length };
        var extendedDrive = engine.TensorConcatenate(new[] { drive, carry }, 0);
        var extendedDones = engine.TensorConcatenate(new[] { dones, new Tensor<T>(new[] { 1 }) }, 0);

        var position = engine.TensorArange<T>(0, length, 1);
        var lag = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(position, new[] { 1, length }), square),
            engine.TensorBroadcastTo(engine.Reshape(position, new[] { length, 1 }), square));
        var decays = new Tensor<T>(square);
        engine.TensorFill(decays, decay);
        var discounts = engine.TensorTriu(engine.TensorPower(decays, engine.TensorClampMin(lag, NumOps.Zero)), 0);

        var donesBefore = engine.TensorSubtract(engine.TensorCumSum(extendedDones, 0), extendedDones);
        var sameEpisode = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(donesBefore, new[] { 1, length }), square),
            engine.TensorBroadcastTo(engine.Reshape(donesBefore, new[] { length, 1 }), square));

        var solved = engine.TensorMatMul(
            engine.TensorMultiply(discounts, sameEpisode), engine.Reshape(extendedDrive, new[] { length, 1 }));
        return engine.TensorSlice(engine.Reshape(solved, new[] { length }), new[] { 0 }, new[] { m });
    }
}
