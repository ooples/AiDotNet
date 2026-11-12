namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Whole-vector helpers shared by the diffusion schedulers (sigma tables, flow time steps and
/// linear multistep coefficients), expressed with engine operations.
/// </summary>
internal static class SchedulerTensorOps
{
    /// <summary>
    /// Vector of <paramref name="length"/> elements all equal to <paramref name="value"/>.
    /// </summary>
    internal static Vector<T> Full<T>(int length, T value)
    {
        var result = new Vector<T>(length);
        result.Fill(value);
        return result;
    }

    /// <summary>
    /// The entries of <paramref name="table"/> at the given integer <paramref name="indices"/>.
    /// </summary>
    internal static Vector<T> Gather<T>(Vector<T> table, int[] indices)
    {
        if (indices.Length == 0)
            return new Vector<T>(0);
        var index = new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices));
        return AiDotNetEngine.Current.TensorIndexSelect(table.AsTensor(), index, 0).ToVector();
    }

    /// <summary>
    /// The k-diffusion noise levels sigma_t = sqrt((1 - alpha_cumprod_t) / alpha_cumprod_t) at the given
    /// timesteps followed by a terminal sigma of zero (diffusers
    /// <c>((1 - alphas_cumprod) / alphas_cumprod) ** 0.5</c> with a trailing 0).
    /// </summary>
    internal static Vector<T> KarrasSigmas<T>(Vector<T> alphasCumprod, int[] timesteps)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var alphaCumprod = Gather(alphasCumprod, timesteps);
        var oneMinusAlpha = engine.Subtract(Full(alphaCumprod.Length, ops.One), alphaCumprod);
        var sigmas = engine.Sqrt(engine.Divide(oneMinusAlpha, alphaCumprod));
        return Vector<T>.Concatenate(sigmas, new Vector<T>(1));
    }

    /// <summary>
    /// The values L_j(x) = prod_{k != j} (x - s_k) / (s_j - s_k) of the Lagrange basis polynomials
    /// of the (distinct) <paramref name="nodes"/> s at the point <paramref name="x"/>.
    /// </summary>
    internal static Vector<T> LagrangeBasis<T>(Vector<T> nodes, T x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = nodes.Length;
        if (m == 1)
            return Full(1, ops.One);

        var square = new[] { m, m };
        // rows[j, k] = s_j, cols[j, k] = s_k.
        var rows = engine.TensorBroadcastTo(nodes.AsTensor().Reshape(new[] { m, 1 }), square);
        var cols = engine.TensorBroadcastTo(nodes.AsTensor().Reshape(new[] { 1, m }), square);
        var eye = engine.TensorEye<T>(m);
        var ones = Tensor<T>.CreateDefault(square, ops.One);
        // The k == j factor is excluded from the product: replace it with 1 in both numerator and denominator.
        var numerator = engine.TensorWhere(eye, ones, engine.ScalarMinusTensor(x, cols));
        var denominator = engine.TensorWhere(eye, ones, engine.TensorSubtract(rows, cols));
        var products = engine.TensorCumProd(engine.TensorDivide(numerator, denominator), 1);
        return engine.TensorSelect(products, 1, m - 1).ToVector();
    }

    /// <summary>
    /// The integrals of the Lagrange basis polynomials of <paramref name="nodes"/> over
    /// [<paramref name="from"/>, <paramref name="to"/>] (the linear multistep coefficients of
    /// k-diffusion / diffusers <c>get_lms_coefficient</c>). The basis has degree at most 3 for up to four
    /// nodes, so two-point Gauss-Legendre quadrature integrates it exactly.
    /// </summary>
    internal static Vector<T> IntegrateLagrangeBasis<T>(Vector<T> nodes, T from, T to)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T half = ops.Divide(ops.Subtract(to, from), ops.FromDouble(2.0));
        T mid = ops.Add(from, half);
        T offset = ops.Multiply(half, ops.FromDouble(1.0 / Math.Sqrt(3.0)));
        var atLow = LagrangeBasis(nodes, ops.Subtract(mid, offset));
        var atHigh = LagrangeBasis(nodes, ops.Add(mid, offset));
        return engine.Multiply(engine.Add(atLow, atHigh), half);
    }

    /// <summary>
    /// <paramref name="sample"/> + sum_j coefficients[j] * terms[j], with the terms stacked into one matrix
    /// product.
    /// </summary>
    internal static Vector<T> AddWeightedSum<T>(Vector<T> sample, Vector<T> coefficients, IReadOnlyList<Vector<T>> terms)
    {
        var engine = AiDotNetEngine.Current;
        int m = coefficients.Length;
        var stacked = new Tensor<T>[m];
        for (int j = 0; j < m; j++)
            stacked[j] = terms[j].AsTensor();
        var matrix = engine.TensorStack(stacked, 0);
        var weighted = engine.TensorMatMul(coefficients.AsTensor().Reshape(new[] { 1, m }), matrix);
        return engine.Add(sample, weighted.Reshape(new[] { sample.Length }).ToVector());
    }

    /// <summary>
    /// The continuous flow time of a training timestep, t = timestep / (trainTimesteps - 1), so the
    /// last training timestep is pure noise (t = 1) and timestep 0 is clean data (t = 0).
    /// </summary>
    internal static double FlowTime(int timestep, int trainTimesteps)
        => trainTimesteps > 1 ? (double)timestep / (trainTimesteps - 1) : 0.0;

    /// <summary>
    /// The flow time between <paramref name="timestep"/> and the next timestep of the schedule (t = 0 after
    /// the last one), i.e. the Euler step size of the probability-flow ODE dx/dt = v. Returns
    /// <paramref name="fallback"/> when the timestep is not part of the schedule.
    /// </summary>
    internal static double FlowStepSize(int[] timesteps, int timestep, int trainTimesteps, double fallback)
    {
        int index = Array.IndexOf(timesteps, timestep);
        if (index < 0)
            return fallback;
        double next = index + 1 < timesteps.Length ? FlowTime(timesteps[index + 1], trainTimesteps) : 0.0;
        return FlowTime(timestep, trainTimesteps) - next;
    }
}
