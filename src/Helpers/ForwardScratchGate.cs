namespace AiDotNet.Helpers;

/// <summary>
/// Process-wide gate for the #1672 destination-buffer (<c>*Into</c>) forward path.
/// </summary>
/// <remarks>
/// <para>
/// The DiT / SiT inference forward allocates a fresh tensor for every op — SDPA scores
/// (~4 GB/forward), the AdaLN/gate broadcasts (~1.9 GB), the MLP/QKV projections — which
/// stalls the cores on memory bandwidth (issue #1672). When this gate is <b>ON</b>, the
/// affected call sites route through the new <c>AiDotNet.Tensors</c> <c>*Into</c> engine
/// overloads that write into a per-predictor reusable scratch buffer instead of allocating.
/// </para>
/// <para>
/// <b>OFF by default.</b> Enable with environment variable <c>AIDOTNET_FWD_SCRATCH=1</c>.
/// The <c>*Into</c> path is the same math into a reused buffer, so output is bit-identical;
/// the flag exists only to give a clean isolated A/B measurement and to keep the scratch
/// reuse strictly opt-in (it is safe only on the no-tape inference forward — the scratch is a
/// dedicated per-op buffer that no other live tensor aliases, distinct from the #1668 shared
/// arena). When OFF, the original allocating path runs unchanged.
/// </para>
/// </remarks>
public static class ForwardScratchGate
{
    /// <summary>
    /// True when <c>AIDOTNET_FWD_SCRATCH=1</c>. Read once at process start so the gate
    /// is a constant branch on the hot path.
    /// </summary>
    private static readonly bool s_envEnabled =
        string.Equals(
            System.Environment.GetEnvironmentVariable("AIDOTNET_FWD_SCRATCH"),
            "1",
            System.StringComparison.Ordinal);

    private static bool Sub(string name) =>
        !string.Equals(System.Environment.GetEnvironmentVariable(name), "0", System.StringComparison.Ordinal);

    private static readonly bool s_envSdpa = s_envEnabled && Sub("AIDOTNET_FWD_SCRATCH_SDPA");
    private static readonly bool s_envAdaLn = s_envEnabled && Sub("AIDOTNET_FWD_SCRATCH_ADALN");
    private static readonly bool s_envFusedLinear = s_envEnabled && Sub("AIDOTNET_FWD_SCRATCH_FUSEDLINEAR");
    private static readonly bool s_envFusedQkv = s_envEnabled && Sub("AIDOTNET_FWD_SCRATCH_FUSEDQKV");

    /// <summary>
    /// Optional in-process override for the gate, used by the correctness A/B probe to flip the
    /// path on the SAME model instance (weights are random per process, so a cross-process checksum
    /// can't prove bit-identicality). Null = use the environment-variable value. Not used in
    /// production (env flag is the real switch); leaving it null keeps the env behavior.
    /// </summary>
    public static bool? Override;

    /// <summary>True when the #1672 scratch path is enabled at all.</summary>
    public static bool Enabled => Override ?? s_envEnabled;

    /// <summary>SDPA *Into sub-gate. ON when <see cref="Enabled"/> unless AIDOTNET_FWD_SCRATCH_SDPA=0 (diagnostic).</summary>
    public static bool Sdpa => Override.HasValue ? Override.Value : s_envSdpa;

    /// <summary>AdaLN/gate broadcast *Into sub-gate. ON when <see cref="Enabled"/> unless AIDOTNET_FWD_SCRATCH_ADALN=0 (diagnostic).</summary>
    public static bool AdaLn => Override.HasValue ? Override.Value : s_envAdaLn;

    /// <summary>
    /// FusedLinear *Into sub-gate. ON when <see cref="Enabled"/> unless AIDOTNET_FWD_SCRATCH_FUSEDLINEAR=0
    /// (diagnostic). Routes Dense/FeedForward inference matmuls through
    /// <c>Engine.FusedLinearInto</c> writing into a per-layer resident scratch buffer instead of
    /// allocating the <c>[batch, outputSize]</c> output each call — the dominant residual allocator
    /// on the DiT/SiT forward (#1672).
    /// </summary>
    public static bool FusedLinear => Override.HasValue ? Override.Value : s_envFusedLinear;

    /// <summary>
    /// FusedQKV sub-gate. ON when <see cref="Enabled"/> unless AIDOTNET_FWD_SCRATCH_FUSEDQKV=0
    /// (diagnostic). Fuses the three separate Q/K/V projection matmuls in
    /// <c>SelfAttentionLayer.Forward</c> into ONE <c>[seq, D] @ [D, 3D]</c> GEMM against a cached
    /// horizontally-concatenated <c>[D, 3D]</c> weight, then slices the <c>[seq, 3D]</c> result back
    /// into Q/K/V. One engine dispatch instead of three (+ three fewer allocations) per attention
    /// block — the parallel-GEMM park/wakeup that gates the DiT/SiT inference forward (#1672).
    /// Bit-identical: the per-output-column dot product is unchanged by the wider N.
    /// </summary>
    public static bool FusedQkv => Override.HasValue ? Override.Value : s_envFusedQkv;
}
