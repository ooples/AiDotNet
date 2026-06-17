using System;
using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Conditioning signal that Helix's slow System-2 VLM produces for the fast System-1 controller.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Per Helix (Figure AI, 2025, "Helix: A Vision-Language-Action Model for Generalist Humanoid Control",
/// arXiv:2502.07092) the dual-system architecture splits responsibilities:
/// </para>
/// <list type="bullet">
///   <item><b>System 2</b>: a 7B-parameter VLM that runs at 7–9 Hz and emits a semantic latent encoding the high-level intent.</item>
///   <item><b>System 1</b>: an 80M-parameter visuomotor transformer that runs at 200 Hz and produces continuous joint commands.</item>
/// </list>
/// <para>
/// This class captures the per-tick S2 latent (timestamp, latent tensor, freshness counter) so the
/// dual-system runner can decide whether to re-invoke S2 or simply reuse the cached latent for the
/// next S1 step.
/// </para>
/// </remarks>
public sealed class HelixSystem2Latent<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>The latent feature vector produced by S2 — typically <c>DecoderDim</c>-sized.</summary>
    public Tensor<T> Latent { get; }

    /// <summary>S1 tick on which this latent was produced. Used by the runner to decide when to re-invoke S2.</summary>
    public int ProducedAtTick { get; }

    /// <summary>Maximum number of S1 ticks this latent remains valid; after that the runner must produce a fresh one.</summary>
    public int ValidForTicks { get; }

    /// <summary>Returns true when the latent has expired and S2 must be re-invoked.</summary>
    public bool IsStaleAt(int currentTick) => currentTick - ProducedAtTick >= ValidForTicks;

    public HelixSystem2Latent(Tensor<T> latent, int producedAtTick, int validForTicks)
    {
        if (latent is null)
            throw new ArgumentNullException(nameof(latent));
        if (validForTicks <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(validForTicks),
                validForTicks,
                "validForTicks must be positive."
            );

        Latent = latent;
        ProducedAtTick = producedAtTick;
        ValidForTicks = validForTicks;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Returns the L2-norm of the latent — useful for monitoring "S2 confidence" between invocations.
    /// </summary>
    public double L2Norm()
    {
        double sum = 0.0;
        for (int i = 0; i < Latent.Length; i++)
        {
            double v = _numOps.ToDouble(Latent[i]);
            sum += v * v;
        }
        return Math.Sqrt(sum);
    }
}
