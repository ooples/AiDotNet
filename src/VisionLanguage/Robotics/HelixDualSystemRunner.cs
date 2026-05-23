using System;
using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Streaming controller that coordinates Helix's fast System-1 visuomotor policy (200 Hz) and
/// slow System-2 VLM (7–9 Hz) per Figure AI's dual-rate dual-system architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Per Helix (Figure AI 2025, arXiv:2502.07092), high-level intent runs slowly while low-level
/// control runs fast. This runner is the explicit coordinator: each call to <see cref="Step"/>
/// advances one S1 tick and lazily re-invokes S2 when the cached latent has gone stale (every
/// <see cref="System2TicksValid"/> S1 ticks — default 22 to match S1:S2 = 200Hz : ~9Hz).
/// </para>
/// <para>
/// The runner is generic over the model's S2 and S1 callbacks so the same coordination logic is
/// reusable: <see cref="Helix{T}"/> wires its native VLM into <c>system2Forward</c> and its action
/// head into <c>system1Forward</c>. Future Helix-class models (e.g. Figure's later iterations) can
/// reuse this runner unchanged.
/// </para>
/// </remarks>
public sealed class HelixDualSystemRunner<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Func<Tensor<T>, string, Tensor<T>> _system2Forward;
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>> _system1Forward;
    private readonly int _system2TicksValid;
    private HelixSystem2Latent<T>? _cachedLatent;
    private int _currentTick;

    /// <summary>Number of S1 ticks an S2 latent remains valid (default 22 — paper §4.1: S1 @ 200 Hz, S2 @ 7–9 Hz).</summary>
    public int System2TicksValid => _system2TicksValid;

    /// <summary>Current S1 tick counter since the runner was created or <see cref="Reset"/> was called.</summary>
    public int CurrentTick => _currentTick;

    /// <summary>The currently cached S2 latent, or null if no S1 step has been taken yet.</summary>
    public HelixSystem2Latent<T>? CachedLatent => _cachedLatent;

    /// <summary>
    /// Constructs a runner.
    /// </summary>
    /// <param name="system2Forward">Slow VLM: <c>(observation, instruction) → latent tensor</c>. Called every <see cref="System2TicksValid"/> ticks.</param>
    /// <param name="system1Forward">Fast policy: <c>(observation, cached_S2_latent) → action tensor</c>. Called every tick.</param>
    /// <param name="system2TicksValid">How many S1 ticks one S2 invocation remains valid. Default 22 (matches paper rate ratio).</param>
    public HelixDualSystemRunner(
        Func<Tensor<T>, string, Tensor<T>> system2Forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> system1Forward,
        int system2TicksValid = 22)
    {
        if (system2TicksValid <= 0) throw new ArgumentOutOfRangeException(nameof(system2TicksValid), system2TicksValid, "system2TicksValid must be positive.");

        _system2Forward = system2Forward ?? throw new ArgumentNullException(nameof(system2Forward));
        _system1Forward = system1Forward ?? throw new ArgumentNullException(nameof(system1Forward));
        _system2TicksValid = system2TicksValid;
        _numOps = MathHelper.GetNumericOperations<T>();
        _currentTick = 0;
    }

    /// <summary>
    /// Advances one S1 tick. Re-invokes S2 first if the cached latent is missing or stale.
    /// Returns the S1 action tensor.
    /// </summary>
    public Tensor<T> Step(Tensor<T> observation, string instruction)
    {
        if (observation is null) throw new ArgumentNullException(nameof(observation));
        if (instruction is null) throw new ArgumentNullException(nameof(instruction));

        if (_cachedLatent is null || _cachedLatent.IsStaleAt(_currentTick))
        {
            var s2Latent = _system2Forward(observation, instruction);
            _cachedLatent = new HelixSystem2Latent<T>(s2Latent, _currentTick, _system2TicksValid);
        }

        var s1Action = _system1Forward(observation, _cachedLatent.Latent);
        _currentTick++;
        return s1Action;
    }

    /// <summary>
    /// Runs <paramref name="numSteps"/> consecutive S1 ticks against the same observation/instruction
    /// (e.g. for batch evaluation or simulation). Returns the per-tick action tensors.
    /// </summary>
    public Tensor<T>[] Rollout(Tensor<T> observation, string instruction, int numSteps)
    {
        if (numSteps <= 0) throw new ArgumentOutOfRangeException(nameof(numSteps), numSteps, "numSteps must be positive.");
        var rollout = new Tensor<T>[numSteps];
        for (int i = 0; i < numSteps; i++)
            rollout[i] = Step(observation, instruction);
        return rollout;
    }

    /// <summary>Resets the tick counter and invalidates the cached S2 latent. Use between episodes.</summary>
    public void Reset()
    {
        _cachedLatent = null;
        _currentTick = 0;
    }
}
