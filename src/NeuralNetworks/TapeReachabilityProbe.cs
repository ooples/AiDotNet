using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Asks the one question a parameter-movement test cannot: was a given tensor actually connected to
/// the loss that was just differentiated?
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the analogue of <c>torch.autograd.grad(loss, params)</c>, which raises "One of the
/// differentiated Tensors appears to not have been used in the graph" when a parameter is
/// unreachable. Reachability is the precise question for a severed tape, and nothing outside the
/// model can ask it otherwise: the tape is created and disposed inside the training call, and the
/// published gradient surface cannot tell "received no gradient" apart from a layer accessor that
/// manufactures zeros (most of the ~170 overrides of <c>GetParameterGradients</c> predate the tape
/// and return a freshly allocated zero vector).
/// </para>
/// <para>
/// Why it is needed at all: a component can be trained by the WRONG HALF of its objective and still
/// move every step, so "did the parameters change" cannot detect the defect. SAC's actor loss is
/// <c>alpha * log pi - min(Q1,Q2)</c>; when the critics are read through <c>Predict</c> (which runs
/// inside a <c>NoGradScope</c>) the Q term is a detached constant, yet the entropy term still carries
/// real gradient to the actor. The weights move, every liveness check passes, and the policy never
/// follows the reward. Reachability sees it immediately: the critic tensors simply are not in the
/// graph.
/// </para>
/// <para>
/// Results are recorded PER BACKWARD PASS, tagged with the network that owned it. That distinction is
/// essential rather than cosmetic: one agent training step runs several backward passes, and a tensor
/// legitimately reached by its own network's update would otherwise mask the fact that a later,
/// different update never reached it. SAC trains the critics before the actor, so a single
/// accumulated "was it ever reached" flag would report success for exactly the defect above.
/// </para>
/// <para>
/// Arm it around a training call, then inspect <see cref="Observations"/>. It is inert when not armed,
/// and scoped to the calling async flow so concurrent fixtures cannot see each other's probes.
/// </para>
/// </remarks>
internal sealed class TapeReachabilityProbe<T> : IDisposable
{
    private static readonly System.Threading.AsyncLocal<TapeReachabilityProbe<T>?> Scope = new();

    private readonly TapeReachabilityProbe<T>? _previous;
    private readonly List<Observation> _observations = new();

    private TapeReachabilityProbe(IReadOnlyList<Tensor<T>> requested)
    {
        Requested = requested;
        _previous = Scope.Value;
        Scope.Value = this;
    }

    /// <summary>The probe armed for the current async flow, or null when none is.</summary>
    internal static TapeReachabilityProbe<T>? Current => Scope.Value;

    /// <summary>The tensors whose reachability is being asked about.</summary>
    internal IReadOnlyList<Tensor<T>> Requested { get; }

    /// <summary>
    /// One entry per backward pass that ran while this probe was armed, in order, each tagged with
    /// the network whose training step produced it.
    /// </summary>
    internal IReadOnlyList<Observation> Observations => _observations;

    /// <summary>Arms a probe for the requested tensors until the returned scope is disposed.</summary>
    internal static TapeReachabilityProbe<T> Arm(IReadOnlyList<Tensor<T>> requested)
        => new(requested ?? throw new ArgumentNullException(nameof(requested)));

    /// <summary>
    /// Records which requested tensors this backward pass reached. A returned gradient that is
    /// entirely zero counts as NOT reached: a detached term and a genuinely zero gradient are
    /// indistinguishable downstream, and treating zeros as reachable would let the very defect this
    /// exists to catch report success.
    /// </summary>
    /// <param name="owner">The network whose training step ran this backward pass.</param>
    /// <param name="gradients">The gradients the tape produced.</param>
    internal void Record(object owner, IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var reached = new HashSet<Tensor<T>>(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

        if (gradients is not null && gradients.Count > 0)
        {
            foreach (var requested in Requested)
            {
                if (!gradients.TryGetValue(requested, out var gradient) || gradient is null) continue;

                for (int i = 0; i < gradient.Length; i++)
                {
                    if (!ops.Equals(gradient.GetFlat(i), ops.Zero))
                    {
                        reached.Add(requested);
                        break;
                    }
                }
            }
        }

        _observations.Add(new Observation(owner, reached));
    }

    /// <inheritdoc />
    public void Dispose() => Scope.Value = _previous;

    /// <summary>The outcome of a single backward pass observed by the probe.</summary>
    internal sealed class Observation
    {
        internal Observation(object owner, IReadOnlyCollection<Tensor<T>> reached)
        {
            Owner = owner;
            Reached = reached;
        }

        /// <summary>The network whose training step ran this backward pass.</summary>
        internal object Owner { get; }

        /// <summary>The requested tensors this particular backward pass produced a gradient for.</summary>
        internal IReadOnlyCollection<Tensor<T>> Reached { get; }
    }
}
