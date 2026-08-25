namespace AiDotNet.Control;

/// <summary>
/// Whether a system evolves in discrete steps or in continuous time.
/// </summary>
/// <remarks>
/// <para>
/// The distinction changes the equations rather than merely their interpretation, so it has to be
/// stated rather than inferred: the discrete and continuous Riccati equations have different forms
/// and different solutions, and the optimal gain is read off each differently.
/// </para>
/// <para><b>For Beginners:</b> Discrete time is a system you sample and command at a fixed rate —
/// anything running on a computer, ultimately. Continuous time is the underlying physics, described
/// by differential equations. If you are writing a control loop that runs every 10 milliseconds, you
/// are almost certainly in discrete time.
/// </para>
/// </remarks>
public enum ControlTimeDomain
{
    /// <summary>
    /// The state advances one step at a time: <c>x[k+1] = A·x[k] + B·u[k]</c>.
    /// </summary>
    Discrete,

    /// <summary>
    /// The state evolves continuously: <c>ẋ(t) = A·x(t) + B·u(t)</c>.
    /// </summary>
    Continuous,
}
