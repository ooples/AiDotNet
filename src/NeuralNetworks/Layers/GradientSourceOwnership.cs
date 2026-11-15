namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Who owns the gradient tensors handed to a layer's scatter, which decides whether the published snapshot may
/// reference them or must copy them.
/// </summary>
internal enum GradientSourceOwnership
{
    /// <summary>
    /// Supplied through the public scatter. The caller may change them after publication, so device gradients are
    /// copied to keep the snapshot fixed.
    /// </summary>
    Caller,

    /// <summary>
    /// A tape step's own gradients. They outlive the tape, and nothing writes them once the step publishes them, so
    /// they are referenced and retained on the tape.
    /// </summary>
    Tape,

    /// <summary>
    /// A compiled plan's gradient buffers. They stay valid until the plan's next step publishes them again, so they
    /// are referenced on the device and on the host.
    /// </summary>
    Plan,
}
