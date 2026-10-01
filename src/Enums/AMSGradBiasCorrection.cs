namespace AiDotNet.Enums;

/// <summary>
/// Which bias corrections AMSGrad applies to its moment estimates.
/// </summary>
/// <remarks>
/// <para>
/// AMSGrad (Reddi, Kale and Kumar, "On the Convergence of Adam and Beyond", ICLR 2018) keeps a running maximum of
/// the second moment so the effective learning rate never increases. The paper's Algorithm 2 applies no bias
/// correction at all. Most frameworks instead layer AMSGrad onto Adam's bias correction: PyTorch's
/// <c>Adam(amsgrad=True)</c> divides the first moment by (1 - beta1^t) and the running maximum by (1 - beta2^t).
/// </para>
/// <para><b>For Beginners:</b> Both moving averages start at zero, so early on they are biased toward zero.
/// "Bias correction" scales them up to compensate. The original paper skips that step; PyTorch includes it, which
/// makes the first few updates larger. Pick <see cref="Paper"/> to reproduce the paper, or <see cref="PyTorch"/> to
/// match results from PyTorch.</para>
/// </remarks>
public enum AMSGradBiasCorrection
{
    /// <summary>
    /// No bias correction, as in the paper's Algorithm 2: update = lr * m / (sqrt(vMax) + epsilon).
    /// </summary>
    Paper,

    /// <summary>
    /// PyTorch <c>amsgrad=True</c>: update = lr * (m / (1 - beta1^t)) / (sqrt(vMax / (1 - beta2^t)) + epsilon).
    /// </summary>
    PyTorch
}
