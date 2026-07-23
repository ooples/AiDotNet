using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Defines the contract for committing to gradient values before revealing them.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In FL, a malicious client could wait to see other clients' updates
/// and then craft its own update to manipulate the result (an "adaptive attack"). Commitment
/// schemes prevent this: each client first sends a "commitment" (a cryptographic lock on its
/// gradient), and only after all commitments are received does anyone reveal the actual values.</para>
///
/// <para><b>Properties:</b></para>
/// <list type="bullet">
/// <item><description><b>Hiding:</b> The commitment reveals nothing about the gradient value.</description></item>
/// <item><description><b>Binding:</b> The client cannot change its gradient after committing.</description></item>
/// </list>
///
/// <para><b>Homomorphic commitments</b> (like Pedersen) have an additional property: the server
/// can verify that the sum of committed values matches the committed sum, without seeing
/// individual values.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IGradientCommitment<T>
{
    /// <summary>
    /// Creates a commitment to a gradient tensor.
    /// </summary>
    /// <param name="gradient">The gradient tensor to commit to.</param>
    /// <returns>A commitment that can later be opened with <see cref="Open"/>.</returns>
    GradientCommitmentData<T> Commit(Tensor<T> gradient);

    /// <summary>
    /// Opens a commitment, revealing the original gradient value.
    /// </summary>
    /// <param name="commitment">The commitment data.</param>
    /// <returns>The gradient tensor that was committed to, or null if the commitment is invalid.</returns>
    Tensor<T>? Open(GradientCommitmentData<T> commitment);

    /// <summary>
    /// Verifies that an opened commitment matches the original commitment.
    /// </summary>
    /// <param name="commitment">The commitment data (with the gradient revealed).</param>
    /// <returns>True if the opened value matches the commitment.</returns>
    bool Verify(GradientCommitmentData<T> commitment);

    /// <summary>
    /// Verifies that the sum of individual commitments matches a claimed aggregate commitment.
    /// Only supported by homomorphic commitment schemes (e.g., Pedersen).
    /// </summary>
    /// <param name="individualCommitments">The individual client commitments.</param>
    /// <param name="aggregateCommitment">The claimed sum commitment.</param>
    /// <returns>True if the aggregate matches the sum of individuals.</returns>
    bool VerifyAggregation(
        IReadOnlyList<GradientCommitmentData<T>> individualCommitments,
        GradientCommitmentData<T> aggregateCommitment);
}

/// <summary>
/// Contains the data for a gradient commitment.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class GradientCommitmentData<T>
{
    /// <summary>Gets or sets the commitment value (cryptographic hash or group element).</summary>
    public byte[] CommitmentValue { get; set; } = Array.Empty<byte>();

    /// <summary>Gets or sets the randomness used to create the commitment (kept secret until open).</summary>
    public byte[] Randomness { get; set; } = Array.Empty<byte>();

    /// <summary>Gets or sets the committed gradient tensor (null until opened).</summary>
    public Tensor<T>? Gradient { get; set; }

    /// <summary>Gets or sets the client ID that created this commitment.</summary>
    public int ClientId { get; set; }

    /// <summary>Gets or sets the round number when the commitment was created.</summary>
    public int Round { get; set; }
}
