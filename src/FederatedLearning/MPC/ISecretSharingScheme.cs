using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Defines the contract for a secret sharing scheme that splits and recombines tensor values.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Secret sharing lets you split a secret number into multiple
/// "shares" that individually look random. The original secret can only be recovered when
/// enough shares are combined. This is the foundation of MPC.</para>
///
/// <para><b>Types of secret sharing:</b></para>
/// <list type="bullet">
/// <item><description><b>Additive:</b> Shares sum to the secret. Any single missing share
/// makes reconstruction impossible. Fast but fragile (all parties must participate).</description></item>
/// <item><description><b>Shamir (threshold):</b> Any t-out-of-n shares suffice to reconstruct.
/// More robust but slightly slower due to polynomial evaluation.</description></item>
/// </list>
///
/// <para><b>Extends:</b> This generic interface works with tensors and complements the existing
/// <c>ShamirSecretSharing</c> class (which operates on raw byte arrays for crypto-level operations).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISecretSharingScheme<T>
{
    /// <summary>
    /// Splits a tensor value into secret shares for the specified number of parties.
    /// </summary>
    /// <param name="secret">The plaintext tensor to share.</param>
    /// <param name="numberOfParties">The number of parties to create shares for.</param>
    /// <returns>An array of share tensors, one per party.</returns>
    Tensor<T>[] Split(Tensor<T> secret, int numberOfParties);

    /// <summary>
    /// Reconstructs the original tensor from a set of shares.
    /// </summary>
    /// <param name="shares">The shares to combine.</param>
    /// <returns>The reconstructed plaintext tensor.</returns>
    Tensor<T> Combine(Tensor<T>[] shares);

    /// <summary>
    /// Gets the minimum number of shares required to reconstruct the secret.
    /// For additive sharing, this equals the total number of parties.
    /// For threshold sharing, this is the threshold parameter.
    /// </summary>
    int ReconstructionThreshold { get; }

    /// <summary>
    /// Gets the name of this secret sharing scheme.
    /// </summary>
    string SchemeName { get; }
}
