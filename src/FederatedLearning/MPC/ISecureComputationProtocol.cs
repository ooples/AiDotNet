using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Defines the contract for a multi-party computation protocol that can perform secure
/// arithmetic and comparison operations on secret-shared values.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Imagine several parties each hold a piece of a number (a "share").
/// None of them know the original number, but together they can add, multiply, or compare
/// secret numbers by exchanging specially crafted messages — all without ever revealing the
/// actual values. This interface defines those operations.</para>
///
/// <para><b>How it works in FL:</b></para>
/// <list type="bullet">
/// <item><description><b>Share:</b> Convert a plaintext value into shares distributed to parties.</description></item>
/// <item><description><b>SecureAdd/Multiply:</b> Perform arithmetic on shares (the result is still shared).</description></item>
/// <item><description><b>SecureCompare:</b> Test if one secret value is greater than another.</description></item>
/// <item><description><b>Reconstruct:</b> Combine shares back into the plaintext result.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISecureComputationProtocol<T>
{
    /// <summary>
    /// Splits a plaintext value into secret shares for the specified number of parties.
    /// </summary>
    /// <param name="value">The plaintext value to share.</param>
    /// <param name="numberOfParties">How many parties will hold shares.</param>
    /// <returns>An array of shares, one per party.</returns>
    Tensor<T>[] Share(Tensor<T> value, int numberOfParties);

    /// <summary>
    /// Reconstructs the plaintext value from a set of shares.
    /// </summary>
    /// <param name="shares">The shares from each party.</param>
    /// <returns>The reconstructed plaintext value.</returns>
    Tensor<T> Reconstruct(Tensor<T>[] shares);

    /// <summary>
    /// Performs element-wise secure addition of two sets of shares.
    /// The result is a new set of shares representing the sum.
    /// </summary>
    /// <param name="sharesA">Shares of the first operand.</param>
    /// <param name="sharesB">Shares of the second operand.</param>
    /// <returns>Shares of the element-wise sum.</returns>
    Tensor<T>[] SecureAdd(Tensor<T>[] sharesA, Tensor<T>[] sharesB);

    /// <summary>
    /// Performs element-wise secure multiplication of two sets of shares.
    /// Requires interaction between parties (e.g., Beaver triples).
    /// </summary>
    /// <param name="sharesA">Shares of the first operand.</param>
    /// <param name="sharesB">Shares of the second operand.</param>
    /// <returns>Shares of the element-wise product.</returns>
    Tensor<T>[] SecureMultiply(Tensor<T>[] sharesA, Tensor<T>[] sharesB);

    /// <summary>
    /// Performs secure comparison: is <paramref name="sharesA"/> element-wise greater than <paramref name="sharesB"/>?
    /// Returns shares of a binary tensor (1 where true, 0 where false).
    /// </summary>
    /// <param name="sharesA">Shares of the first operand.</param>
    /// <param name="sharesB">Shares of the second operand.</param>
    /// <returns>Shares of a binary tensor with 1s where A &gt; B.</returns>
    Tensor<T>[] SecureCompare(Tensor<T>[] sharesA, Tensor<T>[] sharesB);

    /// <summary>
    /// Multiplies shares by a public (non-secret) scalar.
    /// This is a local operation — no interaction needed.
    /// </summary>
    /// <param name="shares">The secret-shared tensor.</param>
    /// <param name="scalar">The public scalar value.</param>
    /// <returns>Shares of the scaled tensor.</returns>
    Tensor<T>[] ScalarMultiply(Tensor<T>[] shares, T scalar);
}
