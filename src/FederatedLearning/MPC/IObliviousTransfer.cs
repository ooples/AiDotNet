namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Defines the contract for an oblivious transfer (OT) protocol.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Oblivious transfer is a fundamental cryptographic building block.
/// A sender has two messages (m0 and m1). A receiver has a choice bit (0 or 1). After the
/// protocol:</para>
/// <list type="bullet">
/// <item><description>The receiver learns the message corresponding to its choice bit, but NOT the other message.</description></item>
/// <item><description>The sender learns nothing about which message the receiver chose.</description></item>
/// </list>
///
/// <para><b>Why this matters for FL:</b> OT is used as a building block for garbled circuits
/// and general MPC. When evaluating a garbled circuit, the evaluator needs to obtain the
/// correct wire labels for its input bits without revealing those bits to the garbler.</para>
///
/// <para><b>Performance:</b> Base OT is expensive (uses public-key crypto). OT extension lets
/// you amortize a small number of base OTs into many cheap OTs using symmetric crypto only.</para>
/// </remarks>
public interface IObliviousTransfer
{
    /// <summary>
    /// Performs a 1-out-of-2 oblivious transfer.
    /// </summary>
    /// <param name="message0">The first message (sent if receiver chooses 0).</param>
    /// <param name="message1">The second message (sent if receiver chooses 1).</param>
    /// <param name="choiceBit">The receiver's choice: 0 or 1.</param>
    /// <returns>The selected message (message0 if choice=0, message1 if choice=1).</returns>
    byte[] Transfer(byte[] message0, byte[] message1, int choiceBit);

    /// <summary>
    /// Performs a batch of 1-out-of-2 oblivious transfers.
    /// </summary>
    /// <param name="messages0">Array of first messages.</param>
    /// <param name="messages1">Array of second messages.</param>
    /// <param name="choiceBits">Array of choice bits (0 or 1 each).</param>
    /// <returns>An array where each entry is the selected message for that index.</returns>
    byte[][] BatchTransfer(byte[][] messages0, byte[][] messages1, int[] choiceBits);

    /// <summary>
    /// Gets the number of base OTs this protocol has performed (for accounting/extension).
    /// </summary>
    int BaseTransferCount { get; }
}
