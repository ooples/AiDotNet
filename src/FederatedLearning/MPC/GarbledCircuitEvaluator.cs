using AiDotNet.FederatedLearning.Cryptography;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Evaluates garbled circuits produced by <see cref="GarbledCircuitGenerator"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The evaluator is the second party in a garbled circuit protocol.
/// It receives the garbled circuit and wire labels for its inputs (via oblivious transfer),
/// then processes each gate to compute the output — without learning the garbler's inputs
/// or any intermediate values.</para>
///
/// <para><b>Workflow:</b></para>
/// <list type="bullet">
/// <item><description>Receive garbled circuit from garbler.</description></item>
/// <item><description>Run OT to get wire labels for your input bits.</description></item>
/// <item><description>Evaluate each gate using the garbled tables.</description></item>
/// <item><description>Decode output labels to get the actual result bits.</description></item>
/// </list>
///
/// <para><b>Security:</b> The evaluator learns only the output. It cannot determine the garbler's
/// inputs because each wire label is a random cryptographic value that reveals nothing about
/// the underlying bit.</para>
/// </remarks>
public class GarbledCircuitEvaluator : IGarbledCircuit
{
    private readonly bool _enableFreeXor;
    private readonly int _labelLength;

    /// <summary>
    /// Initializes a new instance of <see cref="GarbledCircuitEvaluator"/>.
    /// </summary>
    /// <param name="enableFreeXor">Whether Free XOR optimization was used during garbling.</param>
    /// <param name="labelLengthBits">Wire label length in bits (must match garbler).</param>
    public GarbledCircuitEvaluator(bool enableFreeXor = true, int labelLengthBits = 128)
    {
        _enableFreeXor = enableFreeXor;
        _labelLength = labelLengthBits / 8;
    }

    /// <summary>
    /// Not supported by the evaluator — use <see cref="GarbledCircuitGenerator"/> to garble.
    /// </summary>
    public GarbledCircuitData Garble(IReadOnlyList<CircuitGate> gates, int inputWireCount, int outputWireCount)
    {
        throw new NotSupportedException("The evaluator cannot garble circuits. Use GarbledCircuitGenerator instead.");
    }

    /// <inheritdoc/>
    public byte[][] Evaluate(GarbledCircuitData garbledData, byte[][] inputLabels)
    {
        if (garbledData is null)
        {
            throw new ArgumentNullException(nameof(garbledData));
        }

        if (inputLabels is null)
        {
            throw new ArgumentNullException(nameof(inputLabels));
        }

        if (inputLabels.Length < garbledData.InputWireCount)
        {
            throw new ArgumentException(
                $"Expected at least {garbledData.InputWireCount} input labels, got {inputLabels.Length}.",
                nameof(inputLabels));
        }

        // Wire values during evaluation
        var wireValues = new byte[garbledData.TotalWireCount][];

        // Set input wire values from provided labels
        for (int w = 0; w < garbledData.InputWireCount; w++)
        {
            wireValues[w] = inputLabels[w];
        }

        // Evaluate each gate in topological order
        for (int g = 0; g < garbledData.Gates.Count; g++)
        {
            var gate = garbledData.Gates[g];
            var table = garbledData.GarbledTables[g];

            if (gate.Type == GateType.Xor && _enableFreeXor)
            {
                // Free XOR: output label = input0_label XOR input1_label
                wireValues[gate.OutputWire] = XorBytes(
                    wireValues[gate.InputWire0],
                    wireValues[gate.InputWire1]);
            }
            else if (gate.Type == GateType.Not)
            {
                // NOT: just copy (the garbler already swapped the label semantics)
                wireValues[gate.OutputWire] = wireValues[gate.InputWire0];
            }
            else if (table.Length > 0)
            {
                // Decrypt the appropriate garbled table entry
                wireValues[gate.OutputWire] = DecryptGateEntry(
                    wireValues[gate.InputWire0],
                    gate.InputWire1 >= 0 ? wireValues[gate.InputWire1] : null,
                    table,
                    g);
            }
            else
            {
                wireValues[gate.OutputWire] = wireValues[gate.InputWire0];
            }
        }

        // Extract output wire labels
        int firstOutput = garbledData.TotalWireCount - garbledData.OutputWireCount;
        var outputLabels = new byte[garbledData.OutputWireCount][];
        for (int i = 0; i < garbledData.OutputWireCount; i++)
        {
            byte[]? wireVal = wireValues[firstOutput + i];
            outputLabels[i] = wireVal ?? new byte[_labelLength];
        }

        return outputLabels;
    }

    /// <inheritdoc/>
    public int[] Decode(byte[][] outputLabels, byte[][] decodingTable)
    {
        if (outputLabels is null || decodingTable is null)
        {
            throw new ArgumentNullException(outputLabels is null ? nameof(outputLabels) : nameof(decodingTable));
        }

        if (outputLabels.Length != decodingTable.Length)
        {
            throw new ArgumentException("Output labels and decoding table must have the same length.");
        }

        var outputs = new int[outputLabels.Length];
        for (int i = 0; i < outputLabels.Length; i++)
        {
            // The decoding table entry is the 0-label
            // If our label matches, the output bit is 0; otherwise 1
            outputs[i] = ConstantTimeEquals(outputLabels[i], decodingTable[i]) ? 0 : 1;
        }

        return outputs;
    }

    /// <summary>
    /// Obtains input wire labels for the evaluator's input bits using oblivious transfer.
    /// </summary>
    /// <param name="garbledData">The garbled circuit data.</param>
    /// <param name="evaluatorInputBits">The evaluator's input bits.</param>
    /// <param name="evaluatorWireStart">The first wire index belonging to the evaluator.</param>
    /// <param name="ot">The oblivious transfer protocol to use.</param>
    /// <returns>The wire labels for the evaluator's inputs.</returns>
    public byte[][] ObtainInputLabels(
        GarbledCircuitData garbledData,
        int[] evaluatorInputBits,
        int evaluatorWireStart,
        IObliviousTransfer ot)
    {
        if (garbledData is null)
        {
            throw new ArgumentNullException(nameof(garbledData));
        }

        if (evaluatorInputBits is null)
        {
            throw new ArgumentNullException(nameof(evaluatorInputBits));
        }

        if (ot is null)
        {
            throw new ArgumentNullException(nameof(ot));
        }

        var labels = new byte[evaluatorInputBits.Length][];
        for (int i = 0; i < evaluatorInputBits.Length; i++)
        {
            int wireIdx = evaluatorWireStart + i;
            if (wireIdx >= garbledData.InputWireLabels.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(evaluatorWireStart),
                    "Evaluator wire index exceeds input wire count.");
            }

            // Use OT to get the label for the evaluator's choice bit
            labels[i] = ot.Transfer(
                garbledData.InputWireLabels[wireIdx][0],
                garbledData.InputWireLabels[wireIdx][1],
                evaluatorInputBits[i]);
        }

        return labels;
    }

    private byte[] DecryptGateEntry(byte[] inputLabel0, byte[]? inputLabel1, byte[][] table, int gateIndex)
    {
        // Point-and-permute: use last bit of labels to select row
        int bit0 = inputLabel0[_labelLength - 1] & 0x01;
        int bit1 = inputLabel1 is not null ? inputLabel1[_labelLength - 1] & 0x01 : 0;

        int rowIndex;
        if (table.Length == 4)
        {
            rowIndex = (bit0 << 1) | bit1;
        }
        else if (table.Length == 2)
        {
            rowIndex = bit0;
        }
        else
        {
            rowIndex = 0;
        }

        if (rowIndex >= table.Length)
        {
            rowIndex = 0;
        }

        // Derive decryption key: H(inputLabel0 || inputLabel1 || gateIndex)
        var key = DeriveKey(inputLabel0, inputLabel1 ?? Array.Empty<byte>(), gateIndex);

        return XorBytes(key, table[rowIndex]);
    }

    private byte[] DeriveKey(byte[] label0, byte[] label1, int gateIndex)
    {
        var input = new byte[label0.Length + label1.Length + 4];
        Buffer.BlockCopy(label0, 0, input, 0, label0.Length);
        Buffer.BlockCopy(label1, 0, input, label0.Length, label1.Length);
        var gateBytes = BitConverter.GetBytes(gateIndex);
        Buffer.BlockCopy(gateBytes, 0, input, label0.Length + label1.Length, 4);

        var salt = new byte[] { 0x47, 0x43, 0x48, 0x41, 0x53, 0x48 }; // "GCHASH"
        return HkdfSha256.DeriveKey(input, salt, Array.Empty<byte>(), _labelLength);
    }

    private static byte[] XorBytes(byte[] a, byte[] b)
    {
        int len = Math.Min(a.Length, b.Length);
        var result = new byte[len];
        for (int i = 0; i < len; i++)
        {
            result[i] = (byte)(a[i] ^ b[i]);
        }

        return result;
    }

    private static bool ConstantTimeEquals(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff |= a[i] ^ b[i];
        }

        return diff == 0;
    }
}
