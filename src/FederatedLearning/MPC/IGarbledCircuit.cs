namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Defines the contract for garbled circuit generation and evaluation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Garbled circuits are a technique for two parties to compute any
/// function on their combined inputs without revealing their inputs to each other.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item><description><b>Garbler:</b> Encodes a boolean circuit into an "encrypted" version where each
/// wire carries a random label instead of 0/1. Publishes the garbled truth tables for each gate.</description></item>
/// <item><description><b>Evaluator:</b> Given the garbled circuit and wire labels for its inputs (obtained
/// via oblivious transfer), evaluates the circuit gate-by-gate to get the output label.</description></item>
/// <item><description><b>Decoding:</b> The output label maps back to the actual output bit.</description></item>
/// </list>
///
/// <para><b>Optimizations supported:</b></para>
/// <list type="bullet">
/// <item><description><b>Free XOR:</b> XOR gates are "free" (no garbled table needed).</description></item>
/// <item><description><b>Half-gates:</b> AND gates need only 2 ciphertexts instead of 4.</description></item>
/// <item><description><b>Point-and-permute:</b> Each label carries a permutation bit for fast table lookup.</description></item>
/// </list>
/// </remarks>
public interface IGarbledCircuit
{
    /// <summary>
    /// Garbles a boolean circuit represented as a list of gate operations.
    /// </summary>
    /// <param name="gates">The circuit gates to garble.</param>
    /// <param name="inputWireCount">Number of input wires.</param>
    /// <param name="outputWireCount">Number of output wires.</param>
    /// <returns>The garbled circuit data.</returns>
    GarbledCircuitData Garble(IReadOnlyList<CircuitGate> gates, int inputWireCount, int outputWireCount);

    /// <summary>
    /// Evaluates a garbled circuit given input wire labels.
    /// </summary>
    /// <param name="garbledData">The garbled circuit produced by <see cref="Garble"/>.</param>
    /// <param name="inputLabels">The wire labels for the evaluator's inputs.</param>
    /// <returns>The output wire labels.</returns>
    byte[][] Evaluate(GarbledCircuitData garbledData, byte[][] inputLabels);

    /// <summary>
    /// Decodes the output wire labels to actual output bits.
    /// </summary>
    /// <param name="outputLabels">The output wire labels from evaluation.</param>
    /// <param name="decodingTable">The decoding table from the garbler.</param>
    /// <returns>The output bits (0 or 1 each).</returns>
    int[] Decode(byte[][] outputLabels, byte[][] decodingTable);
}

/// <summary>
/// Represents a single gate in a boolean circuit.
/// </summary>
public class CircuitGate
{
    /// <summary>Gets or sets the type of gate (AND, XOR, NOT, OR).</summary>
    public GateType Type { get; set; }

    /// <summary>Gets or sets the index of the first input wire.</summary>
    public int InputWire0 { get; set; }

    /// <summary>Gets or sets the index of the second input wire (-1 for unary gates like NOT).</summary>
    public int InputWire1 { get; set; } = -1;

    /// <summary>Gets or sets the index of the output wire.</summary>
    public int OutputWire { get; set; }
}

/// <summary>
/// Types of boolean gates supported in garbled circuits.
/// </summary>
public enum GateType
{
    /// <summary>AND gate — output is 1 only if both inputs are 1.</summary>
    And,

    /// <summary>XOR gate — output is 1 if inputs differ. Free in garbled circuits.</summary>
    Xor,

    /// <summary>NOT gate (unary) — output is the inverse of input.</summary>
    Not,

    /// <summary>OR gate — output is 1 if either input is 1.</summary>
    Or
}

/// <summary>
/// Contains the garbled circuit data produced by the garbler.
/// </summary>
public class GarbledCircuitData
{
    /// <summary>Gets or sets the garbled truth tables for each gate.</summary>
    public IReadOnlyList<byte[][]> GarbledTables { get; set; } = Array.Empty<byte[][]>();

    /// <summary>Gets or sets the wire labels for the garbler's input wires (both 0 and 1 labels).</summary>
    public byte[][][] InputWireLabels { get; set; } = Array.Empty<byte[][]>();

    /// <summary>Gets or sets the decoding table for output wires.</summary>
    public byte[][] DecodingTable { get; set; } = Array.Empty<byte[]>();

    /// <summary>Gets or sets the number of input wires.</summary>
    public int InputWireCount { get; set; }

    /// <summary>Gets or sets the number of output wires.</summary>
    public int OutputWireCount { get; set; }

    /// <summary>Gets or sets the total number of wires in the circuit.</summary>
    public int TotalWireCount { get; set; }

    /// <summary>Gets or sets the circuit gates.</summary>
    public IReadOnlyList<CircuitGate> Gates { get; set; } = Array.Empty<CircuitGate>();
}
