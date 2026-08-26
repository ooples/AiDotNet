namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The shape of one layer's parameter surface, as recorded in a v5 checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// A flat parameter vector cannot say how many tensors it came from or what shape each had, so
/// restore was left inferring that from the vector's length. The inference is not always possible:
/// <c>EmbeddingLayer</c>'s input projection exists only when the data turned out to be continuous
/// rather than token indices, so a freshly constructed clone cannot know whether the checkpoint
/// holds one tensor or two. Recording the layout removes the guessing for every layer at once.
/// </para>
/// <para>
/// Parsed as a tree rather than applied straight off the stream because deserialization reads a
/// layer's bytes BEFORE constructing the layer they describe.
/// </para>
/// </remarks>
internal sealed class ParameterLayoutNode
{
    /// <summary>Length of the layer's own flat <c>Parameters</c> slot.</summary>
    public int OwnLength { get; private set; }

    /// <summary>
    /// The layer's resolved per-sample input shape, or an empty array when it had nothing concrete
    /// to record. Carried per NODE rather than once per checkpoint so a lazy layer nested at any
    /// depth can rebuild -- a composite's parameter surface IS its children's, and a child that
    /// never learns its input shape contributes nothing to it.
    /// </summary>
    public int[] ResolvedInputShape { get; private set; } = [];

    /// <summary>Shapes of the layer's trainable tensors, in fold order.</summary>
    public int[][] TensorShapes { get; private set; } = [];

    /// <summary>Named non-trainable buffers and their shapes.</summary>
    public (string Name, int[] Shape)[] Buffers { get; private set; } = [];

    /// <summary>Layouts of the registered sub-layers, in fold order.</summary>
    public ParameterLayoutNode[] Children { get; private set; } = [];

    /// <summary>Reads one layer's layout, recursing into its sub-layers.</summary>
    public static ParameterLayoutNode Read(System.IO.BinaryReader reader)
    {
        var node = new ParameterLayoutNode { OwnLength = reader.ReadInt32() };
        node.ResolvedInputShape = ReadShape(reader);

        int tensorCount = reader.ReadInt32();
        var shapes = new int[tensorCount][];
        for (int i = 0; i < tensorCount; i++) shapes[i] = ReadShape(reader);
        node.TensorShapes = shapes;

        int bufferCount = reader.ReadInt32();
        var buffers = new (string, int[])[bufferCount];
        for (int i = 0; i < bufferCount; i++)
        {
            string name = reader.ReadString();
            buffers[i] = (name, ReadShape(reader));
        }
        node.Buffers = buffers;

        int subCount = reader.ReadInt32();
        var children = new ParameterLayoutNode[subCount];
        for (int i = 0; i < subCount; i++) children[i] = Read(reader);
        node.Children = children;

        return node;
    }

    /// <summary>Largest tensor rank this reader will believe from a stream.</summary>
    /// <remarks>
    /// Not a limit on real tensors -- nothing in the library approaches it. It is the point past
    /// which the value can only have come from a misaligned or corrupt stream.
    /// </remarks>
    private const int MaxCredibleRank = 64;

    private static int[] ReadShape(System.IO.BinaryReader reader)
    {
        int rank = reader.ReadInt32();

        // A rank read off a misaligned stream is arbitrary. Unchecked, `new int[rank]` turned that
        // into an OverflowException or a multi-gigabyte allocation thrown from deep inside the
        // reader, which said nothing about the real problem -- the bytes did not line up. Failing
        // here names the cause, and does so before allocating anything.
        if (rank < 0 || rank > MaxCredibleRank)
        {
            throw new System.IO.InvalidDataException(
                $"Parameter layout is corrupt: read a tensor rank of {rank}, which is outside "
                    + $"0..{MaxCredibleRank}. The stream is misaligned -- the layout was written by "
                    + "a different format version, or a preceding field was read at the wrong width.");
        }

        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int dimension = reader.ReadInt32();
            if (dimension < 0)
            {
                throw new System.IO.InvalidDataException(
                    $"Parameter layout is corrupt: dimension {i} of a rank-{rank} shape is "
                        + $"{dimension}. A negative extent is unrepresentable; the usual cause is an "
                        + "int product that overflowed before it was written.");
            }

            shape[i] = dimension;
        }

        return shape;
    }
}
