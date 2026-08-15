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

    private static int[] ReadShape(System.IO.BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++) shape[i] = reader.ReadInt32();
        return shape;
    }
}
