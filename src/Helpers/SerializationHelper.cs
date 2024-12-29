namespace AiDotNet.Helpers;

public static class SerializationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static void SerializeNode(DecisionTreeNode<T>? node, BinaryWriter writer)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.IsLeaf);

        if (node.IsLeaf)
        {
            writer.Write(Convert.ToDouble(node.Prediction));
        }
        else
        {
            writer.Write(node.FeatureIndex);
            writer.Write(Convert.ToDouble(node.SplitValue));
            SerializeNode(node.Left, writer);
            SerializeNode(node.Right, writer);
        }
    }

    public static DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        if (!reader.ReadBoolean())
        {
            return null;
        }

        var node = new DecisionTreeNode<T>
        {
            IsLeaf = reader.ReadBoolean()
        };

        if (node.IsLeaf)
        {
            node.Prediction = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
        }
        else
        {
            node.FeatureIndex = reader.ReadInt32();
            node.SplitValue = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
            node.Left = DeserializeNode(reader);
            node.Right = DeserializeNode(reader);
        }

        return node;
    }

    public static void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);

        foreach (var value in vector)
        {
            writer.Write(Convert.ToDouble(value));
        }
    }

    public static Vector<T> DeserializeVector(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        T[] values = new T[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
        }

        return new Vector<T>(values);
    }
}