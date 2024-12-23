namespace AiDotNet.Helpers;

public static class TreeSerializationHelper
{
    public static void SerializeNode<T>(DecisionTreeNode<T>? node, BinaryWriter writer)
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

    public static DecisionTreeNode<T>? DeserializeNode<T>(BinaryReader reader)
    {
        if (!reader.ReadBoolean())
        {
            return null;
        }

        var node = new DecisionTreeNode<T>();
        node.IsLeaf = reader.ReadBoolean();

        if (node.IsLeaf)
        {
            node.Prediction = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
        }
        else
        {
            node.FeatureIndex = reader.ReadInt32();
            node.SplitValue = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
            node.Left = DeserializeNode<T>(reader);
            node.Right = DeserializeNode<T>(reader);
        }

        return node;
    }
}