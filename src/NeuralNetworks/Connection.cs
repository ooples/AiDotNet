namespace AiDotNet.NeuralNetworks;

public class Connection<T>
{
    public int FromNode { get; }
    public int ToNode { get; }
    public T Weight { get; set; }
    public bool IsEnabled { get; set; }
    public int Innovation { get; }

    public Connection(int fromNode, int toNode, T weight, bool isEnabled, int innovation)
    {
        FromNode = fromNode;
        ToNode = toNode;
        Weight = weight;
        IsEnabled = isEnabled;
        Innovation = innovation;
    }
}