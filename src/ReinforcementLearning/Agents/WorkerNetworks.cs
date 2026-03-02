using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.ReinforcementLearning.Agents.A3C;

/// <summary>
/// Worker-local networks for A3C agent.
/// Each worker maintains its own copy of policy and value networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class WorkerNetworks<T>
{
    public INeuralNetwork<T> PolicyNetwork { get; }
    public INeuralNetwork<T> ValueNetwork { get; }
    public List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> Trajectory { get; set; } = new();

    public WorkerNetworks(INeuralNetwork<T> policyNetwork, INeuralNetwork<T> valueNetwork)
    {
        PolicyNetwork = policyNetwork ?? throw new ArgumentNullException(nameof(policyNetwork));
        ValueNetwork = valueNetwork ?? throw new ArgumentNullException(nameof(valueNetwork));
    }
}
