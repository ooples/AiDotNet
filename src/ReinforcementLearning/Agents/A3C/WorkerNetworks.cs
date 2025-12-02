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
    public NeuralNetwork<T> PolicyNetwork { get; set; } = null!;
    public NeuralNetwork<T> ValueNetwork { get; set; } = null!;
    public List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> Trajectory { get; set; } = new();
}
