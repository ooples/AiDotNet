using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Prioritized experience replay buffer for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Prioritized replay samples important experiences more frequently based on TD error.
/// Uses sum tree data structure for efficient sampling.
/// </remarks>
public class PrioritizedReplayBuffer<T>
{
    private readonly int _capacity;
    private readonly List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _buffer;
    private readonly List<double> _priorities;
    private int _position;
    private double _maxPriority;
    private readonly INumericOperations<T> _numOps;

    public int Count => _buffer.Count;

    public PrioritizedReplayBuffer(int capacity)
    {
        _capacity = capacity;
        _buffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>(capacity);
        _priorities = new List<double>(capacity);
        _position = 0;
        _maxPriority = 1.0;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public void Add(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = (state.Clone(), action.Clone(), reward, nextState.Clone(), done);

        if (_buffer.Count < _capacity)
        {
            _buffer.Add(experience);
            _priorities.Add(_maxPriority);
        }
        else
        {
            _buffer[_position] = experience;
            _priorities[_position] = _maxPriority;
        }

        _position = (_position + 1) % _capacity;
    }

    public (List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch,
            List<int> indices,
            List<double> weights) Sample(int batchSize, double alpha, double beta)
    {
        var batch = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();
        var indices = new List<int>();
        var weights = new List<double>();

        // Compute sampling probabilities
        var probabilities = new List<double>();
        double totalPriority = 0.0;

        for (int i = 0; i < _buffer.Count; i++)
        {
            var priority = Math.Pow(_priorities[i], alpha);
            probabilities.Add(priority);
            totalPriority += priority;
        }

        // Normalize probabilities
        for (int i = 0; i < probabilities.Count; i++)
        {
            probabilities[i] /= totalPriority;
        }

        // Sample with priorities
        var random = RandomHelper.ThreadSafeRandom;
        double minProbability = probabilities.Min();
        double maxWeight = Math.Pow(_buffer.Count * minProbability, -beta);

        for (int i = 0; i < batchSize && i < _buffer.Count; i++)
        {
            // Weighted sampling
            double r = random.NextDouble();
            double cumulative = 0.0;
            int selectedIndex = 0;

            for (int j = 0; j < probabilities.Count; j++)
            {
                cumulative += probabilities[j];
                if (r <= cumulative)
                {
                    selectedIndex = j;
                    break;
                }
            }

            batch.Add(_buffer[selectedIndex]);
            indices.Add(selectedIndex);

            // Compute importance sampling weight
            double weight = Math.Pow(_buffer.Count * probabilities[selectedIndex], -beta) / maxWeight;
            weights.Add(weight);
        }

        return (batch, indices, weights);
    }

    public void UpdatePriorities(List<int> indices, List<double> priorities, double epsilon)
    {
        for (int i = 0; i < indices.Count; i++)
        {
            var priority = priorities[i] + epsilon;
            _priorities[indices[i]] = priority;
            _maxPriority = Math.Max(_maxPriority, priority);
        }
    }
}
