
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Associative Memory for nested learning.
/// Models both backpropagation (data point → local error) and
/// attention mechanisms (query → key-value) as associative memory.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class AssociativeMemory<T> : NestedLearningBase<T>, IAssociativeMemory<T>
{
    private readonly int _capacity;
    private readonly int _dimension;
    private readonly double _inverseTemperature;
    private readonly List<(Vector<T> Input, Vector<T> Target)> _memories;

    public AssociativeMemory(int dimension, int capacity = 1000, double inverseTemperature = 8.0)
    {
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be at least 1.");
        if (inverseTemperature <= 0 || double.IsNaN(inverseTemperature) || double.IsInfinity(inverseTemperature))
            throw new ArgumentOutOfRangeException(nameof(inverseTemperature), inverseTemperature, "Must be a finite positive value.");
        _dimension = dimension;
        _capacity = capacity;
        _inverseTemperature = inverseTemperature;
        _memories = new List<(Vector<T>, Vector<T>)>();
    }

    public void Associate(Vector<T> input, Vector<T> target)
    {
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        // Add to memory buffer
        _memories.Add((input.Clone(), target.Clone()));

        // Maintain capacity limit (FIFO)
        if (_memories.Count > _capacity)
        {
            _memories.RemoveAt(0);
        }
    }

    public Vector<T> Retrieve(Vector<T> query)
    {
        if (query.Length != _dimension)
            throw new ArgumentException($"Query length {query.Length} must match memory dimension {_dimension}.", nameof(query));

        // Modern continuous Hopfield retrieval per Ramsauer et al. 2021:
        // Scores keys (Input) against query, returns weighted sum of values (Target).
        // new_state = softmax(β * keys^T @ query) @ values
        // Falls back to association matrix when no memories are stored.
        if (_memories.Count > 0)
        {
            var scores = new double[_memories.Count];
            double maxScore = double.NegativeInfinity;

            // Compute similarity scores: β * <key_i, query>
            for (int m = 0; m < _memories.Count; m++)
            {
                double dot = 0;
                var key = _memories[m].Input;
                for (int d = 0; d < _dimension; d++)
                    dot += NumOps.ToDouble(NumOps.Multiply(key[d], query[d]));
                scores[m] = _inverseTemperature * dot;
                if (scores[m] > maxScore) maxScore = scores[m];
            }

            // Softmax (numerically stable)
            double sumExp = 0;
            for (int m = 0; m < _memories.Count; m++)
            {
                scores[m] = Math.Exp(scores[m] - maxScore);
                sumExp += scores[m];
            }
            for (int m = 0; m < _memories.Count; m++)
                scores[m] /= (sumExp + 1e-10);

            // Weighted sum of stored values (Target)
            var result = new Vector<T>(_dimension);
            for (int m = 0; m < _memories.Count; m++)
            {
                T weight = NumOps.FromDouble(scores[m]);
                var value = _memories[m].Target;
                for (int d = 0; d < _dimension; d++)
                    result[d] = NumOps.Add(result[d], NumOps.Multiply(weight, value[d]));
            }
            return result;
        }

        // No memories stored — return zero vector
        return new Vector<T>(_dimension);
    }

    public void Update(Vector<T> input, Vector<T> target, T learningRate)
    {
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        // Scale target by learning rate before storing — allows gradual memory updates.
        // learningRate=1.0 stores the full target; smaller values blend toward zero.
        var targetTensor = Tensor<T>.FromVector(target);
        var scaledTensor = Engine.TensorMultiplyScalar(targetTensor, learningRate);

        Associate(input, scaledTensor.ToVector());
    }

    public int Capacity => _capacity;

    public void Clear()
    {
        _memories.Clear();
    }

    /// <summary>
    /// Computes the Hebbian association matrix from stored memories: W = Σ target_i * input_i^T.
    /// This is a read-only view for diagnostics/testing — Retrieve uses softmax attention.
    /// </summary>
    public Matrix<T> GetAssociationMatrix()
    {
        var matrix = new Matrix<T>(_dimension, _dimension);
        foreach (var (input, target) in _memories)
        {
            for (int i = 0; i < _dimension; i++)
                for (int j = 0; j < _dimension; j++)
                    matrix[i, j] = NumOps.Add(matrix[i, j], NumOps.Multiply(target[i], input[j]));
        }
        return matrix;
    }

    /// <summary>
    /// Gets the number of stored memories.
    /// </summary>
    public int MemoryCount => _memories.Count;
}
