
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Associative Memory for nested learning.
/// Models both backpropagation (data point → local error) and
/// attention mechanisms (query → key-value) as associative memory.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class AssociativeMemory<T> : IAssociativeMemory<T>
{
    private readonly int _capacity;
    private readonly int _dimension;
    private readonly double _inverseTemperature;
    private readonly List<(Vector<T> Input, Vector<T> Target)> _memories;
    private Matrix<T> _associationMatrix;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public AssociativeMemory(int dimension, int capacity = 1000, double inverseTemperature = 8.0)
    {
        _dimension = dimension;
        _capacity = capacity;
        _inverseTemperature = inverseTemperature;
        _memories = new List<(Vector<T>, Vector<T>)>();
        _associationMatrix = new Matrix<T>(dimension, dimension);
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

        // Update association matrix using Hebbian-like learning
        UpdateAssociationMatrix(input, target, _numOps.FromDouble(0.01));
    }

    public Vector<T> Retrieve(Vector<T> query)
    {
        if (query.Length != _dimension)
            throw new ArgumentException("Query must match memory dimension");

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
                    dot += _numOps.ToDouble(_numOps.Multiply(key[d], query[d]));
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
                T weight = _numOps.FromDouble(scores[m]);
                var value = _memories[m].Target;
                for (int d = 0; d < _dimension; d++)
                    result[d] = _numOps.Add(result[d], _numOps.Multiply(weight, value[d]));
            }
            return result;
        }

        // Fallback: association matrix retrieval when no memory buffer entries
        var retrieved = _associationMatrix.Multiply(query);

        // Check for exact or near matches in memory buffer
        T bestSimilarity = _numOps.FromDouble(double.NegativeInfinity);
        Vector<T>? bestMatch = null;

        foreach (var (input, target) in _memories)
        {
            T similarity = ComputeSimilarity(query, input);
            if (_numOps.GreaterThan(similarity, bestSimilarity))
            {
                bestSimilarity = similarity;
                bestMatch = target;
            }
        }

        // Blend matrix-based retrieval with buffer-based retrieval
        if (bestMatch != null && _numOps.GreaterThan(bestSimilarity, _numOps.FromDouble(0.8)))
        {
            T blendFactor = _numOps.FromDouble(0.3);
            var blended = new Vector<T>(_dimension);

            for (int i = 0; i < _dimension; i++)
            {
                T matrixPart = _numOps.Multiply(retrieved[i],
                    _numOps.Subtract(_numOps.One, blendFactor));
                T bufferPart = _numOps.Multiply(bestMatch[i], blendFactor);
                blended[i] = _numOps.Add(matrixPart, bufferPart);
            }

            return blended;
        }

        return retrieved;
    }

    public void Update(Vector<T> input, Vector<T> target, T learningRate)
    {
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        UpdateAssociationMatrix(input, target, learningRate);
    }

    private void UpdateAssociationMatrix(Vector<T> input, Vector<T> target, T learningRate)
    {
        // Hebbian learning rule: Δw_ij = η * target_i * input_j
        // This models how backpropagation maps data points to local errors
        for (int i = 0; i < _dimension; i++)
        {
            for (int j = 0; j < _dimension; j++)
            {
                T update = _numOps.Multiply(_numOps.Multiply(target[i], input[j]), learningRate);
                _associationMatrix[i, j] = _numOps.Add(_associationMatrix[i, j], update);
            }
        }
    }

    private T ComputeSimilarity(Vector<T> a, Vector<T> b)
    {
        return _numOps.FromDouble(VectorHelper.CosineSimilarity(a, b));
    }

    public int Capacity => _capacity;

    public void Clear()
    {
        _memories.Clear();
        _associationMatrix = new Matrix<T>(_dimension, _dimension);
    }

    /// <summary>
    /// Gets the association matrix for inspection/debugging.
    /// </summary>
    public Matrix<T> GetAssociationMatrix() => _associationMatrix;

    /// <summary>
    /// Gets the number of stored memories.
    /// </summary>
    public int MemoryCount => _memories.Count;
}
