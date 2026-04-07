
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Associative Memory for nested learning using modern continuous
/// Hopfield retrieval (Ramsauer et al. 2021). Stores key-value pairs and retrieves
/// via softmax attention: new_state = softmax(β * keys^T @ query) @ values.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class AssociativeMemory<T> : NestedLearningBase<T>, IAssociativeMemory<T>
{
    private readonly int _capacity;
    private readonly int _dimension;
    private readonly double _inverseTemperature;
    private readonly List<(Vector<T> Input, Vector<T> Target)> _memories;
    private Matrix<T>? _cachedAssociationMatrix;

    /// <summary>Cosine similarity threshold for treating two keys as duplicates in Update.</summary>
    private const double DuplicateKeyThreshold = 0.99;

    /// <summary>Small epsilon to prevent division by zero in cosine similarity.</summary>
    private const double CosineEpsilon = 1e-10;
    private Tensor<T>? _cachedValuesTensor;

    public AssociativeMemory(int dimension, int capacity = 1000, double inverseTemperature = 8.0)
    {
        if (dimension < 1)
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "Dimension must be at least 1.");
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
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        _memories.Add((input.Clone(), target.Clone()));
        _cachedAssociationMatrix = null;
        _cachedValuesTensor = null;

        if (_memories.Count > _capacity)
        {
            _memories.RemoveAt(0);
        }
    }

    public Vector<T> Retrieve(Vector<T> query)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (query.Length != _dimension)
            throw new ArgumentException($"Query length {query.Length} must match memory dimension {_dimension}.", nameof(query));

        if (_memories.Count == 0)
            return new Vector<T>(_dimension);

        // Modern continuous Hopfield retrieval per Ramsauer et al. 2021:
        // new_state = softmax(β * keys^T @ query) @ values
        //
        // Softmax is computed in double for numerical stability — the exp/log operations
        // in softmax benefit from double precision regardless of T. This matches PyTorch's
        // F.softmax which upcasts to float32 even for float16 inputs.
        var scores = new double[_memories.Count];
        double maxScore = double.NegativeInfinity;

        for (int m = 0; m < _memories.Count; m++)
        {
            T dot = Engine.DotProduct(_memories[m].Input, query);
            scores[m] = _inverseTemperature * NumOps.ToDouble(dot);
            if (scores[m] > maxScore) maxScore = scores[m];
        }

        double sumExp = 0;
        for (int m = 0; m < _memories.Count; m++)
        {
            scores[m] = Math.Exp(scores[m] - maxScore);
            sumExp += scores[m];
        }
        for (int m = 0; m < _memories.Count; m++)
            scores[m] /= (sumExp + 1e-10);

        // Use cached values tensor to avoid per-call O(M×D) allocation
        int M = _memories.Count;
        if (_cachedValuesTensor == null || _cachedValuesTensor.Shape[0] != M)
        {
            _cachedValuesTensor = new Tensor<T>([M, _dimension]);
            for (int m = 0; m < M; m++)
            {
                var target = _memories[m].Target;
                for (int d = 0; d < _dimension; d++)
                    _cachedValuesTensor[m, d] = target[d];
            }
        }

        var weights = new Tensor<T>([1, M]);
        for (int m = 0; m < M; m++)
            weights[0, m] = NumOps.FromDouble(scores[m]);

        var resultTensor = Engine.TensorMatMul(weights, _cachedValuesTensor); // [1, D]
        return resultTensor.Reshape([_dimension]).ToVector();
    }

    public void Update(Vector<T> input, Vector<T> target, T learningRate)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        double lr = NumOps.ToDouble(learningRate);
        if (lr < 0 || lr > 1 || double.IsNaN(lr) || double.IsInfinity(lr))
            throw new ArgumentOutOfRangeException(nameof(learningRate), lr, "Learning rate must be in [0, 1].");

        // If a matching key already exists (cosine similarity > 0.99), blend its target
        // instead of appending a duplicate that splits softmax attention
        T inputNormSq = Engine.DotProduct(input, input);
        double inputNorm = Math.Sqrt(NumOps.ToDouble(inputNormSq));

        for (int m = 0; m < _memories.Count; m++)
        {
            var key = _memories[m].Input;
            T dot = Engine.DotProduct(key, input);
            T keyNormSq = Engine.DotProduct(key, key);
            double cosine = NumOps.ToDouble(dot) / (Math.Sqrt(NumOps.ToDouble(keyNormSq)) * inputNorm + CosineEpsilon);

            if (cosine > DuplicateKeyThreshold)
            {
                // Blend: updated = (1-lr) * existing + lr * target
                var existingTensor = Tensor<T>.FromVector(_memories[m].Target);
                var targetTensor = Tensor<T>.FromVector(target);
                T oneMinusLr = NumOps.Subtract(NumOps.One, learningRate);
                var kept = Engine.TensorMultiplyScalar(existingTensor, oneMinusLr);
                var added = Engine.TensorMultiplyScalar(targetTensor, learningRate);
                var blended = Engine.TensorAdd(kept, added);
                _memories[m] = (key, blended.ToVector());
                _cachedAssociationMatrix = null;
                _cachedValuesTensor = null;
                return;
            }
        }

        // No matching key — store full target (not scaled, since retrieval expects
        // actual values, and learningRate only modulates blend strength for existing keys)
        Associate(input, target);
    }

    public int Capacity => _capacity;

    public void Clear()
    {
        _memories.Clear();
        _cachedAssociationMatrix = null;
        _cachedValuesTensor = null;
    }

    /// <summary>
    /// Computes the Hebbian association matrix from stored memories: W = Σ target_i * input_i^T.
    /// Cached and invalidated on Associate/Update/Clear. For diagnostics/testing only.
    /// </summary>
    public Matrix<T> GetAssociationMatrix()
    {
        if (_cachedAssociationMatrix == null)
        {
            // W = Σ target_i ⊗ input_i  (outer product sum via Engine matmul)
            var result = new Tensor<T>([_dimension, _dimension]);
            foreach (var (input, target) in _memories)
            {
                var tCol = Tensor<T>.FromVector(target).Reshape([_dimension, 1]);
                var iRow = Tensor<T>.FromVector(input).Reshape([1, _dimension]);
                var outer = Engine.TensorMatMul(tCol, iRow);
                result = Engine.TensorAdd(result, outer);
            }

            var matrix = new Matrix<T>(_dimension, _dimension);
            for (int i = 0; i < _dimension; i++)
                for (int j = 0; j < _dimension; j++)
                    matrix[i, j] = result[i, j];
            _cachedAssociationMatrix = matrix;
        }

        // Return a copy so callers can't mutate the cache
        return _cachedAssociationMatrix.Clone();
    }

    /// <summary>
    /// Gets the number of stored memories.
    /// </summary>
    public int MemoryCount => _memories.Count;
}
