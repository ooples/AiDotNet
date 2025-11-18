using AiDotNet.Helpers;

namespace AiDotNet.Prototypes;

/// <summary>
/// Simple linear regression model for prototype validation.
/// Demonstrates GPU acceleration for traditional machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type for weights and calculations.</typeparam>
/// <remarks>
/// <para>
/// Linear regression model: y = X @ weights + bias
/// </para>
/// <para>
/// This implementation uses vectorized operations throughout, enabling GPU
/// acceleration when using float type with GPU engine enabled.
/// </para>
/// </remarks>
public class SimpleLinearRegression<T>
{
    private PrototypeVector<T>? _weights;
    private T? _bias;

    private readonly INumericOperations<T> _numOps;
    private readonly int _numFeatures;

    /// <summary>
    /// Gets whether the model has been trained.
    /// </summary>
    public bool IsTrained => _weights != null;

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures => _numFeatures;

    /// <summary>
    /// Initializes a new instance of the SimpleLinearRegression.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    public SimpleLinearRegression(int numFeatures)
    {
        if (numFeatures <= 0) throw new ArgumentException("Number of features must be positive", nameof(numFeatures));

        _numFeatures = numFeatures;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize with zeros
        _weights = PrototypeVector<T>.Zeros(numFeatures);
        _bias = _numOps.Zero;
    }

    /// <summary>
    /// Trains the model using gradient descent.
    /// </summary>
    /// <param name="X">Training data (flattened: numSamples * numFeatures).</param>
    /// <param name="y">Target values.</param>
    /// <param name="numSamples">Number of samples.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <param name="numEpochs">Number of training epochs.</param>
    /// <param name="verbose">Whether to print progress.</param>
    public void Train(
        PrototypeVector<T> X,
        PrototypeVector<T> y,
        int numSamples,
        double learningRate = 0.01,
        int numEpochs = 100,
        bool verbose = false)
    {
        if (X.Length != numSamples * _numFeatures)
        {
            throw new ArgumentException($"Expected X length {numSamples * _numFeatures}, got {X.Length}");
        }
        if (y.Length != numSamples)
        {
            throw new ArgumentException($"Expected y length {numSamples}, got {y.Length}");
        }

        var lr = _numOps.FromDouble(learningRate);
        var invNumSamples = _numOps.FromDouble(1.0 / numSamples);

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            // Forward pass: predictions = X @ weights + bias
            var predictions = PredictBatch(X, numSamples);

            // Compute loss: MSE = mean((predictions - y)^2)
            var loss = ComputeMSE(predictions, y);

            if (verbose && (epoch % 10 == 0 || epoch == numEpochs - 1))
            {
                Console.WriteLine($"Epoch {epoch + 1}/{numEpochs}, Loss: {_numOps.ToDouble(loss):F6}");
            }

            // Compute gradients
            // error = predictions - y
            var error = predictions.Subtract(y);

            // weight_gradient = (X^T @ error) / numSamples
            var weightGrad = MatrixVectorMultiplyTranspose(X, error, numSamples, _numFeatures);
            weightGrad = weightGrad.Multiply(invNumSamples);

            // bias_gradient = mean(error)
            var biasGrad = _numOps.Zero;
            for (int i = 0; i < error.Length; i++)
            {
                biasGrad = _numOps.Add(biasGrad, error[i]);
            }
            biasGrad = _numOps.Multiply(biasGrad, invNumSamples);

            // Update weights: weights -= lr * weight_gradient
            var weightUpdate = weightGrad.Multiply(lr);
            _weights = _weights!.Subtract(weightUpdate);

            // Update bias: bias -= lr * bias_gradient
            _bias = _numOps.Subtract(_bias!, _numOps.Multiply(biasGrad, lr));
        }
    }

    /// <summary>
    /// Makes a prediction for a single sample.
    /// </summary>
    /// <param name="features">Feature vector.</param>
    /// <returns>Predicted value.</returns>
    public T Predict(PrototypeVector<T> features)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before prediction");
        }
        if (features.Length != _numFeatures)
        {
            throw new ArgumentException($"Expected {_numFeatures} features, got {features.Length}");
        }

        // prediction = dot(weights, features) + bias
        var dotProduct = _numOps.Zero;
        for (int i = 0; i < _numFeatures; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(_weights![i], features[i]));
        }

        return _numOps.Add(dotProduct, _bias!);
    }

    /// <summary>
    /// Makes predictions for a batch of samples.
    /// </summary>
    /// <param name="X">Feature matrix (flattened: numSamples * numFeatures).</param>
    /// <param name="numSamples">Number of samples.</param>
    /// <returns>Vector of predictions.</returns>
    public PrototypeVector<T> PredictBatch(PrototypeVector<T> X, int numSamples)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before prediction");
        }
        if (X.Length != numSamples * _numFeatures)
        {
            throw new ArgumentException($"Expected X length {numSamples * _numFeatures}, got {X.Length}");
        }

        // predictions = X @ weights + bias
        var predictions = MatrixVectorMultiply(X, _weights!, numSamples, _numFeatures);

        // Add bias to all predictions
        var biasVec = new T[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            biasVec[i] = _bias!;
        }

        return predictions.Add(new PrototypeVector<T>(biasVec));
    }

    /// <summary>
    /// Computes Mean Squared Error.
    /// </summary>
    public T ComputeMSE(PrototypeVector<T> predictions, PrototypeVector<T> targets)
    {
        var diff = predictions.Subtract(targets);
        var squared = diff.Multiply(diff);

        var sum = _numOps.Zero;
        for (int i = 0; i < squared.Length; i++)
        {
            sum = _numOps.Add(sum, squared[i]);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(squared.Length));
    }

    /// <summary>
    /// Computes R² score (coefficient of determination).
    /// </summary>
    public T ComputeR2Score(PrototypeVector<T> predictions, PrototypeVector<T> targets)
    {
        // Compute mean of targets
        var targetSum = _numOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            targetSum = _numOps.Add(targetSum, targets[i]);
        }
        var targetMean = _numOps.Divide(targetSum, _numOps.FromDouble(targets.Length));

        // Compute SS_res (residual sum of squares)
        var ssRes = _numOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            var diff = _numOps.Subtract(targets[i], predictions[i]);
            ssRes = _numOps.Add(ssRes, _numOps.Multiply(diff, diff));
        }

        // Compute SS_tot (total sum of squares)
        var ssTot = _numOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            var diff = _numOps.Subtract(targets[i], targetMean);
            ssTot = _numOps.Add(ssTot, _numOps.Multiply(diff, diff));
        }

        // R² = 1 - (SS_res / SS_tot)
        var ratio = _numOps.Divide(ssRes, ssTot);
        return _numOps.Subtract(_numOps.One, ratio);
    }

    /// <summary>
    /// Gets the learned weights.
    /// </summary>
    public PrototypeVector<T>? GetWeights() => _weights;

    /// <summary>
    /// Gets the learned bias.
    /// </summary>
    public T? GetBias() => _bias;

    #region Helper Methods

    /// <summary>
    /// Matrix-vector multiplication: result = matrix @ vector
    /// Matrix is stored in row-major order (flattened).
    /// </summary>
    private PrototypeVector<T> MatrixVectorMultiply(PrototypeVector<T> matrix, PrototypeVector<T> vector, int rows, int cols)
    {
        var result = new T[rows];
        for (int i = 0; i < rows; i++)
        {
            var sum = _numOps.Zero;
            for (int j = 0; j < cols; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i * cols + j], vector[j]));
            }
            result[i] = sum;
        }
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Matrix-vector multiplication with transposed matrix: result = matrix^T @ vector
    /// </summary>
    private PrototypeVector<T> MatrixVectorMultiplyTranspose(PrototypeVector<T> matrix, PrototypeVector<T> vector, int rows, int cols)
    {
        var result = new T[cols];
        for (int i = 0; i < cols; i++)
        {
            var sum = _numOps.Zero;
            for (int j = 0; j < rows; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[j * cols + i], vector[j]));
            }
            result[i] = sum;
        }
        return new PrototypeVector<T>(result);
    }

    #endregion

    /// <summary>
    /// Returns a string representation of the model.
    /// </summary>
    public override string ToString()
    {
        if (!IsTrained)
        {
            return $"SimpleLinearRegression<{typeof(T).Name}>(features={_numFeatures}, untrained)";
        }

        var weightsStr = _weights!.Length <= 5
            ? string.Join(", ", _weights.ToArray().Select(w => $"{_numOps.ToDouble(w):F4}"))
            : $"[{_weights.Length} weights]";

        return $"SimpleLinearRegression<{typeof(T).Name}>(features={_numFeatures}, " +
               $"weights={weightsStr}, bias={_numOps.ToDouble(_bias!):F4})";
    }
}
