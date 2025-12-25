using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Regression;

/// <summary>
/// A neural network regression model that can learn complex non-linear relationships in data.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class implements a fully connected feedforward neural network for regression tasks.
/// It supports multiple hidden layers with customizable activation functions and uses
/// gradient-based optimization to learn from data.
/// </para>
/// <para>
/// The neural network architecture is defined by specifying the number of neurons in each layer,
/// with the first layer corresponding to the input features and the last layer to the output.
/// </para>
/// <para>
/// For Beginners:
/// A neural network is a machine learning model inspired by the human brain. It consists of layers
/// of interconnected "neurons" that process input data to make predictions. Each connection has a
/// "weight" that determines its importance, and these weights are adjusted during training to improve
/// the model's accuracy. This process is similar to how we learn from experience.
/// </para>
/// </remarks>
public class NeuralNetworkRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the neural network regression model.
    /// </summary>
    /// <value>
    /// Contains settings like layer sizes, learning rate, batch size, and activation functions.
    /// </value>
    private readonly NeuralNetworkRegressionOptions<T, Matrix<T>, Vector<T>> _options;

    /// <summary>
    /// The weight matrices for each layer of the neural network.
    /// </summary>
    /// <value>
    /// A list of matrices where each matrix represents the connections between two consecutive layers.
    /// </value>
    private readonly List<Matrix<T>> _weights;

    /// <summary>
    /// The bias vectors for each layer of the neural network.
    /// </summary>
    /// <value>
    /// A list of vectors where each vector represents the biases for a layer.
    /// </value>
    private readonly List<Vector<T>> _biases;

    /// <summary>
    /// The optimization algorithm used to update the model parameters during training.
    /// </summary>
    /// <value>
    /// An implementation of the IOptimizer interface that determines how weights and biases are updated.
    /// </value>
    private readonly IOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    /// <summary>
    /// Initializes a new instance of the NeuralNetworkRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the neural network. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the neural network with random weights and biases according to the Xavier/Glorot initialization method,
    /// which helps with training stability.
    /// </para>
    /// <para>
    /// If no options are provided, a default configuration will be used. If no optimizer is specified in the options,
    /// the Adam optimizer will be used by default.
    /// </para>
    /// <para>
    /// For Beginners:
    /// When you create a neural network, it starts with random values for its internal parameters (weights and biases).
    /// The Xavier/Glorot initialization is a smart way to set these initial random values to help the network learn more effectively.
    /// Think of it like setting up a balanced starting point for the learning process.
    /// </para>
    /// </remarks>
    public NeuralNetworkRegression(NeuralNetworkRegressionOptions<T, Matrix<T>, Vector<T>>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new NeuralNetworkRegressionOptions<T, Matrix<T>, Vector<T>>();
        _optimizer = _options.Optimizer ?? new AdamOptimizer<T, Matrix<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _weights = new List<Matrix<T>>();
        _biases = new List<Vector<T>>();

        InitializeNetwork();
    }

    /// <summary>
    /// Initializes the neural network by creating weight matrices and bias vectors for each layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates the weight matrices and bias vectors based on the layer sizes specified in the options.
    /// </para>
    /// <para>
    /// The weights are initialized using the Xavier/Glorot initialization method, which scales the random values
    /// based on the number of input and output connections to help with training convergence.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method sets up the structure of the neural network by creating the connections (weights) between neurons
    /// and the bias values for each neuron. The Xavier/Glorot initialization is a technique that helps determine
    /// good starting values for these weights, making it easier for the network to learn effectively.
    /// </para>
    /// </remarks>
    private void InitializeNetwork()
    {
        for (int i = 0; i < _options.LayerSizes.Count - 1; i++)
        {
            int inputSize = _options.LayerSizes[i];
            int outputSize = _options.LayerSizes[i + 1];

            Matrix<T> weight = Matrix<T>.CreateRandom(outputSize, inputSize);
            Vector<T> bias = Vector<T>.CreateRandom(outputSize);

            // Xavier/Glorot initialization
            T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inputSize + outputSize)));
            weight = weight.Transform((w, row, col) => NumOps.Multiply(w, scale));

            _weights.Add(weight);
            _biases.Add(bias);
        }
    }

    /// <summary>
    /// Trains the neural network on the provided data.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method trains the neural network using mini-batch gradient descent. The data is divided into batches,
    /// and for each batch, the model parameters are updated based on the computed gradients.
    /// </para>
    /// <para>
    /// The training process runs for the number of epochs specified in the options, and the data is shuffled
    /// before each epoch to improve training stability.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training a neural network is like teaching it to make accurate predictions by showing it many examples.
    /// An "epoch" is one complete pass through all the training data. The data is divided into smaller "batches"
    /// to make the training more efficient. Before each epoch, the data is shuffled (like shuffling flashcards)
    /// to help the network learn better patterns rather than memorizing the order of examples.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> X, Vector<T> y)
    {
        int batchSize = _options.BatchSize;
        int numBatches = (X.Rows + batchSize - 1) / batchSize;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Shuffle the data
            int[] indices = Enumerable.Range(0, X.Rows).ToArray();
            ShuffleArray(indices);

            T totalLoss = NumOps.Zero;

            for (int batch = 0; batch < numBatches; batch++)
            {
                int startIdx = batch * batchSize;
                int endIdx = Math.Min(startIdx + batchSize, X.Rows);

                Matrix<T> batchX = GetBatchRows(X, indices, startIdx, endIdx);
                Vector<T> batchY = GetBatchElements(y, indices, startIdx, endIdx);

                T batchLoss = TrainOnBatch(batchX, batchY);
                totalLoss = NumOps.Add(totalLoss, batchLoss);
            }

            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss}");
            }
        }
    }

    /// <summary>
    /// Randomly shuffles an array using the Fisher-Yates algorithm.
    /// </summary>
    /// <param name="array">The array to shuffle.</param>
    /// <remarks>
    /// <para>
    /// This method is used to randomize the order of training examples before each epoch.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method randomly rearranges the order of items in an array, similar to shuffling a deck of cards.
    /// In machine learning, shuffling the training data helps prevent the model from learning patterns
    /// based on the order of examples, which could lead to poor generalization.
    /// </para>
    /// </remarks>
    private void ShuffleArray(int[] array)
    {
        var random = RandomHelper.CreateSecureRandom();
        int n = array.Length;
        for (int i = n - 1; i > 0; i--)
        {
            int j = random.Next(0, i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    /// <summary>
    /// Extracts a batch of rows from a matrix based on the provided indices.
    /// </summary>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="indices">Array of indices indicating which rows to extract.</param>
    /// <param name="startIdx">The starting index in the indices array.</param>
    /// <param name="endIdx">The ending index (exclusive) in the indices array.</param>
    /// <returns>A new matrix containing the selected rows.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// This method creates a smaller subset of the training data by selecting specific rows from the original data.
    /// It's like picking certain flashcards from a stack to study in one session, based on a list of card numbers.
    /// </para>
    /// </remarks>
    private Matrix<T> GetBatchRows(Matrix<T> matrix, int[] indices, int startIdx, int endIdx)
    {
        int batchSize = endIdx - startIdx;
        Matrix<T> result = new Matrix<T>(batchSize, matrix.Columns);
        for (int i = 0; i < batchSize; i++)
        {
            result.SetRow(i, matrix.GetRow(indices[startIdx + i]));
        }

        return result;
    }

    /// <summary>
    /// Extracts a batch of elements from a vector based on the provided indices.
    /// </summary>
    /// <param name="vector">The source vector.</param>
    /// <param name="indices">Array of indices indicating which elements to extract.</param>
    /// <param name="startIdx">The starting index in the indices array.</param>
    /// <param name="endIdx">The ending index (exclusive) in the indices array.</param>
    /// <returns>A new vector containing the selected elements.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Similar to GetBatchRows, this method creates a smaller subset of the target values that correspond
    /// to the selected training examples. It ensures that each input example is paired with its correct target value.
    /// </para>
    /// </remarks>
    private Vector<T> GetBatchElements(Vector<T> vector, int[] indices, int startIdx, int endIdx)
    {
        int batchSize = endIdx - startIdx;
        Vector<T> result = new(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = vector[indices[startIdx + i]];
        }

        return result;
    }

    /// <summary>
    /// Trains the neural network on a single batch of data.
    /// </summary>
    /// <param name="X">The batch input features matrix.</param>
    /// <param name="y">The batch target values vector.</param>
    /// <returns>The average loss for the batch.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the following steps for each example in the batch:
    /// 1. Forward pass to compute predictions
    /// 2. Compute the loss
    /// 3. Backward pass to compute gradients
    /// 4. Accumulate gradients across all examples
    /// </para>
    /// <para>
    /// After processing all examples in the batch, the accumulated gradients are used to update
    /// the model parameters (weights and biases).
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method is the core of the learning process. For each example in the batch:
    /// 1. The model makes a prediction (forward pass)
    /// 2. The error between the prediction and the actual target is calculated (loss)
    /// 3. The model figures out how to adjust its internal values to reduce this error (backward pass)
    /// 4. These adjustments are collected for all examples in the batch
    /// 
    /// Finally, all the collected adjustments are applied to update the model's internal values,
    /// making it slightly better at its task.
    /// </para>
    /// </remarks>
    private T TrainOnBatch(Matrix<T> X, Vector<T> y)
    {
        List<Matrix<T>> weightGradients = new List<Matrix<T>>();
        List<Vector<T>> biasGradients = new List<Vector<T>>();

        T batchLoss = NumOps.Zero;

        for (int i = 0; i < X.Rows; i++)
        {
            Vector<T> input = X.GetRow(i);
            Vector<T> target = new([y[i]]);

            // Forward pass
            List<Vector<T>> activations = ForwardPass(input);

            // Compute loss
            T loss = _options.LossFunction.CalculateLoss(activations[activations.Count - 1], target);
            batchLoss = NumOps.Add(batchLoss, loss);

            // Backward pass
            List<Vector<T>> deltas = BackwardPass(activations, target);

            // Accumulate gradients
            AccumulateGradients(activations, deltas, weightGradients, biasGradients);
        }

        // Update parameters
        UpdateParameters(weightGradients, biasGradients, X.Rows);

        return NumOps.Divide(batchLoss, NumOps.FromDouble(X.Rows));
    }

    /// <summary>
    /// Performs a forward pass through the neural network.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>A list of activation vectors for each layer, including the input layer.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes the output of each layer in the network by applying the following steps:
    /// 1. Multiply the input by the weight matrix
    /// 2. Add the bias vector
    /// 3. Apply the activation function
    /// </para>
    /// <para>
    /// The result is a list of activation vectors, where the first element is the input and the last element
    /// is the network's output.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The forward pass is how the neural network makes predictions. Information flows from the input layer
    /// through the hidden layers to the output layer. At each layer, the network performs a weighted sum of inputs
    /// (like a weighted average), adds a bias value (like a baseline), and then applies an activation function
    /// (which introduces non-linearity, allowing the network to learn complex patterns).
    /// </para>
    /// </remarks>
    private List<Vector<T>> ForwardPass(Vector<T> input)
    {
        List<Vector<T>> activations = new List<Vector<T>> { input };

        for (int i = 0; i < _weights.Count; i++)
        {
            Vector<T> z = _weights[i].Multiply(activations[i]).Add(_biases[i]);
            Vector<T> a = ApplyActivation(z, i == _weights.Count - 1);
            activations.Add(a);
        }

        return activations;
    }

    /// <summary>
    /// Performs a backward pass through the neural network to compute the gradients.
    /// </summary>
    /// <param name="activations">The list of activation vectors from the forward pass.</param>
    /// <param name="target">The target output vector.</param>
    /// <returns>A list of delta vectors for each layer (excluding the input layer).</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes the error gradients (deltas) for each layer in the network, starting from
    /// the output layer and moving backward to the first hidden layer.
    /// </para>
    /// <para>
    /// For the output layer, the delta is computed using the derivative of the loss function with respect to
    /// the output, multiplied by the derivative of the activation function.
    /// </para>
    /// <para>
    /// For hidden layers, the delta is computed by propagating the error from the next layer, multiplied by
    /// the derivative of the activation function.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The backward pass is how the neural network learns from its mistakes. After making a prediction,
    /// the network calculates how much error it made and then works backward through the layers to determine
    /// how each connection contributed to that error. This is like tracing back through a series of decisions
    /// to figure out which ones led to a mistake. The network uses this information to adjust its internal values
    /// (weights and biases) to make better predictions in the future.
    /// </para>
    /// </remarks>
    private List<Vector<T>> BackwardPass(List<Vector<T>> activations, Vector<T> target)
    {
        var deltas = new List<Vector<T>>();

        // Output layer error
        int lastIndex = activations.Count - 1;
        Vector<T> error = _options.LossFunction.CalculateDerivative(activations[lastIndex], target);
        Vector<T> delta = error.PointwiseMultiply(ApplyActivationDerivative(activations[lastIndex], true));
        deltas.Add(delta);

        // Hidden layers
        for (int i = _weights.Count - 1; i > 0; i--)
        {
            delta = _weights[i].Transpose().Multiply(delta).PointwiseMultiply(ApplyActivationDerivative(activations[i], false));
            deltas.Insert(0, delta);
        }

        return deltas;
    }

    /// <summary>
    /// Accumulates gradients for the weights and biases based on the activations and deltas.
    /// </summary>
    /// <param name="activations">The list of activation vectors from the forward pass.</param>
    /// <param name="deltas">The list of delta vectors from the backward pass.</param>
    /// <param name="weightGradients">The list to store accumulated weight gradients.</param>
    /// <param name="biasGradients">The list to store accumulated bias gradients.</param>
    /// <remarks>
    /// <para>
    /// This method computes the gradients for each weight matrix and bias vector in the network and
    /// accumulates them in the provided lists.
    /// </para>
    /// <para>
    /// For each layer, the weight gradient is computed as the outer product of the delta vector and
    /// the activation vector from the previous layer. The bias gradient is simply the delta vector.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method calculates how much each weight and bias in the network should be adjusted to reduce
    /// the prediction error. It collects these adjustments (gradients) for all connections in the network.
    /// Think of it as figuring out which knobs to turn and by how much to improve the network's performance.
    /// </para>
    /// </remarks>
    private void AccumulateGradients(List<Vector<T>> activations, List<Vector<T>> deltas,
                                     List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients)
    {
        for (int i = 0; i < _weights.Count; i++)
        {
            Matrix<T> weightGradient = deltas[i].OuterProduct(activations[i]);
            Vector<T> biasGradient = deltas[i];

            if (weightGradients.Count <= i)
            {
                weightGradients.Add(weightGradient);
                biasGradients.Add(biasGradient);
            }
            else
            {
                weightGradients[i] = weightGradients[i].Add(weightGradient);
                biasGradients[i] = biasGradients[i].Add(biasGradient);
            }
        }
    }

    /// <summary>
    /// Updates the network parameters (weights and biases) using the accumulated gradients.
    /// </summary>
    /// <param name="weightGradients">The list of accumulated weight gradients.</param>
    /// <param name="biasGradients">The list of accumulated bias gradients.</param>
    /// <param name="batchSize">The number of examples in the batch.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the network using the average gradients computed
    /// from the batch. The update is performed using the optimizer specified in the options.
    /// </para>
    /// <para>
    /// If a gradient-based optimizer is used, it will handle the update according to its specific algorithm.
    /// Otherwise, a simple gradient descent update is applied using the learning rate from the options.
    /// </para>
    /// <para>
    /// After updating the parameters, regularization is applied to help prevent overfitting.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method applies the adjustments calculated during the backward pass to update the network's
    /// internal values (weights and biases). It first averages the adjustments across all examples in the batch,
    /// then uses an optimization algorithm to apply these adjustments in a way that helps the network learn
    /// efficiently. Finally, it applies regularization, which is like adding a penalty for complexity to prevent
    /// the network from becoming too specialized to the training data (overfitting).
    /// </para>
    /// </remarks>
    private void UpdateParameters(List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients, int batchSize)
    {
        T scaleFactor = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < _weights.Count; i++)
        {
            Matrix<T> avgWeightGradient = weightGradients[i].Transform((g, _, _) => NumOps.Multiply(g, scaleFactor));
            Vector<T> avgBiasGradient = biasGradients[i].Transform(g => NumOps.Multiply(g, scaleFactor));

            if (_optimizer is IGradientBasedOptimizer<T, Matrix<T>, Vector<T>> gradientOptimizer)
            {
                _weights[i] = gradientOptimizer.UpdateParameters(_weights[i], avgWeightGradient);
                _biases[i] = gradientOptimizer.UpdateParameters(_biases[i], avgBiasGradient);
            }
            else
            {
                // For non-gradient-based optimizers, we'll use a simple update rule
                _weights[i] = _weights[i].Subtract(avgWeightGradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
                _biases[i] = _biases[i].Subtract(avgBiasGradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
            }

            // Apply regularization
            _weights[i] = Regularization.Regularize(_weights[i]);
            _biases[i] = Regularization.Regularize(_biases[i]);
        }
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="X">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network for each input example and returns
    /// the predicted values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data. It runs each example through
    /// the trained network and returns the predicted values. This is like using what the network has learned
    /// to make educated guesses about new situations it hasn't seen before.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        Vector<T> predictions = new(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            Vector<T> input = X.GetRow(i);
            List<Vector<T>> activations = ForwardPass(input);
            predictions[i] = activations[activations.Count - 1][0];
        }

        return predictions;
    }

    /// <summary>
    /// Applies the appropriate activation function to the input vector.
    /// </summary>
    /// <param name="input">The input vector to apply the activation function to.</param>
    /// <param name="isOutputLayer">Whether the activation is for the output layer.</param>
    /// <returns>A new vector with the activation function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This method applies either the output activation function or the hidden activation function
    /// to each element of the input vector, depending on the isOutputLayer parameter.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.
    /// Different activation functions are often used for hidden layers versus the output layer, depending on
    /// the task. For example, in regression problems, the output layer might use a linear activation function,
    /// while hidden layers might use functions like ReLU (Rectified Linear Unit) or sigmoid.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyActivation(Vector<T> input, bool isOutputLayer)
    {
        // First check if vector activation is available
        var vectorActivation = isOutputLayer
            ? _options.OutputVectorActivation
            : _options.HiddenVectorActivation;

        if (vectorActivation != null)
        {
            return vectorActivation.Activate(input);
        }

        // Fall back to scalar activation
        var scalarActivation = isOutputLayer
            ? _options.OutputActivationFunction
            : _options.HiddenActivationFunction;

        if (scalarActivation != null)
        {
            return input.Transform(x => scalarActivation.Activate(x));
        }

        // If no activation function is specified, return the input unchanged (identity function)
        return input;
    }

    /// <summary>
    /// Applies the derivative of the appropriate activation function to the input vector.
    /// </summary>
    /// <param name="input">The input vector to apply the activation function derivative to.</param>
    /// <param name="isOutputLayer">Whether the activation is for the output layer.</param>
    /// <returns>A new vector with the activation function derivative applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This method applies either the derivative of the output activation function or the derivative of the
    /// hidden activation function to each element of the input vector, depending on the isOutputLayer parameter.
    /// </para>
    /// <para>
    /// For Beginners:
    /// During the backward pass (learning process), the network needs to know how sensitive the output is to
    /// small changes in each neuron's value. This is calculated using the derivative of the activation function.
    /// The derivative tells us the rate of change at a specific point, which helps determine how much to adjust
    /// each weight and bias.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyActivationDerivative(Vector<T> input, bool isOutputLayer)
    {
        // First check if vector activation is available
        var vectorActivation = isOutputLayer
            ? _options.OutputVectorActivation
            : _options.HiddenVectorActivation;

        if (vectorActivation != null)
        {
            return vectorActivation.Derivative(input).ToVector();
        }

        // Fall back to scalar activation
        var scalarActivation = isOutputLayer
            ? _options.OutputActivationFunction
            : _options.HiddenActivationFunction;

        if (scalarActivation != null)
        {
            return input.Transform(x => scalarActivation.Derivative(x));
        }

        // If no activation function is specified, return ones (derivative of identity function)
        return input.Transform(_ => NumOps.One);
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for neural network regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates this is a neural network regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.NeuralNetworkRegression;

    /// <summary>
    /// Optimizes the model parameters using the provided training data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method is called by the base class during the fitting process to optimize the model parameters.
    /// It simply calls the Train method to perform the optimization.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This is an internal method that's part of the model fitting process. It delegates the actual training
    /// to the Train method, which handles the details of updating the model parameters to fit the data.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        Train(x, y);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the neural network specific data,
    /// including the layer sizes, weights, and biases.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize NeuralNetworkRegression specific data
        writer.Write(_options.LayerSizes.Count);
        foreach (var size in _options.LayerSizes)
        {
            writer.Write(size);
        }

        writer.Write(_weights.Count);
        foreach (var weight in _weights)
        {
            writer.Write(weight.Rows);
            writer.Write(weight.Columns);
            foreach (var value in weight.Flatten())
            {
                writer.Write(Convert.ToDouble(value));
            }
        }

        writer.Write(_biases.Count);
        foreach (var bias in _biases)
        {
            writer.Write(bias.Length);
            foreach (var value in bias)
            {
                writer.Write(Convert.ToDouble(value));
            }
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the neural network specific data,
    /// reconstructing the layer sizes, weights, and biases.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize NeuralNetworkRegression specific data
        int layerCount = reader.ReadInt32();
        _options.LayerSizes = new List<int>();
        for (int i = 0; i < layerCount; i++)
        {
            _options.LayerSizes.Add(reader.ReadInt32());
        }

        int weightCount = reader.ReadInt32();
        _weights.Clear();
        for (int i = 0; i < weightCount; i++)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var weightData = new T[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    weightData[r, c] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            _weights.Add(new Matrix<T>(weightData));
        }

        int biasCount = reader.ReadInt32();
        _biases.Clear();
        for (int i = 0; i < biasCount; i++)
        {
            int length = reader.ReadInt32();
            var biasData = new T[length];
            for (int j = 0; j < length; j++)
            {
                biasData[j] = NumOps.FromDouble(reader.ReadDouble());
            }
            _biases.Add(new Vector<T>(biasData));
        }

        InitializeNetwork();
    }

    /// <summary>
    /// Creates a new instance of the Neural Network Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Neural Network Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Neural Network Regression model, including its 
    /// layer structure, weights, biases, optimizer configuration, and regularization settings.
    /// The new instance is completely independent of the original, allowing modifications without
    /// affecting the original model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained neural network.
    /// 
    /// Think of it like cloning your neural network with all its learned knowledge:
    /// - It copies the structure (how many neurons in each layer)
    /// - It duplicates all the connection strengths (weights) between neurons
    /// - It preserves all the bias values for each neuron
    /// - It maintains the same learning algorithm (optimizer) and settings
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further training or experimentation
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        var newModel = new NeuralNetworkRegression<T>(_options, Regularization);

        // Clear the auto-initialized weights and biases
        newModel._weights.Clear();
        newModel._biases.Clear();

        // Deep copy all weight matrices
        foreach (var weight in _weights)
        {
            newModel._weights.Add(weight.Clone());
        }

        // Deep copy all bias vectors
        foreach (var bias in _biases)
        {
            newModel._biases.Add(bias.Clone());
        }

        return newModel;
    }
}
