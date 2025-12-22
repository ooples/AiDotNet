using Newtonsoft.Json;

namespace AiDotNet.Regression;

/// <summary>
/// Represents a multilayer perceptron (neural network) for regression problems.
/// </summary>
/// <remarks>
/// <para>
/// The MultilayerPerceptronRegression is a neural network-based regression model that can capture complex non-linear
/// relationships between features and the target variable. It consists of an input layer, one or more hidden layers,
/// and an output layer, with each layer connected by weights and biases. The model learns by adjusting these weights
/// and biases through a process called backpropagation, minimizing the prediction error.
/// </para>
/// <para><b>For Beginners:</b> A multilayer perceptron is like a digital brain that can learn complex patterns.
/// 
/// Think of it as a system of interconnected layers:
/// - The input layer receives your data (like house features if predicting house prices)
/// - The hidden layers process this information through a series of mathematical transformations
/// - The output layer produces the final prediction (like the predicted house price)
/// 
/// Each connection between neurons has a "weight" (importance) that gets adjusted as the network learns.
/// For example, the network might learn that square footage has a bigger impact on house prices than
/// the age of the house, so it assigns a larger weight to that feature.
/// 
/// The network improves by comparing its predictions to actual values and adjusting the weights
/// to reduce the difference between them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MultilayerPerceptronRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// The configuration options for the multilayer perceptron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the architecture and training behavior of the neural network, including the number and size
    /// of layers, activation functions, learning rate, batch size, and convergence criteria.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the neural network is built and trained.
    /// 
    /// Key settings include:
    /// - Layer sizes: how many "neurons" are in each layer of the network
    /// - Activation functions: mathematical operations that introduce non-linearity
    /// - Learning rate: how quickly the model updates its understanding
    /// - Batch size: how many examples the model looks at before updating
    /// - Max epochs: maximum number of complete passes through the training data
    /// 
    /// These settings are like the knobs on a complex machine that you can adjust
    /// to get better performance for your specific problem.
    /// </para>
    /// </remarks>
    private readonly MultilayerPerceptronOptions<T, Matrix<T>, Vector<T>> _options;

    /// <summary>
    /// The weights connecting the layers of the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each matrix in this list represents the weights connecting two adjacent layers in the network. The element at
    /// position (i,j) in a weight matrix represents the weight of the connection from the j-th neuron in the
    /// previous layer to the i-th neuron in the current layer.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "importance values" for connections between neurons.
    /// 
    /// Weights determine:
    /// - How strongly each input affects each output
    /// - What patterns the network has learned
    /// - How information flows through the network
    /// 
    /// Think of weights like the strength of relationships. A high weight means "this input strongly
    /// affects this output," while a weight near zero means "this input doesn't matter much
    /// for this output."
    /// </para>
    /// </remarks>
    private readonly List<Matrix<T>> _weights;

    /// <summary>
    /// The bias values for each layer of the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each vector in this list represents the bias values for a layer in the network. Biases allow the neural network
    /// to shift the activation function, enabling it to fit the data better by adjusting the threshold for neuron activation.
    /// </para>
    /// <para><b>For Beginners:</b> These are like "baseline values" added to each neuron's calculation.
    /// 
    /// Biases help by:
    /// - Shifting the activation threshold up or down
    /// - Allowing neurons to produce meaningful outputs even when inputs are zero
    /// - Giving the network more flexibility in learning patterns
    /// 
    /// Think of biases like the "default tendency" of a neuron. With no input, the bias determines
    /// whether the neuron tends to fire or remain inactive.
    /// </para>
    /// </remarks>
    private readonly List<Vector<T>> _biases;

    /// <summary>
    /// The optimization algorithm used to update the weights and biases during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The optimizer determines how the weights and biases are updated based on the calculated gradients during training.
    /// Different optimizers can lead to faster convergence or better final performance depending on the problem.
    /// </para>
    /// <para><b>For Beginners:</b> This is the strategy used to adjust the network's weights and biases during learning.
    /// 
    /// The optimizer:
    /// - Determines how quickly and in what direction weights change
    /// - Can adapt the learning speed based on past updates
    /// - Helps navigate complex error landscapes to find better solutions
    /// 
    /// It's like having a smart coach that adjusts your training routine based on your progress,
    /// instead of following a fixed plan regardless of results.
    /// </para>
    /// </remarks>
    private IOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultilayerPerceptronRegression{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the neural network. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new neural network with the specified options and regularization. If no options are provided,
    /// default values are used. The network structure (weights and biases) is initialized based on the layer sizes specified
    /// in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new neural network with your chosen settings.
    /// 
    /// When creating a neural network:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from memorizing the training data
    /// - The network is initialized with random weights, scaled to appropriate values
    /// 
    /// This is like setting up a new brain with the right number of connections, ready to learn
    /// but not yet trained on any data.
    /// </para>
    /// </remarks>
    public MultilayerPerceptronRegression(MultilayerPerceptronOptions<T, Matrix<T>, Vector<T>>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new MultilayerPerceptronOptions<T, Matrix<T>, Vector<T>>();
        _optimizer = _options.Optimizer ?? new AdamOptimizer<T, Matrix<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            LearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _weights = new List<Matrix<T>>();
        _biases = new List<Vector<T>>();

        InitializeNetwork();
    }

    /// <summary>
    /// Initializes the neural network structure with random weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates and initializes the weights and biases for each layer in the neural network. It uses Xavier/Glorot
    /// initialization to set the initial weights to appropriate values that prevent vanishing or exploding gradients during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial "brain wiring" of the neural network.
    /// 
    /// During initialization:
    /// - Random values are assigned to all weights and biases
    /// - These values are scaled based on the layer sizes (Xavier/Glorot initialization)
    /// - This scaling helps the network train more effectively
    /// 
    /// The initial random values give the network a starting point from which to learn.
    /// The special scaling prevents signals from becoming too large or too small as they
    /// pass through multiple layers, which would make learning difficult.
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
            T scaleFactor = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inputSize + outputSize)));
            weight = weight.Transform((w, _, _) => NumOps.Multiply(w, scaleFactor));

            _weights.Add(weight);
            _biases.Add(bias);
        }
    }

    /// <summary>
    /// Trains the neural network using the provided features and target values.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <exception cref="ArgumentException">Thrown when the input feature size does not match the first layer size.</exception>
    /// <remarks>
    /// <para>
    /// This method trains the neural network using mini-batch gradient descent. It iteratively processes batches of the
    /// training data, computes the gradients of the loss with respect to the weights and biases, and updates the model
    /// parameters accordingly. Training continues until the maximum number of epochs is reached or the loss falls below
    /// the specified tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the network learns from your data.
    /// 
    /// During training:
    /// 1. The data is divided into small batches
    /// 2. For each batch:
    ///    - The network makes predictions
    ///    - The error between predictions and actual values is calculated
    ///    - The weights and biases are adjusted to reduce this error
    /// 3. This process repeats for multiple passes (epochs) through the data
    /// 4. Training stops when the error gets small enough or after a maximum number of epochs
    /// 
    /// It's like learning a skill through repeated practice and feedback, gradually
    /// improving performance by adjusting your approach based on the results.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> X, Vector<T> y)
    {
        int numSamples = X.Rows;
        int numFeatures = X.Columns;

        if (_options.LayerSizes[0] != numFeatures)
        {
            throw new ArgumentException("Input feature size does not match the first layer size.");
        }

        for (int epoch = 0; epoch < _options.MaxEpochs; epoch++)
        {
            T totalLoss = NumOps.Zero;

            // Mini-batch gradient descent
            for (int i = 0; i < numSamples; i += _options.BatchSize)
            {
                int batchSize = Math.Min(_options.BatchSize, numSamples - i);
                Matrix<T> batchX = X.GetRowRange(i, batchSize);
                Vector<T> batchY = y.GetSubVector(i, batchSize);

                (T batchLoss, List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients) = ComputeGradients(batchX, batchY);
                UpdateParameters(weightGradients, biasGradients, batchSize);

                totalLoss = NumOps.Add(totalLoss, batchLoss);
            }

            T averageLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(numSamples));

            if (_options.Verbose && epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}, Average Loss: {averageLoss}");
            }

            if (NumOps.LessThan(averageLoss, NumOps.FromDouble(_options.Tolerance)))
            {
                if (_options.Verbose)
                {
                    Console.WriteLine($"Converged at epoch {epoch}");
                }
                break;
            }
        }
    }

    /// <summary>
    /// Computes the gradients of the loss with respect to the weights and biases.
    /// </summary>
    /// <param name="X">The batch feature matrix.</param>
    /// <param name="y">The batch target vector.</param>
    /// <returns>A tuple containing the batch loss, weight gradients, and bias gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass to compute the predictions, calculates the loss, and then performs a backward pass
    /// (backpropagation) to compute the gradients of the loss with respect to each weight and bias in the network. These
    /// gradients indicate how the weights and biases should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out how to adjust the network to improve its predictions.
    /// 
    /// The process works in three steps:
    /// 1. Forward pass: Run data through the network to get predictions
    /// 2. Loss calculation: Measure how far off the predictions are from the actual values
    /// 3. Backward pass (backpropagation): Calculate how changing each weight and bias would affect the error
    /// 
    /// It's like tracking how a ball rolls down a hill to determine the steepest path
    /// downward, which helps find the fastest way to reduce errors.
    /// </para>
    /// </remarks>
    private (T loss, List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients) ComputeGradients(Matrix<T> X, Vector<T> y)
    {
        int numLayers = _weights.Count;
        var activations = new List<Vector<T>>(numLayers + 1);
        var zs = new List<Vector<T>>(numLayers);

        // Forward pass
        activations.Add(X.Transpose().GetColumn(0));  // Input layer
        for (int i = 0; i < numLayers; i++)
        {
            Vector<T> z = _weights[i].Multiply(activations[i]).Add(_biases[i]);
            zs.Add(z);
            activations.Add(ApplyActivation(z, i == numLayers - 1));
        }

        // Compute loss
        T loss = ComputeLoss(activations[activations.Count - 1], y);

        // Backward pass
        List<Matrix<T>> weightGradients = new(numLayers);
        List<Vector<T>> biasGradients = new(numLayers);

        Vector<T> delta = ComputeOutputLayerDelta(activations[activations.Count - 1], y, zs[zs.Count - 1]);

        for (int i = numLayers - 1; i >= 0; i--)
        {
            Matrix<T> weightGradient = delta.OuterProduct(activations[i]);
            weightGradients.Insert(0, weightGradient);
            biasGradients.Insert(0, delta);

            if (i > 0)
            {
                delta = _weights[i].Transpose().Multiply(delta).Transform((d, index) =>
                    NumOps.Multiply(d, ApplyActivationDerivative(zs[i - 1], false)[index]));
            }
        }

        return (loss, weightGradients, biasGradients);
    }

    /// <summary>
    /// Updates the weights and biases based on the computed gradients.
    /// </summary>
    /// <param name="weightGradients">The gradients of the loss with respect to the weights.</param>
    /// <param name="biasGradients">The gradients of the loss with respect to the biases.</param>
    /// <param name="batchSize">The size of the batch used to compute the gradients.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the neural network based on the computed gradients and the optimizer's
    /// update rule. It scales the gradients by the batch size to compute the average gradients, applies the optimizer's
    /// update rule, and then applies regularization to prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's weights and biases to reduce prediction errors.
    /// 
    /// The updating process:
    /// 1. Scales the gradients by the batch size to get the average gradient
    /// 2. Uses the optimizer to determine how much to change each weight and bias
    /// 3. Applies regularization to prevent the weights from growing too large
    /// 
    /// It's like adjusting the recipe for a dish based on feedback, where the gradients
    /// indicate which ingredients need more or less, and the optimizer determines
    /// how much to change each one.
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
                _weights[i] = _weights[i].Subtract(avgWeightGradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
                _biases[i] = _biases[i].Subtract(avgBiasGradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
            }

            // Apply regularization
            _weights[i] = Regularization.Regularize(_weights[i]);
            _biases[i] = Regularization.Regularize(_biases[i]);
        }
    }

    /// <summary>
    /// Generates predictions for new data points using the trained neural network.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A vector containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the neural network for each sample in the input feature matrix
    /// to generate predictions. The forward pass applies the learned weights and biases along with the activation
    /// functions to transform the input features into a predicted output value.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the network uses what it learned to make predictions on new data.
    /// 
    /// For each data point:
    /// 1. The features are fed into the input layer
    /// 2. The network processes them through the hidden layers
    /// 3. The output layer produces the final prediction
    /// 
    /// It's like a factory assembly line where raw materials (features) go in one end,
    /// and a finished product (prediction) comes out the other end.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        Vector<T> predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            Vector<T> input = X.GetRow(i);
            Vector<T> output = ForwardPass(input);
            predictions[i] = output[0];
        }

        return predictions;
    }

    /// <summary>
    /// Performs a forward pass through the neural network for a single input vector.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The output vector from the neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method propagates the input vector through the neural network by applying the weights, biases, and activation
    /// functions at each layer. It returns the final output from the network after passing through all layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method pushes a single data point through the entire network.
    /// 
    /// The forward pass:
    /// - Takes the input features
    /// - For each layer, calculates: activation = activation_function(weights * previous_activation + biases)
    /// - Repeats this process through all layers
    /// - Returns the final output from the last layer
    /// 
    /// It's like following a series of instructions to transform the input data
    /// step by step until reaching the final prediction.
    /// </para>
    /// </remarks>
    private Vector<T> ForwardPass(Vector<T> input)
    {
        Vector<T> activation = input;
        for (int i = 0; i < _weights.Count; i++)
        {
            activation = ApplyActivation(_weights[i].Multiply(activation).Add(_biases[i]), i == _weights.Count - 1);
        }

        return activation;
    }

    /// <summary>
    /// Applies the activation function to an input vector.
    /// </summary>
    /// <param name="input">The input vector before activation.</param>
    /// <param name="isOutputLayer">Whether the input is for the output layer.</param>
    /// <returns>The vector after applying the activation function.</returns>
    private Vector<T> ApplyActivation(Vector<T> input, bool isOutputLayer)
    {
        // First check if vector activation is available
        var vectorActivation = isOutputLayer ? _options.OutputVectorActivation : _options.HiddenVectorActivation;
        if (vectorActivation != null)
        {
            return vectorActivation.Activate(input);
        }

        // Fall back to scalar activation
        var activation = isOutputLayer ? _options.OutputActivation : _options.HiddenActivation;

        // Use the existing IdentityActivation if no activation is specified
        activation ??= new IdentityActivation<T>();

        // Apply scalar activation element-wise to the vector
        return input.Transform(x => activation.Activate(x));
    }

    /// <summary>
    /// Applies the derivative of the activation function to an input vector.
    /// </summary>
    /// <param name="input">The input vector before activation.</param>
    /// <param name="isOutputLayer">Whether the input is for the output layer.</param>
    /// <returns>The vector containing the derivatives of the activation function.</returns>
    private Vector<T> ApplyActivationDerivative(Vector<T> input, bool isOutputLayer)
    {
        // First check if vector activation is available
        var vectorActivation = isOutputLayer ? _options.OutputVectorActivation : _options.HiddenVectorActivation;
        if (vectorActivation != null)
        {
            // Vector activation derivative returns a vector for the input vector
            return vectorActivation.Derivative(input).ToVector();
        }

        // Fall back to scalar activation
        var activation = isOutputLayer ? _options.OutputActivation : _options.HiddenActivation;

        // Use the existing IdentityActivation if no activation is specified
        activation ??= new IdentityActivation<T>();

        // Apply scalar activation derivative element-wise to the vector
        return input.Transform(x => activation.Derivative(x));
    }

    /// <summary>
    /// Computes the mean squared error loss between predictions and targets.
    /// </summary>
    /// <param name="predictions">The predicted values from the neural network.</param>
    /// <param name="targets">The actual target values.</param>
    /// <returns>The mean squared error loss.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the mean squared error (MSE) loss, which is the average of the squared differences between
    /// the predicted values and the actual target values. MSE is a common loss function for regression problems because it
    /// heavily penalizes large errors.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how accurate the predictions are.
    /// 
    /// The mean squared error:
    /// - Calculates the difference between each prediction and the actual value
    /// - Squares these differences to make all values positive and emphasize large errors
    /// - Computes the average of these squared differences
    /// 
    /// Lower values indicate better predictions. Squaring the errors means that
    /// being off by twice as much is penalized four times as heavily, making the
    /// model focus on reducing large errors.
    /// </para>
    /// </remarks>
    private T ComputeLoss(Vector<T> predictions, Vector<T> targets)
    {
        if (_options.LossFunction != null)
        {
            return _options.LossFunction.CalculateLoss(predictions, targets);
        }

        // Default to MSE if no loss function is specified
        T sumSquaredErrors = predictions.Subtract(targets).Transform(x => NumOps.Multiply(x, x)).Sum();
        return NumOps.Divide(sumSquaredErrors, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Computes the error delta for the output layer during backpropagation.
    /// </summary>
    /// <param name="predictions">The predicted values from the neural network.</param>
    /// <param name="targets">The actual target values.</param>
    /// <param name="outputLayerZ">The pre-activation values of the output layer.</param>
    /// <returns>The error delta for the output layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the error delta for the output layer, which is the product of the prediction error and the
    /// derivative of the activation function. This delta is used to start the backpropagation process, propagating the error
    /// backwards through the network to compute the gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the starting point for the learning process.
    /// 
    /// The output layer delta:
    /// - Measures the error at the output layer (predictions - targets)
    /// - Adjusts this error by how responsive the output neurons are to changes
    /// - Provides the starting point for backpropagation to adjust all weights
    /// 
    /// It's like determining how much correction is needed at the final stage of a process,
    /// before working backward to adjust earlier stages.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeOutputLayerDelta(Vector<T> predictions, Vector<T> targets, Vector<T> outputLayerZ)
    {
        Vector<T> error;

        if (_options.LossFunction != null)
        {
            error = _options.LossFunction.CalculateDerivative(predictions, targets);
        }
        else
        {
            // Default to MSE derivative if no loss function is specified
            error = predictions.Subtract(targets);
        }

        Vector<T> activationDerivative = ApplyActivationDerivative(outputLayerZ, true);

        return error.Transform((e, i) => NumOps.Multiply(e, activationDerivative[i]));
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type, in this case, MultilayerPerceptronRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an enumeration value indicating that this is a multilayer perceptron regression model. This is used
    /// for type identification when working with different regression models in a unified manner.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies what kind of model this is.
    /// 
    /// It returns a label (MultilayerPerceptronRegression) that:
    /// - Identifies this specific type of model
    /// - Helps other code handle the model appropriately
    /// - Is used for model identification and categorization
    /// 
    /// It's like a name tag that lets other parts of the program know what kind of model they're working with.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.MultilayerPerceptronRegression;

    /// <summary>
    /// Optimizes the model by training it on the provided data.
    /// </summary>
    /// <param name="x">The feature matrix for training.</param>
    /// <param name="y">The target vector for training.</param>
    /// <remarks>
    /// <para>
    /// This method implements the abstract method from the base class by calling the Train method. It is used to optimize
    /// the model parameters based on the provided training data.
    /// </para>
    /// <para><b>For Beginners:</b> This method trains the model with your data.
    /// 
    /// It's a simple method that:
    /// - Is required by the base class
    /// - Calls the Train method to do the actual work
    /// - Allows the model to fit into the standard optimization framework
    /// 
    /// This consistent interface makes it easier to work with different types of models
    /// in a uniform way.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        Train(x, y);
    }

    /// <summary>
    /// Serializes the neural network model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the entire neural network model, including its parameters, weights, biases, and configuration,
    /// into a byte array that can be stored in a file or database, or transmitted over a network. The model can later be
    /// restored using the Deserialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to a format that can be stored or shared.
    /// 
    /// Serialization:
    /// - Converts the model into a sequence of bytes
    /// - Preserves all the important information (weights, biases, architecture, etc.)
    /// - Allows you to save the trained model to a file
    /// - Lets you load the model later without having to retrain it
    /// 
    /// It's like taking a complete snapshot of the model that you can use later or share with others.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new();
        using BinaryWriter writer = new(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize MultilayerPerceptronRegression specific data
        writer.Write(_options.LayerSizes.Count);
        foreach (var size in _options.LayerSizes)
        {
            writer.Write(size);
        }
        writer.Write(_options.MaxEpochs);
        writer.Write(_options.BatchSize);
        writer.Write(Convert.ToDouble(_options.LearningRate));
        writer.Write(Convert.ToDouble(_options.Tolerance));
        writer.Write(_options.Verbose);

        // Serialize weights and biases
        writer.Write(_weights.Count);
        foreach (var weight in _weights)
        {
            byte[] weightData = weight.Serialize();
            writer.Write(weightData.Length);
            writer.Write(weightData);
        }

        writer.Write(_biases.Count);
        foreach (var bias in _biases)
        {
            byte[] biasData = bias.Serialize();
            writer.Write(biasData.Length);
            writer.Write(biasData);
        }

        // Serialize optimizer
        writer.Write((int)OptimizerFactory<T, Matrix<T>, Vector<T>>.GetOptimizerType(_optimizer));
        byte[] optimizerData = _optimizer.Serialize();
        writer.Write(optimizerData.Length);
        writer.Write(optimizerData);

        // Serialize optimizer options
        string optionsJson = JsonConvert.SerializeObject(_optimizer.GetOptions());
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the neural network model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model data.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    /// <remarks>
    /// <para>
    /// This method restores a neural network model from a serialized byte array, reconstructing its parameters, weights,
    /// biases, and configuration. This allows a previously trained model to be loaded from storage or after being received
    /// over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method rebuilds the model from a saved format.
    /// 
    /// Deserialization:
    /// - Takes a sequence of bytes that represents a model
    /// - Reconstructs the original neural network with all its learned knowledge
    /// - Restores the weights, biases, layer sizes, and other settings
    /// - Allows you to use a previously trained model without retraining
    /// 
    /// It's like unpacking a complete model that was packed up for storage or sharing,
    /// so you can use it again exactly as it was when saved, with all its learned patterns intact.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new(data);
        using BinaryReader reader = new(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize MultilayerPerceptronRegression specific data
        int layerCount = reader.ReadInt32();
        _options.LayerSizes = new List<int>();
        for (int i = 0; i < layerCount; i++)
        {
            _options.LayerSizes.Add(reader.ReadInt32());
        }
        _options.MaxEpochs = reader.ReadInt32();
        _options.BatchSize = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.Tolerance = reader.ReadDouble();
        _options.Verbose = reader.ReadBoolean();

        // Deserialize weights and biases
        int weightCount = reader.ReadInt32();
        _weights.Clear();
        for (int i = 0; i < weightCount; i++)
        {
            int weightDataLength = reader.ReadInt32();
            byte[] weightData = reader.ReadBytes(weightDataLength);
            _weights.Add(Matrix<T>.Deserialize(weightData));
        }

        int biasCount = reader.ReadInt32();
        _biases.Clear();
        for (int i = 0; i < biasCount; i++)
        {
            int biasDataLength = reader.ReadInt32();
            byte[] biasData = reader.ReadBytes(biasDataLength);
            _biases.Add(Vector<T>.Deserialize(biasData));
        }

        // Deserialize optimizer
        OptimizerType optimizerType = (OptimizerType)reader.ReadInt32();
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);

        // Deserialize optimizer options
        string optionsJson = reader.ReadString();
        var options = JsonConvert.DeserializeObject<OptimizationAlgorithmOptions<T, Matrix<T>, Vector<T>>>(optionsJson);

        if (options == null)
        {
            throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }

        // Create optimizer using factory
        _optimizer = OptimizerFactory<T, Matrix<T>, Vector<T>>.CreateOptimizer(optimizerType, options);
        _optimizer.Deserialize(optimizerData);
    }

    /// <summary>
    /// Creates a new instance of the <see cref="MultilayerPerceptronRegression{T}"/> class with the same options and regularization as this instance.
    /// </summary>
    /// <returns>A new instance of the <see cref="MultilayerPerceptronRegression{T}"/> class.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new, uninitialized instance of the multilayer perceptron regression model with the same configuration
    /// as the current instance. It is used for creating copies or clones of the model without copying the learned parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new neural network with the same architecture as this one.
    /// 
    /// The new network:
    /// - Uses the same layer sizes and structure
    /// - Uses the same activation functions and learning parameters
    /// - Uses the same regularization approach to prevent overfitting
    /// - Is uninitialized (not trained yet) with fresh random weights
    /// 
    /// It's like creating a duplicate blueprint of the current network structure without copying
    /// any of the learned knowledge, which is useful for ensemble methods or cross-validation.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new MultilayerPerceptronRegression<T>(_options, Regularization);
    }
}
