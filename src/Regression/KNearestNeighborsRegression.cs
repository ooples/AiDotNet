using AiDotNet.Autodiff;

namespace AiDotNet.Regression;

/// <summary>
/// Implements K-Nearest Neighbors algorithm for regression, which predicts target values
/// by averaging the values of the K closest training examples.
/// </summary>
/// <remarks>
/// <para>
/// K-Nearest Neighbors (KNN) is a non-parametric and instance-based learning algorithm that makes
/// predictions based on the similarity between the input and training samples. For regression, it computes
/// the average of the target values of the K nearest neighbors to the query point. The algorithm doesn't
/// build an explicit model but instead stores all training examples and performs computations at prediction time.
/// </para>
/// <para><b>For Beginners:</b> K-Nearest Neighbors is like asking your neighbors for advice.
/// 
/// Imagine you want to guess the price of a house:
/// - You look at the K most similar houses to yours (the "nearest neighbors") 
/// - You take the average of their prices as your prediction
/// 
/// The "K" is just how many neighbors you consider. If K=3, you look at the 3 most similar houses.
/// 
/// This approach is:
/// - Simple to understand: similar inputs should have similar outputs
/// - Makes no assumptions about the data's structure
/// - Works well when similar examples in your data actually have similar target values
/// 
/// Unlike most machine learning algorithms, KNN doesn't "learn" patterns during training - it simply
/// remembers all examples and does the real work at prediction time by finding similar examples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class KNearestNeighborsRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the K-Nearest Neighbors algorithm.
    /// </summary>
    private readonly KNearestNeighborsOptions _options;

    /// <summary>
    /// Matrix containing the feature vectors of the training samples.
    /// </summary>
    private Matrix<T> _xTrain;

    /// <summary>
    /// Vector containing the target values of the training samples.
    /// </summary>
    private Vector<T> _yTrain;

    /// <summary>
    /// Initializes a new instance of the <see cref="KNearestNeighborsRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the KNN algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new K-Nearest Neighbors regression model with the specified options and
    /// regularization strategy. If no options are provided, default values are used. If no regularization
    /// is specified, no regularization is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new K-Nearest Neighbors model.
    /// 
    /// The key setting is K - the number of neighbors to consider when making predictions:
    /// - Smaller K values (like 1 or 2): More sensitive to noise in the data
    /// - Larger K values (like 10 or 20): Smoother predictions but might miss important patterns
    /// 
    /// If you don't specify any options, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a KNN model with default settings
    /// var knn = new KNearestNeighborsRegression&lt;double&gt;();
    /// 
    /// // Create a KNN model with custom options
    /// var options = new KNearestNeighborsOptions { K = 5 };
    /// var customKnn = new KNearestNeighborsRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public KNearestNeighborsRegression(KNearestNeighborsOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new KNearestNeighborsOptions();
        _xTrain = new Matrix<T>(0, 0);
        _yTrain = new Vector<T>(0);
        SoftKNNTemperature = NumOps.One; // Default temperature = 1.0
    }

    /// <summary>
    /// Optimizes the KNN model by storing the training data for later use in predictions.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method "trains" the KNN model by storing the training data for later use during prediction.
    /// Unlike many other machine learning algorithms, KNN doesn't build a parametric model during training.
    /// Instead, it simply stores the training data and uses it to compute predictions at runtime by
    /// finding the K nearest neighbors to each query point.
    /// </para>
    /// <para><b>For Beginners:</b> KNN doesn't really "learn" during training - it just memorizes the examples.
    /// 
    /// While most machine learning models try to extract patterns during training, KNN takes a different approach:
    /// 1. It simply stores all the training examples (both features and target values)
    /// 2. When asked to make a prediction, it does the actual work of finding similar examples
    /// 
    /// Think of it like studying for an exam by memorizing all the examples in a textbook, 
    /// rather than trying to understand the underlying rules. When given a new problem,
    /// you solve it by finding the most similar examples from the ones you memorized.
    /// 
    /// This is why KNN is sometimes called a "lazy learner" - it doesn't do much work during training,
    /// but has to work harder at prediction time.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Apply regularization to the training data
        if (Regularization != null)
        {
            _xTrain = Regularization.Regularize(x);
            _yTrain = Regularization.Regularize(y);
        }
        else
        {
            _xTrain = x;
            _yTrain = y;
        }
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained KNN model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by finding the K nearest neighbors from the
    /// training data for each input sample and computing the average of their target values. The method
    /// applies any specified regularization to the input data before making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// For each input example, it:
    /// 1. Calculates how similar the new example is to each training example (using distance)
    /// 2. Finds the K most similar training examples
    /// 3. Takes the average of their target values as the prediction
    /// 
    /// This method handles multiple inputs at once, making a separate prediction for each one.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = knn.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Apply regularization to the input data if available
        Matrix<T> regularizedInput = Regularization != null ? Regularization.Regularize(input) : input;

        var predictions = new Vector<T>(regularizedInput.Rows);
        for (int i = 0; i < regularizedInput.Rows; i++)
        {
            predictions[i] = PredictSingle(regularizedInput.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Predicts the target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The feature vector of the sample to predict.</param>
    /// <returns>The predicted value for the input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the target value for a single input feature vector by finding the K nearest
    /// neighbors from the training data and computing the average of their target values. The distance
    /// between the input and each training sample is computed using Euclidean distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a prediction for a single new data point.
    /// 
    /// The prediction process works like this:
    /// 1. Calculate the distance between the new point and every training example
    /// 2. Find the K training examples with the smallest distances (the nearest neighbors)
    /// 3. Calculate the average of their target values
    /// 4. Return this average as the prediction
    /// 
    /// For example, if you want to predict a house price and K=3, this method would:
    /// - Find the 3 most similar houses from the training data
    /// - Calculate the average of their prices
    /// - Return that average as the predicted price
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        var distances = new List<(int index, T distance)>();

        for (int i = 0; i < _xTrain.Rows; i++)
        {
            T distance = CalculateDistance(input, _xTrain.GetRow(i));
            distances.Add((i, distance));
        }

        var nearestNeighbors = distances
            .OrderBy(x => x.distance)
            .Take(_options.K)
            .ToList();

        T sum = NumOps.Zero;
        foreach (var (index, distance) in nearestNeighbors)
        {
            sum = NumOps.Add(sum, _yTrain[index]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(_options.K));
    }

    /// <summary>
    /// Calculates the Euclidean distance between two feature vectors.
    /// </summary>
    /// <param name="v1">The first feature vector.</param>
    /// <param name="v2">The second feature vector.</param>
    /// <returns>The Euclidean distance between the two vectors.</returns>
    private T CalculateDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Gets the model type of the K-Nearest Neighbors Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType() => ModelType.KNearestNeighbors;

    /// <summary>
    /// Serializes the K-Nearest Neighbors Regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the KNN model into a byte array that can be stored in a file, database,
    /// or transmitted over a network. The serialized data includes the base class data, the number of
    /// neighbors (K), and the training data that is used for making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model as a sequence of bytes.
    /// 
    /// Serialization allows you to:
    /// - Save your model to a file
    /// - Store your model in a database
    /// - Send your model over a network
    /// - Keep your model for later use without having to retrain it
    /// 
    /// The serialized data includes:
    /// - The value of K (number of neighbors)
    /// - All the training examples (both features and target values)
    /// 
    /// Since KNN stores all training data, the serialized model can be quite large
    /// compared to other machine learning models.
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = knn.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("knn.model", modelData);
    /// ```
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

        // Serialize KNN specific data
        writer.Write(_options.K);

        // Serialize training data
        writer.Write(_xTrain.Rows);
        writer.Write(_xTrain.Columns);
        for (int i = 0; i < _xTrain.Rows; i++)
            for (int j = 0; j < _xTrain.Columns; j++)
                writer.Write(Convert.ToDouble(_xTrain[i, j]));

        writer.Write(_yTrain.Length);
        for (int i = 0; i < _yTrain.Length; i++)
            writer.Write(Convert.ToDouble(_yTrain[i]));

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized K-Nearest Neighbors Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a KNN model from a byte array that was previously created using the
    /// Serialize method. It restores the base class data, the number of neighbors (K), and the training
    /// data that is used for making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - The value of K is restored
    /// - All training examples are loaded back into memory
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("knn.model");
    /// 
    /// // Deserialize the model
    /// var knn = new KNearestNeighborsRegression&lt;double&gt;();
    /// knn.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = knn.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize KNN specific data
        _options.K = reader.ReadInt32();

        // Deserialize training data
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _xTrain = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _xTrain[i, j] = NumOps.FromDouble(reader.ReadDouble());

        int yLength = reader.ReadInt32();
        _yTrain = new Vector<T>(yLength);
        for (int i = 0; i < yLength; i++)
            _yTrain[i] = NumOps.FromDouble(reader.ReadDouble());

        // Apply regularization to the deserialized data if available
        if (Regularization != null)
        {
            _xTrain = Regularization.Regularize(_xTrain);
            _yTrain = Regularization.Regularize(_yTrain);
        }
    }

    /// <summary>
    /// Creates a new instance of the KNearestNeighborsRegression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new KNearestNeighborsRegression instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the KNearestNeighborsRegression model with the same configuration options
    /// and regularization settings as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new KNearestNeighborsRegression<T>(_options, Regularization);
    }

    // ===== Soft KNN Support for JIT Compilation =====

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) KNN mode for JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable soft KNN mode with attention-weighted outputs for JIT support;
    /// <c>false</c> (default) for traditional hard K-nearest neighbors.
    /// </value>
    /// <remarks>
    /// <para><b>Soft KNN:</b> Instead of selecting exactly K nearest neighbors and averaging their
    /// labels, soft KNN computes attention weights over ALL training samples based on distances.
    /// This makes the algorithm differentiable and JIT-compilable.</para>
    /// <para><b>Formula:</b> weights = softmax(-distances / temperature)</para>
    /// <para><b>Output:</b> weighted_output = sum(weights * labels)</para>
    /// <para><b>Trade-offs:</b></para>
    /// <list type="bullet">
    /// <item><description>Soft KNN is differentiable and JIT-compilable</description></item>
    /// <item><description>Results are smooth approximations of hard K selection</description></item>
    /// <item><description>Lower temperature = sharper attention (closer to hard K selection)</description></item>
    /// <item><description>Higher temperature = softer attention (considers more neighbors)</description></item>
    /// </list>
    /// <para><b>Computational Note:</b> Soft KNN computes attention over ALL training samples,
    /// which can be expensive for large training sets. The JIT-compiled version embeds all
    /// support vectors as constants, so the computation graph size scales with training set size.</para>
    /// </remarks>
    public bool UseSoftKNN { get; set; } = false;

    /// <summary>
    /// Gets or sets the temperature parameter for soft KNN mode.
    /// </summary>
    /// <value>
    /// The temperature for softmax attention. Lower values produce sharper attention.
    /// Default is 1.0.
    /// </value>
    public T SoftKNNTemperature { get; set; }

    /// <summary>
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when <see cref="UseSoftKNN"/> is enabled and training data is available;
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// When <see cref="UseSoftKNN"/> is enabled, KNN can be exported as a differentiable
    /// computation graph using attention-weighted averaging. The training data is embedded
    /// as constants in the computation graph.
    /// </para>
    /// <para>
    /// When <see cref="UseSoftKNN"/> is disabled, JIT compilation is not supported because
    /// traditional hard KNN requires dynamic neighbor selection that cannot be represented
    /// as a static computation graph.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => UseSoftKNN && _xTrain.Rows > 0;

    /// <summary>
    /// Exports the model's computation as a graph of operations.
    /// </summary>
    /// <param name="inputNodes">The input nodes for the computation graph.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when <see cref="UseSoftKNN"/> is false.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no training data is available.
    /// </exception>
    /// <remarks>
    /// <para>
    /// When soft KNN mode is enabled, this exports the KNN model as a differentiable computation
    /// graph using <see cref="TensorOperations{T}.SoftKNN"/> operations. The training data
    /// (support vectors and labels) are embedded as constants in the graph.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!UseSoftKNN)
        {
            throw new NotSupportedException(
                "KNearestNeighborsRegression does not support JIT compilation in hard KNN mode because it " +
                "requires dynamic neighbor selection based on distances at prediction time.\n\n" +
                "To enable JIT compilation, set UseSoftKNN = true to use soft (differentiable) KNN " +
                "with attention-weighted outputs.");
        }

        if (_xTrain.Rows == 0)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the KNN model has not been trained. " +
                "Call Train() first to store the training data.");
        }

        int numFeatures = _xTrain.Columns;
        int numSamples = _xTrain.Rows;

        // Create input variable node
        var inputTensor = new Tensor<T>(new[] { numFeatures });
        var input = TensorOperations<T>.Variable(inputTensor, "input");
        inputNodes.Add(input);

        // Create constants for support vectors (training features)
        var supportVectorsTensor = new Tensor<T>(new[] { numSamples, numFeatures });
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                supportVectorsTensor[i * numFeatures + j] = _xTrain[i, j];
            }
        }
        var supportVectors = TensorOperations<T>.Constant(supportVectorsTensor, "support_vectors");

        // Create constants for labels (training targets)
        var labelsTensor = new Tensor<T>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            labelsTensor[i] = _yTrain[i];
        }
        var labels = TensorOperations<T>.Constant(labelsTensor, "labels");

        // Use SoftKNN operation: output = sum(softmax(-distances / temp) * labels)
        return TensorOperations<T>.SoftKNN(input, supportVectors, labels, SoftKNNTemperature);
    }
}
