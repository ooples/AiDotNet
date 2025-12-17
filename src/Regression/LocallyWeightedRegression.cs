using AiDotNet.Autodiff;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Locally Weighted Regression, a non-parametric approach that creates a different model
/// for each prediction point based on the weighted influence of nearby training examples.
/// </summary>
/// <remarks>
/// <para>
/// Locally Weighted Regression (LWR) is a memory-based, non-parametric method that creates a unique model
/// for each prediction point. Unlike global regression methods that find a single model for all data,
/// LWR fits a separate weighted regression model for each query point, giving higher influence to
/// nearby training examples. This approach provides excellent flexibility for modeling complex, nonlinear
/// relationships without specifying a fixed functional form.
/// </para>
/// <para><b>For Beginners:</b> Locally Weighted Regression is like having a personalized prediction for each point.
/// 
/// Instead of creating a single model for all data (like linear regression does), LWR:
/// - Creates a new, custom model for each prediction point you want to estimate
/// - Gives more importance to training examples that are close to your prediction point
/// - Gives less importance to training examples that are far away
/// 
/// Imagine predicting house prices: When estimating the price of a specific house, LWR would:
/// - Give most influence to similar houses in the same neighborhood
/// - Give moderate influence to somewhat similar houses in nearby areas
/// - Give little or no influence to very different houses in distant locations
/// 
/// This approach is flexible and works well for complex patterns, but requires keeping all training
/// data around for making predictions, which can be computationally intensive for large datasets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LocallyWeightedRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Locally Weighted Regression algorithm.
    /// </summary>
    private readonly LocallyWeightedRegressionOptions _options;

    /// <summary>
    /// Matrix containing the feature vectors of the training samples.
    /// </summary>
    private Matrix<T> _xTrain;

    /// <summary>
    /// Vector containing the target values of the training samples.
    /// </summary>
    private Vector<T> _yTrain;

    /// <summary>
    /// Initializes a new instance of the <see cref="LocallyWeightedRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the Locally Weighted Regression algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Locally Weighted Regression model with the specified options and regularization
    /// strategy. If no options are provided, default values are used. If no regularization is specified, no regularization
    /// is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Locally Weighted Regression model.
    /// 
    /// The most important option is the "bandwidth", which controls how quickly the influence of training points
    /// drops off with distance:
    /// - Smaller bandwidth: Only very nearby points have influence (more local, potentially more wiggly)
    /// - Larger bandwidth: Points farther away also have some influence (smoother, potentially less accurate for complex patterns)
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a Locally Weighted Regression model with default settings
    /// var lwr = new LocallyWeightedRegression&lt;double&gt;();
    /// 
    /// // Create a model with custom options
    /// var options = new LocallyWeightedRegressionOptions { Bandwidth = 0.5 };
    /// var customLwr = new LocallyWeightedRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public LocallyWeightedRegression(LocallyWeightedRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new LocallyWeightedRegressionOptions();
        _xTrain = Matrix<T>.Empty();
        _yTrain = Vector<T>.Empty();
    }

    /// <summary>
    /// Stores the training data for later use in making predictions.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method "optimizes" the Locally Weighted Regression model by simply storing the training data for later use
    /// during prediction. Unlike global regression methods that compute a fixed set of parameters during training,
    /// LWR defers the actual model fitting until prediction time, when a unique model is created for each query point.
    /// </para>
    /// <para><b>For Beginners:</b> Unlike most regression models, LWR doesn't compute a model during training.
    /// 
    /// Instead of finding a single global model during training, LWR simply:
    /// - Stores all the training examples (both features and target values)
    /// - Waits until prediction time to create a custom model for each point
    /// 
    /// This is similar to K-Nearest Neighbors but more sophisticated, as it creates a weighted regression
    /// model for each prediction rather than just averaging nearby points.
    /// 
    /// Because it doesn't do much work during training, LWR is sometimes called a "lazy learner" - it
    /// postpones the real work until prediction time.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // In LWR, we don't pre-compute a global model. Instead, we store the training data.
        _xTrain = x;
        _yTrain = y;
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained Locally Weighted Regression model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by creating a unique weighted regression model
    /// for each input sample. It processes each input row separately, creating a custom model that gives higher
    /// weight to training examples that are closer to the query point, then uses that model to make a prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your training data to make predictions on new data.
    /// 
    /// For each input example you want to predict:
    /// 1. The method creates a custom model just for that example
    /// 2. It computes predictions using this personalized model
    /// 3. It repeats this process for each example
    /// 
    /// This approach can be more accurate than global models for complex patterns, but
    /// it's also more computationally intensive because a new model is created for each prediction.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = lwr.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
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
    /// This method predicts the target value for a single input feature vector by creating a weighted least squares
    /// model. First, it computes weights for each training example based on their distance to the input point,
    /// giving higher weight to closer points. Then it solves a weighted linear regression problem using these
    /// weights to find the best coefficients, and uses those coefficients to make a prediction for the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a personalized prediction model for a single data point.
    /// 
    /// The prediction process for a single point works like this:
    /// 1. Calculate weights for all training examples based on how close they are to the input point
    ///    (nearby examples get higher weights)
    /// 2. Use these weights to create a custom weighted regression model
    /// 3. Use this custom model to make a prediction for the input point
    /// 
    /// The bandwidth parameter controls how quickly the weights decrease with distance:
    /// - Small bandwidth: Only very close points get significant weight
    /// - Large bandwidth: Even somewhat distant points get some weight
    /// 
    /// This personalized approach allows the model to adapt to local patterns in different regions of the data.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        // Compute weights for each training point
        var weights = ComputeWeights(input);

        // Create the weighted design matrix and target vector
        var weightedX = _xTrain.PointwiseMultiply(weights.CreateDiagonal());
        var weightedY = _yTrain.PointwiseMultiply(weights);

        // Add regularization
        weightedX = Regularization.Regularize(weightedX);

        // Solve the weighted least squares problem
        var xTx = weightedX.Transpose().Multiply(weightedX);
        var xTy = weightedX.Transpose().Multiply(weightedY);
        var coefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _options.DecompositionType);

        // Apply regularization to coefficients
        coefficients = Regularization.Regularize(coefficients);

        // Make prediction
        return input.DotProduct(coefficients);
    }

    /// <summary>
    /// Computes weights for each training example based on their distance to the input point.
    /// </summary>
    /// <param name="input">The input feature vector for which to compute weights.</param>
    /// <returns>A vector of weights for each training example.</returns>
    private Vector<T> ComputeWeights(Vector<T> input)
    {
        var weights = new Vector<T>(_xTrain.Rows);
        var bandwidth = NumOps.FromDouble(_options.Bandwidth);

        for (int i = 0; i < _xTrain.Rows; i++)
        {
            var distance = EuclideanDistance(input, _xTrain.GetRow(i));
            weights[i] = KernelFunction(NumOps.Divide(distance, bandwidth));
        }

        return weights;
    }

    /// <summary>
    /// Calculates the Euclidean distance between two feature vectors.
    /// </summary>
    /// <param name="v1">The first feature vector.</param>
    /// <param name="v2">The second feature vector.</param>
    /// <returns>The Euclidean distance between the two vectors.</returns>
    private T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        return NumOps.Sqrt(v1.Subtract(v2).PointwiseMultiply(v1.Subtract(v2)).Sum());
    }

    /// <summary>
    /// Applies a kernel function to transform distances into weights.
    /// </summary>
    /// <param name="u">The normalized distance value to transform.</param>
    /// <returns>The weight value after applying the kernel function.</returns>
    private T KernelFunction(T u)
    {
        // Tricube kernel function
        var absU = NumOps.Abs(u);
        if (NumOps.GreaterThan(absU, NumOps.One))
            return NumOps.Zero;
        var temp = NumOps.Subtract(NumOps.One, NumOps.Power(absU, NumOps.FromDouble(3)));
        return NumOps.Power(temp, NumOps.FromDouble(3));
    }

    /// <summary>
    /// Gets the model type of the Locally Weighted Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType() => ModelType.LocallyWeightedRegression;

    /// <summary>
    /// Serializes the Locally Weighted Regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Locally Weighted Regression model into a byte array that can be stored in a file,
    /// database, or transmitted over a network. The serialized data includes the base class data, the bandwidth
    /// parameter, and the training data that is used for making predictions.
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
    /// - The bandwidth parameter that controls the locality of the weighted regression
    /// - All the training examples (both features and target values)
    /// 
    /// Since Locally Weighted Regression stores all training data, the serialized model can be
    /// quite large compared to parametric models like linear regression.
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = lwr.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("lwr.model", modelData);
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

        // Serialize LWR specific data
        writer.Write(_options.Bandwidth);

        // Serialize _xTrain
        writer.Write(_xTrain.Rows);
        writer.Write(_xTrain.Columns);
        for (int i = 0; i < _xTrain.Rows; i++)
        {
            for (int j = 0; j < _xTrain.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_xTrain[i, j]));
            }
        }

        // Serialize _yTrain
        writer.Write(_yTrain.Length);
        for (int i = 0; i < _yTrain.Length; i++)
        {
            writer.Write(Convert.ToDouble(_yTrain[i]));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized Locally Weighted Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a Locally Weighted Regression model from a byte array that was previously created
    /// using the Serialize method. It restores the base class data, the bandwidth parameter, and the training data
    /// that is used for making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - The bandwidth parameter is restored
    /// - All training examples are loaded back into memory
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("lwr.model");
    /// 
    /// // Deserialize the model
    /// var lwr = new LocallyWeightedRegression&lt;double&gt;();
    /// lwr.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = lwr.Predict(newFeatures);
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

        // Deserialize LWR specific data
        _options.Bandwidth = reader.ReadDouble();

        // Deserialize _xTrain
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _xTrain = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _xTrain[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize _yTrain
        int length = reader.ReadInt32();
        _yTrain = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            _yTrain[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Creates a new instance of the LocallyWeightedRegression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new LocallyWeightedRegression instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the LocallyWeightedRegression model with the same configuration options
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
        return new LocallyWeightedRegression<T>(_options, Regularization);
    }

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) mode for JIT compilation support.
    /// </summary>
    /// <value><c>true</c> to enable soft mode; <c>false</c> (default) for traditional LWR behavior.</value>
    /// <remarks>
    /// <para>
    /// When enabled, LocallyWeightedRegression uses a differentiable approximation that embeds
    /// all training data as constants in the computation graph and computes attention-weighted
    /// predictions using the softmax of negative squared distances.
    /// </para>
    /// <para><b>For Beginners:</b> Soft mode allows this model to be JIT compiled for faster inference.
    /// Traditional LWR solves a new weighted least squares problem for each prediction, which
    /// cannot be represented as a static computation graph. Soft mode uses a simplified approach
    /// that enables JIT compilation while giving similar results for smooth data.
    /// </para>
    /// </remarks>
    public bool UseSoftMode
    {
        get => _options.UseSoftMode;
        set => _options.UseSoftMode = value;
    }

    /// <summary>
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when <see cref="UseSoftMode"/> is enabled and training data is available;
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// When <see cref="UseSoftMode"/> is enabled, LWR can be exported as a differentiable
    /// computation graph using attention-weighted averaging. The training data is embedded
    /// as constants in the computation graph.
    /// </para>
    /// <para>
    /// When <see cref="UseSoftMode"/> is disabled, JIT compilation is not supported because
    /// traditional LWR requires solving a weighted least squares problem for each query point,
    /// which cannot be represented as a static computation graph.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => UseSoftMode && _xTrain.Rows > 0;

    /// <summary>
    /// Exports the model's computation as a graph of operations.
    /// </summary>
    /// <param name="inputNodes">The input nodes for the computation graph.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when <see cref="UseSoftMode"/> is false.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no training data is available.
    /// </exception>
    /// <remarks>
    /// <para>
    /// When soft mode is enabled, this exports the LWR model as a differentiable computation
    /// graph using <see cref="TensorOperations{T}.SoftLocallyWeighted"/> operations. The training data
    /// (features and targets) are embedded as constants in the graph.
    /// </para>
    /// <para>
    /// The soft LWR approximation computes:
    /// - distances[i] = ||input - xTrain[i]||²
    /// - weights = softmax(-distances / bandwidth)
    /// - output = Σ weights[i] * yTrain[i]
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!UseSoftMode)
        {
            throw new NotSupportedException(
                "LocallyWeightedRegression does not support JIT compilation in traditional mode because it " +
                "solves a new weighted least squares problem for each query point.\n\n" +
                "To enable JIT compilation, set UseSoftMode = true to use soft (differentiable) LWR " +
                "with attention-weighted outputs.");
        }

        if (_xTrain.Rows == 0)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the LWR model has not been trained. " +
                "Call Train() first to store the training data.");
        }

        int numFeatures = _xTrain.Columns;
        int numSamples = _xTrain.Rows;

        // Create input variable node
        var inputTensor = new Tensor<T>(new[] { numFeatures });
        var input = TensorOperations<T>.Variable(inputTensor, "input");
        inputNodes.Add(input);

        // Create constants for training features
        var xTrainTensor = new Tensor<T>(new[] { numSamples, numFeatures });
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                xTrainTensor[i * numFeatures + j] = _xTrain[i, j];
            }
        }
        var xTrainNode = TensorOperations<T>.Constant(xTrainTensor, "x_train");

        // Create constants for training targets
        var yTrainTensor = new Tensor<T>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            yTrainTensor[i] = _yTrain[i];
        }
        var yTrainNode = TensorOperations<T>.Constant(yTrainTensor, "y_train");

        // Use SoftLocallyWeighted operation with bandwidth parameter
        var bandwidth = NumOps.FromDouble(_options.Bandwidth);
        return TensorOperations<T>.SoftLocallyWeighted(input, xTrainNode, yTrainNode, bandwidth);
    }
}
