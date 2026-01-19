using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// A tool that allows agents to use trained machine learning models for predictions.
/// Integrates with AiModelResult for inference on trained models.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
/// <typeparam name="TInput">The input type for the model (e.g., Matrix&lt;T&gt;, Vector&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output type for the model (e.g., Vector&lt;T&gt;, Scalar&lt;T&gt;).</typeparam>
/// <remarks>
/// For Beginners:
/// This tool bridges AI agents with traditional machine learning models, enabling agents to:
/// - Make predictions using trained ML models
/// - Analyze data patterns
/// - Classify inputs
/// - Generate numerical forecasts
///
/// Use cases:
/// - **Predictive Agent**: "What will sales be next quarter?"
///   → Agent uses regression model trained on historical data
///
/// - **Classification Agent**: "Is this email spam?"
///   → Agent uses classification model on email features
///
/// - **Anomaly Detection**: "Are there any unusual patterns in this data?"
///   → Agent uses anomaly detection model
///
/// - **Recommendation**: "Which product should we recommend?"
///   → Agent uses recommendation model
///
/// Example:
/// <code>
/// // Train a model
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;();
/// var trainedModel = builder
///     .ConfigureModel(new LinearRegressionModel&lt;double&gt;())
///     .Build(trainX, trainY);
///
/// // Create tool from trained model
/// var predictionTool = new PredictionModelTool&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(
///     trainedModel,
///     "SalesPredictor",
///     "Predicts quarterly sales based on features: marketing_spend, season, previous_sales"
/// );
///
/// // Agent can now use the ML model
/// var agent = new Agent&lt;double&gt;(chatModel, new[] { predictionTool });
/// var result = await agent.RunAsync(
///     "Predict sales for Q4 with marketing spend of $50k");
/// </code>
/// </remarks>
public class PredictionModelTool<T, TInput, TOutput> : ITool
{
    private readonly AiModelResult<T, TInput, TOutput> _model;
    private readonly string _name;
    private readonly string _description;
    private readonly Func<string, TInput> _inputParser;

    /// <summary>
    /// Initializes a new instance of the <see cref="PredictionModelTool{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="model">The trained prediction model to use.</param>
    /// <param name="name">The name of the tool (e.g., "SalesPredictor", "SpamClassifier").</param>
    /// <param name="description">Description of what the model predicts and expected input format.</param>
    /// <param name="inputParser">Function to parse string input into model input format.</param>
    /// <exception cref="ArgumentNullException">Thrown when any required parameter is null.</exception>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Parameters:**
    /// - model: Your trained AiModelResult
    /// - name: Give it a clear, descriptive name
    /// - description: Explain what it predicts and what input format to use
    /// - inputParser: Convert agent's string input to model's expected format
    ///
    /// **Input parsing:**
    /// The inputParser is crucial - it converts natural language/JSON from the agent
    /// into the format your model expects (Matrix, Vector, etc.).
    ///
    /// Example parsers:
    /// ```csharp
    /// // For Vector input - JSON array
    /// inputParser: input => {
    ///     var array = JsonSerializer.Deserialize&lt;double[]&gt;(input);
    ///     return new Vector&lt;double&gt;(array);
    /// }
    ///
    /// // For Matrix input - JSON 2D array
    /// inputParser: input => {
    ///     var array = JsonSerializer.Deserialize&lt;double[][]&gt;(input);
    ///     return new Matrix&lt;double&gt;(array);
    /// }
    /// ```
    /// </remarks>
    public PredictionModelTool(
        AiModelResult<T, TInput, TOutput> model,
        string name,
        string description,
        Func<string, TInput> inputParser)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _inputParser = inputParser ?? throw new ArgumentNullException(nameof(inputParser));

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty.", nameof(name));
        }

        if (string.IsNullOrWhiteSpace(description))
        {
            throw new ArgumentException("Description cannot be null or empty.", nameof(description));
        }

        _name = name;
        _description = description;
    }

    /// <inheritdoc/>
    public string Name => _name;

    /// <inheritdoc/>
    public string Description => _description;

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Input cannot be empty. " +
                   "Please provide input in the format specified in the tool description.";
        }

        try
        {
            // Parse the input
            TInput parsedInput = _inputParser(input);

            // Make prediction
            TOutput prediction = _model.Predict(parsedInput);

            // Format the output
            return FormatPrediction(prediction);
        }
        catch (JsonReaderException ex)
        {
            return $"Error parsing input JSON: {ex.Message}. " +
                   "Please ensure input follows the expected format specified in the description.";
        }
        catch (InvalidOperationException ex)
        {
            return $"Error making prediction: {ex.Message}. " +
                   "The model may not be properly initialized.";
        }
        catch (ArgumentException ex)
        {
            return $"Error: Invalid input format - {ex.Message}";
        }
        catch (Exception ex)
        {
            return $"Unexpected error during prediction: {ex.Message}";
        }
    }

    /// <summary>
    /// Formats the model's prediction output into a readable string.
    /// </summary>
    /// <param name="prediction">The prediction output from the model.</param>
    /// <returns>A formatted string representation of the prediction.</returns>
    /// <remarks>
    /// For Beginners:
    /// This converts the model's output (which might be a Vector, Matrix, or other type)
    /// into text that the agent can understand and use in its reasoning.
    /// </remarks>
    private string FormatPrediction(TOutput prediction)
    {
        if (prediction == null)
        {
            return "Prediction: null (no output generated)";
        }

        // Handle common output types
        switch (prediction)
        {
            case Vector<T> vector:
                return FormatVector(vector);

            case Matrix<T> matrix:
                return FormatMatrix(matrix);

            case T scalar:
                return $"Prediction: {scalar}";

            default:
                return $"Prediction: {prediction}";
        }
    }

    /// <summary>
    /// Formats a Vector output.
    /// </summary>
    private string FormatVector(Vector<T> vector)
    {
        if (vector.Length == 1)
        {
            return $"Prediction: {vector[0]}";
        }

        var values = new List<string>();
        for (int i = 0; i < Math.Min(vector.Length, 10); i++)
        {
            string? valueStr = vector[i]?.ToString();
            if (valueStr != null)
            {
                values.Add(valueStr);
            }
            else
            {
                values.Add("0");
            }
        }

        if (vector.Length > 10)
        {
            values.Add($"... ({vector.Length - 10} more values)");
        }

        return $"Predictions: [{string.Join(", ", values)}]";
    }

    /// <summary>
    /// Formats a Matrix output.
    /// </summary>
    private string FormatMatrix(Matrix<T> matrix)
    {
        var result = new System.Text.StringBuilder();
        result.AppendLine($"Predictions (Matrix {matrix.Rows}x{matrix.Columns}):");

        int rowsToShow = Math.Min(matrix.Rows, 5);
        int colsToShow = Math.Min(matrix.Columns, 5);

        for (int i = 0; i < rowsToShow; i++)
        {
            var row = new List<string>();
            for (int j = 0; j < colsToShow; j++)
            {
                string? valueStr = matrix[i, j]?.ToString();
                if (valueStr != null)
                {
                    row.Add(valueStr);
                }
                else
                {
                    row.Add("0");
                }
            }

            if (matrix.Columns > colsToShow)
            {
                row.Add("...");
            }

            result.AppendLine($"  [{string.Join(", ", row)}]");
        }

        if (matrix.Rows > rowsToShow)
        {
            result.AppendLine($"  ... ({matrix.Rows - rowsToShow} more rows)");
        }

        return result.ToString().TrimEnd();
    }

    /// <summary>
    /// Creates a prediction tool with a simple JSON array parser for Vector input.
    /// </summary>
    /// <param name="model">The trained model.</param>
    /// <param name="name">Tool name.</param>
    /// <param name="description">Tool description including expected features.</param>
    /// <returns>A configured PredictionModelTool.</returns>
    /// <remarks>
    /// For Beginners:
    /// This is a convenience factory method for the common case of models that take
    /// a Vector as input (like most regression and classification models).
    ///
    /// Example:
    /// <code>
    /// var tool = PredictionModelTool&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;
    ///     .CreateVectorInputTool(
    ///         trainedModel,
    ///         "HousePricePredictor",
    ///         "Predicts house prices. Input: JSON array [sqft, bedrooms, bathrooms, age]"
    ///     );
    /// </code>
    /// </remarks>
    public static PredictionModelTool<T, Vector<T>, TOutput> CreateVectorInputTool(
        AiModelResult<T, Vector<T>, TOutput> model,
        string name,
        string description)
    {
        return new PredictionModelTool<T, Vector<T>, TOutput>(
            model,
            name,
            description,
            input =>
            {
                // Parse JSON array to Vector
                var array = JsonConvert.DeserializeObject<T[]>(input);
                if (array == null || array.Length == 0)
                {
                    throw new ArgumentException("Input must be a non-empty JSON array of numbers.");
                }

                return new Vector<T>(array);
            });
    }

    /// <summary>
    /// Creates a prediction tool with a JSON 2D array parser for Matrix input.
    /// </summary>
    /// <param name="model">The trained model.</param>
    /// <param name="name">Tool name.</param>
    /// <param name="description">Tool description including expected features.</param>
    /// <returns>A configured PredictionModelTool.</returns>
    /// <remarks>
    /// For Beginners:
    /// Convenience method for models that take a Matrix as input (batch predictions).
    ///
    /// Example:
    /// <code>
    /// var tool = PredictionModelTool&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;
    ///     .CreateMatrixInputTool(
    ///         trainedModel,
    ///         "BatchPredictor",
    ///         "Predicts multiple outputs. Input: 2D JSON array [[f1,f2,f3], [f1,f2,f3], ...]"
    ///     );
    /// </code>
    /// </remarks>
    public static PredictionModelTool<T, Matrix<T>, TOutput> CreateMatrixInputTool(
        AiModelResult<T, Matrix<T>, TOutput> model,
        string name,
        string description)
    {
        return new PredictionModelTool<T, Matrix<T>, TOutput>(
            model,
            name,
            description,
            input =>
            {
                // Parse JSON 2D array to Matrix
                var array = JsonConvert.DeserializeObject<T[][]>(input);
                if (array == null || array.Length == 0)
                {
                    throw new ArgumentException("Input must be a non-empty 2D JSON array.");
                }

                return new Matrix<T>(array);
            });
    }
}
