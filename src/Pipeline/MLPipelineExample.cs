using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Example demonstrating a complete ML pipeline using custom data types
    /// </summary>
    public class MLPipelineExample
    {
        /// <summary>
        /// Demonstrates a complete ML pipeline with double precision
        /// </summary>
        public static async Task RunDoublePrecisionExample()
        {
            Console.WriteLine("=== ML Pipeline Example (Double Precision) ===\n");
            
            // Step 1: Data Loading
            var dataLoader = new DataLoadingStep<double>(
                "data/housing_prices.csv", 
                DataSourceType.CSV,
                new DataLoadingOptions
                {
                    HasHeader = true,
                    LabelColumnIndex = -1, // Last column is the target
                    Delimiter = ','
                });
            
            var (data, labels) = await dataLoader.LoadDataAsync();
            Console.WriteLine($"Loaded data: {data.Rows} samples, {data.Columns} features");
            
            // Step 2: Data Cleaning
            var cleaner = new DataCleaningStep<double>(new DataCleaningConfig
            {
                HandleMissingValues = true,
                ImputationStrategy = ImputationStrategy.Mean,
                HandleOutliers = true,
                OutlierMethod = OutlierDetectionMethod.IQR,
                RemoveDuplicates = true
            });
            
            await cleaner.FitAsync(data, labels);
            data = await cleaner.TransformAsync(data);
            Console.WriteLine($"After cleaning: {data.Rows} samples");
            
            // Step 3: Data Splitting
            var splitter = new DataSplittingStep<double>(new DataSplittingConfig
            {
                TrainRatio = 0.7,
                ValidationRatio = 0.15,
                TestRatio = 0.15,
                Shuffle = true,
                RandomSeed = 42,
                SplitMethod = SplitMethod.Random
            });
            
            await splitter.FitAsync(data, labels);
            var (trainData, trainLabels) = splitter.GetTrainData(data, labels);
            var (valData, valLabels) = splitter.GetValidationData(data, labels);
            var (testData, testLabels) = splitter.GetTestData(data, labels);
            
            Console.WriteLine($"Split data: Train={trainData.Rows}, Val={valData.Rows}, Test={testData.Rows}");
            
            // Step 4: Feature Engineering
            var featureEng = new FeatureEngineeringStep<double>(new FeatureEngineeringConfig<double>
            {
                AutoGenerate = true,
                GeneratePolynomialFeatures = true,
                PolynomialDegree = 2,
                GenerateInteractionFeatures = true,
                MaxInteractionFeatures = 10
            });
            
            await featureEng.FitAsync(trainData, trainLabels);
            trainData = await featureEng.TransformAsync(trainData);
            valData = await featureEng.TransformAsync(valData);
            testData = await featureEng.TransformAsync(testData);
            
            Console.WriteLine($"After feature engineering: {trainData.Columns} features");
            
            // Step 5: Normalization
            var normalizer = new NormalizationStep<double>(NormalizationMethod.MinMax);
            await normalizer.FitAsync(trainData, trainLabels);
            trainData = await normalizer.TransformAsync(trainData);
            valData = await normalizer.TransformAsync(valData);
            testData = await normalizer.TransformAsync(testData);
            
            Console.WriteLine("Data normalized using MinMax normalization");
            
            // Step 6: Model Training
            var model = new MultipleRegression<double>();
            var trainer = new ModelTrainingStep<double>(new ModelTrainingConfig<double>
            {
                Model = model,
                ModelType = ModelType.MultipleRegression,
                UseEarlyStopping = false, // Multiple regression trains in one step
                CalculateAdvancedMetrics = true
            });
            
            await trainer.FitAsync(trainData, trainLabels);
            Console.WriteLine("\nModel trained successfully!");
            
            // Step 7: Evaluation
            var trainMetrics = trainer.GetTrainingMetrics();
            Console.WriteLine($"\nTraining Metrics:");
            Console.WriteLine($"  MSE: {trainMetrics["TrainMSE"]:F4}");
            Console.WriteLine($"  RMSE: {trainMetrics["TrainRMSE"]:F4}");
            Console.WriteLine($"  R²: {trainMetrics["TrainR2"]:F4}");
            
            // Make predictions on test set
            var testPredictions = trainer.Predict(testData);
            
            // Calculate test metrics
            var testMSE = CalculateMSE(testLabels!, testPredictions);
            Console.WriteLine($"\nTest Metrics:");
            Console.WriteLine($"  MSE: {testMSE:F4}");
            Console.WriteLine($"  RMSE: {Math.Sqrt(testMSE):F4}");
            
            // Save the model
            await trainer.SaveModelAsync("models/housing_price_model.bin");
            Console.WriteLine("\nModel saved to 'models/housing_price_model.bin'");
        }
        
        /// <summary>
        /// Demonstrates a complete ML pipeline with float precision for memory efficiency
        /// </summary>
        public static async Task RunFloatPrecisionExample()
        {
            Console.WriteLine("\n=== ML Pipeline Example (Float Precision) ===\n");
            
            // Create pipeline with float precision for reduced memory usage
            var dataLoader = new DataLoadingStep<float>(
                async () => GenerateSyntheticData<float>(),
                new DataLoadingOptions());
            
            var (data, labels) = await dataLoader.LoadDataAsync();
            Console.WriteLine($"Generated synthetic data: {data.Rows} samples, {data.Columns} features");
            
            // Use the same pipeline steps but with float precision
            var cleaner = new DataCleaningStep<float>(new DataCleaningConfig
            {
                HandleMissingValues = true,
                HandleOutliers = true,
                OutlierMethod = OutlierDetectionMethod.ZScore
            });
            
            var normalizer = new NormalizationStep<float>(NormalizationMethod.ZScore);
            
            // Process data through pipeline
            await cleaner.FitAsync(data, labels);
            data = await cleaner.TransformAsync(data);
            
            await normalizer.FitAsync(data, labels);
            data = await normalizer.TransformAsync(data);
            
            Console.WriteLine($"Processed data ready for training (using {typeof(float).Name} precision)");
            Console.WriteLine($"Memory usage is approximately {(float.IsNaN(0) ? "32" : "32")}% of double precision");
        }
        
        /// <summary>
        /// Generates synthetic data for testing
        /// </summary>
        private static (Matrix<T> data, Vector<T> labels) GenerateSyntheticData<T>()
        {
            var rng = new Random(42);
            var samples = 1000;
            var features = 5;
            
            var data = new Matrix<T>(samples, features);
            var labels = new Vector<T>(samples);
            
            // Generate random data
            for (int i = 0; i < samples; i++)
            {
                var label = default(T);
                for (int j = 0; j < features; j++)
                {
                    var value = (T)Convert.ChangeType(rng.NextDouble() * 10, typeof(T));
                    data[i, j] = value;
                    
                    // Simple linear relationship for labels
                    var coef = (T)Convert.ChangeType(j + 1, typeof(T));
                    var contrib = (T)Convert.ChangeType(
                        Convert.ToDouble(value) * Convert.ToDouble(coef), 
                        typeof(T));
                    
                    if (j == 0)
                        label = contrib;
                    else
                        label = (T)Convert.ChangeType(
                            Convert.ToDouble(label) + Convert.ToDouble(contrib), 
                            typeof(T));
                }
                
                // Add some noise
                var noise = (T)Convert.ChangeType(rng.NextDouble() - 0.5, typeof(T));
                labels[i] = (T)Convert.ChangeType(
                    Convert.ToDouble(label) + Convert.ToDouble(noise), 
                    typeof(T));
            }
            
            return (data, labels);
        }
        
        /// <summary>
        /// Calculates Mean Squared Error
        /// </summary>
        private static double CalculateMSE<T>(Vector<T> actual, Vector<T> predicted)
        {
            double sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                var diff = Convert.ToDouble(actual[i]) - Convert.ToDouble(predicted[i]);
                sum += diff * diff;
            }
            return sum / actual.Length;
        }
        
        /// <summary>
        /// Main entry point for the example
        /// </summary>
        public static async Task Main(string[] args)
        {
            try
            {
                // Run double precision example
                await RunDoublePrecisionExample();
                
                // Run float precision example
                await RunFloatPrecisionExample();
                
                Console.WriteLine("\n=== Pipeline Examples Completed Successfully ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}