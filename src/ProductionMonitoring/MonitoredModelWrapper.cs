using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Wrapper that adds monitoring capabilities to any model
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    public class MonitoredModelWrapper<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
    {
        private readonly IFullModel<T, TInput, TOutput> wrappedModel;
        private readonly DefaultProductionMonitor<T> monitor;
        private readonly INumericOperations<T> ops;
        
        public MonitoredModelWrapper(IFullModel<T, TInput, TOutput> model)
        {
            this.wrappedModel = model ?? throw new ArgumentNullException(nameof(model));
            this.monitor = new DefaultProductionMonitor<T>();
            this.ops = MathHelper.GetNumericOperations<T>();
            
            // Copy metadata from wrapped model
            var wrappedMetadata = model.GetModelMetadata();
            this.ModelMetadata = new ModelMetadata<T>
            {
                ModelType = wrappedMetadata.ModelType,
                FeatureCount = wrappedMetadata.FeatureCount,
                Complexity = wrappedMetadata.Complexity,
                Description = $"Monitored {wrappedMetadata.Description}"
            };
        }
        
        public ModelMetadata<T> ModelMetadata { get; set; }
        
        public async Task<TOutput> PredictAsync(TInput inputs)
        {
            // Monitor input data if it's a Tensor
            if (inputs is Tensor<T> tensorInput)
            {
                await monitor.MonitorInputDataAsync(tensorInput);
            }
            
            // Get prediction from wrapped model
            var result = wrappedModel.Predict(inputs);
            
            // Monitor predictions if output is a Tensor
            if (result is Tensor<T> tensorOutput)
            {
                await monitor.MonitorPredictionsAsync(tensorOutput);
            }
            
            return result;
        }
        
        public void Train(TInput inputs, TOutput outputs)
        {
            wrappedModel.Train(inputs, outputs);
        }
        
        public TOutput Predict(TInput inputs)
        {
            return PredictAsync(inputs).GetAwaiter().GetResult();
        }
        
        public void Save(string filePath)
        {
            var data = wrappedModel.Serialize();
            System.IO.File.WriteAllBytes(filePath, data);
        }
        
        public void Load(string filePath)
        {
            var data = System.IO.File.ReadAllBytes(filePath);
            wrappedModel.Deserialize(data);
        }
        
        public void Dispose()
        {
            // IFullModel doesn't implement IDisposable
            // Only dispose monitor if it implements IDisposable
            if (monitor is IDisposable disposableMonitor)
            {
                disposableMonitor.Dispose();
            }
        }
        
        /// <summary>
        /// Gets monitoring alerts
        /// </summary>
        public List<DataDriftAlert> GetAlerts()
        {
            return monitor.GetRecentAlerts();
        }
        
        /// <summary>
        /// Checks if retraining is needed
        /// </summary>
        public bool NeedsRetraining()
        {
            return monitor.GetRetrainingRecommendation();
        }
        
        /// <summary>
        /// Gets model health score
        /// </summary>
        public double GetHealthScore()
        {
            return monitor.GetHealthScore();
        }
        
        // IModel interface implementation
        public ModelMetadata<T> GetModelMetadata()
        {
            return ModelMetadata;
        }
        
        // IModelSerializer implementation
        public byte[] Serialize()
        {
            return wrappedModel.Serialize();
        }
        
        public void Deserialize(byte[] data)
        {
            wrappedModel.Deserialize(data);
        }
        
        // IParameterizable implementation
        public Vector<T> GetParameters()
        {
            return wrappedModel.GetParameters();
        }
        
        public void SetParameters(Vector<T> parameters)
        {
            wrappedModel.SetParameters(parameters);
        }
        
        public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        {
            var newWrappedModel = wrappedModel.WithParameters(parameters);
            return new MonitoredModelWrapper<T, TInput, TOutput>(newWrappedModel);
        }
        
        // IFeatureAware implementation
        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return wrappedModel.GetActiveFeatureIndices();
        }
        
        public bool IsFeatureUsed(int featureIndex)
        {
            return wrappedModel.IsFeatureUsed(featureIndex);
        }
        
        public void SetActiveFeatureIndices(IEnumerable<int> activeIndices)
        {
            wrappedModel.SetActiveFeatureIndices(activeIndices);
        }
        
        // ICloneable implementation
        public IFullModel<T, TInput, TOutput> DeepCopy()
        {
            return new MonitoredModelWrapper<T, TInput, TOutput>(wrappedModel.DeepCopy());
        }
        
        public IFullModel<T, TInput, TOutput> Clone()
        {
            return DeepCopy();
        }
        
        // IInterpretableModel implementation - delegate to wrapped model
        public async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
            return await wrappedModel.GetGlobalFeatureImportanceAsync();
        }
        
        public async Task<Matrix<T>> GetShapValuesAsync(TInput input)
        {
            return await wrappedModel.GetShapValuesAsync(input);
        }
        
        public async Task<LimeExplanation<T>> GetLimeExplanationAsync(TInput input, int numFeatures = 10)
        {
            return await wrappedModel.GetLimeExplanationAsync(input, numFeatures);
        }
        
        public async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(TInput input, TOutput desiredOutput, int maxChanges = 5)
        {
            return await wrappedModel.GetCounterfactualAsync(input, desiredOutput, maxChanges);
        }
        
        public async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
            return await wrappedModel.GetPartialDependenceAsync(featureIndices, gridResolution);
        }
        
        public async Task<FairnessMetrics<T>> ValidateFairnessAsync(TInput inputs, int sensitiveFeatureIndex)
        {
            return await wrappedModel.ValidateFairnessAsync(inputs, sensitiveFeatureIndex);
        }
        
        public async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
            return await wrappedModel.GetModelSpecificInterpretabilityAsync();
        }
        
        public async Task<string> GenerateTextExplanationAsync(TInput input, TOutput prediction)
        {
            return await wrappedModel.GenerateTextExplanationAsync(input, prediction);
        }
        
        public async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
            return await wrappedModel.GetFeatureInteractionAsync(feature1Index, feature2Index);
        }
        
        public async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(TInput input, T threshold)
        {
            return await wrappedModel.GetAnchorExplanationAsync(input, threshold);
        }
        
        public async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(TInput input)
        {
            return await wrappedModel.GetLocalFeatureImportanceAsync(input);
        }
        
        public void SetBaseModel(IModel<TInput, TOutput, ModelMetadata<T>> model)
        {
            wrappedModel.SetBaseModel(model);
        }
        
        public void EnableMethod(params InterpretationMethod[] methods)
        {
            wrappedModel.EnableMethod(methods);
        }
        
        public void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
            wrappedModel.ConfigureFairness(sensitiveFeatures, fairnessMetrics);
        }
    }
}