using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Deployment
{
    /// <summary>
    /// Cached model wrapper for cloud deployment
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    internal class CachedModel<T> : InterpretableModelBase<T, Tensor<T>, Tensor<T>>, IFullModel<T, Tensor<T>, Tensor<T>>
    {
        private readonly IFullModel<T, Tensor<T>, Tensor<T>> baseModel;
        private readonly Dictionary<string, Tensor<T>> cache;
        private readonly int maxCacheSize = 1000;
        
        public CachedModel(IFullModel<T, Tensor<T>, Tensor<T>> baseModel)
        {
            this.baseModel = baseModel;
            this.cache = new Dictionary<string, Tensor<T>>();
        }
        
        public ModelMetadata<T> ModelMetadata 
        { 
            get => baseModel.GetModelMetadata();
            set { /* Setting model metadata not supported on base model */ }
        }
        
        public override Tensor<T> Predict(Tensor<T> inputs)
        {
            var key = ComputeCacheKey(inputs);
            
            if (cache.TryGetValue(key, out var cachedResult))
            {
                return cachedResult;
            }
            
            var result = baseModel.Predict(inputs);
            
            // Add to cache if not full
            if (cache.Count < maxCacheSize)
            {
                cache[key] = result;
            }
            
            return result;
        }
        
        public override async Task<Tensor<T>> PredictAsync(Tensor<T> input)
        {
            var key = ComputeCacheKey(input);
            
            if (cache.TryGetValue(key, out var cachedResult))
            {
                return cachedResult;
            }
            
            var result = baseModel.Predict(input);
            
            // Add to cache if not full
            if (cache.Count < maxCacheSize)
            {
                cache[key] = result;
            }
            
            return result;
        }
        
        private string ComputeCacheKey(Tensor<T> inputs)
        {
            // Simple hash of input values
            var hash = 0;
            for (int i = 0; i < Math.Min(inputs.Length, 100); i++)
            {
                hash = hash * 31 + inputs[i].GetHashCode();
            }
            return hash.ToString();
        }
        
        public override void Train(Tensor<T> inputs, Tensor<T> outputs)
        {
            baseModel.Train(inputs, outputs);
            cache.Clear(); // Invalidate cache after training
        }
        
        public override async Task TrainAsync(Tensor<T> inputs, Tensor<T> targets)
        {
            await baseModel.TrainAsync(inputs, targets);
            cache.Clear(); // Invalidate cache after training
        }
        
        public override void Save(string filePath)
        {
            baseModel.Save(filePath);
        }
        
        public override void Load(string filePath)
        {
            baseModel.Load(filePath);
            cache.Clear();
        }
        
        public byte[] Serialize()
        {
            return baseModel.Serialize();
        }
        
        public void Deserialize(byte[] data)
        {
            baseModel.Deserialize(data);
            cache.Clear();
        }
        
        public override void Dispose()
        {
            if (baseModel is IDisposable disposable)
            {
                disposable.Dispose();
            }
            cache.Clear();
        }
        
        // IModel interface implementation
        public override ModelMetadata<T> GetModelMetadata()
        {
            return baseModel.GetModelMetadata();
        }
        
        public override void SetModelMetadata(ModelMetadata<T> metadata)
        {
            // Setting model metadata not supported on base model
        }
        
        // IParameterizable implementation
        public Vector<T> GetParameters()
        {
            return baseModel.GetParameters();
        }
        
        public void SetParameters(Vector<T> parameters)
        {
            baseModel.SetParameters(parameters);
            cache.Clear();
        }
        
        public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var newModel = new CachedModel<T>(baseModel.WithParameters(parameters));
            return newModel;
        }
        
        // IFeatureAware implementation
        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return baseModel.GetActiveFeatureIndices();
        }
        
        public bool IsFeatureUsed(int featureIndex)
        {
            return baseModel.IsFeatureUsed(featureIndex);
        }
        
        public void SetActiveFeatureIndices(IEnumerable<int> indices)
        {
            baseModel.SetActiveFeatureIndices(indices);
            cache.Clear();
        }
        
        // ICloneable implementation
        public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            return new CachedModel<T>(baseModel.DeepCopy());
        }
        
        public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            return DeepCopy();
        }
        
        // Override interpretability methods to delegate to base model
        public override async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
            return await baseModel.GetGlobalFeatureImportanceAsync();
        }
        
        public override async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
        {
            return await baseModel.GetLocalFeatureImportanceAsync(input);
        }
        
        public override async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
        {
            return await baseModel.GetShapValuesAsync(inputs);
        }
        
        public override async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
        {
            return await baseModel.GetLimeExplanationAsync(input, numFeatures);
        }
        
        public override async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
            return await baseModel.GetPartialDependenceAsync(featureIndices, gridResolution);
        }
        
        public override async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
        {
            return await baseModel.GetCounterfactualAsync(input, desiredOutput, maxChanges);
        }
        
        public override async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
            return await baseModel.GetModelSpecificInterpretabilityAsync();
        }
        
        public override async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
        {
            return await baseModel.GenerateTextExplanationAsync(input, prediction);
        }
        
        public override async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
            return await baseModel.GetFeatureInteractionAsync(feature1Index, feature2Index);
        }
        
        public override async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
        {
            return await baseModel.ValidateFairnessAsync(inputs, sensitiveFeatureIndex);
        }
        
        public override async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
        {
            return await baseModel.GetAnchorExplanationAsync(input, threshold);
        }
        
        public override void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
        {
            if (model is IFullModel<T, Tensor<T>, Tensor<T>> fullModel)
            {
                baseModel = fullModel;
            }
        }
        
        public override void EnableMethod(params Enums.InterpretationMethod[] methods)
        {
            baseModel.EnableMethod(methods);
        }
        
        public override void ConfigureFairness(Vector<int> sensitiveFeatures, params Enums.FairnessMetric[] fairnessMetrics)
        {
            baseModel.ConfigureFairness(sensitiveFeatures, fairnessMetrics);
        }
    }
}