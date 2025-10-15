#!/bin/bash

echo "Fixing truncated files..."

# Fix FoundationModelAdapter.cs
echo "Fixing FoundationModelAdapter.cs..."
cat >> "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs" << 'EOF'

        #endregion
    }
}
EOF

# Fix CloudOptimizer.cs  
echo "Fixing CloudOptimizer.cs..."
cat >> "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs" << 'EOF'

        // IModel interface implementation
        public override ModelMetadata<T> GetModelMetadata()
        {
            return baseModel.GetModelMetadata();
        }
        
        // IModelSerializer implementation is above
        
        // IParameterizable implementation
        public virtual Vector<T> GetParameters()
        {
            return baseModel.GetParameters();
        }
        
        public virtual void SetParameters(Vector<T> parameters)
        {
            baseModel.SetParameters(parameters);
            cache.Clear();
        }
        
        public virtual IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        {
            var cloned = (CachedModel<T, TInput, TOutput>)MemberwiseClone();
            cloned.baseModel = baseModel.WithParameters(parameters);
            cloned.cache = new Dictionary<int, TOutput>();
            return cloned;
        }
        
        // IFeatureAware implementation
        public virtual IEnumerable<int> GetActiveFeatureIndices()
        {
            return baseModel.GetActiveFeatureIndices();
        }
        
        public virtual bool IsFeatureUsed(int featureIndex)
        {
            return baseModel.IsFeatureUsed(featureIndex);
        }
        
        public virtual void SetActiveFeatureIndices(IEnumerable<int> activeIndices)
        {
            baseModel.SetActiveFeatureIndices(activeIndices);
            cache.Clear();
        }
        
        // ICloneable implementation
        public virtual IFullModel<T, TInput, TOutput> Clone()
        {
            var cloned = new CachedModel<T, TInput, TOutput>(baseModel.Clone(), maxCacheSize);
            return cloned;
        }
        
        public virtual IFullModel<T, TInput, TOutput> DeepCopy()
        {
            return Clone();
        }
    }
}
EOF

echo "Done fixing truncated files."