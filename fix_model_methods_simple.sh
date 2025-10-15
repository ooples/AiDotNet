#!/bin/bash

# Script to fix model methods more simply

echo "Fixing model methods..."

# Fix EnsembleModelBase.cs - it already has Serialize/Deserialize, just need to move Save/Load logic there
echo "Fixing EnsembleModelBase.cs..."
# Remove the Save and Load methods (they're already using Serialize/Deserialize internally)
sed -i '/public virtual void Save(string path)/,/^    }$/d' "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"
sed -i '/public virtual void Load(string path)/,/^    }$/d' "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"

# Now update the existing Serialize/Deserialize placeholders
# First, let's update the Serialize method
sed -i '/public virtual byte\[\] Serialize()/,/return new byte\[0\];/{
    /public virtual byte\[\] Serialize()/!{
        /return new byte\[0\];/!d
    }
}' "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"

# Insert the actual implementation
sed -i '/public virtual byte\[\] Serialize()/{
a\    {\
        using (var stream = new MemoryStream())\
        using (var writer = new BinaryWriter(stream))\
        {\
            // Write ensemble metadata\
            writer.Write("ENSEMBLE_V1"); // Version marker\
            writer.Write(_baseModels.Count);\
            writer.Write((int)_options.Strategy);\
            \
            // Write weights\
            writer.Write(_modelWeights.Length);\
            foreach (var weight in _modelWeights)\
            {\
                SerializationHelper<T>.WriteValue(writer, weight);\
            }\
            \
            // Write each model\
            foreach (var model in _baseModels)\
            {\
                writer.Write(model.GetType().AssemblyQualifiedName ?? model.GetType().FullName!);\
                \
                // Serialize model\
                var modelBytes = model.Serialize();\
                writer.Write(modelBytes.Length);\
                writer.Write(modelBytes);\
            }\
            \
            // Write options\
            WriteOptions(writer);\
            \
            return stream.ToArray();\
        }
}' "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"

# Update the Deserialize method
sed -i '/public virtual void Deserialize(byte\[\] data)/,/\/\/ For now, do nothing/{
    /public virtual void Deserialize(byte\[\] data)/!{
        /\/\/ For now, do nothing/!d
    }
}' "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"

# Insert the actual implementation
sed -i '/public virtual void Deserialize(byte\[\] data)/{
a\    {\
        using (var stream = new MemoryStream(data))\
        using (var reader = new BinaryReader(stream))\
        {\
            // Read and verify version\
            var version = reader.ReadString();\
            if (version != "ENSEMBLE_V1")\
            {\
                throw new InvalidOperationException($"Unsupported ensemble version: {version}");\
            }\
            \
            // Read ensemble metadata\
            var modelCount = reader.ReadInt32();\
            var strategy = (EnsembleStrategy)reader.ReadInt32();\
            \
            // Clear existing models\
            _baseModels.Clear();\
            _modelIdentifiers.Clear();\
            \
            // Read weights\
            var weightCount = reader.ReadInt32();\
            var weights = new T[weightCount];\
            for (int i = 0; i < weightCount; i++)\
            {\
                weights[i] = SerializationHelper<T>.ReadValue(reader);\
            }\
            _modelWeights = new Vector<T>(weights);\
            \
            // Read each model\
            for (int i = 0; i < modelCount; i++)\
            {\
                var typeName = reader.ReadString();\
                var modelType = Type.GetType(typeName);\
                if (modelType == null)\
                {\
                    throw new InvalidOperationException($"Cannot find type: {typeName}");\
                }\
                \
                var model = Activator.CreateInstance(modelType) as IFullModel<T, TInput, TOutput>;\
                if (model == null)\
                {\
                    throw new InvalidOperationException($"Cannot create instance of type: {typeName}");\
                }\
                \
                // Read model data\
                var modelBytesLength = reader.ReadInt32();\
                var modelBytes = reader.ReadBytes(modelBytesLength);\
                \
                // Deserialize model\
                model.Deserialize(modelBytes);\
                \
                _baseModels.Add(model);\
                _modelIdentifiers[i] = $"{model.GetType().Name}_{Guid.NewGuid():N}";\
            }\
            \
            // Read options\
            ReadOptions(reader);\
            \
            // Recreate combination strategy\
            _combinationStrategy = CreateCombinationStrategy();\
        }
}' "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"

# Remove async methods and other non-interface methods from all files
for file in "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs" \
            "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelBase.cs" \
            "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs" \
            "/home/ooples/AiDotNet/src/Genetics/ModelIndividual.cs"; do
    echo "Removing non-interface methods from $(basename $file)..."
    
    # Remove PredictAsync methods
    sed -i '/public virtual.*Task.*PredictAsync(/,/^    }$/d' "$file"
    
    # Remove TrainAsync methods
    sed -i '/public virtual.*Task.*TrainAsync(/,/^    }$/d' "$file"
    
    # Remove Dispose methods
    sed -i '/public virtual void Dispose()/,/^    }$/d' "$file"
    
    # Remove SetModelMetadata methods
    sed -i '/public virtual void SetModelMetadata(/,/^    }$/d' "$file"
    
    # Remove Save methods
    sed -i '/public virtual void Save(string/,/^    }$/d' "$file"
    
    # Remove Load methods
    sed -i '/public virtual void Load(string/,/^    }$/d' "$file"
done

echo "Done fixing model methods."