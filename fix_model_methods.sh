#!/bin/bash

# Script to fix model methods - convert Save/Load to use Serialize/Deserialize and remove non-interface methods

echo "Fixing model methods..."

# Function to fix a file
fix_file() {
    local file=$1
    local class_name=$2
    echo "Processing: $file ($class_name)"
    
    # Create a temporary file for the modifications
    local temp_file="${file}.temp"
    
    # Process the file to fix methods
    awk -v class="$class_name" '
    BEGIN {
        in_save_method = 0
        in_load_method = 0
        in_async_method = 0
        in_dispose_method = 0
        in_setmetadata_method = 0
        brace_count = 0
        skip_lines = 0
    }
    
    # Skip lines if needed
    skip_lines > 0 {
        skip_lines--
        next
    }
    
    # Detect Save method
    /public virtual void Save\(string/ {
        in_save_method = 1
        brace_count = 0
        print "    /// <summary>"
        print "    /// Serializes the model to a byte array."
        print "    /// </summary>"
        print "    public override byte[] Serialize()"
        print "    {"
        print "        using (var stream = new MemoryStream())"
        print "        using (var writer = new BinaryWriter(stream))"
        print "        {"
        # Skip the method signature line
        next
    }
    
    # Detect Load method
    /public virtual void Load\(string/ {
        in_load_method = 1
        brace_count = 0
        print "    /// <summary>"
        print "    /// Deserializes the model from a byte array."
        print "    /// </summary>"
        print "    public override void Deserialize(byte[] data)"
        print "    {"
        print "        using (var stream = new MemoryStream(data))"
        print "        using (var reader = new BinaryReader(stream))"
        print "        {"
        # Skip the method signature line
        next
    }
    
    # Detect async methods to remove
    /public virtual async Task.*PredictAsync\(/ ||
    /public virtual async Task.*TrainAsync\(/ ||
    /public virtual Task.*PredictAsync\(/ ||
    /public virtual Task.*TrainAsync\(/ {
        in_async_method = 1
        brace_count = 0
        next
    }
    
    # Detect Dispose method to remove
    /public virtual void Dispose\(\)/ {
        in_dispose_method = 1
        brace_count = 0
        next
    }
    
    # Detect SetModelMetadata method to remove
    /public virtual void SetModelMetadata\(/ {
        in_setmetadata_method = 1
        brace_count = 0
        next
    }
    
    # Handle method bodies
    in_save_method || in_load_method || in_async_method || in_dispose_method || in_setmetadata_method {
        # Count braces
        gsub(/{/, "{", $0)
        open_braces = gsub(/{/, "{", $0)
        close_braces = gsub(/}/, "}", $0)
        brace_count += open_braces - close_braces
        
        # For Save method, replace file operations
        if (in_save_method) {
            # Skip logger lines
            if (/logger.*[Ss]aving.*model/) {
                next
            }
            if (/logger.*[Ss]aved.*successfully/) {
                next
            }
            # Replace file operations
            if (/BinaryWriter.*File\.Open.*path.*FileMode\.Create/) {
                next  # Already handled in method signature
            }
            # Keep the content but adjust for stream
            print $0
        }
        # For Load method, replace file operations
        else if (in_load_method) {
            # Skip logger lines
            if (/logger.*[Ll]oading.*model/) {
                next
            }
            if (/logger.*[Ll]oaded.*successfully/) {
                next
            }
            # Replace file operations
            if (/BinaryReader.*File\.Open.*path.*FileMode\.Open/) {
                next  # Already handled in method signature
            }
            # Keep the content but adjust for stream
            print $0
        }
        # For methods to remove, skip everything
        else if (in_async_method || in_dispose_method || in_setmetadata_method) {
            # Do nothing - skip the line
        }
        
        # Check if method ends
        if (brace_count == 0 && (open_braces > 0 || close_braces > 0)) {
            if (in_save_method) {
                print "        }"
                print "        return stream.ToArray();"
                print "    }"
                in_save_method = 0
            }
            else if (in_load_method) {
                print "        }"
                print "    }"
                in_load_method = 0
            }
            else if (in_async_method) {
                in_async_method = 0
            }
            else if (in_dispose_method) {
                in_dispose_method = 0
            }
            else if (in_setmetadata_method) {
                in_setmetadata_method = 0
            }
        }
    }
    # Normal lines
    !in_save_method && !in_load_method && !in_async_method && !in_dispose_method && !in_setmetadata_method {
        print $0
    }
    ' "$file" > "$temp_file"
    
    # Replace the original file
    mv "$temp_file" "$file"
    echo "  ✓ Fixed methods in $class_name"
}

# Fix EnsembleModelBase.cs
fix_file "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs" "EnsembleModelBase"

# Fix FoundationModelAdapter.cs
fix_file "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs" "FoundationModelAdapter"

# Fix FoundationModelBase.cs
fix_file "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelBase.cs" "FoundationModelBase"

# Fix CloudOptimizer.cs (CachedModel class)
fix_file "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs" "CachedModel"

# Fix ModelIndividual.cs
fix_file "/home/ooples/AiDotNet/src/Genetics/ModelIndividual.cs" "ModelIndividual"

echo "Done fixing model methods."