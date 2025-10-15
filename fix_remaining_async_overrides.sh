#!/bin/bash

# Script to fix remaining async override issues

echo "Fixing remaining async override issues..."

# Fix FoundationModelAdapter.cs - line 438
echo "Fixing FoundationModelAdapter.cs..."
sed -i '438s/public override async Task<Vector<T>> PredictAsync/public virtual async Task<Vector<T>> PredictAsync/' "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs"

# Fix FoundationModelBase.cs - line 674
echo "Fixing FoundationModelBase.cs..."
sed -i '674s/public override async Task<string> PredictAsync/public virtual async Task<string> PredictAsync/' "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelBase.cs"

# Fix ModelIndividual.cs - line 945
echo "Fixing ModelIndividual.cs..."
sed -i '945s/public override async Task<TOutput> PredictAsync/public virtual async Task<TOutput> PredictAsync/' "/home/ooples/AiDotNet/src/Genetics/ModelIndividual.cs"

echo "Done fixing remaining async override issues."