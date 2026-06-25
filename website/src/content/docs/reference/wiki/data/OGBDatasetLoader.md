---
title: "OGBDatasetLoader<T>"
description: "Loads datasets from the Open Graph Benchmark (OGB) for standardized evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Loads datasets from the Open Graph Benchmark (OGB) for standardized evaluation.

## For Beginners

OGB provides standard benchmarks for fair comparison.

**What is OGB?**

- Collection of real-world graph datasets
- Standardized train/val/test splits
- Automated evaluation metrics
- Enables fair comparison between different GNN methods

**Why OGB matters:**

- **Reproducibility**: Everyone uses same data splits
- **Realism**: Real-world graphs, not toy datasets
- **Scale**: Large graphs that test scalability
- **Diversity**: Multiple domains and tasks

**OGB Dataset Categories:**

**1. Node Property Prediction:**

- ogbn-arxiv: Citation network (169K papers)
- ogbn-products: Amazon product co-purchasing network (2.4M products)
- ogbn-proteins: Protein association network (132K proteins)

**2. Link Property Prediction:**

- ogbl-collab: Author collaboration network
- ogbl-citation2: Citation network
- ogbl-ddi: Drug-drug interaction network

**3. Graph Property Prediction:**

- ogbg-molhiv: Molecular graphs for HIV activity prediction (41K molecules)
- ogbg-molpcba: Molecular graphs for biological assays (437K molecules)
- ogbg-ppa: Protein association graphs

## How It Works

The Open Graph Benchmark (OGB) is a collection of realistic, large-scale graph datasets
with standardized evaluation protocols for graph machine learning research.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OGBDatasetLoader(String,OGBDatasetLoader<>.OGBTask,Int32,String,Boolean)` | Initializes a new instance of the `OGBDatasetLoader` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |
| `NumClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CollectGraphLabels(List<GraphData<>>,Int32)` | Collects graph labels into a single tensor. |
| `ConvertEdgesToTensor(ValueTuple<Int32,Int32>[])` | Converts an array of edge tuples to a tensor of shape [num_edges, 2]. |
| `CreateGraphClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `CreateLinkPredictionTask(Double,Double,Nullable<Int32>)` |  |
| `CreateNodeClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `DownloadDatasetAsync(String,CancellationToken)` | Downloads the dataset from the standard OGB source. |
| `EnsureDataExistsAsync(String,CancellationToken)` | Ensures the dataset files exist locally, downloading if necessary. |
| `FindFile(String,String[])` | Finds a file with one of the given names. |
| `FindSubdirectory(String,String)` | Finds a subdirectory with the given name. |
| `GetDefaultDataPath` | Gets the default data cache path. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ParseBatchedGraphsAsync(String,String,String,CancellationToken)` | Parses batched graphs from edge file and graph index file. |
| `ParseDatasetAsync(String,CancellationToken)` | Parses the OGB dataset files. |
| `ParseEdgeFileAsync(String,CancellationToken)` | Parses edge file in CSV format. |
| `ParseGraphDatasetAsync(String,CancellationToken)` | Parses graph-level prediction dataset. |
| `ParseLabelsAsync(String,CancellationToken)` | Parses labels file in CSV format. |
| `ParseLinkDatasetAsync(String,CancellationToken)` | Parses link prediction dataset. |
| `ParseNodeDatasetAsync(String,CancellationToken)` | Parses node-level prediction dataset. |
| `ParseNodeFeaturesAsync(String,CancellationToken)` | Parses node features file in CSV format. |
| `ParseSMILES(String)` | Simple SMILES parser for molecular graphs. |
| `ParseSmilesFileAsync(String,CancellationToken)` | Parses a SMILES file into molecular graphs. |
| `UnloadDataCore` |  |

