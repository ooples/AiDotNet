---
title: "MolecularDatasetLoader<T>"
description: "Loads molecular graph datasets (ZINC, QM9) for graph-level property prediction and generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Loads molecular graph datasets (ZINC, QM9) for graph-level property prediction and generation.

## For Beginners

Molecular graphs represent chemistry as networks.

**Graph Representation of Molecules:**
```
Water (H₂O):

- Nodes: 3 atoms (O, H, H)
- Edges: 2 bonds (O-H, O-H)
- Node features: Atom type, charge, hybridization
- Edge features: Bond type (single, double, triple)

```

**Why model molecules as graphs?**

- **Structure matters**: Same atoms, different arrangement = different properties
* Example: Diamond vs Graphite (both pure carbon!)
- **Bonds are relationships**: Like social networks, but for atoms
- **GNNs excel**: Message passing mimics electron delocalization

**Major Molecular Datasets:**

**ZINC:**

- **Size**: 250,000 drug-like molecules (subset: 12,000)
- **Source**: ZINC database (commercially available compounds)
- **Tasks**: Graph regression on constrained solubility
- **Features**:
* Atoms: C, N, O, F, P, S, Cl, Br, I (28 atom types)
* Bonds: Single, double, triple, aromatic
- **Use case**: Drug discovery, molecular generation

**QM9:**

- **Size**: 134,000 small organic molecules
- **Source**: Quantum mechanical calculations
- **Tasks**: Regression on 19 quantum properties
* Energy, enthalpy, heat capacity
* HOMO/LUMO gap (electronic properties)
* Dipole moment, polarizability
- **Atoms**: C, H, N, O, F (up to 9 heavy atoms)
- **Use case**: Property prediction, molecular design

## How It Works

Molecular datasets represent molecules as graphs where atoms are nodes and chemical bonds are edges.
These datasets are fundamental benchmarks for graph neural networks in drug discovery and
materials science.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MolecularDatasetLoader(MolecularDatasetLoader<>.MolecularDataset,Int32,String,Boolean)` | Initializes a new instance of the `MolecularDatasetLoader` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |
| `NumClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildMolecularGraph(List<Int32>,List<ValueTuple<Int32,Int32,Int32>>,List<ValueTuple<Double,Double,Double>>)` | Builds a molecular graph from parsed atom and bond data. |
| `CreateGraphClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `CreateGraphGenerationTask` | Creates a graph generation task for molecular generation. |
| `CreateGraphLabels(Int32,Int32,Boolean,Random)` | Creates graph-level labels. |
| `CreateLinkPredictionTask(Double,Double,Nullable<Int32>)` |  |
| `CreateNodeClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `DownloadDatasetAsync(String,CancellationToken)` | Downloads the dataset from the standard source. |
| `EnsureDataExistsAsync(String,CancellationToken)` | Ensures the dataset files exist locally, downloading if necessary. |
| `ExtractTarAsync(Stream,String,CancellationToken)` | Extracts a tar archive from a stream. |
| `ExtractTarGzAsync(String,String,CancellationToken)` | Extracts a .tar.gz archive to the specified directory. |
| `GetDataFilePath(String)` | Gets the path to the data file for the current dataset. |
| `GetDefaultDataPath` | Gets the default data cache path. |
| `InferBondsFromDistances(List<Int32>,List<ValueTuple<Double,Double,Double>>)` | Infers bonds from atomic distances using covalent radii. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ParseDatasetAsync(String,CancellationToken)` | Parses the dataset files and builds graph structures. |
| `ParseQM9DatasetAsync(String,CancellationToken)` | Parses the QM9 dataset in SDF format. |
| `ParseSDFFileAsync(String,CancellationToken)` | Parses an SDF file containing multiple molecules. |
| `ParseSMILES(String)` | Parses a SMILES string into a molecular graph. |
| `ParseSmilesCSVAsync(String,CancellationToken)` | Parses a CSV file containing SMILES strings. |
| `ParseSmilesFileAsync(String,CancellationToken)` | Parses a file containing SMILES strings (one per line). |
| `ParseXYZFileAsync(String,CancellationToken)` | Parses an XYZ file for a single molecule. |
| `ParseZINC250KDatasetAsync(String,CancellationToken)` | Parses the ZINC 250K dataset from CSV. |
| `ParseZINCDatasetAsync(String,CancellationToken)` | Parses the ZINC benchmark dataset. |
| `UnloadDataCore` |  |
| `ValidateMolecularGraph(GraphData<>)` | Validates that a generated molecular graph follows chemical rules. |

