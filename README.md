# 🌄 Graia

An *experimental* neural network library.

## Prerequisites

To run Graia along with a F# notebook you will need:

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download)
- [Polyglot Notebook VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode)

## Installation

1. `cd src`
1. `dotnet restore` to install the dependencies.
1. `dotnet build` to create the Graia library.

## Notebook usage

### MNIST

1. Download [the MNIST dataset provided in a easy-to-use CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
1. Extract the two CSV files from the zip archive and place them in a folder named `datasets` inside the `notebooks` folder.

## TODO

- Graia: use parallel operations on arrays https://fsharp.github.io/fsharp-core-docs/reference/fsharp-collections-arraymodule-parallel.html
- showHistory
- fixed scale charts
- use .NET 9.0
- `model.fit` instead of `fit model`
- use builder pattern for model generation??? https://sporto.github.io/elm-patterns/basic/builder-pattern.html
- **tests**
  - excitation/inhibition: weights (00,01,10,11) with inputs (0,0,0,0) and (1,1,1,1)
