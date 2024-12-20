# 🌄 Graia

An *experimental* neural network library.

## Prerequisites

To run Graia along with a F# notebook you will need:

- [.NET 9 SDK](https://dotnet.microsoft.com/en-us/download)
- [Polyglot Notebook VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode)

## Installation

1. `cd src`
1. `dotnet restore` to install the dependencies.
1. `dotnet tool restore` to install the tools.
1. `dotnet build` to create the Graia library.

## Notebook usage

### MNIST

1. Download [the MNIST dataset provided in a easy-to-use CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
1. Extract the two CSV files from the zip archive and place them in a folder named `datasets` inside the `notebooks` folder.

## TODO

- byte outputs (no bit count of pools, just 8 bits to byte)
- byte inputs?
- fixed scale colors in weight charts
- use builder pattern for model generation??? https://sporto.github.io/elm-patterns/basic/builder-pattern.html
- **tests**
  - excitation/inhibition: weights (00,01,10,11) with inputs (0,0,0,0) and (1,1,1,1)
