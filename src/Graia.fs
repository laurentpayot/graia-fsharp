module Graia

let VERSION = "0.0.1"

printfn $"🌄 Graia v{VERSION}"

open System.Runtime.Intrinsics
open System

printfn $"Vector128: {Vector128.IsHardwareAccelerated}"
printfn $"Vector256: {Vector256.IsHardwareAccelerated}"
printfn $"Vector512: {Vector512.IsHardwareAccelerated}"

// open System.Collections
// open System

// let a: BitArray = BitArray(3)
// let b: BitArray = BitArray(3)

// a.Set(0, true)
// b.Set(2, true)

// for x in a do
//     printfn $"%A{x}"

// printfn $"a = %A{a}"
// printfn $"b = %A{b}"


type Config = {
    inputs: int
    outputs: int
    layerNodes: int
    layers: int
    seed: int option
}

type History = {
    loss: array<float>
    accuracy: array<float>
}

type Model = {
    graiaVersion: string
    config: Config
    inputWeights: array<array<int>>
    hiddenWeights: array<array<array<int>>>
    outputWeights: array<array<int>>
    history: History
}

let modelInit (config: Config) : Model =
    let {
            inputs = inputs
            outputs = outputs
            layerNodes = layerNodes
            layers = layers
            seed = seed
        } =
        config

    let parametersNb: int =
        (inputs * layerNodes)
        + (layerNodes * layerNodes * (layers - 1))
        + (layerNodes * outputs)

    printfn $"🌄 Graia model with {parametersNb} parameters ready."

    {
        graiaVersion = VERSION
        config = config
        inputWeights = [||]
        hiddenWeights = [||]
        outputWeights = [||]
        history = { loss = [||]; accuracy = [||] }
    }
