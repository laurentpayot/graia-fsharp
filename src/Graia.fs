module Graia

open System.Collections
open System.Numerics
open System


let VERSION = "0.0.1"

printfn $"🌄 Graia v{VERSION}"

// open System.Collections

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

type NodeValues = BitArray

// excitatory bits * inhibitory bits
type Weights = array<BitArray * BitArray>

type Model = {
    graiaVersion: string
    config: Config
    inputWeights: Weights
    hiddenWeights: array<Weights>
    outputWeights: Weights
    history: History
}


let init (config: Config) : Model =
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

    let rnd =
        match seed with
        | Some seed -> Random(seed)
        | None -> Random()

    let randomBitArray (length: int) : BitArray =
        let bytes = Array.zeroCreate length
        rnd.NextBytes bytes
        bytes |> Array.map (fun x -> x > 127uy) |> BitArray

    let randomWeights (dim1: int) (dim2: int) : Weights =
        Array.init dim2 (fun _ -> randomBitArray dim1, randomBitArray dim1)

    let model = {
        graiaVersion = VERSION
        config = config
        inputWeights = randomWeights inputs layerNodes
        hiddenWeights = Array.init layers (fun _ -> randomWeights layerNodes layerNodes)
        outputWeights = randomWeights layerNodes outputs
        history = { loss = [||]; accuracy = [||] }
    }

    printfn $"🌄 Graia model with {parametersNb} parameters ready."
    model


//! waiting for native bitArray PopCount https://github.com/dotnet/runtime/issues/104299
let private bitArrayPopCount (ba: BitArray) =
    // https://stackoverflow.com/a/67248403/2675387
    let intArray = Array.create ((ba.Count >>> 5) + 1) 0u
    ba.CopyTo(intArray, 0)
    intArray |> Array.sumBy BitOperations.PopCount


let fit (xs: array<NodeValues>) (ys: array<int>) (epochs: int) (model: Model) : Model =
    // TODO
    model
