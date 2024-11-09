﻿module Graia

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
    inputBits: int
    outputBytes: int
    layerNodes: int
    layers: int
    seed: int option
}

type History = {
    loss: array<float>
    accuracy: array<float>
}

type NodeBits = BitArray

// excitatory bits * inhibitory bits
type Weights = array<BitArray * BitArray>

type Model = {
    graiaVersion: string
    config: Config
    inputWeights: Weights
    hiddenLayersWeights: array<Weights>
    outputWeights: Weights
    lastOutputs: array<byte>
    lastIntermediateOutputs: array<NodeBits>
    history: History
}


let init (config: Config) : Model =
    let {
            inputBits = inputBits
            outputBytes = outputBytes
            layerNodes = layerNodes
            layers = layers
            seed = seed
        } =
        config

    let outputBits = outputBytes * 8

    let parametersNb: int =
        (inputBits * layerNodes)
        + (layerNodes * layerNodes * (layers - 1))
        + (layerNodes * outputBits)

    let rnd =
        match seed with
        | Some seed -> Random(seed)
        | None -> Random()

    let randomBitArray (length: int) : BitArray =
        let bytes = Array.zeroCreate length
        rnd.NextBytes bytes
        bytes |> Array.map (fun x -> x > 127uy) |> BitArray

    let randomWeights (inputDim: int) (outputDim: int) : Weights =
        Array.init outputDim (fun _ -> randomBitArray inputDim, randomBitArray inputDim)

    let model = {
        graiaVersion = VERSION
        config = config
        inputWeights = randomWeights inputBits layerNodes
        hiddenLayersWeights = Array.init (layers - 1) (fun _ -> randomWeights layerNodes layerNodes)
        outputWeights = randomWeights layerNodes outputBits
        lastOutputs = [||]
        lastIntermediateOutputs = [||]
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

let private layerOutputs (wts: Weights) (layerInputs: NodeBits) : NodeBits =
    wts
    |> Array.map (fun (plusBits, minusBits) ->
        let positives = bitArrayPopCount (layerInputs.And(plusBits))
        let negatives = bitArrayPopCount (layerInputs.And(minusBits))

        positives > negatives)
    |> BitArray

let maxByteIndex (xs: array<byte>) : int =
    xs |> Array.indexed |> Array.maxBy snd |> fst

let getLoss (finalBytes: array<byte>) (y: int) : float =
    let idealBytes: array<byte> =
        Array.init finalBytes.Length (fun i -> if i = y then 255uy else 0uy)

    let maxByte = Array.max finalBytes
    let normalizationCoef: float = if maxByte = 0uy then 1.0 else 1.0 / (float maxByte)

    Array.zip finalBytes idealBytes
    // mean absolute error
    |> Array.sumBy (fun (a, b) -> abs (float a - float b))
    |> (fun x -> x * normalizationCoef / (float finalBytes.Length))

type State = {
    inputWeights: Weights
    hiddenLayersWeights: array<Weights>
    outputWeights: Weights
    intermediateBits: array<NodeBits>
    totalLoss: float
    totalCorrect: int
}

let private rowFit (state: State) (xs: NodeBits) (y: int) : State =
    let inputLayerBits = layerOutputs state.inputWeights xs

    // intermediate bits = input layer bits + hidden layers bits
    let intermediateBits =
        state.hiddenLayersWeights
        |> Array.fold
            (fun layerBits (wts: Weights) ->
                let lastLayerBits = Array.last layerBits
                let lastLayerOutputs = layerOutputs wts lastLayerBits
                Array.append layerBits [| lastLayerOutputs |])
            [| inputLayerBits |]

    let finalBits = layerOutputs state.outputWeights (Array.last intermediateBits)
    let finalBytes: array<byte> = Array.zeroCreate (finalBits.Count / 8)
    finalBits.CopyTo(finalBytes, 0)

    let answer = maxByteIndex finalBytes
    let isCorrect = (answer = y) && finalBytes[answer] > 0uy

    {
        state with

            // TODO weight teaching!!!
            inputWeights = state.inputWeights
            hiddenLayersWeights = state.hiddenLayersWeights
            outputWeights = state.outputWeights

            intermediateBits = intermediateBits
            totalLoss = state.totalLoss + getLoss finalBytes y
            totalCorrect =
                if isCorrect then
                    state.totalCorrect + 1
                else
                    state.totalCorrect
    }

let rec fit (xsRows: array<NodeBits>) (yRows: array<int>) (epochs: int) (model: Model) : Model =
    if epochs < 1 then
        model
    else
        // let curr = model.history.loss.Length + 1
        // let total = model.history.loss.Length + epochs
        // let progress = String.replicate (12 * curr / total) "█"
        // let rest = String.replicate (12 - progress.Length) "░"
        // let progressBar = progress + rest
        // //! no way to update the same line https://stackoverflow.com/questions/47675136/is-there-a-way-to-update-the-same-line-in-f-interactive-using-printf
        // // printf "\u001b[1G"
        // // Console.SetCursorPosition(0, 0)
        // printfn
        //     $"Epoch {curr} of {total}\t {progressBar}\t Accuracy {100 * 0}%%\t Loss (MAE) {100 * 0}%%"

        let initialState: State = {
            inputWeights = model.inputWeights
            hiddenLayersWeights = model.hiddenLayersWeights
            outputWeights = model.outputWeights
            intermediateBits =
                Array.init model.config.layers (fun _ -> BitArray(model.config.layerNodes))
            totalLoss = 0.0
            totalCorrect = 0
        }

        let state = Array.fold2 rowFit initialState xsRows yRows

        fit xsRows yRows (epochs - 1) {
            model with
                inputWeights = state.inputWeights
                hiddenLayersWeights = state.hiddenLayersWeights
                outputWeights = state.outputWeights
                history = {
                    loss =
                        Array.append model.history.loss [| state.totalLoss / float xsRows.Length |]
                    accuracy =
                        Array.append model.history.accuracy [|
                            float state.totalCorrect / float xsRows.Length
                        |]
                }
        }
