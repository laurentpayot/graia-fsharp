module Graia

open System.Collections
open System.Numerics
open System


let VERSION = "0.0.1"
printfn $"🌄 Graia v{VERSION}"

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
    mutable inputWeights: Weights
    mutable hiddenLayersWeights: array<Weights>
    mutable outputWeights: Weights
    mutable lastOutputs: array<byte>
    mutable lastIntermediateOutputs: array<NodeBits>
    mutable lastEpochTotalLoss: float
    mutable lastEpochTotalCorrect: int
    mutable history: History
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
        lastEpochTotalLoss = 0.0
        lastEpochTotalCorrect = 0
        history = { loss = [||]; accuracy = [||] }
    }

    printfn $"🌄 Graia model with {parametersNb} parameters ready."
    model


//! waiting for native bitArray PopCount https://github.com/dotnet/runtime/issues/104299
let private bitArrayPopCount (ba: BitArray) : int =
    // 32 = 2^5 (BitOperations.PopCount only works with integers)
    let uint32s: array<uint32> = Array.zeroCreate ((ba.Count >>> 5) + 1)
    ba.CopyTo(uint32s, 0)
    uint32s |> Array.sumBy BitOperations.PopCount

let private layerOutputs (weights: Weights) (layerInputs: NodeBits) : NodeBits =
    weights
    |> Array.map (fun (plusBits, minusBits) ->
        let positiveInputsClone = layerInputs.Clone() :?> BitArray
        let negativesInputsClone = layerInputs.Clone() :?> BitArray
        let positives = bitArrayPopCount (positiveInputsClone.And(plusBits))
        let negatives = bitArrayPopCount (negativesInputsClone.And(minusBits))

        // if layerInputs.Count < 784 then
        //     printfn $"Positives: {positives} Negatives: {negatives}"

        positives > negatives)
    |> BitArray

let maxByteIndex (xs: array<byte>) : int =
    xs |> Array.indexed |> Array.maxBy snd |> fst

let getLoss (finalBytes: array<byte>) (y: int) : float =
    let idealBytes: array<byte> =
        Array.init finalBytes.Length (fun i -> if i = y then 255uy else 0uy)

    let maxByte = Array.max finalBytes

    let normalizationCoef: float =
        if maxByte = 0uy then 1.0 else (1.0 / (float maxByte))

    Array.zip finalBytes idealBytes
    // mean absolute error
    |> Array.averageBy (fun (final, ideal) -> normalizationCoef * abs (float final - float ideal))

let private mutateWeights
    (wasCorrect: bool)
    (inputBits: NodeBits)
    (outputBits: NodeBits)
    (weights: Weights)
    =
    weights
    |> Array.mapi (fun i (plusBits, minusBits) ->
        let wasNodeTriggered = outputBits[i]
        let plusInputsClone = inputBits.Clone() :?> BitArray
        let minusInputsClone = inputBits.Clone() :?> BitArray
        let activatedPlusBits = plusInputsClone.And(plusBits)
        let activatedMinusBits = minusInputsClone.And(minusBits)

        if wasCorrect then
            if wasNodeTriggered then
                minusBits.Xor(activatedMinusBits)
            else
                plusBits.Xor(activatedPlusBits)
        else if wasNodeTriggered then
            plusBits.Xor(activatedPlusBits)
        else
            minusBits.Xor(activatedMinusBits))

let private rowFit (model: Model) (xs: NodeBits) (y: int) : Model =
    let inputLayerBits = layerOutputs model.inputWeights xs

    // intermediate outputs = input layer bits (included by Array.scan) + hidden layers bits
    model.lastIntermediateOutputs <-
        model.hiddenLayersWeights
        |> Array.scan (fun layerBits weights -> layerOutputs weights layerBits) inputLayerBits

    let finalBits =
        layerOutputs model.outputWeights (Array.last model.lastIntermediateOutputs)

    let finalBytes: array<byte> = Array.zeroCreate (finalBits.Count / 8)
    finalBits.CopyTo(finalBytes, 0)

    let answer = maxByteIndex finalBytes
    let isCorrect = (answer = y) && finalBytes[answer] > 0uy
    let teach = mutateWeights isCorrect

    model.inputWeights |> teach xs inputLayerBits |> ignore

    model.hiddenLayersWeights
    |> Array.map2 (fun (i, o) w -> teach i o w) (Array.pairwise model.lastIntermediateOutputs)
    |> ignore

    model.outputWeights
    |> teach (Array.last model.lastIntermediateOutputs) finalBits
    |> ignore

    model.lastEpochTotalLoss <- model.lastEpochTotalLoss + getLoss finalBytes y

    model.lastEpochTotalCorrect <-
        if isCorrect then
            model.lastEpochTotalCorrect + 1
        else
            model.lastEpochTotalCorrect

    model

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


        Array.fold2 rowFit model xsRows yRows |> ignore

        model.history <- {
            loss =
                Array.append model.history.loss [| model.lastEpochTotalLoss / float xsRows.Length |]
            accuracy =
                Array.append model.history.accuracy [|
                    (float model.lastEpochTotalCorrect) / float xsRows.Length
                |]
        }

        fit xsRows yRows (epochs - 1) model
