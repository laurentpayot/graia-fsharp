module Graia

open System.Collections
open System.Numerics
open System
open System.Reflection.Metadata.Ecma335


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
type NodeWeights = BitArray * BitArray

type LayerWeights = array<NodeWeights>

type Model = {
    graiaVersion: string
    config: Config
    mutable inputLayerWeights: LayerWeights
    mutable hiddenLayersWeights: array<LayerWeights>
    mutable outputLayerWeights: LayerWeights
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

    let randomLayerWeights (inputDim: int) (outputDim: int) : LayerWeights =
        Array.init outputDim (fun _ -> randomBitArray inputDim, randomBitArray inputDim)

    let model = {
        graiaVersion = VERSION
        config = config
        inputLayerWeights = randomLayerWeights inputBits layerNodes
        hiddenLayersWeights =
            Array.init (layers - 1) (fun _ -> randomLayerWeights layerNodes layerNodes)
        outputLayerWeights = randomLayerWeights layerNodes outputBits
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

type ActivatedWeightBits = {
    plus: BitArray
    minus: BitArray
    both: BitArray
    plusOnly: BitArray
    minusOnly: BitArray
    noBits: BitArray option
}

let getActivatedWeightBits
    (getActivatedNoBits: Boolean)
    (inputBits: NodeBits)
    ((plusWeightBits, minusWeightBits): NodeWeights)
    : ActivatedWeightBits =
    let plus = (plusWeightBits.Clone() :?> BitArray).And(inputBits)
    let minus = (minusWeightBits.Clone() :?> BitArray).And(inputBits)
    let both = (plus.Clone() :?> BitArray).And(minus)
    let plusOnly = (plus.Clone() :?> BitArray).Xor(both)
    let minusOnly = (minus.Clone() :?> BitArray).Xor(both)

    {
        plus = plus
        minus = minus
        both = both
        plusOnly = plusOnly
        minusOnly = minusOnly
        noBits =
            if getActivatedNoBits then
                Some((plus.Clone() :?> BitArray).Or(minus).Not())
            else
                None
    }

let private layerOutputs (layerWeights: LayerWeights) (inputBits: NodeBits) : NodeBits =
    layerWeights
    |> Array.map (fun nodeWeights ->
        let activatedBits = getActivatedWeightBits false inputBits nodeWeights

        (((bitArrayPopCount activatedBits.both) <<< 1)
         + bitArrayPopCount activatedBits.plusOnly) > bitArrayPopCount activatedBits.minusOnly)
    |> BitArray

let maxByteIndex (xs: array<byte>) : int =
    xs |> Array.indexed |> Array.maxBy snd |> fst

let getLoss (finalBytes: array<byte>) (y: int) : float =
    let idealNorm: array<float> =
        Array.init finalBytes.Length (fun i -> if i = y then 1. else 0.)

    let maxByte = Array.max finalBytes

    if maxByte = 0uy then
        1.0
    else
        finalBytes
        |> Array.map (fun x -> float x / float maxByte)
        |> Array.zip idealNorm
        // mean absolute error
        |> Array.averageBy (fun (ideal, final) -> abs (final - ideal))

// effectful function
let exciteActivatedNodeWeights (inputBits: NodeBits) (nodeWeights: NodeWeights) : unit =
    // TODO
    ()

// effectful function
let inhibitActivatedNodeWeights (inputBits: NodeBits) (nodeWeights: NodeWeights) : unit =
    // TODO
    ()

let mutateLayerWeights
    (wasCorrect: bool)
    (inputBits: NodeBits)
    (outputBits: NodeBits)
    (layerWeights: LayerWeights)
    =
    layerWeights
    |> Array.mapi (fun i nodeWeights ->
        let wasNodeTriggered = outputBits[i]


        // TODO !!!!!!!!!!!!!!!!!!!


        let activatedPlusBits = (plusBits.Clone() :?> BitArray).And(inputBits)
        let activatedMinusBits = (minusBits.Clone() :?> BitArray).And(inputBits)

        let activatedBothBits =
            (activatedPlusBits.Clone() :?> BitArray).And(activatedMinusBits)

        let activatedNoBits =
            (activatedPlusBits.Clone() :?> BitArray).Or(activatedMinusBits).Not()

        let activatedPlusBitsOnly =
            (activatedPlusBits.Clone() :?> BitArray).Xor(activatedBothBits)

        let activatedMinusBitsOnly =
            (activatedMinusBits.Clone() :?> BitArray).Xor(activatedBothBits)

        if wasCorrect then
            if wasNodeTriggered then // excitation

                // remove single activated minus bits
                minusBits.Xor(activatedMinusBitsOnly) |> ignore
                // turn activated plus only bits into both bits
                minusBits.Or(activatedPlusBitsOnly) |> ignore
                // turn activated no bits into plus only bits
                plusBits.Or(activatedNoBits)

            else // inhibition

                // remove single activated plus bits
                plusBits.Xor(activatedPlusBitsOnly) |> ignore
                // turn activated both bits into plus only bits
                minusBits.Xor(activatedBothBits) |> ignore
                // turn activated no bits into minus only bits
                minusBits.Or(activatedNoBits)

        else if wasNodeTriggered then // inhibition

            // remove single activated plus bits
            plusBits.Xor(activatedPlusBitsOnly) |> ignore
            // turn activated both bits into plus only bits
            minusBits.Xor(activatedBothBits) |> ignore
            // turn activated no bits into minus only bits
            minusBits.Or(activatedNoBits)

        else // excitation

            // remove single activated minus bits
            minusBits.Xor(activatedMinusBitsOnly) |> ignore
            // turn activated plus only bits into both bits
            minusBits.Or(activatedPlusBitsOnly) |> ignore
            // turn activated no bits into plus only bits
            plusBits.Or(activatedNoBits)

    )

let private rowFit (model: Model) (xs: NodeBits) (y: int) : Model =
    let inputLayerBits = layerOutputs model.inputLayerWeights xs

    // intermediate outputs = input layer bits (included by Array.scan) + hidden layers bits
    model.lastIntermediateOutputs <-
        model.hiddenLayersWeights
        |> Array.scan
            (fun layerBits layerWeights -> layerOutputs layerWeights layerBits)
            inputLayerBits

    let finalBits =
        layerOutputs model.outputLayerWeights (Array.last model.lastIntermediateOutputs)

    let finalBytes: array<byte> = Array.zeroCreate (finalBits.Count / 8)
    finalBits.CopyTo(finalBytes, 0)

    let answer = maxByteIndex finalBytes
    let isCorrect = (answer = y) && finalBytes[answer] > 0uy
    let teachLayer = mutateLayerWeights isCorrect

    model.inputLayerWeights |> teachLayer xs inputLayerBits |> ignore

    model.hiddenLayersWeights
    |> Array.map2 (fun (i, o) w -> teachLayer i o w) (Array.pairwise model.lastIntermediateOutputs)
    |> ignore

    model.outputLayerWeights
    |> teachLayer (Array.last model.lastIntermediateOutputs) finalBits
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

        model.lastEpochTotalLoss <- 0.0
        model.lastEpochTotalCorrect <- 0
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
