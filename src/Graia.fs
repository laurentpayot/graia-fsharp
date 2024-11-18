module Graia

// disabling incomplete pattern match warnings (for getActiveWeightBits `asked` argument)
#nowarn "25"

open System.Collections
open System.Numerics
open System

let VERSION = "0.0.1"
printfn $"🌄 Graia v{VERSION}"

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

// for inference
type LayerBytes = array<byte>
// for training
type LayerBits = BitArray

// excitatory bits * inhibitory bits
type NodeWeights = BitArray * BitArray

type LayerWeights = array<NodeWeights>

type Model = {
    graiaVersion: string
    config: Config
    mutable inputLayerWeights: LayerWeights
    mutable hiddenLayersWeights: array<LayerWeights>
    mutable outputLayerWeights: LayerWeights
    mutable lastOutputs: LayerBytes
    mutable lastIntermediateOutputs: array<LayerBytes>
    mutable lastEpochTotalLoss: float
    mutable lastEpochTotalCorrect: int
    mutable history: History
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

    let outputBits = outputs * 32

    let parametersNb: int =
        (inputs * layerNodes)
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
        Array.init outputDim (fun _ ->
            let plusBits = randomBitArray inputDim
            let minusBits = randomBitArray inputDim
            let plusBits' = (plusBits.Clone() :?> BitArray).Not()
            // make minus bits false when plus bits are true
            minusBits.And(plusBits') |> ignore
            plusBits, minusBits

        )

    let model = {
        graiaVersion = VERSION
        config = config
        inputLayerWeights = randomLayerWeights inputs layerNodes
        hiddenLayersWeights =
            Array.init (layers - 1) (fun _ -> randomLayerWeights layerNodes layerNodes)
        outputLayerWeights = randomLayerWeights layerNodes outputBits
        lastOutputs = [||]
        lastIntermediateOutputs = Array.init layers (fun _ -> Array.zeroCreate layerNodes)
        lastEpochTotalLoss = 0.0
        lastEpochTotalCorrect = 0
        history = { loss = [||]; accuracy = [||] }
    }

    printfn $"🌄 Graia model with {parametersNb} parameters ready."
    model


//! waiting for native bitArray PopCount https://github.com/dotnet/runtime/issues/104299
let bitArrayPopCount (ba: BitArray) : int =
    // 32 = 2^5 (BitOperations.PopCount only works with integers)
    let uint32s: array<uint32> = Array.zeroCreate ((ba.Count >>> 5) + 1)
    ba.CopyTo(uint32s, 0)
    uint32s |> Array.sumBy BitOperations.PopCount

let layerOutputs (layerWeights: LayerWeights) (inputBytes: LayerBytes) : LayerBytes =
    layerWeights
    |> Array.Parallel.map (fun (plusWeightBits, minusWeightBits) ->

        let sum =
            inputBytes
            |> Array.indexed
            |> (Array.fold
                    (fun sum (i, value) ->
                        if plusWeightBits[i] then sum + int value
                        else if minusWeightBits[i] then sum + int value
                        else sum)
                    0)
        // activation condition
        sum |> max 0 |> min 255 |> byte)

let maxIntIndex (xs: array<int>) : int =
    xs |> Array.indexed |> Array.maxBy snd |> fst

let getLoss (outputs: array<int>) (labelIndex: int) : float =
    let idealNorm: array<float> =
        Array.init outputs.Length (fun i -> if i = labelIndex then 1. else 0.)

    let maxOutput = Array.max outputs

    if maxOutput = 0 then
        1.0
    else
        outputs
        |> Array.map (fun x -> float x / float maxOutput)
        |> Array.zip idealNorm
        // mean absolute error
        |> Array.averageBy (fun (ideal, final) -> abs (final - ideal))

// effectful function
let exciteNodeWeightsWithActiveInput (inputBits: LayerBits) (nodeWeights: NodeWeights) : unit =
    let (plusWeightBits, minusWeightBits) = nodeWeights
    let plus' = (plusWeightBits.Clone() :?> BitArray).And(inputBits)
    let minus' = (minusWeightBits.Clone() :?> BitArray).And(inputBits)
    let noBits' = (inputBits.Clone() :?> BitArray).Xor(plus').Xor(minus')

    // remove active minus bits
    minusWeightBits.Xor(minus') |> ignore
    // turn "active no bits" into plus only bits
    plusWeightBits.Or(noBits') |> ignore

// effectful function
let inhibitNodeWeightsWithActiveInput (inputBits: LayerBits) (nodeWeights: NodeWeights) : unit =
    let (plusWeightBits, minusWeightBits) = nodeWeights
    let plus' = (plusWeightBits.Clone() :?> BitArray).And(inputBits)
    let minus' = (minusWeightBits.Clone() :?> BitArray).And(inputBits)
    let noBits' = (inputBits.Clone() :?> BitArray).Xor(plus').Xor(minus')

    // remove active plus bits
    plusWeightBits.Xor(plus') |> ignore
    // turn "active no bits" into minus only bits
    minusWeightBits.Or(noBits') |> ignore

let mutateLayerWeights
    (wasCorrect: bool)
    (inputBits: LayerBits)
    (outputBits: LayerBits)
    (layerWeights: LayerWeights)
    : unit =
    layerWeights
    |> Array.Parallel.mapi (fun i nodeWeights ->
        let wasNodeTriggered = outputBits[i]

        (inputBits, nodeWeights)
        ||>
        //  Hebbian learning rule
        if wasCorrect then
            if wasNodeTriggered then
                // correct + node triggered = excite active inputs
                exciteNodeWeightsWithActiveInput
            else
                // correct + node not triggered = inhibit active inputs
                inhibitNodeWeightsWithActiveInput
        else if wasNodeTriggered then
            // incorrect + node triggered = inhibit active inputs
            inhibitNodeWeightsWithActiveInput
        else
            // incorrect + node not triggered = excite active inputs
            exciteNodeWeightsWithActiveInput

    )
    |> ignore

let rowFit (model: Model) (xs: LayerBytes) (labelIndex: int) : Model =
    let inputLayerBits = layerOutputs model.inputLayerWeights xs

    // intermediate outputs = input layer bits (included by Array.scan) + hidden layers bits
    model.lastIntermediateOutputs <-
        model.hiddenLayersWeights
        |> Array.scan
            (fun layerBits layerWeights -> layerOutputs layerWeights layerBits)
            inputLayerBits

    let finalBits =
        layerOutputs model.outputLayerWeights (Array.last model.lastIntermediateOutputs)

    let final32BitsSections: array<uint32> = Array.zeroCreate (finalBits.Count / 32)
    finalBits.CopyTo(final32BitsSections, 0)
    // outputs max value possible is 32
    let outputs: array<int> = final32BitsSections |> Array.map BitOperations.PopCount

    let answer = maxIntIndex outputs
    let isCorrect = (answer = labelIndex) && outputs[answer] > 0
    let loss = getLoss outputs labelIndex
    let teachLayer = mutateLayerWeights isCorrect

    model.inputLayerWeights |> teachLayer xs inputLayerBits |> ignore

    model.hiddenLayersWeights
    |> Array.map2 (fun (i, o) w -> teachLayer i o w) (Array.pairwise model.lastIntermediateOutputs)
    |> ignore

    model.outputLayerWeights
    |> teachLayer (Array.last model.lastIntermediateOutputs) finalBits
    |> ignore

    model.lastOutputs <- outputs
    model.lastEpochTotalLoss <- model.lastEpochTotalLoss + loss

    model.lastEpochTotalCorrect <-
        if isCorrect then
            model.lastEpochTotalCorrect + 1
        else
            model.lastEpochTotalCorrect

    model

let rec fit
    (xsRows: array<LayerBytes>)
    (labelIndexRows: array<int>)
    (epochs: int)
    (model: Model)
    : Model =
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
        Array.fold2 rowFit model xsRows labelIndexRows |> ignore

        model.history <- {
            loss =
                Array.append model.history.loss [| model.lastEpochTotalLoss / float xsRows.Length |]
            accuracy =
                Array.append model.history.accuracy [|
                    (float model.lastEpochTotalCorrect) / float xsRows.Length
                |]
        }

        fit xsRows labelIndexRows (epochs - 1) model
