module Graia

// disabling incomplete pattern match warnings (for getActiveWeightBits `asked` argument)
#nowarn "25"

open System.Collections
open System.Numerics
open System

let VERSION = "0.0.1"
printfn $"🌄 Graia v{VERSION}"

type Config = {
    inputBits: int
    outputs: int
    layerNodes: int
    layers: int
    thresholdRatio: int
    seed: int option
}

type History = {
    loss: array<float>
    accuracy: array<float>
}

type LayerBits = BitArray
// excitatory bits * inhibitory bits
type NodeWeights = BitArray * BitArray

type LayerWeights = array<NodeWeights>


//! waiting for native bitArray PopCount https://github.com/dotnet/runtime/issues/104299
let bitArrayPopCount (ba: BitArray) : int =
    // 32 = 2^5 (BitOperations.PopCount only works with integers)
    let uint32s: array<uint32> = Array.zeroCreate ((ba.Count >>> 5) + 1)
    ba.CopyTo(uint32s, 0)
    uint32s |> Array.sumBy BitOperations.PopCount

type WeightPairKind =
    | Plus
    | Minus
    | Both
    | PlusOnly
    | MinusOnly
    | NoBits

let getWeightBitsWithActiveInput
    (asked: WeightPairKind array)
    (inputBits: LayerBits)
    ((plusWeightBits, minusWeightBits): NodeWeights)
    : BitArray array =
    let plus = (plusWeightBits.Clone() :?> BitArray).And(inputBits)
    let minus = (minusWeightBits.Clone() :?> BitArray).And(inputBits)
    let both = (plus.Clone() :?> BitArray).And(minus)

    asked
    |> Array.map (function
        | Plus -> plus
        | Minus -> minus
        | Both -> both
        | PlusOnly -> (plus.Clone() :?> BitArray).Xor(both)
        | MinusOnly -> (minus.Clone() :?> BitArray).Xor(both)
        | NoBits -> (inputBits.Clone() :?> BitArray).Xor(plus).Xor(minus))

let layerOutputsForTR
    (thresholdRatio: int)
    (layerWeights: LayerWeights)
    (inputBits: LayerBits)
    : LayerBits =
    let threshold =
        if thresholdRatio = 0 then
            0
        else
            inputBits.Count / thresholdRatio

    layerWeights
    |> Array.Parallel.map (fun nodeWeights ->
        let [| plus; both; minusOnly |] =
            getWeightBitsWithActiveInput [| Plus; Both; MinusOnly |] inputBits nodeWeights

        // activation condition
        (bitArrayPopCount plus + bitArrayPopCount both) - bitArrayPopCount minusOnly > threshold

    )
    |> BitArray

// effectful function
let exciteNodeWeightsWithActiveInput (inputBits: LayerBits) (nodeWeights: NodeWeights) : unit =
    let [| plusOnly; minusOnly; noBits |] =
        getWeightBitsWithActiveInput [| PlusOnly; MinusOnly; NoBits |] inputBits nodeWeights

    let (plusWeightBits, minusWeightBits) = nodeWeights

    // remove single minus bits
    minusWeightBits.Xor(minusOnly) |> ignore
    // turn plus only bits into both bits
    minusWeightBits.Or(plusOnly) |> ignore
    // turn no bits into plus only bits
    plusWeightBits.Or(noBits) |> ignore

// effectful function
let inhibitNodeWeightsWithActiveInput (inputBits: LayerBits) (nodeWeights: NodeWeights) : unit =
    let [| plusOnly; noBits; both |] =
        getWeightBitsWithActiveInput [| PlusOnly; NoBits; Both |] inputBits nodeWeights

    let (plusWeightBits, minusWeightBits) = nodeWeights

    // remove single plus bits
    plusWeightBits.Xor(plusOnly) |> ignore
    // order is important!
    // turn no bits into minus only bits
    minusWeightBits.Or(noBits) |> ignore
    // turn both bits into plus only bits
    minusWeightBits.Xor(both) |> ignore

let mutateLayerWeights
    (wasGood: bool)
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
        if wasGood then
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

let getOutputs (outputBits: LayerBits) : array<int> =
    let output32BitsPools: array<uint32> = Array.zeroCreate (outputBits.Count / 32)
    outputBits.CopyTo(output32BitsPools, 0)
    // outputs maximum value possible is 32
    output32BitsPools |> Array.map BitOperations.PopCount

type Prediction = {
    intermediateOutputBits: array<LayerBits>
    outputBits: LayerBits
} with

    member this.outputs = getOutputs this.outputBits
    member this.result = maxIntIndex this.outputs

type Evaluation = { isCorrect: bool; loss: float }

let evaluate (prediction: Prediction) (answer: int) : Evaluation = {
    isCorrect = prediction.result = answer
    loss = getLoss prediction.outputs answer
}

type Model = {
    graiaVersion: string
    config: Config
    mutable inputLayerWeights: LayerWeights
    mutable hiddenLayersWeights: array<LayerWeights>
    mutable outputLayerWeights: LayerWeights
    lastPrediction: Prediction
    lastAnswer: int
    history: History
}

let init (config: Config) : Model =
    let {
            inputBits = inputBitsNb
            outputs = outputsNb
            layerNodes = layerNodesNb
            layers = layersNb
            seed = seed
        } =
        config

    let outputBitsNb = outputsNb * 32

    let parametersNb: int =
        (inputBitsNb * layerNodesNb)
        + (layerNodesNb * layerNodesNb * (layersNb - 1))
        + (layerNodesNb * outputBitsNb)

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
        inputLayerWeights = randomLayerWeights inputBitsNb layerNodesNb
        hiddenLayersWeights =
            Array.init (layersNb - 1) (fun _ -> randomLayerWeights layerNodesNb layerNodesNb)
        outputLayerWeights = randomLayerWeights layerNodesNb outputBitsNb
        lastPrediction = {
            intermediateOutputBits = Array.init layersNb (fun _ -> BitArray(layerNodesNb))
            outputBits = BitArray(outputBitsNb)
        }
        lastAnswer = -1
        history = { loss = [||]; accuracy = [||] }
    }

    printfn $"🌄 Graia model with {parametersNb} parameters ready."
    model

let predict (model: Model) (xs: LayerBits) : Prediction =
    let layerOutputs = layerOutputsForTR model.config.thresholdRatio

    let inputLayerBits = layerOutputs model.inputLayerWeights xs

    // intermediate outputs = input layer bits (included by Array.scan) + hidden layers bits
    let intermediateOutputBits =
        model.hiddenLayersWeights
        |> Array.scan
            (fun layerBits layerWeights -> layerOutputs layerWeights layerBits)
            inputLayerBits

    let outputBits =
        layerOutputs model.outputLayerWeights (Array.last intermediateOutputBits)

    {
        intermediateOutputBits = intermediateOutputBits
        outputBits = outputBits
    }

let teachModel (isGood: bool) (model: Model) (inputBits: LayerBits) (pred: Prediction) : unit =
    let teachLayer = mutateLayerWeights isGood

    model.inputLayerWeights
    |> teachLayer inputBits pred.intermediateOutputBits[0]
    |> ignore

    model.hiddenLayersWeights
    |> Array.map2 (fun (i, o) w -> teachLayer i o w) (Array.pairwise pred.intermediateOutputBits)
    |> ignore

    model.outputLayerWeights
    |> teachLayer (Array.last pred.intermediateOutputBits) pred.outputBits
    |> ignore

type EpochData = { totalLoss: float; totalCorrect: int }

let rowFit
    (model: Model, data: EpochData)
    (inputBits: LayerBits)
    (labelIndex: int)
    : Model * EpochData =
    let pred: Prediction = predict model inputBits

    let { loss = loss; isCorrect = isCorrect } = evaluate pred labelIndex
    let { loss = previousLoss } = evaluate model.lastPrediction model.lastAnswer

    let isBetter = loss < previousLoss
    teachModel isBetter model inputBits pred

    {
        model with
            lastPrediction = pred
            lastAnswer = labelIndex
    },
    {
        totalLoss = data.totalLoss + loss
        totalCorrect =
            if isCorrect then
                data.totalCorrect + 1
            else
                data.totalCorrect
    }

let rec fit
    (inputBitsRows: array<LayerBits>)
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

        let epochModel, epochData =
            Array.fold2
                rowFit
                (model, { totalLoss = 0.0; totalCorrect = 0 })
                inputBitsRows
                labelIndexRows

        fit inputBitsRows labelIndexRows (epochs - 1) {
            epochModel with
                history = {
                    loss =
                        Array.append model.history.loss [|
                            epochData.totalLoss / float inputBitsRows.Length
                        |]
                    accuracy =
                        Array.append model.history.accuracy [|
                            (float epochData.totalCorrect) / float inputBitsRows.Length
                        |]
                }
        }
