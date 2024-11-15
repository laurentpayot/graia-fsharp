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
        Array.init outputDim (fun _ ->
            let plusBits = randomBitArray inputDim
            let minusBits = randomBitArray inputDim
            let plusBits' =(plusBits.Clone() :?> BitArray).Not()
            // make minus bits false when plus bits are true
            minusBits.And(plusBits') |> ignore
            plusBits, minusBits

        )

    let model = {
        graiaVersion = VERSION
        config = config
        inputLayerWeights = randomLayerWeights inputBits layerNodes
        hiddenLayersWeights =
            Array.init (layers - 1) (fun _ -> randomLayerWeights layerNodes layerNodes)
        outputLayerWeights = randomLayerWeights layerNodes outputBits
        lastOutputs = [||]
        lastIntermediateOutputs = Array.init layers (fun _ -> BitArray(layerNodes))
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

type WeightPairKind =
    | Plus
    | Minus
    | Both
    | PlusOnly
    | MinusOnly
    | NoBits

let getWeightBitsWithInput
    (isInputActive: bool)
    (asked: WeightPairKind array)
    (inputBits: NodeBits)
    ((plusWeightBits, minusWeightBits): NodeWeights)
    : BitArray array =
    let inputBits' = if isInputActive then inputBits else (inputBits.Clone() :?> BitArray).Not()
    let plus = (plusWeightBits.Clone() :?> BitArray).And(inputBits')
    let minus = (minusWeightBits.Clone() :?> BitArray).And(inputBits')
    let both = (plus.Clone() :?> BitArray).And(minus)

    asked
    |> Array.map (function
        | Plus -> plus
        | Minus -> minus
        | Both -> both
        | PlusOnly -> (plus.Clone() :?> BitArray).Xor(both)
        | MinusOnly -> (minus.Clone() :?> BitArray).Xor(both)
        | NoBits -> (inputBits'.Clone() :?> BitArray).Xor(plus).Xor(minus))

// let getActiveWeightBits = getWeightBitsWithInput true
// let getInactiveWeightBits = getWeightBitsWithInput false

let layerOutputs (layerWeights: LayerWeights) (inputBits: NodeBits) : NodeBits =
    layerWeights
    |> Array.Parallel.map (fun nodeWeights ->
        let [| plus; minus; |] =
            getWeightBitsWithInput true [| Plus; Minus; |] inputBits nodeWeights

        // activation condition
        bitArrayPopCount plus > bitArrayPopCount minus

    )
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
let exciteNodeWeightsWithInput (isInputActive: bool) (inputBits: NodeBits) (nodeWeights: NodeWeights) : unit =
    let [| plusOnly; minusOnly; noBits |] =
        getWeightBitsWithInput isInputActive [| PlusOnly; MinusOnly; NoBits |] inputBits nodeWeights

    let (plusWeightBits, minusWeightBits) = nodeWeights

    // remove single minus bits
    minusWeightBits.Xor(minusOnly) |> ignore
    // turn plus only bits into both bits
    minusWeightBits.Or(plusOnly) |> ignore
    // turn no bits into plus only bits
    plusWeightBits.Or(noBits) |> ignore

let exciteNodeWeightsWithActiveInput (inputBits: NodeBits) (nodeWeights: NodeWeights) : unit =
    let [| minusOnly; noBits |] =
        getWeightBitsWithInput true [| MinusOnly; NoBits |] inputBits nodeWeights

    let (plusWeightBits, minusWeightBits) = nodeWeights

    // remove single minus bits
    minusWeightBits.Xor(minusOnly) |> ignore
    // turn no bits into plus only bits
    plusWeightBits.Or(noBits) |> ignore

// effectful function
let inhibitNodeWeightsWithInput (isInputActive: bool) (inputBits: NodeBits) (nodeWeights: NodeWeights) : unit =
    let [| plusOnly; noBits; both |] =
        getWeightBitsWithInput isInputActive [| PlusOnly; NoBits; Both |] inputBits nodeWeights

    let (plusWeightBits, minusWeightBits) = nodeWeights

    // remove single plus bits
    plusWeightBits.Xor(plusOnly) |> ignore
    // order is important!
    // turn no bits into minus only bits
    minusWeightBits.Or(noBits) |> ignore
    // turn both bits into plus only bits
    minusWeightBits.Xor(both) |> ignore

let inhibitNodeWeightsWithActiveInput (inputBits: NodeBits) (nodeWeights: NodeWeights) : unit =
    let [| plusOnly; noBits |] =
        getWeightBitsWithInput true [| PlusOnly; NoBits; |] inputBits nodeWeights

    let (plusWeightBits, minusWeightBits) = nodeWeights

    // remove single plus bits
    plusWeightBits.Xor(plusOnly) |> ignore
    // turn no bits into minus only bits
    minusWeightBits.Or(noBits) |> ignore
    // turn both bits into plus only bits
    // minusWeightBits.Xor(both) |> ignore

let mutateLayerWeights
    (wasCorrect: bool)
    (inputBits: NodeBits)
    (outputBits: NodeBits)
    (layerWeights: LayerWeights)
    : unit
    =
    layerWeights
    |> Array.Parallel.mapi (fun i nodeWeights ->
        let wasNodeTriggered = outputBits[i]

        //  Hebbian learning rule
        if wasCorrect then
            if wasNodeTriggered then
                // correct + node triggered = excite active inputs
                exciteNodeWeightsWithActiveInput inputBits nodeWeights
            else
                // correct + node not triggered = inhibit active inputs
                inhibitNodeWeightsWithActiveInput inputBits nodeWeights
        else if wasNodeTriggered then
            // incorrect + node triggered = inhibit active inputs
            inhibitNodeWeightsWithActiveInput inputBits nodeWeights
        else
            // incorrect + node not triggered = excite active inputs
            exciteNodeWeightsWithActiveInput inputBits nodeWeights

    )
    |> ignore

let rowFit (model: Model) (xs: NodeBits) (y: int) : Model =
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
    model.lastOutputs <- finalBytes

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
