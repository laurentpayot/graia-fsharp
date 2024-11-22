module Graia

open System
open System.Collections


let VERSION = "0.0.1"
printfn $"🌄 Graia v{VERSION}"


type Config = {
    inputBools: int
    outputs: int
    layerNodes: int
    layers: int
    learningRate: float
    threshold: int
    seed: int option
}

type History = {
    loss: array<float>
    accuracy: array<float>
}

type Outputs = array<byte>
type Activations = array<bool>
// weights are signed bytes from -126 to 126
type NodeWeights = array<sbyte>

type LayerWeights = array<NodeWeights>


let getActivationsAbove
    (threshold: int)
    (layerWeights: LayerWeights)
    (inputBools: Activations)
    : Activations =
    layerWeights
    |> Array.Parallel.map (fun nodeWeights ->
        (0, inputBools, nodeWeights)
        |||> Array.fold2 (fun sum isActive weight ->
            if isActive = false || weight = 0y then
                sum
            else
                sum + int weight)

        // activation function
        |> fun sum -> sum > threshold

    )

// effectful function
let exciteActivatedNodeWeights
    (step: sbyte)
    (inputBools: Activations)
    (nodeWeights: NodeWeights)
    : unit =
    Array.iteri2
        (fun i isActive weight ->
            if isActive then
                nodeWeights[i] <- min 126y (weight + step))
        inputBools
        nodeWeights

// effectful function
let inhibitActivatedNodeWeights
    (step: sbyte)
    (inputBools: Activations)
    (nodeWeights: NodeWeights)
    : unit =
    Array.iteri2
        (fun i isActive weight ->
            if isActive then
                nodeWeights[i] <- max -126y (weight - step))
        inputBools
        nodeWeights

let mutateLayerWeights
    (wasGood: bool)
    (step: sbyte)
    (inputBools: Activations)
    (outputBools: Activations)
    (layerWeights: LayerWeights)
    : unit =
    layerWeights
    |> Array.Parallel.mapi (fun i nodeWeights ->
        let wasNodeTriggered = outputBools[i]

        (step, inputBools, nodeWeights)
        |||>

        //  Hebbian learning rule
        if wasGood then
            if wasNodeTriggered then
                // correct + node triggered = excite active inputs
                exciteActivatedNodeWeights
            else
                // correct + node not triggered = inhibit active inputs
                inhibitActivatedNodeWeights
        else if wasNodeTriggered then
            // incorrect + node triggered = inhibit active inputs
            inhibitActivatedNodeWeights
        else
            // incorrect + node not triggered = excite active inputs
            exciteActivatedNodeWeights

    )

    |> ignore

let maxOutputIndex (xs: Outputs) : int =
    xs |> Array.indexed |> Array.maxBy snd |> fst

let getLoss (outputs: Outputs) (labelIndex: int) : float =
    let idealNorm: array<float> =
        Array.init outputs.Length (fun i -> if i = labelIndex then 1. else 0.)

    let maxOutput = Array.max outputs

    if maxOutput = 0uy then
        1.0
    else
        outputs
        |> Array.map (fun x -> float x / float maxOutput)
        |> Array.zip idealNorm
        // mean absolute error
        |> Array.averageBy (fun (ideal, final) -> abs (final - ideal))

let getOutputs (outputBools: Activations) : Outputs =
    let outputs: Outputs = Array.zeroCreate (outputBools.Length / 8)
    BitArray(outputBools).CopyTo(outputs, 0)
    outputs


type Prediction = {
    intermediateActivations: array<Activations>
    outputActivations: Activations
} with

    member this.outputs = getOutputs this.outputActivations
    member this.result = maxOutputIndex this.outputs

type Evaluation = { isCorrect: bool; loss: float }

let evaluate (prediction: Prediction) (answer: int) : Evaluation = {
    isCorrect = prediction.result = answer
    loss = getLoss prediction.outputs answer
}

type Model = {
    graiaVersion: string
    config: Config
    inputLayerWeights: LayerWeights
    hiddenLayersWeights: array<LayerWeights>
    outputLayerWeights: LayerWeights
    lastPrediction: Prediction
    lastAnswer: int
    history: History
}

let init (config: Config) : Model =
    let {
            inputBools = inputBoolsNb
            outputs = outputsNb
            layerNodes = layerNodesNb
            layers = layersNb
            seed = seed
        } =
        config

    let outputBoolsNb = outputsNb * 8

    let billionParameters =
        (inputBoolsNb * layerNodesNb)
        + (layerNodesNb * layerNodesNb * (layersNb - 1))
        + (layerNodesNb * outputBoolsNb)
        |> float
        |> fun n -> n / 1_000_000_000.

    let rnd =
        match seed with
        | Some seed -> Random(seed)
        | None -> Random()

    let randomWeights (length: int) : NodeWeights =
        Array.init length (fun _ -> rnd.Next(-126, 127) |> sbyte)

    let randomLayerWeights (inputDim: int) (outputDim: int) : LayerWeights =
        Array.init outputDim (fun _ -> randomWeights inputDim)

    let model = {
        graiaVersion = VERSION
        config = config
        inputLayerWeights = randomLayerWeights inputBoolsNb layerNodesNb
        hiddenLayersWeights =
            Array.init (layersNb - 1) (fun _ -> randomLayerWeights layerNodesNb layerNodesNb)
        outputLayerWeights = randomLayerWeights layerNodesNb outputBoolsNb
        lastPrediction = {
            intermediateActivations = Array.zeroCreate layersNb
            outputActivations = Array.zeroCreate outputBoolsNb
        }
        lastAnswer = -1
        history = { loss = [||]; accuracy = [||] }
    }

    printfn $"🌄 Graia model with {billionParameters} billion parameters ready."
    model

let predict (model: Model) (xs: Activations) : Prediction =
    let layerOutputs = getActivationsAbove model.config.threshold

    let inputActivations = layerOutputs model.inputLayerWeights xs

    // intermediate outputs = input layer bits (included by Array.scan) + hidden layers bits
    let intermediateActivations =
        model.hiddenLayersWeights
        |> Array.scan
            (fun layerBits layerWeights -> layerOutputs layerWeights layerBits)
            inputActivations

    let outputActivations =
        layerOutputs model.outputLayerWeights (Array.last intermediateActivations)

    {
        intermediateActivations = intermediateActivations
        outputActivations = outputActivations
    }

let teachModel (model: Model) (loss: float) (inputBools: Activations) (pred: Prediction) : unit =
    let { loss = previousLoss } = evaluate model.lastPrediction model.lastAnswer
    let isBetter = loss < previousLoss
    let step = 126. * loss * model.config.learningRate |> sbyte
    let teachLayer = mutateLayerWeights isBetter step

    model.inputLayerWeights
    |> teachLayer inputBools pred.intermediateActivations[0]
    |> ignore

    model.hiddenLayersWeights
    |> Array.map2 (fun (i, o) w -> teachLayer i o w) (Array.pairwise pred.intermediateActivations)
    |> ignore

    model.outputLayerWeights
    |> teachLayer (Array.last pred.intermediateActivations) pred.outputActivations
    |> ignore

type EpochData = { totalLoss: float; totalCorrect: int }

let rowFit
    (model: Model, data: EpochData)
    (inputBools: Activations)
    (labelIndex: int)
    : Model * EpochData =
    let pred: Prediction = predict model inputBools

    let { loss = loss; isCorrect = isCorrect } = evaluate pred labelIndex

    teachModel model loss inputBools pred

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
    (inputBoolsRows: array<Activations>)
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
                inputBoolsRows
                labelIndexRows

        fit inputBoolsRows labelIndexRows (epochs - 1) {
            epochModel with
                history = {
                    loss =
                        Array.append model.history.loss [|
                            epochData.totalLoss / float inputBoolsRows.Length
                        |]
                    accuracy =
                        Array.append model.history.accuracy [|
                            (float epochData.totalCorrect) / float inputBoolsRows.Length
                        |]
                }
        }
