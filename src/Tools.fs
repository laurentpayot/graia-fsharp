module Tools

open System.IO
open System
open System.Collections

open Plotly.NET
open Plotly.NET.StyleParam
open Microsoft.DotNet.Interactive

open Graia


type ByteRow = array<byte>

// parallel treatment of lines does NOT ensure the order of (label, data) pairs to be preserved
let loadMnistCsvUnsafeOrder (path: string) : array<int> * array<ByteRow> =
    File.ReadAllLines(path)
    // remove header row
    |> Array.skip 1
    |> Array.Parallel.reduceBy
        (fun row ->
            let columns = row.Split(",")
            let label = Array.head columns |> int

            let data =
                columns
                // remove label column
                |> Array.skip 1
                |> Array.map byte

            [| (label, data) |])
        Array.append
    |> Array.unzip

let byteRowsToBitArraysBinarized (threshold: byte) (byteRows: array<ByteRow>) : array<BitArray> =
    byteRows
    |> Array.Parallel.map ((Array.map (fun v -> v >= threshold)) >> BitArray)

let byteRowsToBitArrays (byteRows: array<ByteRow>) : array<BitArray> =
    byteRows |> Array.Parallel.map BitArray

let layerWeightsToMatrix (layerWeights: LayerWeights) : array<array<int>> =
    layerWeights
    |> Array.Parallel.map (fun (plusBits, minusBits) ->
        let plusBools: array<Boolean> = Array.zeroCreate plusBits.Count
        plusBits.CopyTo(plusBools, 0)
        let minusBools: array<Boolean> = Array.zeroCreate minusBits.Count
        minusBits.CopyTo(minusBools, 0)

        Array.map2
            (fun plus minus ->
                match plus, minus with
                | true, true -> 2
                | true, false -> 1
                | false, false -> 0
                | false, true -> -1)
            plusBools
            minusBools)

let showRowDigitBinarized (threshold: byte) (row: ByteRow) : DisplayedValue =
    let image =
        row
        |> Array.Parallel.map (fun b ->
            let i = int b

            if threshold = 0uy then [| i; i; i |]
            else if b >= threshold then [| 255; 255; 255 |]
            else [| 0; 0; 0 |])
        |> Array.chunkBySize 28

    let chart =
        Chart.Image(image)
        |> Chart.withSize (100., 100.)
        |> Chart.withMarginSize (0., 0., 0., 0.)

    chart.Display()

let showRowDigit (row: ByteRow) : DisplayedValue = showRowDigitBinarized 0uy row

let showLayerWeights (title: string) (layerWeights: LayerWeights) : DisplayedValue =
    let matrix = layerWeights |> layerWeightsToMatrix
    let xSize = 80 + max 300 (matrix.Length / 4)
    let ySize = max 300 (matrix[0].Length / 4)

    let chart =
        Chart.Heatmap(matrix, ColorScale = Colorscale.Picnic)
        |> Chart.withTitle title
        |> Chart.withXAxisStyle ("Inputs")
        |> Chart.withYAxisStyle ("Nodes")
        |> Chart.withColorBarStyle (TitleText = "Weights")
        |> Chart.withSize (xSize, ySize)
        |> Chart.withMarginSize (80., 10., 50., 10.)

    chart.Display()

let bitArraysToMatrix (bitArrays: array<BitArray>) : array<array<int>> =
    bitArrays
    |> Array.Parallel.map (fun ba ->
        let bools: array<Boolean> = Array.zeroCreate ba.Count
        ba.CopyTo(bools, 0)
        Array.map (fun bool -> if bool then 1 else 0) bools)

let showIntermediateOutputs (title: string) (outputs: array<BitArray>) : DisplayedValue =
    let matrix = bitArraysToMatrix outputs

    let chart =
        Chart.Heatmap(
            zData = matrix,
            ColorScale = Colorscale.Blackbody,
            ShowScale = false,
            ReverseYAxis = true
        // rowNames = Array.map string [| 0 .. outputs.Length - 1 |]
        )
        |> Chart.withTitle title
        |> Chart.withXAxisStyle ("Nodes")
        |> Chart.withYAxisStyle ("Layers")
        |> Chart.withColorBarStyle (TitleText = "Outputs")
        |> Chart.withSize (1000., 100. + 20. * float outputs.Length)
        |> Chart.withMarginSize (80., 10., 50., 10.)

    chart.Display()

let showOutputs (title: string) (outputs: array<int>) : DisplayedValue =
    let intOutputs = Array.map int outputs
    let strLabels = Array.map string [| 0 .. outputs.Length - 1 |]

    let chart =
        Chart.Column(values = intOutputs, Keys = strLabels)
        |> Chart.withTitle title
        |> Chart.withXAxisStyle ("Digits")
        |> Chart.withYAxisStyle ("Values")
        |> Chart.withSize (1000., 250.)
        |> Chart.withMarginSize (80., 10., 50., 10.)

    chart.Display()

let showHistory (model: Model) : DisplayedValue =
    let accuracyChart =
        Chart.Line(
            xy = [
                for e in 1 .. model.history.accuracy.Length -> (e, model.history.accuracy[e - 1])
            ]
        )
        |> Chart.withTraceInfo (Name = "Accuracy")

    let lossChart =
        Chart.Line(
            xy = [ for e in 1 .. model.history.loss.Length -> (e, model.history.loss[e - 1]) ]
        )
        |> Chart.withTraceInfo (Name = "Loss")

    let chart =
        Chart.combine [ accuracyChart; lossChart ]
        |> Chart.withTitle $"Training History"
        |> Chart.withXAxisStyle ("Epochs")
        // |> Chart.withYAxisStyle ("Loss (MAE)")
        |> Chart.withSize (1000., 250.)
        |> Chart.withMarginSize (80., 10., 50., 10.)

    chart.Display()
