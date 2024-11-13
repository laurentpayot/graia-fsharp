module Tools

open System.IO
open System
open System.Collections

open Plotly.NET
open FSharpPlus

open Graia
open Microsoft.DotNet.Interactive


type ByteRow = array<byte>

let loadMnistCsv (path: string) : array<int> * array<ByteRow> =
    File.ReadAllLines(path)
    // remove header row
    |> Array.skip 1
    // turning seq into array for performance reasons https://theburningmonk.com/2012/06/f-speed-test-iter-and-map-operations-with-array-vs-list/
    |> Array.map (String.split [ "," ] >> Array.ofSeq)
    |> Array.fold
        (fun acc row ->
            let label = Array.head row |> int

            let data =
                row
                // remove label column
                |> Array.skip 1
                |> Array.map byte

            Array.append acc [| (label, data) |])
        [||]
    |> Array.unzip

let byteRowsToBitArraysBinarized (threshold: byte) (byteRows: array<ByteRow>) : array<BitArray> =
    byteRows
    |> Array.map ((Seq.map (fun v -> v >= threshold)) >> Array.ofSeq >> BitArray)

let byteRowsToBitArrays (byteRows: array<ByteRow>) : array<BitArray> =
    byteRows |> Array.map (Array.ofSeq >> BitArray)

let layerWeightsToMatrix (layerWeights: LayerWeights) : array<array<int>> =
    layerWeights
    |> Array.map (fun (plusBits, minusBits) ->
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
        |> Seq.map (fun x ->
            if threshold = 0uy then int x
            else if x >= threshold then 1
            else 0)
        |> Seq.chunkBySize 28
        |> Seq.rev

    let chart =
        Chart.Heatmap(image, ColorScale = StyleParam.Colorscale.Greys, ShowScale = false)
        |> Chart.withSize (100., 100.)
        |> Chart.withMarginSize (0., 0., 0., 0.)

    chart.Display()

let showRowDigit (row: ByteRow) : DisplayedValue = showRowDigitBinarized 0uy row

let showLayerWeights (title: string) (layerWeights: LayerWeights) : DisplayedValue =
    let matrix = layerWeights |> layerWeightsToMatrix

    let chart =
        Chart.Heatmap(matrix, ColorScale = StyleParam.Colorscale.Picnic)
        |> Chart.withTitle title
        |> Chart.withXAxisStyle ("Inputs")
        |> Chart.withYAxisStyle ("Nodes")
        |> Chart.withColorBarStyle (TitleText = "Weights")
        |> Chart.withSize (1000., 240.)
        |> Chart.withMarginSize (80., 10., 50., 10.)

    chart.Display()

let bitArraysToMatrix (bitArrays: array<BitArray>) : array<array<int>> =
    bitArrays
    |> Array.map (fun ba ->
        let bools: array<Boolean> = Array.zeroCreate ba.Count
        ba.CopyTo(bools, 0)
        Array.map (fun bool -> if bool then 1 else 0) bools)

let showIntermediateOutputs (title: string) (outputs: array<BitArray>) : DisplayedValue =
    let matrix = bitArraysToMatrix outputs

    let chart =
        Chart.Heatmap(
            zData = matrix,
            ColorScale = StyleParam.Colorscale.Greys,
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

let showOutputs (title: string) (outputs: array<byte>) : DisplayedValue =
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
