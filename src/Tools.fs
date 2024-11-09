module Tools

open System.IO
open System
open System.Collections

open FSharpPlus

open Graia


let loadMnistCsv (path: string) : array<int> * array<seq<byte>> =
    File.ReadAllText(path)
    |> String.split [ "\n" ]
    // stop at the last empty line
    |> Seq.takeWhile (not << String.IsNullOrWhiteSpace)
    // remove header row
    |> Seq.skip 1
    |> Seq.map (String.split [ "," ])
    |> fold
        (fun acc row ->

            let label = Seq.head row |> int

            let data =
                row
                // remove label column
                |> Seq.skip 1
                |> Seq.map byte

            Seq.append acc [| (label, data) |])
        [||]
    |> Array.ofSeq
    |> Array.unzip

let byteRowsToBitArrays (threshold: byte) (byteRows: array<seq<byte>>) : array<BitArray> =
    byteRows
    |> Array.map ((Seq.map (fun v -> v >= threshold)) >> Array.ofSeq >> BitArray)

let weightsToMatrix (weights: Weights) : array<array<int>> =
    weights
    |> Array.map (fun (plusBits, minusBits) ->
        let plusBools: array<Boolean> = Array.zeroCreate plusBits.Count
        plusBits.CopyTo(plusBools, 0)
        let minusBools: array<Boolean> = Array.zeroCreate minusBits.Count
        minusBits.CopyTo(minusBools, 0)

        Array.map2
            (fun plus minus ->
                match plus, minus with
                | true, true -> 0
                | true, false -> 1
                | false, true -> -1
                | false, false -> 0)
            plusBools
            minusBools)
