module Tools

open System.IO
open System
open System.Collections

open FSharpPlus


let loadMnistCsv (path: string) : array<int> * array<seq<int>> =
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
                |> Seq.map int

            Seq.append acc [| (label, data) |])
        [||]
    |> Array.ofSeq
    |> Array.unzip

let intRowsToBitArrays (threshold: int) (intRows: array<seq<int>>) : array<BitArray> =
    intRows
    |> Array.map ((Seq.map (fun v -> v >= threshold)) >> Array.ofSeq >> BitArray)
