﻿module Tools

open System.IO
open System
open System.Collections

open FSharpPlus


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

let bitArraysToMatrix (bitArrays: array<BitArray>) : array<array<byte>> =
    bitArrays
    |> Array.map (fun ba ->
        let bools: array<Boolean> = Array.zeroCreate (ba.Count)
        ba.CopyTo(bools, 0)
        bools |> Array.map (fun b -> if b then 1uy else 0uy))
