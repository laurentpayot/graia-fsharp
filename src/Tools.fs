module Tools

open System.IO

open FSharpPlus


let loadMnistCsv (path: string) : string array * int seq array =
    File.ReadAllText(path)
    |> String.split [ "\n" ]
    // remove header row
    |> Seq.skip 1
    |> Seq.map (String.split [ "," ])
    |> fold
        (fun acc row ->
            let label = Seq.head row
            let data = Seq.skip 1 row |> Seq.map int
            Seq.append acc [| (label, data) |])
        [||]
    |> Array.ofSeq
    |> Array.unzip
