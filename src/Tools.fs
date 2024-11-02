module Tools

open System.IO

open FSharpPlus

let loadMnist (path: string) : string array * byte seq array =
    File.ReadAllText(path)
    |> String.split [ "\n" ]
    // remove header row
    |> Seq.skip 1
    |> Seq.map (String.split [ "," ])
    |> fold
        (fun acc row ->
            let label = Seq.head row
            let data = Seq.skip 1 row |> Seq.map byte
            Seq.append acc [| (label, data) |])
        [||]
    |> Array.ofSeq
    |> Array.unzip

let binarize (threshold: int) (bytes: byte seq) : byte seq =
    let threshold = byte threshold
    bytes |> Seq.map (fun value -> if value >= threshold then 255uy else 0uy)

let toRectangleSvg (size: float) (columns: int) (bytes: byte seq) : string =
    let array = Array.ofSeq bytes
    let rows = array.Length / columns

    let mutable svg =
        $"""<svg width="{(float columns) * size}" height="{(float rows) * size}" viewBox="0 0 {columns} {rows}" xmlns="http://www.w3.org/2000/svg">\n"""

    for y = 0 to rows - 1 do
        for x = 0 to columns - 1 do
            let v = array[x + y * rows]

            svg <- $"""{svg}<rect x="{x}" y="{y}" width="1" height="1" fill="rgb({v},{v},{v})"/>"""

        svg <- svg + "\n"

    svg + "</svg>"


let toSquareSvg (size: float) (bytes: byte seq) : string =
    let array = Array.ofSeq bytes
    let side = array.Length |> sqrt

    toRectangleSvg size side array
