module Graia

printfn "🌄 Graia v0.0.1"

open System.Runtime.Intrinsics
printfn $"Vector128: {Vector128.IsHardwareAccelerated}"
printfn $"Vector256: {Vector256.IsHardwareAccelerated}"
printfn $"Vector512: {Vector512.IsHardwareAccelerated}"

open System.Collections

let a: BitArray = BitArray(3)
let b: BitArray = BitArray(3)

a.Set(0, true)
b.Set(2, true)

for x in a do
    printfn $"%A{x}"

printfn $"a = %A{a}"
printfn $"b = %A{b}"
