using CSV
using DataFrames
using Plots

pyplot()

function plot_results(file, label)
    df = DataFrame(CSV.File(file))

    N = df[!, :N]
    mean_runtime = df[!, :runtime] .* 1e3 # in ps

    tflops = (2 .* N .^ 3) ./ mean_runtime

    plot!(N, tflops, label=label, xscale=:log2, markershape=:circle)
end

plot_results("cudanative.csv", "CUDAnative")
plot_results("cublas.csv", "cuBLAS")
plot_results("cutlass-wmma.csv", "CUTLASS (WMMA)")

title!("Performance of mixed-precision GEMM\nProblem size: N x N x N")
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
