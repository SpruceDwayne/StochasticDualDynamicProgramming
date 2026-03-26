using Plots

# Beta-binomial based probability vector g(t, n)
#
# t ∈ [0, 1]: t≈0 shifts mass toward low indices, t≈1 toward high indices.
# n: number of probabilities returned (beta-binomial with N = n-1 trials).
#
# The sigmoid maps t ∈ [0,1] so that t=0 → m≈0.007 and t=1 → m≈0.993,
# matching the original [1,100] parameterization at its endpoints.

m(t) = 1.0 / (1.0 + exp(-10.0 * (t - 0.5)))

c = 8.0

α(t) = c * m(t)
β(t) = c * (1.0 - m(t))

# Rising factorial: x^{(k)} = x(x+1)···(x+k-1), with x^{(0)} = 1
rising_factorial(x, k) = k == 0 ? 1.0 : prod(x + i for i in 0:(k-1))

# Beta-binomial PMF with N = n-1 trials, shape parameters α(t), β(t).
# P(k) = C(N,k) · α^{(k)} · β^{(N-k)} / (α+β)^{(N)},  k = 0,…,N
function g(t, n)
    a = α(t)
    b = β(t)
    N = n - 1
    denom = rising_factorial(a + b, N)
    return [binomial(N, k) * rising_factorial(a, k) * rising_factorial(b, N - k) / denom
            for k in 0:N]
end

# Print values for n = 3 (recovers original behaviour)
println("=== n = 3 ===")
for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    println("t = $(t): ", round.(g(t, 3); digits=4))
end

println("\n=== n = 5 ===")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]
    println("t = $(t): ", round.(g(t, 5); digits=4))
end

# ---- Plot section ----
n = 20
x = 1:n
plot()

for t in [0.1,0.2, 0.3,0.4,0.5,0.6, 0.7]#, 0.8, 0.9]
    y = g(t, n)
    plot!(x, y, label="t=$t", marker=:o)
end

xlabel!("Index")
ylabel!("Probability")
title!("Beta-binomial probability vectors (n=$n) at selected t")