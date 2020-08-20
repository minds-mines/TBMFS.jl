import MLJBase

using DataFrames
using MLLabelUtils
using LinearAlgebra
using MLJ

using Flux: onehotbatch, onecold

"""
Variables associated with the Task Balanced Multimodal Feature Selection Classifier.
"""
mutable struct TBMFSClassifier <: MLJBase.Deterministic
    Xcut::Array{UnitRange{Int64},1} # How to "cut" X into multimodal data
    C::Float64                      # SVM Regularization
    αₘ::Array{Float64, 1}           # Modality reconstruction hyperparameter
    γₘ::Array{Float64, 1}           # Row-sparsity hyperparamter for each Bₘ
    r::Int64                        # Matrix factorization parameter (inner dimension)
    μ::Float64                      # Augmented Lagrangian penalty parameter
    ρ::Float64                      # Scaling factor on μ
    maxiter::Int64                  # Maximum number of iterations
    tol::Float64                    # Stopping criterion
end

"""
Default constructor for the Task Balanced Multimodal Feture Selection Classifier.

TODO: Add checks to hyperparameters.
"""
function TBMFSClassifier(; Xcut=missing, C=1.0, αₘ=missing, γₘ=missing, r=missing, μ=0.001, ρ=1.2, maxiter=50, tol=1e-5)
    model = TBMFSClassifier(Xcut, C, αₘ, γₘ, r, μ, ρ, maxiter, tol)
    return model
end

"""
Calculate Task Balanced Multimodal Feature Selection objective loss. Equation (5)
"""
function objloss!(obj𝓛s, i, Xₘ, Y_hot, W, b, Z, Bₘ, αₘ, γₘ, C)
    for m in 1:length(Xₘ)
        obj𝓛s[i] += αₘ[m] * l21norm(Xₘ[m] - Bₘ[m]*Z)
        obj𝓛s[i] += γₘ[m] * l21norm(Bₘ[m])
    end
    obj𝓛s[i] += 0.5 * norm(W)^2
    obj𝓛s[i] += C * sum(max.(1 .- (W' * Z .+ b) .* Y_hot, 0))
end

"""
Calculates the Lagrangian loss. Equation (9)
"""
function loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
    𝓛s[i] = 0.0
    Es[i] = 0.0
    Fs[i] = 0.0
    B̂s[i] = 0.0
    Ẑs[i] = 0.0

    𝓛s[i] += C * sum(max.(Y_hot .* E, 0))
    𝓛s[i] += 0.5 * norm(W)^2
    𝓛s[i] += μ/2.0 * norm(E .- Y_hot .+ W'*Z .+ b .+ η/μ)^2
    𝓛s[i] += μ/2.0 * norm(Z - Ẑ + Ω/μ)^2

    Es[i] += norm(E .- Y_hot .+ W' * Z .+ b)
    Ẑs[i] += norm(Z - Ẑ)

    for m in 1:length(Xₘ)
        𝓛s[i] += αₘ[m] * l21norm(Fₘ[m])
        𝓛s[i] += γₘ[m] * l21norm(B̂ₘ[m])
        𝓛s[i] += μ/2.0 * norm(Fₘ[m] - Xₘ[m] + Bₘ[m]*Z + Θₘ[m]/μ)^2
        𝓛s[i] += μ/2.0 * norm(Bₘ[m] - B̂ₘ[m] + Λₘ[m]/μ)^2

        Fs[i] += norm(Fₘ[m] - Xₘ[m] + Bₘ[m]*Z)
        B̂s[i] += norm(Bₘ[m] - B̂ₘ[m])
    end
end

"""
Calculates the ℓ_{2,1} norm for the matrix X.
"""
function l21norm(X)
    res = 0.0
    for row ∈ eachrow(X)
        res += norm(row, 2)
    end
    return res
end

"""
Initializes variables for optimization.

Note that all the variables in X come in concatenated together.
Thus, they must be "cut" into multiple modalities.
"""
function init_var(model::TBMFSClassifier, X, Y)
    K = MLJBase.nrows(classes(Y[1]))
    x = MLJBase.matrix(X)
    N, _ = size(x)

    Xₘ = [x[:,cut]' for cut in model.Xcut]
    Xₘ = [vcat(xₘ, ones(size(xₘ)[2])') for xₘ in Xₘ] # Add intercept to each modality.
    Y_onehot = onehotbatch(Y, classes(Y[1])) .* 2.0 .- 1.0

    M = length(Xₘ)
    dₘ = [size(Xₘ[m])[1] for m in 1:M]

    E = randn(K, N)
    W = randn(model.r, K)
    b = randn(K)
    Z = randn(model.r, N)
    Ẑ = randn(model.r, N)
    Fₘ = [randn(dₘ[i], N) for i in 1:M]
    Bₘ = [randn(dₘ[i], model.r) for i in 1:M]
    B̂ₘ = [randn(dₘ[i], model.r) for i in 1:M]

    Θₘ = [zeros(dₘ[i], N) for i in 1:M]
    Λₘ = [zeros(dₘ[i], model.r) for i in 1:M]
    Ω = zeros(model.r, N)
    η = zeros(K, N)

    return Xₘ, Y_onehot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, N, M
end

"""
Updates each w_k. Equation (11)
"""
function update_W!(W, Y_hot, E, b, Z, η, μ)
    _, K = size(W)
    r, N = size(Z)
    ∑zᵢzᵢᵀ = zeros(r, r)

    for i in 1:N
        ∑zᵢzᵢᵀ += Z[:,i] * Z[:,i]'
    end

    #W[:] = (∑zᵢzᵢᵀ + I/μ)' \ ((Y_hot .- E .- b .- η/μ) * Z')'
    W[:] = (((Y_hot .- E .- b .- η/μ) * Z') * pinv(∑zᵢzᵢᵀ + I/μ))'
end

"""
Updates each b_k. Equation (12)
"""
function update_b!(b, Y_hot, E, W, Z, η, μ)
    K, N = size(Y_hot)

    b[:] = sum(Y_hot .- E .- W' * Z .- η / μ, dims=2) ./ N
end

"""
Updates the each e_{im}. Equation (14)
"""
function update_E!(E, Y_hot, W, b, Z, η, C, μ)
    S = Y_hot .- (W'*Z  .+ b) - η/μ

    gt = (Y_hot .* S) .> C/μ
    mid = 0 .<= (Y_hot .* S) .<= C/μ

    E[:] = S .* .!mid - gt .* Y_hot .* (C/μ)
end

"""
Updates Z. Equation (16)
"""
function update_Z!(Z, Xₘ, Y_hot, E, W, b, Ẑ, Fₘ, Bₘ, Θₘ, Ω, η, μ)
    M = length(Xₘ)
    r, N = size(Z)
    _, K = size(W)

    Tₘ = [x - f - θ/μ for (x, f, θ) in zip(Xₘ, Fₘ, Θₘ)]
    U = Y_hot .- E .- b .- η/μ

    ∑BₘᵀBₘ = zeros(r, r)
    ∑BₘᵀT = zeros(r, N)
    ∑wₖwₖᵀ = zeros(r, r)
    ∑wₖU = W * U

    for m in 1:M
        ∑BₘᵀBₘ += Bₘ[m]' * Bₘ[m]
        ∑BₘᵀT += Bₘ[m]' * Tₘ[m]
    end

    for k in 1:K
        ∑wₖwₖᵀ += W[:,k] * W[:,k]'
    end

    #Z[:] = (∑BₘᵀBₘ + ∑wₖwₖᵀ + I) \ (∑BₘᵀT + ∑wₖU + Ẑ - Ω/μ)
    Z[:] = pinv(∑BₘᵀBₘ + ∑wₖwₖᵀ + I) * (∑BₘᵀT + ∑wₖU + Ẑ - Ω/μ)
end

"""
Updates Ẑ. Equation (18)
"""
function update_Ẑ!(Ẑ, Z, Ω, μ)
    U, Σ, V = svd(Z + Ω/μ)
    Ẑ[:] = U*V'
end

"""
Updates Fₘ. Equation (20)
"""
function update_Fₘ!(Fₘ, Xₘ, Z, Bₘ, Θₘ, μ, αₘ, Lₘ)
    Lₘ = Xₘ - Bₘ*Z - Θₘ/μ
    threshold = [max(1 - αₘ/(μ*norm(lⁱ) + 1e-12), 0) for lⁱ ∈ eachrow(Lₘ)]
    Fₘ[:] = Lₘ .* threshold
end

"""
Updates B̂ₘ. Equation (22)
"""
function update_B̂ₘ!(B̂ₘ, Bₘ, Λₘ, μ, γₘ)
    Oₘ = Bₘ + Λₘ/μ
    threshold = [max(1 - γₘ/(μ*norm(oⁱ) + 1e-12), 0) for oⁱ ∈ eachrow(Oₘ)]
    B̂ₘ[:] = Oₘ .* threshold
end

"""
Updates Bₘ. Equation (24)
"""
function update_Bₘ!(Bₘ, Xₘ, Z, Fₘ, B̂ₘ, Θₘ, Λₘ, μ)
    #Bₘ[:] = ((Z*Z' + I) \ (-(Fₘ-Xₘ+Θₘ/μ)*Z' + (B̂ₘ - Λₘ/μ))')' 
    Bₘ[:] =  (-(Fₘ-Xₘ+Θₘ/μ)*Z' + (B̂ₘ - Λₘ/μ)) * pinv(Z*Z' + I)
end

"""
Fits a Task Balanced Multimodal Feature Selection Classifier from X --> Y.

NOTE: checking the Lagrangian loss function (commented out) before 
and after an update is a nice way to debug a complex multiblock ADMM 
algorithm such as this one. 

TODO: decrease the verbosity of these checks.
"""
function MLJBase.fit(model::TBMFSClassifier, verbosity::Integer, X, y)
    Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, N, M = init_var(model, X, y)

    C = model.C
    αₘ = model.αₘ
    γₘ = model.γₘ
    μ = model.μ

    Lₘ = [zeros(size(x)) for x in Xₘ]

    obj𝓛s = zeros(model.maxiter)
    𝓛s = zeros(model.maxiter)
    Es = zeros(model.maxiter)
    Fs = zeros(model.maxiter)
    B̂s = zeros(model.maxiter)
    Ẑs = zeros(model.maxiter)
    prev_𝓛 = Inf

    for i in 1:model.maxiter
        for m in 1:M
            # Update Fₘ
            # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
            # l1 = 𝓛s[i]
            update_Fₘ!(Fₘ[m], Xₘ[m], Z, Bₘ[m], Θₘ[m], μ, αₘ[m], Lₘ)
            # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
            # l2 = 𝓛s[i]
            # if verbosity>10 println("checking Fₘ ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end

            # Update Bₘ
            # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
            # l1 = 𝓛s[i]
            update_Bₘ!(Bₘ[m], Xₘ[m], Z, Fₘ[m], B̂ₘ[m], Θₘ[m], Λₘ[m], μ)
            # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
            # l2 = 𝓛s[i]
            # if verbosity>10 println("checking Bₘ ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end

            # Update B̂ₘ
            # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
            # l1 = 𝓛s[i]
            update_B̂ₘ!(B̂ₘ[m], Bₘ[m], Λₘ[m], μ, γₘ[m])
            # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
            # l2 = 𝓛s[i]
            # if verbosity>10 println("checking B̂ₘ ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end
        end

        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l1 = 𝓛s[i]
        update_E!(E, Y_hot, W, b, Z, η, C, μ)
        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l2 = 𝓛s[i]
        # if verbosity>10 println("checking E ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end

        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l1 = 𝓛s[i]
        update_Z!(Z, Xₘ, Y_hot, E, W, b, Ẑ, Fₘ, Bₘ, Θₘ, Ω, η, μ)
        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l2 = 𝓛s[i]
        # if verbosity>10 println("checking Z ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end

        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l1 = 𝓛s[i]
        update_Ẑ!(Ẑ, Z, Ω, μ)
        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l2 = 𝓛s[i]
        # if verbosity>10 println("checking Ẑ ", l2, " <= ", l1, " ", l2 <= l1) end; #@assert l2 <= l1 + 1e-8 end

        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l1 = 𝓛s[i]
        update_W!(W, Y_hot, E, b, Z, η, μ)
        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l2 = 𝓛s[i]
        # if verbosity>10 println("checking W ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end

        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l1 = 𝓛s[i]
        update_b!(b, Y_hot, E, W, Z, η, μ)
        # loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        # l2 = 𝓛s[i]
        # if verbosity>10 println("checking b ", l2, " <= ", l1, " ", l2 <= l1); @assert l2 <= l1 + 1e-8 end

        for m in 1:M
            # Update Θₘ
            Θₘ[m] = Θₘ[m] + μ * (Fₘ[m] - Xₘ[m] + Bₘ[m]*Z)
            # Update Λₘ
            Λₘ[m] = Λₘ[m] + μ * (Bₘ[m] - B̂ₘ[m])
        end
        # Update η
        η = η + μ * (E .- Y_hot .+ W' * Z .+ b)
        # Update Ω
        Ω = Ω + μ * (Z - Ẑ)

        # Update μ
        μ = μ*model.ρ

        loss!(𝓛s, Es, Fs, B̂s, Ẑs, i, Xₘ, Y_hot, E, W, b, Z, Ẑ, Fₘ, Bₘ, B̂ₘ, Θₘ, Λₘ, Ω, η, αₘ, γₘ, C, μ)
        objloss!(obj𝓛s, i, Xₘ, Y_hot, W, b, Z, Bₘ, αₘ, γₘ, C)

        if verbosity>5 println("LOSS: ", obj𝓛s[i]) end
        if abs(prev_𝓛 - 𝓛s[i]) < model.tol
            break
        end
        prev_𝓛 = 𝓛s[i]
    end

    fitresult = (Bₘ, W, b, classes(y[1]))
    cache = missing
    report = (losses=(obj𝓛s, 𝓛s, Es, Fs, B̂s, Ẑs))

    return fitresult, cache, report
end

"""
Given new multimodal data (in Xnew) give a classification prediction
using the fitresult.
"""
function MLJBase.predict(model::TBMFSClassifier, fitresult, Xnew)
    DEBUG=false
    xnew = MLJBase.matrix(Xnew)

    Bₘ, W, b, c = fitresult

    Xnewₘ = [xnew[:,cut]' for cut in model.Xcut]
    Xnewₘ = [vcat(xnewₘ, ones(size(xnewₘ)[2])') for xnewₘ in Xnewₘ]

    Znew = zeros(model.r, size(xnew)[1])

    for m in 1:length(Xnewₘ)
        Zpartial = model.αₘ[m] * pinv(Bₘ[m]) * Xnewₘ[m]
        if DEBUG
            display(Zpartial)
        end
        Znew += Zpartial
    end
    Znew = Znew ./ sum(model.αₘ)
    pred = W'*Znew .+ b

    return onecold(pred, c)
end
