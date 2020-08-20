using MLJ
using CategoricalArrays
using Plots
using Plots.PlotMeasures
using LaTeXStrings

import MLJBase

"""
Parameters for the Task Balanced Multimodal Feature Selection Classifier

Xcut::Array{UnitRange{Int64},1}  # How to "cut" X into multimodal data
C::Float64                       # SVM regularization
αₘ::Array{Float64, 1}            # Modality reconstruction hyperparameter
γₘ::Array{Float64, 1}            # Row-sparsity hyperparamter for each Bₘ
r::Int64                         # Matrix factorization parameter (inner dimension)
μ::Float64                       # Augmented Lagrangian weighting factor
ρ::Float64                       # Scaling factor on μ
maxiter::Int64                   # Maximum number of iterations
tol::Float64                     # Stopping criterion

This script trains the TBMFSClassifier on dummy X and y data and plots
the objective value over time.
"""

# Creating some synthetic multimodal data
N = 90
D = 100

X = randn(N, D)
Xcut = [1:40, 41:90, 91:100] # X contains three modalities, here's how to cut
y = recode(rand(0:1, N), 0=>"HC/MCI", 1=>"AD")
y = CategoricalArray(y)

include("TBMFS.jl")
tbmfs = TBMFSClassifier(Xcut=Xcut,
                        C = 1.0,
                        αₘ=[1, 1, 1],
                        γₘ=[1, 1, 1],
                        r=10,
                        μ=1e-4,
                        ρ=1.1,
                        maxiter=1000,
                        tol=1e-8)

tbmfs_classifier = machine(tbmfs, X, y)

# You can either use the fit! and predict methods, this is helpful when debugging the method.
fit!(tbmfs_classifier)
predict(tbmfs_classifier, X)

# Or follow an evaluation scheme like so
evaluate!(tbmfs_classifier, 
          resampling=CV(nfolds=6, shuffle=true), 
          measure=[mcc], 
          verbosity=10)
# See https://alan-turing-institute.github.io/MLJ.jl/stable/ for additional details

# Once the model is trained we can plot its objective
objective = report(tbmfs_classifier).losses[1]
eq5 = plot(objective, label="Eq. (5)", xlabel="Iteration", ylabel="Objective", title="Convergence")

# We can also plot the differences between our introduced variables
Es = report(tbmfs_classifier).losses[3]
Fs = report(tbmfs_classifier).losses[4]
Bs = report(tbmfs_classifier).losses[5]
Zs = report(tbmfs_classifier).losses[6]

diffs = plot(Es, label=L"\sum || e_{ik} - (y_{ik} - (\mathbf{w}_k^T\mathbf{z}_i+b_k)) || ", yaxis=:log, xlabel="Iteration", ylabel="Difference", title="Constraint Differences")
plot!(Fs, label=L"\sum || \mathbf{F}_m - \left(\mathbf{X}_m - \mathbf{B}_m\mathbf{Z}\right) || ")
plot!(Bs, label=L"\sum || \hat{\mathbf{B}}_m - \mathbf{B}_m ||")
plot!(Zs, label=L" || \hat{\mathbf{Z}} - \mathbf{Z} || ")

# Check that Z meets constraints too
Z = report(tbmfs_classifier).Z
hm = heatmap(Z*Z', yflip=true, legendfontsize=2, title=L"\mathbf{Z}\mathbf{Z}^T = \mathbf{I}")

plot(eq5, diffs, hm, layout = (1, 3), size=(1500, 400))
plot!(legend=:topright)
plot!(margin=5mm)


