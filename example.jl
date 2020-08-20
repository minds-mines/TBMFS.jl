using MLJ
using CategoricalArrays
using Plots

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
X = randn(1000, 100)
Xcut = [1:40, 41:90, 91:100] # X contains three modalities, here's how to cut
y = recode(rand(0:1, 1000), 0=>"HC/MCI", 1=>"AD")
y = CategoricalArray(y)

include("TBMFS.jl")
tbmfs = TBMFSClassifier(Xcut=Xcut,
                        C = 1.0,
                        αₘ=[1, 1, 1],
                        γₘ=[1, 1, 1],
                        r=10,
                        μ=1e-4,
                        ρ=1.1,
                        maxiter=3000,
                        tol=1e-4)

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
objective = report(tbmfs_classifier)[1]
plot(objective, xlabel="Iteration", ylabel="Objective, see Eq. (5)", legend=false)
