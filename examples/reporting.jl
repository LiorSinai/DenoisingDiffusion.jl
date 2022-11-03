using Flux: onecold
using Printf

"""
    confusion_matrix(ŷ, y, labels)
    
Rows are ground truth labels, columns are predicted labels.
`y` and `ŷ` are either both vectors or both matrices.
"""
function confusion_matrix(ŷ::AbstractVector, y::AbstractVector, labels)
    nlabels = length(labels) 
    cm = zeros(Int, nlabels, nlabels)
    for (i, groundtruth) in enumerate(labels)
        idxs = (y .== groundtruth)
        for (j, pred) in enumerate(labels)
            cm[i, j] = sum(pred .== ŷ[idxs])
        end
    end
    cm
end

confusion_matrix(ŷ::AbstractMatrix, y::AbstractMatrix, labels) = confusion_matrix(onecold(ŷ), onecold(y), labels)

function recall(cm::AbstractMatrix; target=1)
    tp = cm[target, target]
    tp / sum(cm[target, :])
end

function precision(cm::AbstractMatrix; target=1)
    tp = cm[target, target]
    fp = sum(cm[:, target]) - tp
    tp / (tp + fp)
end

function f1_score(cm::AbstractMatrix; target=1)
    r = recall(cm, target=target)
    p = precision(cm, target=target)
    (p==0 || r==0) ? 0 : 2 * p * r / (r + p) # == 2/(1/r + 1/p)
end

function classification_report(cm::AbstractMatrix, labels=1:size(cm, 1))
    @assert size(cm, 1) == size(cm, 2) "matrix must be square"
    report = Dict{Int, Dict{String, Float64}}()
    report[-1] = Dict{String, Float64}("precision" => 0.0, "recall"=>0.0, "f1"=>0.0, "support"=>sum(cm))
    for i in 1:size(cm, 1)
        rt = Dict{String, Float64}()
        rt["support"] = sum(cm[i, :])
        rt["precision"] = precision(cm, target=i)
        rt["recall"] = recall(cm, target=i)
        rt["f1"] = f1_score(cm, target=i)
        report[i] = rt
        report[-1]["precision"] += rt["support"] * rt["precision"]
        report[-1]["recall"] += rt["support"] * rt["recall"]
        report[-1]["f1"] += rt["support"] * rt["f1"]
    end
    @printf("%12s  precision  recall  f1-score  support\n", "")
    for i in 1:size(cm, 1)
        rt = report[i]
        @printf("%12s  %9.2f  %6.2f  %8.2f  %7d\n", 
            string(labels[i]), rt["precision"], rt["recall"], rt["f1"], rt["support"]
        )
    end
    println("")
    weighted = report[-1]
    weighted["precision"] /= weighted["support"]
    weighted["recall"] /= weighted["support"]
    weighted["f1"] /= weighted["support"]
    @printf("%12s  %9.2f  %6.2f  %8.2f  %7d", 
            "weighted avg", weighted["precision"], weighted["recall"], weighted["f1"], weighted["support"]
        )
end