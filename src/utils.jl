function make_count_matrix(counts)
    idx = findall(counts .> 0)
    countmat = Array{Int}(undef, length(idx), 2)
    countmat[:, 1] = idx
    countmat[:, 2] = counts[idx]
    return countmat
end

function format_counts_lda(countsdf::DataFrame)
    counts = Matrix{Int}[]
    for lib in names(countsdf)
        if lib == :term
            continue
        end

        colcounts = convert(Array, countsdf[lib])
        countmat = make_count_matrix(colcounts)

        push!(counts, countmat)
    end
    return counts
end

function format_counts_ctm(countsdf::DataFrame, cols::Vector{Symbol})
    return format_counts_mmctm([countsdf], cols)
end

function format_counts_mmctm(countdfs::Vector{DataFrame}, cols::Vector{Symbol})
    counts = Vector{Matrix{Int}}[]
    for col in cols
        doc_counts = Matrix{Int}[]
        for df in countdfs
            push!(doc_counts, make_count_matrix(convert(Array, df[!, col])))
        end

        push!(counts, doc_counts)
    end

    return counts
end
