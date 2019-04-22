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

        #
        colcounts = convert(Array, countsdf[lib])
        countmat = make_count_matrix(colcounts)

        push!(counts, countmat)
    end
    return counts
end

function format_counts_ctm(countsdf::DataFrame)
    return format_counts_mmctm(countsdf)
end

function format_counts_mmctm(countsdfs::DataFrame...)
    counts = Vector{Matrix{Int}}[]

    if length(countsdfs) == 0
        return counts
    end

    for lib in names(countsdfs[1])
        if lib == :term
            continue
        end

        libcounts = Matrix{Int}[]
        for countsdf in countsdfs
            modality_counts = convert(Array, countsdf[lib])
            modality_countmat = make_count_matrix(modality_counts)
            push!(libcounts, modality_countmat)
        end

        push!(counts, libcounts)
    end

    return counts
end
