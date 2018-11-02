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

		colcounts = convert(Array, counts_df[lib])
		countmat = makecountmat(colcounts)

        push!(counts, countmat)
    end
    return counts
end

function format_counts_ctm(countsdf::DataFrame)
	counts = Vector{Matrix{Int}}[]
    for lib in names(countsdf)
        if lib == :term
            continue
        end

		colcounts = convert(Array, countsdf[lib])
		countmat = makecountmat(colcounts)

		push!(counts, [countmat])
	end
	return counts
end

function format_counts_mmctm(snv_countsdf::DataFrame,
        sv_countsdf::DataFrame)

    counts = Vector{Matrix{Int}}[]
    for lib in names(snv_countsdf)
        if lib == :term
            continue
        end

        snvcounts = convert(Array, snv_countsdf[lib])
        snvcountmat = make_count_matrix(snvcounts)

        svcounts = convert(Array, sv_countsdf[lib])
        svcountmat = make_count_matrix(svcounts)

        push!(counts, [snvcountmat, svcountmat])
    end

    return counts
end
