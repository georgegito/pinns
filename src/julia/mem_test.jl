using Dates

const mem_size = 0x1_000_000

counter = let
    counter = 0
    start_t = now()

    while (now() - start_t) / Millisecond(1000) < 15
        counter += 1
        arr = Vector{Int32}(undef, mem_size)
        arr[rand(1:mem_size)] = rand(Int32, 1)[begin]
        # sleep(1)
    end
    counter
end

@show counter, mem_size

# gtime --verbose julia --heap-size-hint=100m mem_test.jl