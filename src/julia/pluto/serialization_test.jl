### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 39d61cb2-96a4-11ee-010c-c5c0eaaf80f2
using Serialization

# ╔═╡ ca95d91c-ae17-4602-996f-a02569d18e27
A = [1 2 3 ; 4 5 6]

# ╔═╡ 1a9df39c-33ce-4877-9294-92125c83e2ae
serialize("/Users/ggito/repos/pinns/src/julia/pluto/test", A)

# ╔═╡ acb92b46-b9f6-4dd5-b230-358fdd63ef25
test = deserialize("/Users/ggito/repos/pinns/src/julia/pluto/test")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "a94b38100edafdc55c04e0d083f4a0eb4fb3c631"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
"""

# ╔═╡ Cell order:
# ╠═39d61cb2-96a4-11ee-010c-c5c0eaaf80f2
# ╠═ca95d91c-ae17-4602-996f-a02569d18e27
# ╠═1a9df39c-33ce-4877-9294-92125c83e2ae
# ╠═acb92b46-b9f6-4dd5-b230-358fdd63ef25
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
